#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>

#include "sputnik/vector_utils.h"
#include "dhaziza_gemm.h"
#include "cudaProfiler.h"


#include <cstdio>
#include <cstdlib>
#include <inttypes.h>

/*
Loops unrolling
Vec load

Gotchas:
- Don't use at::TensorAccessor in a loop - super slow
*/


namespace {


template <typename P>
class PredicatedTileIteratorFromSharedMemory : public P {
 public:

    using P::P;

  CUTLASS_DEVICE
  void load_with_byte_offset(typename P::Fragment &frag, typename P::LongIndex byte_offset) {

    using AccessType = typename P::AccessType;
    constexpr auto kAccessesPerVector = P::kAccessesPerVector;

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < P::ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < P::ThreadMap::Iterations::kContiguous; ++c) {

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * P::ThreadMap::Iterations::kContiguous);
          
          P::address_iterator_.set_iteration_index(idx);
          char const *byte_ptr = reinterpret_cast<char const *>(P::address_iterator_.get()) + byte_offset;

          AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_ptr);
          if (P::address_iterator_.valid()) {
              frag_ptr[idx] = *access_ptr;
          }

        //   cutlass::arch::global_load<AccessType,
        //                              sizeof(AccessType)
        //                             >(
        //       frag_ptr[idx], access_ptr, address_iterator_.valid());

          ++P::address_iterator_;
        }
      }
    }
  }
};

template <typename scalar_t_, bool compute_logsumexp_>
struct AttentionKernel {
    using scalar_t = scalar_t_;
    static constexpr bool compute_logsumexp = compute_logsumexp_;

    // Blocks
    static constexpr int64_t kQueriesPerBlock = 16;
    static constexpr int64_t kNumBlocksX = 1;
    static int64_t getNumBlocksY(int64_t num_queries) {
        return ceil_div(num_queries, kQueriesPerBlock);
    }
    // Threads
    static constexpr int64_t kWarpSize = 32;
    static constexpr int64_t kNumWarpsPerBlock = 4;
    static constexpr int64_t kKeysPerWarp = kWarpSize;

    static constexpr int64_t kSiDim1 = kNumWarpsPerBlock * kKeysPerWarp;

    static_assert(kKeysPerWarp % kWarpSize == 0);

    // Sort of sanitizers to ensure we always access within bounds
    template <int kDim>
    struct ArrayWithBoundsChecks {
        using TYPE = scalar_t[kDim];
        __device__ ArrayWithBoundsChecks(TYPE& data, int64_t offset = 0): _data(data), _offset(offset) {}

        TYPE& _data;
        int64_t _offset;

        __device__ scalar_t& operator[](int64_t dim) {
            assert(dim < kDim);
            return _data[_offset + dim];
        }
    };

    template <int kDimFirst, int kDimSecond>
    struct ArrayWithBoundsChecks2d {
        using TYPE = scalar_t[kDimFirst][kDimSecond];
        __device__ ArrayWithBoundsChecks2d(TYPE& data): _data(data) {}

        TYPE& _data;
        __device__ ArrayWithBoundsChecks<kDimSecond> operator[](int64_t dim1) {
            assert(dim1 < kDimFirst);
            return ArrayWithBoundsChecks<kDimSecond>(_data[0], dim1 * kDimSecond);
        }
    };

    static void __device__ attention_kernel(
        at::TensorAccessor<scalar_t, 2> output,
        at::TensorAccessor<scalar_t, 1> logsumexp,
        at::TensorAccessor<scalar_t, 2> query,
        at::TensorAccessor<scalar_t, 2> key,
        at::TensorAccessor<scalar_t, 2> value) {
        // int64_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
        // int64_t block_dim = blockDim.x * blockDim.y;
    
        int8_t lane_id = threadIdx.x;
        int8_t warp_id = threadIdx.y;

        // In this block, we will only ever:
        // - read query[query_start:query_end, :]
        // - write to output[query_start:query_end, :]
        // int64_t query_start = blockIdx.y * kQueriesPerBlock;
        // int64_t query_end = (blockIdx.y + 1) * kQueriesPerBlock;

        int32_t num_keys = key.size(0);
        int32_t num_values = value.size(0);
        int32_t num_queries = query.size(0);
        int32_t K = key.size(1);

        scalar_t __shared__ m_prime[kQueriesPerBlock];
        scalar_t __shared__ mi[kQueriesPerBlock][kNumWarpsPerBlock];
        scalar_t __shared__ s_prime[kQueriesPerBlock];
        scalar_t __shared__ si[kQueriesPerBlock][kSiDim1];
        // ArrayWithBoundsChecks<kQueriesPerBlock> m_prime(m_prime_);
        // ArrayWithBoundsChecks2d<kQueriesPerBlock, kNumWarpsPerBlock> mi(mi_);
        // ArrayWithBoundsChecks<kQueriesPerBlock> s_prime(s_prime_);
        // ArrayWithBoundsChecks2d<kQueriesPerBlock, kNumWarpsPerBlock * kKeysPerWarp> si(si_);

        for (int32_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) {
            mi[q + lane_id][warp_id] = -std::numeric_limits<scalar_t>::infinity();
        }
        if (warp_id == 0) {
            for (int32_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) {
                s_prime[q + lane_id] = 0;
                m_prime[q + lane_id] = -std::numeric_limits<scalar_t>::infinity();
            }
        }

        // Iterate through keys
        for (int32_t iter_key_start = 0; iter_key_start < num_keys; iter_key_start += kNumWarpsPerBlock * kKeysPerWarp) {
            // int64_t iter_key_end = iter_key_start + kNumWarpsPerBlock * kKeysPerWarp;

            // TODO(half): Shared memory banks are organized such that successive 32-bit words are assigned to successive banks and the bandwidth is 32 bits per bank per clock cycle

            __syncthreads(); // Need to have shared memory initialized, and `m_prime` updated from end of prev iter

            // 1. Compute dot-product into shared memory for each query
            compute_dot_product_qk(iter_key_start, query, key, m_prime, si, mi);

            __syncthreads(); // `mi` calculation done based on warp-data

            // 2b. Aggregate max across different warps
            for (int32_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) { // parallel lanes
                scalar_t global_max = mi[q + lane_id][0];
                for(int32_t other_warp = 0; other_warp < kNumWarpsPerBlock; ++other_warp) {
                    global_max = std::max(global_max, mi[q + lane_id][other_warp]);
                }
                mi[q + lane_id][warp_id] = global_max;
            }

            __syncthreads(); // `mi` calculation done based on block data. `mi[a][i] == mi[a][j]` for all (a, i, j)

            // TODO: Maybe this could be parallelized across warps
            // WARNING: This modifies `si` and `m_prime` to store the precalculated exp version
            // so we can reuse it later in `compute_dot_product_att_value`
            static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, ".. or add a condition to loop below");
            for (int32_t q = warp_id; q < kQueriesPerBlock; q += kNumWarpsPerBlock) { // parallel warps
                // 3. Update s_prime
                scalar_t sp = 0;
                scalar_t my_mi = mi[q][warp_id];
                static_assert(kNumWarpsPerBlock * kKeysPerWarp % kWarpSize == 0, ".. or add a condition to loop below");
                for (int32_t key_id = lane_id; key_id < kNumWarpsPerBlock * kKeysPerWarp; key_id += kWarpSize) { // parallel lanes
                    scalar_t si_exp = std::exp(si[q][key_id] - my_mi) * (key_id < num_keys);
                    sp += si_exp;
                    si[q][key_id] = si_exp;
                }
                scalar_t m_prime_exp = std::exp(m_prime[q] - my_mi);
                sp = warpSum(sp) + s_prime[q] * m_prime_exp;

                m_prime[q] = m_prime_exp;
                s_prime[q] = sp;
            }
            __syncthreads(); // `s_prime` done

            // 4. Partial matmull with the values we have and V
            // `v* <- v* . exp(m* - mi) + v_i . exp(si - mi)`
            compute_dot_product_att_value(iter_key_start, value, m_prime, si, mi, output);
            __syncthreads(); // we modify `m_prime` after

            // 5. `m_prime` <- `mi`
            if (warp_id == 0) {
                for (int64_t q = thread_id(); q < kQueriesPerBlock; q += kWarpSize * kNumWarpsPerBlock) { // parallel lanes
                    m_prime[q] = mi[q][0];
                }
            }
        }

        // 6. Divide by s_prime all of the values
        const int32_t output_stride0 = output.stride(0);
        const int32_t iter_col_last = output.size(1) - thread_id();
        const int32_t iter_query_last = std::min((int32_t)kQueriesPerBlock, int32_t(num_queries - query_start()));
        if (iter_col_last > 0 && iter_query_last > 0) {
            // &output[query_start()][thread_id]
            scalar_t* output_line_ptr = output.data() + query_start() * output_stride0 + thread_id();
            for (int32_t q = 0; q < iter_query_last; ++q) {
                scalar_t line_s_prime = s_prime[q];
                for (int32_t value_col = 0; value_col < iter_col_last; value_col += kNumWarpsPerBlock * kWarpSize) { // parallel warps/lanes
                    output_line_ptr[value_col] /= line_s_prime;
                }
                output_line_ptr += output_stride0;
            }
        }
    }

#if 0
    // Naive version
    static __device__ void compute_dot_product_att_value(
        iter_key_start const& iter_key_start,
        at::TensorAccessor<scalar_t, 2> value,
        scalar_t m_prime[kQueriesPerBlock],
        scalar_t si[kQueriesPerBlock][kSiDim1],
        scalar_t mi[kQueriesPerBlock][kNumWarpsPerBlock],
        at::TensorAccessor<scalar_t, 2> output
    ) {
        int64_t K = value.size(1);
        scalar_t* value_start_ptr = &value[iter_key_start][0];
        assert(value.stride(1) == 1);

        scalar_t* output_ptr = &output[query_start()][thread_id()];
        const int64_t output_stride0 = output.stride(0);
        assert(output.stride(1) == 1);

        for (int64_t q = 0; q < kQueriesPerBlock; ++q) {
            scalar_t my_mi = mi[q][warp_id()];
            scalar_t exp_mprime_mi = m_prime[q];

            for (int64_t value_col = 0; value_col < K; value_col += kNumWarpsPerBlock * kWarpSize) { // parallel warps/lanes
                if (value_col + thread_id() >= K) {
                    // TODO: Warp divergence if K is not good
                    break;
                }
                scalar_t current_v = 0;
                scalar_t* value_ptr = value_start_ptr + (value_col + thread_id());
                for (int64_t k = 0; k < kNumWarpsPerBlock * kKeysPerWarp; ++k) {
                    // scalar_t current_value = value[iter_key_start + k][value_col + thread_id()];
                    scalar_t current_value = *(value_ptr + k * K);
                    scalar_t current_si = si[q][k];
                    current_v += current_value * current_si; // LONG SCOREBOARD bottleneck
                }
                // output[query_start() + q][value_col + thread_id()]
                scalar_t v_prime = output_ptr[value_col];
                output_ptr[value_col] = v_prime * exp_mprime_mi + current_v;
            }
            output_ptr += output_stride0;
        }
    }
#else
    // cutlass version
    static __device__ void compute_dot_product_att_value(
        int32_t const& iter_key_start,
        at::TensorAccessor<scalar_t, 2>& value,
        scalar_t m_prime[kQueriesPerBlock],
        scalar_t si[kQueriesPerBlock][kSiDim1],
        scalar_t mi[kQueriesPerBlock][kNumWarpsPerBlock],
        at::TensorAccessor<scalar_t, 2>& output
    ) {
        using ThreadblockShape = cutlass::gemm::GemmShape<kQueriesPerBlock, kNumWarpsPerBlock * kKeysPerWarp, 8>;
        using WarpShape = cutlass::gemm::GemmShape<kQueriesPerBlock, kKeysPerWarp, 8>;
        using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

        // default_mma_core_simt.h
        using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape,                       // ThreadblockShape,
            WarpShape,                              // WarpShape,
            InstructionShape,                       // InstructionShape,
            float,                                  // ElementA,
            cutlass::layout::RowMajor,              // LayoutA,
            float,                                  // ElementB,
            cutlass::layout::RowMajor,              // LayoutB,
            float,                                  // ElementC,
            cutlass::layout::RowMajor,              // LayoutC,
            // Just use `cutlass::arch::OpClassTensorOp` for TensorCores (requires sm>7.0)
            cutlass::arch::OpClassSimt,             // OpClass,
            2,                                      // Stages,
            cutlass::arch::OpMultiplyAdd            // Operator,
            >;

        using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
                cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
                MmaCore::ElementA, MmaCore::LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

        // Define iterators over tiles from the B operand
        using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
                cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
                MmaCore::ElementB, MmaCore::LayoutB, 0, typename MmaCore::IteratorThreadMapB>;


        using Mma = cutlass::gemm::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, MmaCore::ElementC, MmaCore::LayoutC,
            typename MmaCore::MmaPolicy>;


        cutlass::gemm::GemmCoord problem_size(
            std::min((int64_t)kQueriesPerBlock, output.size(0) - query_start()), // M
            value.size(1), // N
            std::min(kNumWarpsPerBlock * kKeysPerWarp, value.size(0) - iter_key_start) // K
        );
        typename IteratorA::Params params_A(kSiDim1);
        typename IteratorA::TensorRef ref_A(
            &si[0][0],
            kSiDim1
        );

        typename IteratorB::Params params_B(typename MmaCore::LayoutB(value.stride(0)));
        typename IteratorB::TensorRef ref_B(
            &value[iter_key_start][0],
            value.stride(0)
        );

        static_assert(MmaCore::WarpCount::kM * MmaCore::WarpCount::kN * MmaCore::WarpCount::kK == kNumWarpsPerBlock);

        const int64_t nBlockN = ceil_div((int64_t)problem_size.n(), int64_t(ThreadblockShape::kN));
        for (int64_t blockN = 0; blockN < nBlockN; ++blockN) {
            // Shared storage needed by threadblock-scoped matrix multiply-accumulate
            __shared__ typename Mma::SharedStorage shared_storage;

            // Compute threadblock location
            cutlass::gemm::GemmCoord tb_tile_offset = {0, blockN, 0};

            cutlass::MatrixCoord tb_offset_A{tb_tile_offset.m() * Mma::Shape::kM,
                                            tb_tile_offset.k()};

            cutlass::MatrixCoord tb_offset_B{tb_tile_offset.k(),
                                            tb_tile_offset.n() * Mma::Shape::kN};

            // Construct iterators to A and B operands
            typename Mma::IteratorA iterator_A(params_A, ref_A.data(),
                                                {problem_size.m(), problem_size.k()},
                                                thread_id(), tb_offset_A);

            typename Mma::IteratorB iterator_B(params_B, ref_B.data(),
                                                {problem_size.k(), problem_size.n()},
                                                thread_id(), tb_offset_B);

            uint8_t my_warp_id = warp_id();
            uint8_t my_lane_id = lane_id();

            // Construct thread-scoped matrix multiply
            Mma mma(shared_storage, thread_id(), my_warp_id, my_lane_id);

            // Output results
            // cutlass::gemm::warp::MmaSimtTileIterator<cutlass::MatrixShape<16, 32>, cutlass::gemm::Operand::kC, float, cutlass::layout::RowMajor, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<1>, cutlass::gemm::GemmShape<4, 4, 1>>, 1, 1>
            typename Mma::Operator::IteratorC iterator_C({&output[query_start()][0], output.stride(0)}, my_lane_id);
            auto iterator_C_offset_m = (tb_tile_offset.m() * Mma::WarpCount::kM) +
                    (my_warp_id % Mma::WarpCount::kM);
            auto iterator_C_offset_n = (tb_tile_offset.n() * Mma::WarpCount::kN) +
                    (my_warp_id / Mma::WarpCount::kM);
            using LaneMmaShape = typename Mma::Policy;
            typename Mma::Operator::IteratorC::Policy::LaneLayout lane_layout = Mma::Operator::IteratorC::Policy::get_lane_layout();
            cutlass::MatrixCoord lane_offset = lane_layout.inverse(my_lane_id) * cutlass::MatrixCoord(Mma::Operator::IteratorC::Policy::LaneMmaShape::kM, Mma::Operator::IteratorC::Policy::LaneMmaShape::kN);
            iterator_C.add_tile_offset({iterator_C_offset_m, iterator_C_offset_n});
        
            typename Mma::FragmentC accum; // cutlass::Array<float, 16, true>
            // TODO: We could avoid all this mess using cutlass's Epilogue concept I think
            // but I got lost in templates and reimplemented everything

            const int32_t thread_offset_m = Mma::WarpGemm::kM * iterator_C_offset_m + lane_offset.row();
            const int32_t thread_offset_n = Mma::WarpGemm::kN * iterator_C_offset_n + lane_offset.column();
            scalar_t* output_ptr = &output[query_start()][0];
            const int32_t output_s0 = output.stride(0);
            const int32_t max_m = output.size(0) - query_start();
            const int32_t max_n = output.size(1);

            // Load data already calculated, and rescale it (as the max value for the softmax might have changed)
            accum.clear();
            iterate_on_frag<Mma::Operator::IteratorC>(accum, thread_offset_m, thread_offset_n, [&] (float& accum_v, int32_t m, int32_t n) {
                if (m < max_m && n < max_n) {
                    accum_v = output_ptr[m * output_s0 + n] * m_prime[m];
                }
            });
            int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

            // Compute threadblock-scoped matrix multiply-add
            mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

            iterate_on_frag<Mma::Operator::IteratorC>(accum, thread_offset_m, thread_offset_n, [&] (float& accum_v, int32_t m, int32_t n) {
                if (m < max_m && n < max_n) {
                    assert(m >= 0 && n >= 0);
                    output_ptr[m * output_s0 + n] = accum_v;
                }
            });
        }
    }

    template <typename Iterator, typename Fragment, typename FN>
    static void __device__ iterate_on_frag(Fragment& frag, int32_t offset_m, int32_t offset_n, FN callback) {
        // TODO: This is quite hacky, and only needed for Simt. For other Mmas, we can use epilogue.
        using Policy = typename Iterator::Policy;
        using Delta = typename Iterator::Delta;
        using Iterations = typename Iterator::Iterations;
        using Element = typename Iterator::Element;

        static_assert(Fragment::kStorageElements == kQueriesPerBlock);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) { // 0
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
                        callback(
                            frag.at(n + Policy::LaneMmaShape::kN * (mma_n + Iterations::kColumn * (m + mma_m * Policy::LaneMmaShape::kM))),
                            offset_m + m + mma_m * Delta::kRow,
                            offset_n + n + mma_n * Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN
                        );
                    }
                }
            }
        }
    }
#endif

#if 0
    static __device__ void compute_dot_product_qk(
        int32_t const& iter_key_start,
        at::TensorAccessor<scalar_t, 2>& query,
        at::TensorAccessor<scalar_t, 2>& key,
        scalar_t m_prime[kQueriesPerBlock],
        scalar_t si[kQueriesPerBlock][kSiDim1],
        scalar_t mi[kQueriesPerBlock][kNumWarpsPerBlock]
    ) {
        /*
        Computes the block-matrix product of:
        (a) query[query_start:query_end, :]
        with
        (b) key[iter_key_start:iter_key_start + kNumWarpsPerBlock * kKeysPerWarp]
        and stores that into `si`
        */
        int64_t num_keys = key.size(0);
        int64_t num_queries = query.size(0);
        int64_t K = key.size(1);

        scalar_t* key_start_ptr = key.data();
        int64_t key_stride0 = key.stride(0);
        scalar_t* query_start_ptr = &query[query_start()][0];
        int64_t query_stride0 = query.stride(0);

        for (int64_t q = 0; q < kQueriesPerBlock; ++q) {
            int64_t key_offset = iter_key_start + warp_id() * kKeysPerWarp + lane_id();
            scalar_t scale = 1.0 / std::sqrt(scalar_t(K));
            scalar_t currentMax = m_prime[q];
            if (query_start() + q < num_queries) {
                for (int64_t key_id = 0; key_id < kKeysPerWarp; key_id += kWarpSize) { // parallel lanes
                    if (key_offset + key_id >= num_keys) {
                        break;
                    }
                    scalar_t dot_product = 0;
                    scalar_t* cur_key = key_start_ptr + (key_offset + key_id) * key_stride0;
                    scalar_t* cur_query = query_start_ptr + q * query_stride0;
                    for (int64_t k = 0; k < K; ++k) {
                        dot_product += cur_key[k] * cur_query[k];
                    }
                    dot_product *= scale;
                    si[q][warp_id() * kKeysPerWarp + key_id + lane_id()] = dot_product;
                    currentMax = std::max(currentMax, dot_product);
                }
            }
            // 2a. At the same time aggregate the max at the warp-level
            mi[q][warp_id()] = warpMax(currentMax);
        }
    }
#else
    static __device__ void compute_dot_product_qk(
        int32_t const& iter_key_start,
        at::TensorAccessor<scalar_t, 2>& query,
        at::TensorAccessor<scalar_t, 2>& key,
        scalar_t m_prime[kQueriesPerBlock],
        scalar_t si[kQueriesPerBlock][kSiDim1],
        scalar_t mi[kQueriesPerBlock][kNumWarpsPerBlock]
    ) {
        /*
        Computes the block-matrix product of:
        (a) query[query_start:query_end, :]
        with
        (b) key[iter_key_start:iter_key_start + kNumWarpsPerBlock * kKeysPerWarp]
        and stores that into `si`
        */

        using ThreadblockShape = cutlass::gemm::GemmShape<kQueriesPerBlock, kNumWarpsPerBlock * kKeysPerWarp, 8>;
        using WarpShape = cutlass::gemm::GemmShape<kQueriesPerBlock, kKeysPerWarp, 8>;
        using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

        using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape,                       // ThreadblockShape,
            WarpShape,                              // WarpShape,
            InstructionShape,                       // InstructionShape,
            float,                                  // ElementA,
            cutlass::layout::RowMajor,              // LayoutA,
            float,                                  // ElementB,
            cutlass::layout::ColumnMajor,           // LayoutB,
            float,                                  // ElementC,
            cutlass::layout::RowMajor,              // LayoutC,
            // Just use `cutlass::arch::OpClassTensorOp` for TensorCores (requires sm>7.0)
            cutlass::arch::OpClassSimt,             // OpClass,
            2,                                      // Stages,
            cutlass::arch::OpMultiplyAdd            // Operator,
            >;

        using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
                cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
                MmaCore::ElementA, MmaCore::LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

        // Define iterators over tiles from the B operand
        using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
                cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
                MmaCore::ElementB, MmaCore::LayoutB, 0, typename MmaCore::IteratorThreadMapB>;


        using Mma = cutlass::gemm::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, MmaCore::ElementC, MmaCore::LayoutC,
            typename MmaCore::MmaPolicy>;


        int64_t num_queries = query.size(0);
        int64_t K = key.size(1);

        cutlass::gemm::GemmCoord problem_size(
            std::min((int64_t)kQueriesPerBlock, num_queries - query_start()),
            std::min(kNumWarpsPerBlock * kKeysPerWarp, key.size(0) - iter_key_start),
            K
        );
        IteratorA::Params params_A(typename MmaCore::LayoutA(query.stride(0)));
        IteratorA::TensorRef ref_A(
            &query[query_start()][0],
            query.stride(0)
        );

        typename IteratorB::Params params_B(typename MmaCore::LayoutB(key.stride(0)));
        typename IteratorB::TensorRef ref_B(
            &key[iter_key_start][0],
            key.stride(0)
        );

        static_assert(MmaCore::WarpCount::kM * MmaCore::WarpCount::kN * MmaCore::WarpCount::kK == kNumWarpsPerBlock);

        cutlass_mma<Mma>(
            problem_size,
            params_A, ref_A,
            params_B, ref_B,
            &si[0][0], kSiDim1
        );
        __syncthreads();

        int64_t num_keys = key.size(0);
        int16_t key_offset = iter_key_start + warp_id() * kKeysPerWarp + lane_id();
        scalar_t scale = 1.0 / std::sqrt(scalar_t(K));
        for (int16_t q = 0; q < kQueriesPerBlock; ++q) {
            scalar_t currentMax = m_prime[q];
            if (query_start() + q < num_queries) {
                CUTLASS_PRAGMA_UNROLL
                for (int64_t key_id = 0; key_id < kKeysPerWarp; key_id += kWarpSize) { // parallel lanes
                    if (key_offset + key_id >= num_keys) {
                        break;
                    }
                    scalar_t dot_product = si[q][warp_id() * kKeysPerWarp + key_id + lane_id()];
                    dot_product *= scale;
                    si[q][warp_id() * kKeysPerWarp + key_id + lane_id()] = dot_product;

                    // 2a. At the same time aggregate the max at the warp-level
                    currentMax = std::max(currentMax, dot_product);
                }
            }
            mi[q][warp_id()] = warpMax(currentMax);
        }
    }
#endif

    template <typename Mma>
    static __device__ void cutlass_mma(cutlass::gemm::GemmCoord problem_size,
                            typename Mma::IteratorA::Params params_A,
                            typename Mma::IteratorA::TensorRef ref_A,
                            typename Mma::IteratorB::Params params_B,
                            typename Mma::IteratorB::TensorRef ref_B,
                            typename Mma::ElementC *ptr_C,
                            typename Mma::LayoutC::Stride::Index ldc) {
        // Shared storage needed by threadblock-scoped matrix multiply-accumulate
        __shared__ typename Mma::SharedStorage shared_storage;

        // Compute threadblock location
        cutlass::gemm::GemmCoord tb_tile_offset = {0, 0, 0};

        cutlass::MatrixCoord tb_offset_A{tb_tile_offset.m() * Mma::Shape::kM,
                                        tb_tile_offset.k()};

        cutlass::MatrixCoord tb_offset_B{tb_tile_offset.k(),
                                        tb_tile_offset.n() * Mma::Shape::kN};

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(params_A, ref_A.data(),
                                            {problem_size.m(), problem_size.k()},
                                            thread_id(), tb_offset_A);

        typename Mma::IteratorB iterator_B(params_B, ref_B.data(),
                                            {problem_size.k(), problem_size.n()},
                                            thread_id(), tb_offset_B);

        auto my_warp_id = warp_id();
        auto my_lane_id = lane_id();

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage, thread_id(), my_warp_id, my_lane_id);

        typename Mma::FragmentC accum;

        accum.clear();

        auto gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

        // Output results
        typename Mma::Operator::IteratorC iterator_C({ptr_C, ldc}, my_lane_id);

        iterator_C.add_tile_offset(
            {(tb_tile_offset.m() * Mma::WarpCount::kM) +
                (my_warp_id % Mma::WarpCount::kM),
            (tb_tile_offset.n() * Mma::WarpCount::kN) +
                (my_warp_id / Mma::WarpCount::kM)});

        iterator_C.store(accum);
    }

    static __device__ __forceinline__ scalar_t warpMax(scalar_t val) {
        for (int stride = kWarpSize / 2; stride > 0; stride >>= 1) {
            scalar_t tmp = __shfl_xor_sync(0xffffffff, val, stride, kWarpSize);
            val = tmp > val ? tmp : val;
        }
        return val;
    }

    static __device__ __forceinline__ scalar_t warpSum(scalar_t val) {
        for (int stride = kWarpSize / 2; stride > 0; stride >>= 1) {
            scalar_t tmp = __shfl_xor_sync(0xffffffff, val, stride, kWarpSize);
            val += tmp;
        }
        return val;
    }

    static __device__ __forceinline__ int8_t lane_id() {
        return threadIdx.x;
    }
    static __device__ __forceinline__ int8_t warp_id() {
        return threadIdx.y;
    }
    static __device__ __forceinline__ int16_t thread_id() {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }
    static __device__ __forceinline__ int32_t query_start() {
        return blockIdx.y * kQueriesPerBlock;
    }
};

template <typename AK>
__global__ void
__launch_bounds__(
    // maxThreadsPerBlock specifies the maximum number of threads per block with which the application will ever launch
    AK::kWarpSize * AK::kNumWarpsPerBlock,
    // minBlocksPerMultiprocessor is optional and specifies the desired minimum number of resident blocks per multiprocessor
    12 / AK::kNumWarpsPerBlock
)
attention_kernel_batched(
    at::PackedTensorAccessor<typename AK::scalar_t, 3> output,
    at::PackedTensorAccessor<typename AK::scalar_t, 2> logsumexp,
    at::PackedTensorAccessor<typename AK::scalar_t, 3> query,
    at::PackedTensorAccessor<typename AK::scalar_t, 3> key,
    at::PackedTensorAccessor<typename AK::scalar_t, 3> value) {
    auto batch_id = blockIdx.z;
    AK::attention_kernel(
        output[batch_id],
        logsumexp[batch_id],
        query[batch_id],
        key[batch_id],
        value[batch_id]
    );
}

std::tuple<at::Tensor, at::Tensor> launch_attention2(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool compute_logsumexp
) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");

  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);
  at::Tensor res = at::zeros({B, M, K}, query.options());
  at::Tensor logsumexp = at::empty({B, M}, query.options());

  typedef float scalar_t;
  // assert(compute_logsumexp == true);
  using AK = AttentionKernel<scalar_t, true>;

  dim3 grid(AK::kNumBlocksX, AK::getNumBlocksY(M), B);
  dim3 block(AK::kWarpSize, AK::kNumWarpsPerBlock, 1);


  attention_kernel_batched<AK><<<grid, block>>>(
        res.packed_accessor<scalar_t, 3>(),
        logsumexp.packed_accessor<scalar_t, 2>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>());
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(res, logsumexp);
}
}

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention2"),
      TORCH_FN(launch_attention2));
}
