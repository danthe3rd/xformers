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
    static constexpr int64_t kNumWarpsPerBlock = 2;
    static constexpr int64_t kKeysPerWarp = kWarpSize;
    static constexpr int64_t kValuesPerWarp = kWarpSize * kNumWarpsPerBlock;
    
    static_assert(kKeysPerWarp % kWarpSize == 0);
    static_assert(kValuesPerWarp % kWarpSize == 0);
    static_assert(kValuesPerWarp == (kWarpSize * kNumWarpsPerBlock), "not implemented yet");

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
    
        int64_t lane_id = threadIdx.x;
        int64_t warp_id = threadIdx.y;

        // In this block, we will only ever:
        // - read query[query_start:query_end, :]
        // - write to output[query_start:query_end, :]
        // int64_t query_start = blockIdx.y * kQueriesPerBlock;
        // int64_t query_end = (blockIdx.y + 1) * kQueriesPerBlock;

        int64_t num_keys = key.size(0);
        int64_t num_values = value.size(0);
        int64_t num_queries = query.size(0);
        int64_t K = key.size(1);

        scalar_t __shared__ m_prime[kQueriesPerBlock];
        scalar_t __shared__ mi[kQueriesPerBlock][kNumWarpsPerBlock];
        scalar_t __shared__ s_prime[kQueriesPerBlock];
        scalar_t __shared__ si[kQueriesPerBlock][kNumWarpsPerBlock * kKeysPerWarp];
        // ArrayWithBoundsChecks<kQueriesPerBlock> m_prime(m_prime_);
        // ArrayWithBoundsChecks2d<kQueriesPerBlock, kNumWarpsPerBlock> mi(mi_);
        // ArrayWithBoundsChecks<kQueriesPerBlock> s_prime(s_prime_);
        // ArrayWithBoundsChecks2d<kQueriesPerBlock, kNumWarpsPerBlock * kKeysPerWarp> si(si_);

        for (int64_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) {
            mi[q + lane_id][warp_id] = -std::numeric_limits<scalar_t>::infinity();
        }
        if (warp_id == 0) {
            for (int64_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) {
                s_prime[q + lane_id] = 0;
                m_prime[q + lane_id] = -std::numeric_limits<scalar_t>::infinity();
            }
        }

        // Iterate through keys
        for (int64_t iter_key_start = 0; iter_key_start < num_keys; iter_key_start += kNumWarpsPerBlock * kKeysPerWarp) {
            // int64_t iter_key_end = iter_key_start + kNumWarpsPerBlock * kKeysPerWarp;

            // TODO(half): Shared memory banks are organized such that successive 32-bit words are assigned to successive banks and the bandwidth is 32 bits per bank per clock cycle

            __syncthreads(); // Need to have shared memory initialized, and `m_prime` updated from end of prev iter

            // 1. Compute dot-product into shared memory for each query
            compute_dot_product_qk(iter_key_start, query, key, m_prime, si, mi);
            // TODO: Optimize this matmull (cutlass?)

            __syncthreads(); // `mi` calculation done based on warp-data

            // 2b. Aggregate max across different warps
            for (int64_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) { // parallel lanes
                scalar_t global_max = mi[q + lane_id][0];
                for(int64_t other_warp = 0; other_warp < kNumWarpsPerBlock; ++other_warp) {
                    global_max = std::max(global_max, mi[q + lane_id][other_warp]);
                }
                mi[q + lane_id][warp_id] = global_max;

            }

            __syncthreads(); // `mi` calculation done based on block data. `mi[a][i] == mi[a][j]` for all (a, i, j)

            // TODO: Maybe this could be parallelized across warps
            if (warp_id == 0) {
                for (int64_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) { // parallel lanes
                    // 3. Update s_prime
                    scalar_t my_mi = mi[q + lane_id][warp_id];
                    scalar_t sp = s_prime[q + lane_id] * std::exp(m_prime[q + lane_id] - my_mi);
                    for (int64_t key_id = 0; key_id < kNumWarpsPerBlock * kKeysPerWarp; ++key_id) {
                        sp += std::exp(si[q + lane_id][key_id] - my_mi);
                    }
                    s_prime[q + lane_id] = sp;
                }
            }
            __syncthreads(); // `s_prime` done

            // 4. Partial matmull with the values we have and V
            // `v* <- v* . exp(m* - mi) + v_i . exp(si - mi)`
            // TODO: Make it efficient matmull
            compute_dot_product_att_value(iter_key_start, value, m_prime, si, mi, output);
            __syncthreads(); // we modify `m_prime` after

            // 5. `m_prime` <- `mi`
            if (warp_id == 0) {
                for (int64_t q = 0; q + lane_id < kQueriesPerBlock; q += kWarpSize) { // parallel lanes
                    m_prime[q + lane_id] = mi[q + lane_id][0];
                }
            }
        }

        // 6. Divide by s_prime all of the values
        const int64_t output_stride0 = output.stride(0);
        const int64_t last_K_iter = K - thread_id();
        // &output[query_start()][thread_id]
        scalar_t* output_line_ptr = output.data() + query_start() * output_stride0 + thread_id();
        for (int64_t q = 0; q < kQueriesPerBlock; ++q) {
            scalar_t line_s_prime = s_prime[q];
            for (int64_t value_col = 0; value_col < last_K_iter; value_col += kNumWarpsPerBlock * kWarpSize) { // parallel warps/lanes
                output_line_ptr[value_col] /= line_s_prime;
            }
            output_line_ptr += output_stride0;
        }
    }

    static __device__ void compute_dot_product_att_value(
        int64_t iter_key_start,
        at::TensorAccessor<scalar_t, 2> value,
        scalar_t m_prime[kQueriesPerBlock],
        scalar_t si[kQueriesPerBlock][kNumWarpsPerBlock * kKeysPerWarp],
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
            scalar_t exp_mprime_mi = std::exp(m_prime[q] - my_mi);
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
                    current_v += current_value * std::exp(current_si - my_mi); // TODO: store in smem?
                }
                // output[query_start() + q][value_col + thread_id()]
                scalar_t v_prime = output_ptr[value_col];
                output_ptr[value_col] = v_prime * exp_mprime_mi + current_v;
            }
            output_ptr += output_stride0;
        }
    }

    static __device__ void compute_dot_product_qk(
        int64_t iter_key_start,
        at::TensorAccessor<scalar_t, 2> query,
        at::TensorAccessor<scalar_t, 2> key,
        scalar_t m_prime[kQueriesPerBlock],
        scalar_t si[kQueriesPerBlock][kNumWarpsPerBlock * kKeysPerWarp],
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

                    // 2a. At the same time aggregate the max at the warp-level
                    scalar_t max_over_warp = std::max(currentMax, warpMax(dot_product));
                    currentMax = std::max(max_over_warp, dot_product); // TODO(remove)
                }
            }
            mi[q][warp_id()] = currentMax;
        }
    }

#if 0
    static __device__ void compute_dot_product_qk_cutlass(
        int64_t iter_key_start,
        at::TensorAccessor<scalar_t, 2> query,
        at::TensorAccessor<scalar_t, 2> key,
        scalar_t m_prime[kQueriesPerBlock],
        scalar_t si[kQueriesPerBlock][kNumWarpsPerBlock * kKeysPerWarp],
        scalar_t mi[kQueriesPerBlock][kNumWarpsPerBlock]
    ) {
        /*
        Computes the block-matrix product of:
        (a) query[query_start:query_end, :]
        with
        (b) key[iter_key_start:iter_key_start + kNumWarpsPerBlock * kKeysPerWarp]
        and stores that into `si`
        */
        using namespace dhaziza_custom_matmull;
        using P = GemmParams<float>;

        int64_t num_queries = query.size(0);
        int64_t K = key.size(1);

        // TODO: Handle non optimal matrix sizes
        cutlass::gemm::GemmCoord problem_size(kQueriesPerBlock, kNumWarpsPerBlock * kKeysPerWarp, K);
        typename P::IteratorA::Params params_A(typename P::LayoutA(query.stride(0)));
        typename P::IteratorA::TensorRef ref_A(
            &query[query_start()][0],
            query.stride(0)
        );

        typename P::IteratorB::Params params_B(typename P::LayoutB(key.stride(0)));
        typename P::IteratorB::TensorRef ref_B(
            &key[iter_key_start][0],
            key.stride(0)
        );

        static_assert(P::ThreadblockShape::kN == kNumWarpsPerBlock * kKeysPerWarp);
        static_assert(P::ThreadblockShape::kM == kQueriesPerBlock);
        static_assert(P::kNumWarps == kNumWarpsPerBlock);
        // scalar_t local_si[kQueriesPerBlock][kNumWarpsPerBlock * kKeysPerWarp]
        kernel_mma<P::Mma>(
            problem_size,
            params_A, ref_A,
            params_B, ref_B,
            &si[0][0], kNumWarpsPerBlock * kKeysPerWarp,
            0, // blockAxisN
            0 // blockAxisM
        );
    }
#endif

    static __device__ __forceinline__ scalar_t warpMax(scalar_t val) {
        for (int stride = kWarpSize / 2; stride > 0; stride >>= 1) {
            scalar_t tmp = __shfl_xor_sync(0xffffffff, val, stride, kWarpSize);
            val = tmp > val ? tmp : val;
        }
        return val;
    }

    static __device__ __forceinline__ int64_t lane_id() {
        return threadIdx.x;
    }
    static __device__ __forceinline__ int64_t warp_id() {
        return threadIdx.y;
    }
    static __device__ __forceinline__ int64_t thread_id() {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }
    static __device__ __forceinline__ int64_t query_start() {
        return blockIdx.y * kQueriesPerBlock;
    }
};

template <typename AK>
__global__ void attention_kernel_batched(
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
