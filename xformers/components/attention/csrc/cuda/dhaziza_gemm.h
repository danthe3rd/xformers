/*
TORCH_CUDA_ARCH_LIST="6.0;7.0" NVCC_FLAGS="-g -G" python3 setup.py develop && \
python3 xformers/benchmarks/benchmark_mem_eff_attention.py --compare main,boundschecks --label nobs

set cuda memcheck on

/public/apps/cuda/11.4/bin/nvcc cuda_hello.cu

gdb -args /public/apps/cuda/11.4/nsight-compute-2021.2.2/target/linux-desktop-glibc_2_11_3-x64/ncu ./a.out
set follow-fork-mode child

salloc --gpus-per-node 1 --nodes 1 --partition devlab -C volta --time 1-00:00:00 --mem-per-gpu 50G
ssh -L localhost:1235:localhost:1235 learnfair0309

CUDA_VISIBLE_DEVICES=0 /public/apps/cuda/11.4/nsight-compute-2021.2.2/target/linux-desktop-glibc_2_11_3-x64/ncu \
--export report.ncu-rep --force-overwrite --target-processes application-only \
--replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 \
--section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart \
--section MemoryWorkloadAnalysis_Tables --section Nvlink_Tables --section Nvlink_Topology --section Occupancy --section SchedulerStats --section SourceCounters \
--section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats \
--sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes \
--import-source yes --check-exit-code yes \
/private/home/dhaziza/.conda/envs/xformers/bin/python3 xformers/benchmarks/benchmark_mem_eff_attention.py

*/
// ...
#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"

#include "cutlass/core_io.h"

#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
//...

namespace {

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}
}

namespace dhaziza_custom_matmull {
constexpr auto NUM_WARPS = 32u;

template <
    typename scalar_t>
__global__ void kernel_naive(
    at::PackedTensorAccessor<scalar_t, 2> query,
    at::PackedTensorAccessor<scalar_t, 2> key,
    at::PackedTensorAccessor<scalar_t, 2> out) {
  int64_t M = query.size(0);
  int64_t K = query.size(1);
  int64_t N = key.size(0);
  for (int64_t n = threadIdx.x; n < N; n += NUM_WARPS) {
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t k = 0; k < K; ++k) {
        out[m][n] += query[m][k] * key[n][k];
      }
    }
  }
}

// Copy pasted from
// https://github.com/NVIDIA/cutlass/blob/v2.9.0/test/unit/gemm/threadblock/mma_pipelined_testbed.h
// and
// https://github.com/NVIDIA/cutlass/blob/v2.9.0/test/unit/gemm/threadblock/mma_pipelined_simt.cu 
template <typename Mma>
__device__ void kernel_mma(cutlass::gemm::GemmCoord problem_size,
                           typename Mma::IteratorA::Params params_A,
                           typename Mma::IteratorA::TensorRef ref_A,
                           typename Mma::IteratorB::Params params_B,
                           typename Mma::IteratorB::TensorRef ref_B,
                           typename Mma::ElementC *ptr_C,
                           typename Mma::LayoutC::Stride::Index ldc,
                           int64_t blockAxisN,
                           int64_t blockAxisM) {
  // Shared storage needed by threadblock-scoped matrix multiply-accumulate
  __shared__ typename Mma::SharedStorage shared_storage;

  // Compute threadblock location
  cutlass::gemm::GemmCoord tb_tile_offset = {
    int(blockAxisM), // int(blockIdx.x), // M
    int(blockAxisN), // int(blockIdx.y), // N
    0 // K
  };

  cutlass::MatrixCoord tb_offset_A{tb_tile_offset.m() * Mma::Shape::kM,
                                   tb_tile_offset.k()};

  cutlass::MatrixCoord tb_offset_B{tb_tile_offset.k(),
                                   tb_tile_offset.n() * Mma::Shape::kN};

  // Compute position within threadblock
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  // Construct iterators to A and B operands
  typename Mma::IteratorA iterator_A(params_A, ref_A.data(),
                                     {problem_size.m(), problem_size.k()},
                                     tb_thread_id, tb_offset_A);

  typename Mma::IteratorB iterator_B(params_B, ref_B.data(),
                                     {problem_size.k(), problem_size.n()},
                                     tb_thread_id, tb_offset_B);

  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;

  // Construct thread-scoped matrix multiply
  Mma mma(shared_storage, tb_thread_id, warp_id, lane_id);

  typename Mma::FragmentC accum;

  accum.clear();

  int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

  // Compute threadblock-scoped matrix multiply-add
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

  // Output results
  typename Mma::Operator::IteratorC iterator_C({ptr_C, ldc}, lane_id);

  iterator_C.add_tile_offset(
      {(tb_tile_offset.m() * Mma::WarpCount::kM) +
           (warp_id % Mma::WarpCount::kM),
       (tb_tile_offset.n() * Mma::WarpCount::kN) +
           (warp_id / Mma::WarpCount::kM)});

  iterator_C.store(accum);
}

template <
    typename scalar_t>
struct GemmParams {
  using ElementA = float; // cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = float; // cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;


  // cutlass::gemm::GemmCoord problem_size(64, 64, 128);
  // using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  // using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  // using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  // NOTE: Ratio between the 2 following shapes gives num_warps
  // Playing with these numbers can greatly improve/degrade performance
  // NOTE: Using 8 as first dim gives incorrect result
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  // Define the MmaCore components
  static const int kStages = 2;
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape,    // ThreadblockShape,
      WarpShape,    // WarpShape,
      InstructionShape,      // InstructionShape,
      ElementA,                                  // ElementA,
      LayoutA,           // LayoutA,
      ElementB,                                  // ElementB,
      LayoutB,              // LayoutB,
      ElementC,                                  // ElementC,
      LayoutC,              // LayoutC,
      // Just use `cutlass::arch::OpClassTensorOp` for TensorCores (requires sm>7.0)
      cutlass::arch::OpClassSimt,             // OpClass,
      kStages,                                      // Stages,
      cutlass::arch::OpMultiplyAdd            // Operator,
      >;
  static constexpr int kNumWarps = MmaCore::WarpCount::kM * MmaCore::WarpCount::kN * MmaCore::WarpCount::kK;
    

  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

  // Define iterators over tiles from the B operand
  using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB>;

  // Define MmaPipeline Single Stage
  using MmaPipelineSingleStage =  cutlass::gemm::threadblock::MmaSingleStage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementC, LayoutC,
      typename MmaCore::MmaPolicy>;

  // Define MmaPipeline Two Stages
  using MmaPipelineTwoStages =  cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementC, LayoutC,
      typename MmaCore::MmaPolicy>;
  
  // Define the threadblock-scoped pipelined matrix multiply (Select between Single vs. Two stages)
  using Mma = typename cutlass::platform::conditional<(kStages==1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;
};

template <
    typename scalar_t>
__device__ void kernel_cutlass_single(
    at::TensorAccessor<scalar_t, 2> query,
    at::TensorAccessor<scalar_t, 2> key,
    at::TensorAccessor<scalar_t, 2> out) {

  using P = GemmParams<scalar_t>;

  cutlass::gemm::GemmCoord problem_size(query.size(0), key.size(0), query.size(1));
  typename P::IteratorA::Params params_A(typename P::LayoutA(query.stride(0)));
  typename P::IteratorA::TensorRef ref_A(
    &query[0][0],
    query.stride(0)
  );

  typename P::IteratorB::Params params_B(typename P::LayoutB(key.stride(0)));
  typename P::IteratorB::TensorRef ref_B(
    &key[0][0],
    key.stride(0)
  );

  int64_t nBlocksN = ceil_div(key.size(0), int64_t(P::ThreadblockShape::kN));
  for (int64_t n = 0; n < nBlocksN; ++n) {
    kernel_mma<P::Mma>(
        problem_size,
        params_A, ref_A,
        params_B, ref_B,
        &out[0][0], out.stride(0),
        n, // blockAxisN
        blockIdx.y // blockAxisM
    );
  }
}

template <
    typename scalar_t>
__global__ void kernel_cutlass(
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> out) {
  at::TensorAccessor<scalar_t, 2> query_s = query[blockIdx.x];
  kernel_cutlass_single<scalar_t>(query_s, key[blockIdx.x], out[blockIdx.x]);
}

}