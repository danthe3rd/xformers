#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>

#include "sputnik/vector_utils.h"


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

template <typename scalar_t>
constexpr __host__ __device__ bool integerIsPowerOf2(scalar_t v) {
  return (v && !(v & (v - 1)));
}

template <typename scalar_t>
__device__ __forceinline__ void iMul(scalar_t x1, float4* out) {
  out[0].x *= x1;
  out[0].y *= x1;
  out[0].z *= x1;
  out[0].w *= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iMul(scalar_t x1, float2* out) {
  out[0].x *= x1;
  out[0].y *= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iMul(scalar_t x1, float* out) {
  out[0] *= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iDiv(scalar_t x1, float4* out) {
  out[0].x /= x1;
  out[0].y /= x1;
  out[0].z /= x1;
  out[0].w /= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iDiv(scalar_t x1, float2* out) {
  out[0].x /= x1;
  out[0].y /= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void iDiv(scalar_t x1, float* out) {
  out[0] /= x1;
}

template <typename scalar_t>
__device__ __forceinline__ void myGpuAtomicAdd(scalar_t* address, float4 val) {
  gpuAtomicAdd(address + 0, val.x);
  gpuAtomicAdd(address + 1, val.y);
  gpuAtomicAdd(address + 2, val.z);
  gpuAtomicAdd(address + 3, val.w);
}

template <typename scalar_t>
__device__ __forceinline__ void myGpuAtomicAdd(scalar_t* address, float2 val) {
  gpuAtomicAdd(address + 0, val.x);
  gpuAtomicAdd(address + 1, val.y);
}

template <typename scalar_t>
__device__ __forceinline__ void myGpuAtomicAdd(scalar_t* address, float val) {
  gpuAtomicAdd(address, val);
}

template <typename scalar_t, int WARP_SIZE>
__device__ __forceinline__ scalar_t warpSum(scalar_t val) {
  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, stride, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, int WARP_SIZE>
__device__ __forceinline__ float2 warpSum(float2 val) {
  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
    val.x += __shfl_xor_sync(0xffffffff, val.x, stride, WARP_SIZE);
    val.y += __shfl_xor_sync(0xffffffff, val.y, stride, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, int WARP_SIZE>
__device__ __forceinline__ float4 warpSum(float4 val) {
  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
    val.x += __shfl_xor_sync(0xffffffff, val.x, stride, WARP_SIZE);
    val.y += __shfl_xor_sync(0xffffffff, val.y, stride, WARP_SIZE);
    val.z += __shfl_xor_sync(0xffffffff, val.z, stride, WARP_SIZE);
    val.w += __shfl_xor_sync(0xffffffff, val.w, stride, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, int WARP_SIZE>
__device__ __forceinline__ scalar_t warpMax(scalar_t val) {
  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
    scalar_t tmp = __shfl_xor_sync(0xffffffff, val, stride, WARP_SIZE);
    val = tmp > val ? tmp : val;
  }
  return val;
}

template <typename scalar_t, typename vec_t, int kBlockSizeK, int kBlockSizeQ>
__device__ void compute_dot(
    vec_t* queries[kBlockSizeQ],
    vec_t* keys,
    scalar_t out[kBlockSizeQ][kBlockSizeK],
    int64_t K) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  scalar_t scale = 1.0 / std::sqrt(scalar_t(K));
  vec_t q_i[kBlockSizeQ];
  for (int64_t k = 0; k < K / kVecSize; k += 1) {
#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      q_i[q_item_idx] = __ldg(queries[q_item_idx] + k);
      iMul(scale, q_i + q_item_idx);
    }
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      vec_t k_i = keys[k + K / kVecSize * k_item_idx];
#pragma unroll
      for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
        sputnik::VectorCompute<vec_t>::Dot(
            q_i[q_item_idx], k_i, &out[q_item_idx][k_item_idx]);
      }
    }
  }
}

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int BUFFER_SIZE>
__device__ void compute_final_mult(
    vec_t* vi,
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK],
    scalar_t m_delta[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
    int64_t K) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);

  for (int64_t k = 0; k < K / kVecSize; k += 1) {
#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      iMul<scalar_t>(m_delta[q_item_idx], &buffer[q_item_idx][k]);
    }
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      vec_t tmp2 = vi[k + K / kVecSize * k_item_idx];

#pragma unroll
      for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
        sputnik::VectorCompute<vec_t>::FMA(
            s_delta[q_item_idx][k_item_idx], tmp2, &buffer[q_item_idx][k]);
      }
    }
  }
}

template <typename scalar_t, int kBlockSizeK, int kBlockSizeQ>
__device__ __forceinline__ void compute_max(
    scalar_t a[kBlockSizeQ][kBlockSizeK],
    scalar_t b[kBlockSizeQ],
    scalar_t out[kBlockSizeQ]) {
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    out[q_item_idx] =
        a[q_item_idx][0] > b[q_item_idx] ? a[q_item_idx][0] : b[q_item_idx];
#pragma unroll
    for (int64_t k_item_idx = 1; k_item_idx < kBlockSizeK; k_item_idx++) {
      out[q_item_idx] = a[q_item_idx][k_item_idx] > out[q_item_idx]
          ? a[q_item_idx][k_item_idx]
          : out[q_item_idx];
    }
  }
}

template <typename scalar_t, int kBlockSizeK, int kBlockSizeQ>
__device__ __forceinline__ void compute_scaling_coeffs(
    scalar_t m_i[kBlockSizeQ],
    scalar_t m_prime[kBlockSizeQ],
    scalar_t si[kBlockSizeQ][kBlockSizeK],
    scalar_t m_delta[kBlockSizeQ],
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK]) {
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++)
    m_delta[q_item_idx] = std::exp(m_prime[q_item_idx] - m_i[q_item_idx]);
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++)
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++)
      s_delta[q_item_idx][k_item_idx] =
          std::exp(si[q_item_idx][k_item_idx] - m_i[q_item_idx]);
}

template <typename scalar_t, int kBlockSizeK, int kBlockSizeQ>
__device__ __forceinline__ void update_scaling_coeffs(
    scalar_t m_delta[kBlockSizeQ],
    scalar_t m_i[kBlockSizeQ],
    scalar_t s_delta[kBlockSizeQ][kBlockSizeK],
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ]) {
#pragma unroll
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    s_prime[q_item_idx] = s_prime[q_item_idx] * m_delta[q_item_idx];
#pragma unroll
    for (int64_t k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++)
      s_prime[q_item_idx] += s_delta[q_item_idx][k_item_idx];

    m_prime[q_item_idx] = m_i[q_item_idx];
  }
}

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int BUFFER_SIZE>
__device__ void compute_loop(
    vec_t* query_block[kBlockSizeQ],
    vec_t* key_i,
    vec_t* value_i,
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
    int64_t K) {
  scalar_t si[kBlockSizeQ][kBlockSizeK] = {0};
  compute_dot<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ>(
      query_block, key_i, si, K);

  scalar_t m_i[kBlockSizeQ];
  compute_max<scalar_t, kBlockSizeK, kBlockSizeQ>(si, m_prime, m_i);

  scalar_t m_delta[kBlockSizeQ];
  scalar_t s_delta[kBlockSizeQ][kBlockSizeK];

  compute_scaling_coeffs<scalar_t, kBlockSizeK, kBlockSizeQ>(
      m_i, m_prime, si, m_delta, s_delta);

  compute_final_mult<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ, BUFFER_SIZE>(
      value_i, s_delta, m_delta, buffer, K);

  update_scaling_coeffs<scalar_t, kBlockSizeK, kBlockSizeQ>(
      m_delta, m_i, s_delta, m_prime, s_prime);
}

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeQ,
    int WARP_SIZE,
    int BUFFER_SIZE>
__device__ __forceinline__ void aggregate_coeffs(
    scalar_t m_prime[kBlockSizeQ],
    scalar_t s_prime[kBlockSizeQ],
    vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
    int64_t K) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    scalar_t m_i = m_prime[q_item_idx];
    scalar_t s_i = s_prime[q_item_idx];
    m_prime[q_item_idx] = warpMax<scalar_t, WARP_SIZE>(m_prime[q_item_idx]);
    scalar_t m_delta = std::exp(m_i - m_prime[q_item_idx]);
    scalar_t s_delta = s_i * m_delta;
    s_delta = warpSum<scalar_t, WARP_SIZE>(s_delta);
    s_prime[q_item_idx] = s_delta;
    for (int64_t k = 0; k < K / kVecSize; k += 1) {
      vec_t tmp = buffer[q_item_idx][k];
      iMul<scalar_t>(m_delta, &tmp);
      tmp = warpSum<vec_t, WARP_SIZE>(tmp);
      buffer[q_item_idx][k] = tmp;
    }
  }
}

template <
    bool first,
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int BUFFER_SIZE,
    int WARP_SIZE>
struct UnrollLoop {
  static __device__ __forceinline__ void eval(
      vec_t* query_block[kBlockSizeQ],
      at::TensorAccessor<scalar_t, 2> key,
      at::TensorAccessor<scalar_t, 2> value,
      scalar_t m_prime[kBlockSizeQ],
      scalar_t s_prime[kBlockSizeQ],
      vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
      int64_t K,
      int64_t N) {
    constexpr int64_t step = kBlockSizeK * WARP_SIZE;
    int64_t l;
    if (first) {
      l = threadIdx.x * kBlockSizeK;
    } else {
      l = N - (N & (2 * step - 1)) + threadIdx.x * kBlockSizeK;
    }
    // this is equivalent to N - N % step, but faster
    // guaranteed to be the same as step is a power of 2
    int64_t end_iter = kBlockSizeK == 1 ? N : N - (N & (step - 1));
    // if (l < end_iter) {
    {
      for (; l < end_iter; l += step) {
        auto key_i = reinterpret_cast<vec_t*>(key[l].data());
        auto value_i = reinterpret_cast<vec_t*>(value[l].data());

        compute_loop<scalar_t, vec_t, kBlockSizeK, kBlockSizeQ, BUFFER_SIZE>(
            query_block, key_i, value_i, m_prime, s_prime, buffer, K);
      }
    }
    {
      UnrollLoop<
          false,
          scalar_t,
          vec_t,
          kBlockSizeK / 2,
          kBlockSizeQ,
          BUFFER_SIZE,
          WARP_SIZE>::
          eval(query_block, key, value, m_prime, s_prime, buffer, K, N);
    }
  }
};

template <
    bool first,
    typename scalar_t,
    typename vec_t,
    int kBlockSizeQ,
    int BUFFER_SIZE,
    int WARP_SIZE>
struct UnrollLoop<
    first,
    scalar_t,
    vec_t,
    0,
    kBlockSizeQ,
    BUFFER_SIZE,
    WARP_SIZE> {
  static __device__ __forceinline__ void eval(
      vec_t* query_block[kBlockSizeQ],
      at::TensorAccessor<scalar_t, 2> key,
      at::TensorAccessor<scalar_t, 2> value,
      scalar_t m_prime[kBlockSizeQ],
      scalar_t s_prime[kBlockSizeQ],
      vec_t buffer[kBlockSizeQ][BUFFER_SIZE] /*TODO [BUFFER_SIZE limitation]*/,
      int64_t K,
      int64_t N) {}
};

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeK,
    int kBlockSizeQ,
    int WARP_SIZE,
    int BUFFER_SIZE,
    bool compute_logsumexp>
__global__ void attention_kernel(
    at::PackedTensorAccessor<scalar_t, 3> output,
    at::PackedTensorAccessor<scalar_t, 2> logsumexp,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value) {
  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);
  static_assert(
      integerIsPowerOf2(kBlockSizeK * WARP_SIZE),
      "kBlockSizeK * WARP_SIZE should be a power of 2");
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  int64_t batch_idx = blockIdx.y;
  int64_t query_idx =
      blockIdx.x * (blockDim.y * kBlockSizeQ) + threadIdx.y * kBlockSizeQ;

  if (query_idx >= M)
    return;

  vec_t* query_block[kBlockSizeQ];
  vec_t* output_block[kBlockSizeQ];
  scalar_t* logsumexp_block[kBlockSizeQ];
  // TODO [BUFFER_SIZE limitation]: the current strategy assumes a
  // statically-known size for K. Ideally we would like to remove this
  // limitation in the future, so that any K is supported
  vec_t buffer[kBlockSizeQ][BUFFER_SIZE] = {0};
  scalar_t s_prime[kBlockSizeQ] = {0};
  scalar_t m_prime[kBlockSizeQ];
  for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    int64_t index = query_idx + q_item_idx;
    index = index >= M ? M - 1 : index;
    query_block[q_item_idx] =
        reinterpret_cast<vec_t*>(query[batch_idx][index].data());
    output_block[q_item_idx] =
        reinterpret_cast<vec_t*>(output[batch_idx][index].data());
    m_prime[q_item_idx] = -std::numeric_limits<scalar_t>::infinity();
    logsumexp_block[q_item_idx] = &logsumexp[batch_idx][index];
  }

  // Computes s_prime, buffer (aka v_prime) and m_prime
  UnrollLoop<true, scalar_t, vec_t, kBlockSizeK, kBlockSizeQ, BUFFER_SIZE, WARP_SIZE>::eval(query_block, key[batch_idx], value[batch_idx], m_prime, s_prime, buffer, K, N);

  aggregate_coeffs<scalar_t, vec_t, kBlockSizeQ, WARP_SIZE, BUFFER_SIZE>(
      m_prime, s_prime, buffer, K);

  for (int64_t k = threadIdx.x; k < K / kVecSize; k += blockDim.x) {
    vec_t tmp;

#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      tmp = buffer[q_item_idx][k];
      iDiv<scalar_t>(s_prime[q_item_idx], &tmp);

      output_block[q_item_idx][k] = tmp;
    }
  }

  if (compute_logsumexp) {
#pragma unroll
    for (int64_t q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      *logsumexp_block[q_item_idx] =
          m_prime[q_item_idx] + std::log(s_prime[q_item_idx]);
    }
  }
}

template <bool compute_logsumexp>
void launch_attention(
    at::Tensor& res,
    at::Tensor& logsumexp,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  constexpr int WARP_SIZE = 4;

 constexpr int kBlockSizeK = 32;
 constexpr int kBlockSizeQ = 2;

  constexpr int TILE_SIZE = 32;
  constexpr int BUFFER_SIZE = 32;
//  constexpr int BUFFER_SIZE = 8;


  dim3 grid(ceil_div(M, int64_t(TILE_SIZE)), B);
  constexpr dim3 block(WARP_SIZE, TILE_SIZE / kBlockSizeQ);
  static_assert(block.x * block.y <= 512);

  using scalar_t = float;

  if ((K % 4) == 0) {
    TORCH_CHECK(
        K / 4 <= BUFFER_SIZE,
        "For now only a certain number of K values are supported. Let us know if you hit this and we will fix it");
    attention_kernel<
        scalar_t,
        float4,
        kBlockSizeK,
        kBlockSizeQ,
        WARP_SIZE,
        BUFFER_SIZE,
        compute_logsumexp><<<grid, block, 0, stream>>>(
        res.packed_accessor<scalar_t, 3>(),
        logsumexp.packed_accessor<scalar_t, 2>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>());
  } else if ((K % 2) == 0) {
    TORCH_CHECK(
        K / 2 <= BUFFER_SIZE,
        "For now only a certain number of K values are supported. Let us know if you hit this and we will fix it");
    attention_kernel<
        scalar_t,
        float2,
        kBlockSizeK,
        kBlockSizeQ,
        WARP_SIZE,
        BUFFER_SIZE,
        compute_logsumexp><<<grid, block, 0, stream>>>(
        res.packed_accessor<scalar_t, 3>(),
        logsumexp.packed_accessor<scalar_t, 2>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>());

  } else {
    TORCH_CHECK(
        K <= BUFFER_SIZE,
        "For now only a certain number of K values are supported. Let us know if you hit this and we will fix it");
    attention_kernel<
        scalar_t,
        float,
        kBlockSizeK,
        kBlockSizeQ,
        WARP_SIZE,
        BUFFER_SIZE,
        compute_logsumexp><<<grid, block, 0, stream>>>(
        res.packed_accessor<scalar_t, 3>(),
        logsumexp.packed_accessor<scalar_t, 2>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>());
  }
}

std::tuple<at::Tensor, at::Tensor> attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool compute_logsumexp
    // const at::Tensor& mask
) {
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  // TORCH_CHECK(query.dim() == mask.dim());
  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(
      query.size(2) ==
      value.size(2)); // TODO: drop this limitation in the future

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");

  // TODO drop this limitation in the future
  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  // TODO: support other dtypes in the future
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::Float,
      "Only float32 type is supported for now");

  at::cuda::CUDAGuard device_guard(query.device());

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor res = at::zeros({B, M, K}, query.options());
  at::Tensor logsumexp = at::empty({B, M}, query.options());

  // have to pass compute_logsumexp as a template parameter
  // otherwise there is a slowdown in the kernel...
  if (compute_logsumexp) {
    launch_attention<true>(res, logsumexp, query, key, value);
  } else {
    launch_attention<false>(res, logsumexp, query, key, value);
  }

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(res, logsumexp);
}

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeQ,
    int kBlockSizeK,
    int TILE_SIZEQ,
    int TILE_SIZEK,
    bool check_bounds>
__global__ void attention_backward_grad_v_kernel(
    at::PackedTensorAccessor<scalar_t, 3> grad_v,
    at::PackedTensorAccessor<scalar_t, 3> grad_out,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value,
    at::PackedTensorAccessor<scalar_t, 2> tmp_sum_i,
    at::PackedTensorAccessor<scalar_t, 2> logsumexp_normalizer) {
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);

  int64_t batch_idx = blockIdx.z;
  int64_t query_idx =
      blockIdx.x * blockDim.x * kBlockSizeQ + threadIdx.x * kBlockSizeQ;
  int64_t l = blockIdx.y * blockDim.y * kBlockSizeK + threadIdx.y * kBlockSizeK;

  __shared__ scalar_t fact[TILE_SIZEQ][TILE_SIZEK + 1];

#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      fact[kBlockSizeQ * threadIdx.x + q_item_idx]
          [kBlockSizeK * threadIdx.y + k_item_idx] = 0;
    }
  }

  scalar_t normalizer[kBlockSizeQ];
  scalar_t tmp_sum[kBlockSizeQ] = {0};

  vec_t *qb[kBlockSizeQ], *kb[kBlockSizeK], *vb[kBlockSizeK], *gb[kBlockSizeQ],
      *gbb[TILE_SIZEQ];
  scalar_t maskQ[kBlockSizeQ], maskK[kBlockSizeK];

  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
    int64_t index = l + k_item_idx;
    maskK[k_item_idx] = index >= N ? scalar_t(0) : scalar_t(1);
    if (check_bounds)
      index = min(index, N - 1);
    kb[k_item_idx] = reinterpret_cast<vec_t*>(key[batch_idx][index].data());
    vb[k_item_idx] = reinterpret_cast<vec_t*>(value[batch_idx][index].data());
  }

  for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    int64_t index = query_idx + q_item_idx;
    maskQ[q_item_idx] = index >= M ? scalar_t(0) : scalar_t(1);
    if (check_bounds)
      index = min(index, M - 1);
    qb[q_item_idx] = reinterpret_cast<vec_t*>(query[batch_idx][index].data());
    gb[q_item_idx] =
        reinterpret_cast<vec_t*>(grad_out[batch_idx][index].data());
  }

  for (int64_t i = 0; i < TILE_SIZEQ; i++) {
    int64_t index = query_idx + i - kBlockSizeQ * threadIdx.x;
    if (check_bounds)
      index = min(index, M - 1);
    gbb[i] = reinterpret_cast<vec_t*>(grad_out[batch_idx][index].data());
  }

  for (int i = 0; i < kBlockSizeQ; i++) {
    int64_t index = query_idx + i;
    if (check_bounds)
      index = min(index, M - 1);
    normalizer[i] = logsumexp_normalizer[batch_idx][index];
  }

  scalar_t attn_v[kBlockSizeQ][kBlockSizeK] = {0};
  scalar_t grad_attn_v[kBlockSizeQ][kBlockSizeK] = {0};
  scalar_t scale = 1.0 / std::sqrt(scalar_t(K));

  for (int64_t k = 0; k < K / kVecSize; k += 1) {
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      vec_t kk = __ldg(kb[k_item_idx] + k);
      iMul(scale, &kk);
      vec_t tt = __ldg(vb[k_item_idx] + k);
#pragma unroll
      for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
        sputnik::VectorCompute<vec_t>::Dot(
            __ldg(qb[q_item_idx] + k), kk, &attn_v[q_item_idx][k_item_idx]);
        sputnik::VectorCompute<vec_t>::Dot(
            __ldg(gb[q_item_idx] + k),
            tt,
            &grad_attn_v[q_item_idx][k_item_idx]);
      }
    }
  }
#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      attn_v[q_item_idx][k_item_idx] =
          std::exp(attn_v[q_item_idx][k_item_idx] - normalizer[q_item_idx]) *
          maskQ[q_item_idx] * maskK[k_item_idx];
    }
  }

#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      fact[kBlockSizeQ * threadIdx.x + q_item_idx]
          [kBlockSizeK * threadIdx.y + k_item_idx] =
              attn_v[q_item_idx][k_item_idx];
      tmp_sum[q_item_idx] +=
          attn_v[q_item_idx][k_item_idx] * grad_attn_v[q_item_idx][k_item_idx];
    }
  }
  __syncthreads();

  for (int64_t k = threadIdx.x; k < K / kVecSize; k += blockDim.x) {
    vec_t res[kBlockSizeK] = {0};
#pragma unroll
    for (int64_t i = 0; i < TILE_SIZEQ; i++) {
      vec_t kk = __ldg(gbb[i] + k);
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
        sputnik::VectorCompute<vec_t>::FMA(
            fact[i][kBlockSizeK * threadIdx.y + k_item_idx],
            kk,
            &res[k_item_idx]);
      }
    }
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      int64_t index = l + k_item_idx;
      if (check_bounds)
        index = min(index, N - 1);
      myGpuAtomicAdd(&grad_v[batch_idx][index][k * kVecSize], res[k_item_idx]);
    }
  }
  for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    int64_t index = query_idx + q_item_idx;
    if (check_bounds)
      index = min(index, M - 1);
    myGpuAtomicAdd(&tmp_sum_i[batch_idx][index], tmp_sum[q_item_idx]);
  }
}

template <
    typename scalar_t,
    typename vec_t,
    int kBlockSizeQ,
    int kBlockSizeK,
    int TILE_SIZEQ,
    int TILE_SIZEK,
    bool check_bounds>
__global__ void attention_backward_grad_qk_kernel(
    at::PackedTensorAccessor<scalar_t, 3> grad_q,
    at::PackedTensorAccessor<scalar_t, 3> grad_k,
    at::PackedTensorAccessor<scalar_t, 3> grad_out,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value,
    at::PackedTensorAccessor<scalar_t, 2> tmp_sum_i,
    at::PackedTensorAccessor<scalar_t, 2> logsumexp_normalizer) {
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  constexpr int kVecSize = sizeof(vec_t) / sizeof(scalar_t);

  int64_t batch_idx = blockIdx.z;
  int64_t query_idx =
      blockIdx.x * blockDim.x * kBlockSizeQ + threadIdx.x * kBlockSizeQ;
  int64_t l = blockIdx.y * blockDim.y * kBlockSizeK + threadIdx.y * kBlockSizeK;

  __shared__ scalar_t fact[TILE_SIZEQ][TILE_SIZEK + 1];

#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      fact[kBlockSizeQ * threadIdx.x + q_item_idx]
          [kBlockSizeK * threadIdx.y + k_item_idx] = 0;
    }
  }

  scalar_t normalizer[kBlockSizeQ];
  scalar_t tmp_sum[kBlockSizeQ];

  vec_t *qb[kBlockSizeQ], *kb[kBlockSizeK], *vb[kBlockSizeK], *gb[kBlockSizeQ],
      *qbb[TILE_SIZEQ], *kbb[TILE_SIZEK];
  scalar_t maskQ[kBlockSizeQ], maskK[kBlockSizeK];

  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
    int64_t index = l + k_item_idx;
    maskK[k_item_idx] = index >= N ? scalar_t(0) : scalar_t(1);
    if (check_bounds)
      index = min(index, N - 1);
    kb[k_item_idx] = reinterpret_cast<vec_t*>(key[batch_idx][index].data());
    vb[k_item_idx] = reinterpret_cast<vec_t*>(value[batch_idx][index].data());
  }

  for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
    int64_t index = query_idx + q_item_idx;
    maskQ[q_item_idx] = index >= M ? scalar_t(0) : scalar_t(1);
    if (check_bounds)
      index = min(index, M - 1);
    qb[q_item_idx] = reinterpret_cast<vec_t*>(query[batch_idx][index].data());
    gb[q_item_idx] =
        reinterpret_cast<vec_t*>(grad_out[batch_idx][index].data());
  }
  for (int64_t i = 0; i < TILE_SIZEQ; i++) {
    int64_t index = query_idx + i - kBlockSizeQ * threadIdx.x;
    if (check_bounds)
      index = min(index, M - 1);
    qbb[i] = reinterpret_cast<vec_t*>(query[batch_idx][index].data());
  }

  for (int64_t i = 0; i < TILE_SIZEK; i++) {
    int64_t index = l + i - kBlockSizeK * threadIdx.y;
    if (check_bounds)
      index = min(index, N - 1);
    kbb[i] = reinterpret_cast<vec_t*>(key[batch_idx][index].data());
  }

  for (int i = 0; i < kBlockSizeQ; i++) {
    int64_t index = query_idx + i;
    if (check_bounds)
      index = min(index, M - 1);
    normalizer[i] = logsumexp_normalizer[batch_idx][index];
    tmp_sum[i] = tmp_sum_i[batch_idx][index];
  }

  scalar_t attn_v[kBlockSizeQ][kBlockSizeK] = {0};
  scalar_t grad_attn_v[kBlockSizeQ][kBlockSizeK] = {0};
  scalar_t scale = 1.0 / std::sqrt(scalar_t(K));

  for (int64_t k = 0; k < K / kVecSize; k += 1) {
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      vec_t kk = __ldg(kb[k_item_idx] + k);
      iMul(scale, &kk);
      vec_t tt = __ldg(vb[k_item_idx] + k);
#pragma unroll
      for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
        sputnik::VectorCompute<vec_t>::Dot(
            __ldg(qb[q_item_idx] + k), kk, &attn_v[q_item_idx][k_item_idx]);
        sputnik::VectorCompute<vec_t>::Dot(
            __ldg(gb[q_item_idx] + k),
            tt,
            &grad_attn_v[q_item_idx][k_item_idx]);
      }
    }
  }
#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      attn_v[q_item_idx][k_item_idx] =
          std::exp(attn_v[q_item_idx][k_item_idx] - normalizer[q_item_idx]) *
          maskQ[q_item_idx] * maskK[k_item_idx];
    }
  }

#pragma unroll
  for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      fact[kBlockSizeQ * threadIdx.x + q_item_idx]
          [kBlockSizeK * threadIdx.y + k_item_idx] =
              attn_v[q_item_idx][k_item_idx] * scale *
          (grad_attn_v[q_item_idx][k_item_idx] - tmp_sum[q_item_idx]);
    }
  }
  __syncthreads();

  for (int64_t k = threadIdx.y; k < K / kVecSize; k += blockDim.y) {
    vec_t res[kBlockSizeQ] = {0};
#pragma unroll
    for (int64_t i = 0; i < TILE_SIZEK; i++) {
      vec_t kk = __ldg(kbb[i] + k);
#pragma unroll
      for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
        sputnik::VectorCompute<vec_t>::FMA(
            fact[kBlockSizeQ * threadIdx.x + q_item_idx][i],
            kk,
            &res[q_item_idx]);
      }
    }
#pragma unroll
    for (int q_item_idx = 0; q_item_idx < kBlockSizeQ; q_item_idx++) {
      int64_t index = query_idx + q_item_idx;
      if (check_bounds)
        index = min(index, M - 1);
      myGpuAtomicAdd(&grad_q[batch_idx][index][k * kVecSize], res[q_item_idx]);
    }
  }

  for (int64_t k = threadIdx.x; k < K / kVecSize; k += blockDim.x) {
    vec_t res[kBlockSizeK] = {0};
#pragma unroll
    for (int64_t i = 0; i < TILE_SIZEQ; i++) {
      vec_t kk = __ldg(qbb[i] + k);
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
        sputnik::VectorCompute<vec_t>::FMA(
            fact[i][kBlockSizeK * threadIdx.y + k_item_idx],
            kk,
            &res[k_item_idx]);
      }
    }
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockSizeK; k_item_idx++) {
      int64_t index = l + k_item_idx;
      if (check_bounds)
        index = min(index, N - 1);
      myGpuAtomicAdd(&grad_k[batch_idx][index][k * kVecSize], res[k_item_idx]);
    }
  }
}

template <typename scalar_t, typename vec_t>
void launch_attention_backward(
    at::Tensor& grad_q,
    at::Tensor& grad_k,
    at::Tensor& grad_v,
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    at::Tensor& tmp_sum_i) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  constexpr int TILE_SIZEQ = 32;
  constexpr int TILE_SIZEK = 32;

  constexpr int64_t kBlockSizeQ = 4;
  constexpr int64_t kBlockSizeK = 8;

  dim3 grid(
      ceil_div(M, int64_t(TILE_SIZEQ)), ceil_div(N, int64_t(TILE_SIZEK)), B);
  dim3 block(TILE_SIZEQ / kBlockSizeQ, TILE_SIZEK / kBlockSizeK);

  constexpr int TILE_SIZEQ2 = 32;
  constexpr int TILE_SIZEK2 = 32;

  constexpr int64_t kBlockSizeQ2 = 4;
  constexpr int64_t kBlockSizeK2 = 4;

  dim3 grid2(
      ceil_div(M, int64_t(TILE_SIZEQ2)), ceil_div(N, int64_t(TILE_SIZEK2)), B);
  dim3 block2(TILE_SIZEQ2 / kBlockSizeQ2, TILE_SIZEK2 / kBlockSizeK2);

  // the bounds checking in device code is very expensive, making the code
  // around 25% slower. So let's skip those checks if possible.
  if ((M % TILE_SIZEQ == 0) && (N % TILE_SIZEK == 0)) {
    attention_backward_grad_v_kernel<
        scalar_t,
        vec_t,
        kBlockSizeQ,
        kBlockSizeK,
        TILE_SIZEQ,
        TILE_SIZEK,
        false><<<grid, block, 0, stream>>>(
        grad_v.packed_accessor<scalar_t, 3>(),
        grad_out.packed_accessor<scalar_t, 3>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>(),
        tmp_sum_i.packed_accessor<scalar_t, 2>(),
        logsumexp.packed_accessor<scalar_t, 2>());
  } else {
    attention_backward_grad_v_kernel<
        scalar_t,
        vec_t,
        kBlockSizeQ,
        kBlockSizeK,
        TILE_SIZEQ,
        TILE_SIZEK,
        true><<<grid, block, 0, stream>>>(
        grad_v.packed_accessor<scalar_t, 3>(),
        grad_out.packed_accessor<scalar_t, 3>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>(),
        tmp_sum_i.packed_accessor<scalar_t, 2>(),
        logsumexp.packed_accessor<scalar_t, 2>());
  }

  if ((M % TILE_SIZEQ2 == 0) && (N % TILE_SIZEK2 == 0)) {
    attention_backward_grad_qk_kernel<
        scalar_t,
        vec_t,
        kBlockSizeQ2,
        kBlockSizeK2,
        TILE_SIZEQ2,
        TILE_SIZEK2,
        false><<<grid2, block2, 0, stream>>>(
        grad_q.packed_accessor<scalar_t, 3>(),
        grad_k.packed_accessor<scalar_t, 3>(),
        grad_out.packed_accessor<scalar_t, 3>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>(),
        tmp_sum_i.packed_accessor<scalar_t, 2>(),
        logsumexp.packed_accessor<scalar_t, 2>());
  } else {
    attention_backward_grad_qk_kernel<
        scalar_t,
        vec_t,
        kBlockSizeQ2,
        kBlockSizeK2,
        TILE_SIZEQ2,
        TILE_SIZEK2,
        true><<<grid2, block2, 0, stream>>>(
        grad_q.packed_accessor<scalar_t, 3>(),
        grad_k.packed_accessor<scalar_t, 3>(),
        grad_out.packed_accessor<scalar_t, 3>(),
        query.packed_accessor<scalar_t, 3>(),
        key.packed_accessor<scalar_t, 3>(),
        value.packed_accessor<scalar_t, 3>(),
        tmp_sum_i.packed_accessor<scalar_t, 2>(),
        logsumexp.packed_accessor<scalar_t, 2>());
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp
    // const at::Tensor& mask
) {
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  // TORCH_CHECK(query.dim() == mask.dim());
  TORCH_CHECK(query.dim() == 3);

  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));
  TORCH_CHECK(query.size(2) == grad_out_.size(2));

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(
      query.size(2) ==
      value.size(2)); // TODO: drop this limitation in the future

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(grad_out_.is_cuda(), "grad_out must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");
  TORCH_CHECK(!grad_out_.is_sparse(), "grad_out must be a dense tensor");

  // TODO drop this limitation in the future
  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  // TODO: support other dtypes in the future
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::Float,
      "Only float32 type is supported for now");

  at::cuda::CUDAGuard device_guard(query.device());

  // handle potentially non-contiguous grad_out through a copy
  auto grad_out = grad_out_.contiguous();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor grad_q = at::zeros_like(query);
  at::Tensor grad_k = at::zeros_like(key);
  at::Tensor grad_v = at::zeros_like(value);

  at::Tensor tmp_sum_i = at::zeros({B, M}, query.options());

  // using scalar_t = float;
  // using vec_t = float4;
  // using vec_t = float;

  if ((K % 4) == 0) {
    launch_attention_backward<float, float4>(
        grad_q,
        grad_k,
        grad_v,
        grad_out,
        query,
        key,
        value,
        logsumexp,
        tmp_sum_i);
  } else if ((K % 2) == 0) {
    launch_attention_backward<float, float2>(
        grad_q,
        grad_k,
        grad_v,
        grad_out,
        query,
        key,
        value,
        logsumexp,
        tmp_sum_i);
  } else {
    launch_attention_backward<float, float>(
        grad_q,
        grad_k,
        grad_v,
        grad_out,
        query,
        key,
        value,
        logsumexp,
        tmp_sum_i);
  }

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(grad_q, grad_k, grad_v);
}

namespace dhaziza_custom_matmull {
constexpr auto NUM_WARPS = 32u;

template <
    typename scalar_t,
    typename vec_t>
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

namespace dhaziza_custom_matmull {

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
                           int64_t blockAxisN) {
  // Shared storage needed by threadblock-scoped matrix multiply-accumulate
  __shared__ typename Mma::SharedStorage shared_storage;

  // Compute threadblock location
  cutlass::gemm::GemmCoord tb_tile_offset = {
    int(blockIdx.y), // M
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
    typename scalar_t,
    typename vec_t>
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
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
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
    typename scalar_t,
    typename vec_t>
__device__ void kernel_cutlass_single(
    at::TensorAccessor<scalar_t, 2> query,
    at::TensorAccessor<scalar_t, 2> key,
    at::TensorAccessor<scalar_t, 2> out) {

  using P = GemmParams<scalar_t, vec_t>;

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
        n
    );
  }
}


template <
    typename scalar_t,
    typename vec_t>
__global__ void kernel_cutlass(
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> out) {
  at::TensorAccessor<scalar_t, 2> query_s = query[blockIdx.x];
  kernel_cutlass_single<scalar_t, vec_t>(query_s, key[blockIdx.x], out[blockIdx.x]);
}

at::Tensor launch(
    const at::Tensor& query,
    const at::Tensor& key
) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");

  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());

  at::cuda::CUDAGuard device_guard(query.device());

  using P = GemmParams<float, float>;

  int64_t M = query.size(1);
  int64_t K = query.size(2);
  int64_t N = key.size(1);
  at::Tensor out = at::zeros({query.size(0), M, N}, query.options());
  dim3 grid(query.size(0), ceil_div(M, int64_t(P::ThreadblockShape::kM)));
  dim3 block(32, P::kNumWarps);

  kernel_cutlass<float, float><<<grid, block>>>(
      query.packed_accessor<float, 3>(),
      key.packed_accessor<float, 3>(),
      out.packed_accessor<float, 3>());
  return out;
}
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention"),
      TORCH_FN(attention));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward"),
      TORCH_FN(attention_backward));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::dhaziza_custom_matmull"),
      TORCH_FN(dhaziza_custom_matmull::launch));
}
