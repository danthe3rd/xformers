#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention(Tensor query, Tensor key, Tensor value, bool compute_logsumexp) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention2(Tensor query, Tensor key, Tensor value, bool compute_logsumexp) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor logsumexp) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::dhaziza_custom_matmull(Tensor query, Tensor key) -> Tensor"));
}
