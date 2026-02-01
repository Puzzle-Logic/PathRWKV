#include <torch/extension.h>
#include "ATen/ATen.h"

template <typename scalar_t>
void cuda_forward(int B, int T, int C, int H, scalar_t *r, scalar_t *k, scalar_t *v, scalar_t *w, scalar_t *u, scalar_t *y);

template <typename scalar_t>
void cuda_backward(int B, int T, int C, int H, scalar_t *r, scalar_t *k, scalar_t *v, scalar_t *w, scalar_t *u, scalar_t *gy, scalar_t *gr, scalar_t *gk, scalar_t *gv, scalar_t *gw, scalar_t *gu);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(r.scalar_type(), "wkv6_forward", ([&] {
        cuda_forward<scalar_t>(B, T, C, H,
            r.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>());
    }));
}

void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(r.scalar_type(), "wkv6_backward", ([&] {
        cuda_backward<scalar_t>(B, T, C, H,
            r.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            gy.data_ptr<scalar_t>(),
            gr.data_ptr<scalar_t>(),
            gk.data_ptr<scalar_t>(),
            gv.data_ptr<scalar_t>(),
            gw.data_ptr<scalar_t>(),
            gu.data_ptr<scalar_t>());
    }));
}

TORCH_LIBRARY(wkv6_parallel, m) {
    m.def("forward(int B, int T, int C, int H, Tensor r, Tensor k, Tensor v, Tensor w, Tensor u, Tensor(a!) y) -> ()");
    m.def("backward(int B, int T, int C, int H, Tensor r, Tensor k, Tensor v, Tensor w, Tensor u, Tensor gy, Tensor(a!) gr, Tensor(b!) gk, Tensor(c!) gv, Tensor(d!) gw, Tensor(e!) gu) -> ()");
}

TORCH_LIBRARY_IMPL(wkv6_parallel, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}