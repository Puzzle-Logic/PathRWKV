#include <stdio.h>
#include <assert.h>
#include <torch/extension.h>
#include "ATen/ATen.h"

template <typename F> __device__ __forceinline__ float float2float(F x) { return static_cast<float>(x); }
__device__ __forceinline__ float float2float(float x) { return x; }
__device__ __forceinline__ float float2float(double x) { return static_cast<float>(x); }

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _u += h*_N_;

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    float state[_N_] = {0};

    __syncthreads();
    u[i] = float2float(_u[i]); 
    __syncthreads();

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        w[i] = __expf(-__expf(float2float(_w[t])));
        r[i] = float2float(_r[t]);
        k[i] = float2float(_k[t]);
        __syncthreads();

        const float v = float2float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            float x = k[j] * v;
            
            y += r[j] * (u[j] * x + s);
            s = s * w[j] + x;
        }
        _y[t] = F(y);
    }
}

template <typename F>
__global__ void kernel_backward_101(const int B, const int L, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ _w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gr, F *__restrict__ const _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float v[_N_], gy[_N_];
    const float u = float2float(_u[h*_N_ + i]);
    float state[_N_] = {0};

    const int t_0 = b*L*C + h*_N_ + i;
    const int t_T = t_0 + L*C;

    float gu = 0;
    for (int t = t_0; t < t_T; t += C)
    {
        __syncthreads();
        v[i] = float2float(_v[t]);
        gy[i] = float2float(_gy[t]);
        __syncthreads();

        const float k = float2float(_k[t]);
        const float w = __expf(-__expf(float2float(_w[t])));
        float gr = 0, gu_ = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            float x = k * v[j];

            gr += (u * x + s) * gy[j];
            gu_ += x * gy[j];
            s = s * w + x;
        }
        _gr[t] = F(gr);
        gu += float2float(_r[t]) * gu_;
    }
    _gu[b*C + h*_N_ + i] = F(gu);
}

template <typename F>
__global__ void kernel_backward_102(const int B, const int L, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ _w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gk)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    
    __shared__ float v[_N_], gy[_N_];
    const float u = float2float(_u[h*_N_ + i]);
    float scccc[_N_] = {0};

    const int t_0 = b*L*C + h*_N_ + i;
    const int t_T_1 = t_0 + (L-1)*C;

    for (int t = t_T_1; t >= t_0; t -= C)
    {
        __syncthreads();
        v[i] = float2float(_v[t]);
        gy[i] = float2float(_gy[t]);
        __syncthreads();

        const float rr = float2float(_r[t]);
        const float w = __expf(-__expf(float2float(_w[t])));
        float gk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = rr * gy[j];
            
            gk += (u * x + s) * v[j];
            s = x + s * w;
        }
        _gk[t] = F(gk);
    }
}

template <typename F>
__global__ void kernel_backward_103(const int B, const int L, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ _w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gv)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _u += h*_N_;

    __shared__ float u_[_N_], r[_N_], k[_N_], w_[_N_];
    __syncthreads();
    u_[i] = float2float(_u[i]);
    __syncthreads();

    float sdddd[_N_] = {0};
    const int t_0 = b*L*C + h*_N_ + i;
    const int t_T_1 = t_0 + (L-1)*C;

    for (int t = t_T_1; t >= t_0; t -= C)
    {
        __syncthreads();
        r[i] = float2float(_r[t]);
        k[i] = float2float(_k[t]);
        w_[i] = __expf(-__expf(float2float(_w[t])));
        __syncthreads();

        const float gyy = float2float(_gy[t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = sdddd[j];
            float x = gyy * r[j];
            
            gv += (u_[j] * x + s) * k[j];
            s = x + s * w_[j];
        }
        _gv[t] = F(gv);
    }
}

template <typename F>
__global__ void kernel_backward_201(const int B, const int L, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ _w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gw)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float v[_N_], gy[_N_];
    float saaaa[_N_] = {0}, sbbbb[_T_-2] = {0}, scccc[_N_] = {0};

    const int t_0 = b*L*C + h*_N_ + i;
    const int t_1 = t_0 + C;
    const int t_2 = t_0 + 2*C;
    const int t_T_1 = t_0 + (L-1)*C;

    for (int t = t_T_1; t > t_1; t -= C)
    {
        __syncthreads();
        gy[i] = float2float(_gy[t]);
        v[i] = float2float(_v[t-2*C]);
        __syncthreads();

        const float r = float2float(_r[t]);
        const float w = __expf(-__expf(float2float(_w[t-C])));
        float sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = saaaa[j];
            float x = r * gy[j];
            s = (s + x) * w;
            sum += s * v[j];
        }
        sbbbb[(t-t_2)/C] = sum * float2float(_k[t-2*C]);
    }

    float sss = sbbbb[0];
    
    _gw[t_0] = F(0.0f);
    _gw[t_1] = F(sss * -__expf(float2float(_w[t_1])));

    for (int t = t_2; t < t_T_1; t += C)
    {
        __syncthreads();
        gy[i] = float2float(_gy[t]);
        v[i] = float2float(_v[t-2*C]);
        __syncthreads();

        const float w = __expf(-__expf(float2float(_w[t-C])));
        const float k = float2float(_k[t-2*C]);
        float sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = k * v[j];
            s = (s + x) * w;
            sum += s * gy[j];
        }
        sss += sbbbb[(t-t_1)/C] - (sum * float2float(_r[t]));
        _gw[t] = F(sss * -__expf(float2float(_w[t])));
    }
    _gw[t_T_1] = F(0.0f);
}

template <typename scalar_t>
void cuda_forward(int B, int T, int C, int H, scalar_t *r, scalar_t *k, scalar_t *v, scalar_t *w, scalar_t *u, scalar_t *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, y);
}

template <typename scalar_t>
void cuda_backward(int B, int T, int C, int H, scalar_t *r, scalar_t *k, scalar_t *v, scalar_t *w, scalar_t *u, scalar_t *gy, scalar_t *gr, scalar_t *gk, scalar_t *gv, scalar_t *gw, scalar_t *gu)
{
    assert(H*_N_ == C);
    kernel_backward_101<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gr, gu);
    kernel_backward_102<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gk);
    kernel_backward_103<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gv);
    kernel_backward_201<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gw);
}

template void cuda_forward<float>(int, int, int, int, float*, float*, float*, float*, float*, float*);
template void cuda_forward<double>(int, int, int, int, double*, double*, double*, double*, double*, double*);
template void cuda_forward<at::Half>(int, int, int, int, at::Half*, at::Half*, at::Half*, at::Half*, at::Half*, at::Half*);
template void cuda_forward<at::BFloat16>(int, int, int, int, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*);

template void cuda_backward<float>(int, int, int, int, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*);
template void cuda_backward<double>(int, int, int, int, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
template void cuda_backward<at::Half>(int, int, int, int, at::Half*, at::Half*, at::Half*, at::Half*, at::Half*, at::Half*, at::Half*, at::Half*, at::Half*, at::Half*, at::Half*);
template void cuda_backward<at::BFloat16>(int, int, int, int, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*, at::BFloat16*);