import os
import torch
from torch import nn
from pathlib import Path

major, minor = torch.cuda.get_device_capability()
arch = f"{major}.{minor}"
os.environ["TORCH_CUDA_ARCH_LIST"] = arch
from torch.utils.cpp_extension import load

from .pe import SlidePE

HEAD_SIZE = 64
MAX_N_TILES_TRAIN = 2000
CHUNK_LEN_INFERENCE = 2000
ROOT_PATH = Path(__file__).resolve().parent

load(
    name="wkv6_parallel",
    is_python_module=False,
    sources=[
        str(ROOT_PATH / "cuda" / "wkv6_op.cpp"),
        str(ROOT_PATH / "cuda" / "wkv6.cu"),
    ],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        f"-D_N_={HEAD_SIZE}",
        f"-D_T_={MAX_N_TILES_TRAIN}",
    ],
)
load(
    name="wkv6_state",
    is_python_module=False,
    sources=[
        str(ROOT_PATH / "cuda" / "wkv6state_op.cpp"),
        str(ROOT_PATH / "cuda" / "wkv6state.cu"),
    ],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        f"-D_N_={HEAD_SIZE}",
        f"-D_T_={CHUNK_LEN_INFERENCE}",
    ],
)


class WKV_6_Parallel(torch.autograd.Function):
    @staticmethod
    def create_tensor(shape, device, requires_grad=False):
        return torch.empty(
            shape,
            device=device,
            requires_grad=requires_grad,
            memory_format=torch.contiguous_format,
        )

    @staticmethod
    def forward(ctx, r, k, v, w, u):
        with torch.no_grad():
            B, T, C = r.size()
            N = C // HEAD_SIZE
            ctx.B, ctx.T, ctx.C = B, T, C
            r, k, v, w, u = [i.float() for i in [r, k, v, w, u]]
            ctx.save_for_backward(r, k, v, w, u)
            y = WKV_6_Parallel.create_tensor((B, T, C), r.device, True)
            torch.ops.wkv6_parallel.forward(B, T, C, N, r, k, v, w, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B, T, C = ctx.B, ctx.T, ctx.C
            N = C // HEAD_SIZE
            r, k, v, w, u = ctx.saved_tensors
            gr, gk, gv, gw = [
                WKV_6_Parallel.create_tensor((B, T, C), gy.device) for _ in range(4)
            ]
            gu = WKV_6_Parallel.create_tensor((B, C), gy.device)
            torch.ops.wkv6_parallel.backward(
                B, T, C, N, r, k, v, w, u, gy, gr, gk, gv, gw, gu
            )
            gu = torch.sum(gu, 0).view(N, HEAD_SIZE)
            return (gr, gk, gv, gw, gu)


def cuda_wkv_parallel(r, k, v, w, u):
    return WKV_6_Parallel.apply(r, k, v, w, u)


class WKV_6_State(torch.autograd.Function):
    @staticmethod
    def create_tensor(shape, device, requires_grad=False):
        return torch.empty(
            shape,
            device=device,
            requires_grad=requires_grad,
            memory_format=torch.contiguous_format,
        )

    @staticmethod
    def forward(ctx, r, k, v, w, u, s):
        with torch.no_grad():
            B, T, C = r.size()
            N = C // HEAD_SIZE
            ctx.B, ctx.T, ctx.C = B, T, C
            r, k, v, w, u, s = [i.float() for i in [r, k, v, w, u, s]]
            ctx.save_for_backward(r, k, v, w, u, s)
            y = WKV_6_State.create_tensor((B, T, C), r.device, True)
            torch.ops.wkv6_state.forward(B, T, C, N, r, k, v, w, u, s, y)
            return y, s

    @staticmethod
    def backward(ctx, gy, gs):
        with torch.no_grad():
            B, T, C = ctx.B, ctx.T, ctx.C
            N = C // HEAD_SIZE
            r, k, v, w, u, s = ctx.saved_tensors
            gr, gk, gv, gw = [
                WKV_6_State.create_tensor((B, T, C), gy.device) for _ in range(4)
            ]
            gu = WKV_6_State.create_tensor((B, C), gy.device)
            torch.ops.wkv6_state.backward(
                B, T, C, N, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs
            )
            gu = torch.sum(gu, 0).view(N, HEAD_SIZE)
            return gr, gk, gv, gw, gu, gs


def cuda_wkv_state(r, k, v, w, u, s):
    return WKV_6_State.apply(r, k, v, w, u, s)


class TimeMix(nn.Module):
    def __init__(self, embed_dim, depth, layer_id):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.layer_id = layer_id
        self.head_size = HEAD_SIZE
        self.n_head = self.embed_dim // self.head_size

        ratio_0_to_1 = layer_id / (self.depth - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / self.depth)
        ddd = torch.ones(1, 1, self.embed_dim)
        for i in range(self.embed_dim):
            ddd[0, 0, i] = i / self.embed_dim

        # Time mix params
        self.miu_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0**0.9))
        self.lambda_ = nn.Parameter(
            torch.stack(
                [
                    1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0),  # lambda_w
                    1.0
                    - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
                    - 0.4 * ratio_0_to_1,  # lambda_k
                    1.0
                    - torch.pow(ddd, 0.4 * ratio_1_to_almost0)
                    - 0.6 * ratio_0_to_1,  # lambda_v
                    1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),  # lambda_r
                    1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),  # lambda_g
                ]
            )
        )

        self.A = nn.Parameter(torch.zeros(self.embed_dim, 32 * 5))
        self.B = nn.Parameter(torch.zeros(5, 32, self.embed_dim).uniform_(-0.01, 0.01))

        # Time decay params
        decay_speed = torch.ones(self.embed_dim)
        for n in range(self.embed_dim):
            decay_speed[n] = -6 + 5.5 * (n / (self.embed_dim - 1)) ** (
                0.85 + 1.0 * ratio_0_to_1**0.5
            )
        self.time_decay_miu = nn.Parameter(decay_speed.reshape(1, 1, self.embed_dim))

        self.time_decay_A = nn.Parameter(torch.zeros(self.embed_dim, 64))
        self.time_decay_B = nn.Parameter(
            torch.zeros(64, self.embed_dim).uniform_(-0.01, 0.01)
        )

        # Bonus
        tmp = torch.zeros(self.embed_dim)
        for n in range(self.embed_dim):
            zigzag = ((n + 1) % 3 - 1) * 0.1
            tmp[n] = ratio_0_to_1 * (2.5 - (n / (self.embed_dim - 1))) + zigzag
        self.u = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.W_k, self.W_v, self.W_r, self.W_o = [
            nn.Linear(self.embed_dim, self.embed_dim, bias=False) for _ in range(4)
        ]
        self.W_r.weight.data.uniform_(
            -0.5 / (self.embed_dim**0.5), 0.5 / (self.embed_dim**0.5)
        )
        self.W_k.weight.data.uniform_(
            -0.05 / (self.embed_dim**0.5), 0.05 / (self.embed_dim**0.5)
        )
        self.W_v.weight.data.uniform_(
            -0.5 / (self.embed_dim**0.5), 0.5 / (self.embed_dim**0.5)
        )
        self.W_o.weight.data.zero_()

        self.W_g_1 = nn.Parameter(
            torch.zeros(self.embed_dim, 160).uniform_(-0.01, 0.01)
        )
        self.W_g_2 = nn.Parameter(
            torch.zeros(160, self.embed_dim).uniform_(-0.01, 0.01)
        )

        self.ln_x = nn.GroupNorm(self.n_head, self.embed_dim, eps=1e-5 * self.n_head)

    @staticmethod
    def lerp(a, b_a, miu):
        return a + b_a * miu

    @staticmethod
    def lora(x, A, B, lambda_=None):
        return (
            lambda_ + torch.tanh(x @ A) @ B
            if lambda_ is not None
            else torch.tanh(x @ A) @ B
        )

    @staticmethod
    def batch_lora(x, A, B, lambda_, batch_size=5):
        b, t, _ = x.size()
        x = torch.tanh(x @ A).view(batch_size, b * t, -1)
        x = torch.bmm(x, B).view(batch_size, b, t, -1)
        x = lambda_ + x
        return x

    @staticmethod
    def ddlerp(a, b, miu_x, A, B, lambda_):
        b_a = b - a
        x = TimeMix.lerp(a, b_a, miu_x)
        miu = TimeMix.batch_lora(x, A, B, lambda_)
        x = TimeMix.lerp(a, b_a, miu)
        return x

    def forward(self, x, state=None):
        B, T, C = x.size()
        x_last = self.time_shift(x)
        x_ddlerp = self.ddlerp(x, x_last, self.miu_x, self.A, self.B, self.lambda_)
        w, k, v, r, g = x_ddlerp.unbind(dim=0)

        w = self.lora(w, self.time_decay_A, self.time_decay_B, self.time_decay_miu)
        k = self.W_k(k) * torch.clamp(w, max=0).exp()
        v, r = self.W_v(v), self.W_r(r)
        g = self.lora(g, self.W_g_1, self.W_g_2)

        if state is None:
            x = cuda_wkv_parallel(r, k, v, w, self.u)
            out_state = None
        else:
            x, out_state = cuda_wkv_state(r, k, v, w, self.u, state)

        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C)
        x = self.W_o(x * g)

        if state is None:
            return x
        else:
            return x, out_state


class ChannelMix(nn.Module):
    def __init__(self, embed_dim, depth, layer_id):
        super().__init__()
        self.layer_id = layer_id

        ratio_1_to_almost0 = 1.0 - (layer_id / depth)  # 1 to ~0
        ddd = torch.ones(1, 1, embed_dim)
        for i in range(embed_dim):
            ddd[0, 0, i] = i / embed_dim
        self.miu_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        self.miu_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_r = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        x_last = self.time_shift(x)
        k = TimeMix.lerp(x, x_last - x, self.miu_k)
        r = TimeMix.lerp(x, x_last - x, self.miu_r)

        k = self.W_k(k)
        k = torch.relu(k) ** 2
        v = self.W_v(k)
        r = torch.sigmoid(self.W_r(r))
        return r * v


class Block(nn.Module):
    def __init__(self, embed_dim, depth, block_id):
        super().__init__()
        self.time_mix = TimeMix(embed_dim, depth, block_id)
        self.channel_mix = ChannelMix(embed_dim, depth, block_id)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, state=None):
        if state is None:
            x = x + self.time_mix(self.ln1(x))
            x = x + self.channel_mix(self.ln2(x))
            return x
        else:
            x_time, new_state = self.time_mix(self.ln1(x), state)
            x = x + x_time
            x = x + self.channel_mix(self.ln2(x))
            return x, new_state


class PathRWKVv6(nn.Module):
    def __init__(self, args, embed_dim=768, n_layers=2, slide_ngrids=1200):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.input_dim = embed_dim
        self.output_dim = embed_dim
        self.MTL_token_num = len(args.tasks)
        self.pos_embed = SlidePE(embed_dim, slide_ngrids)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, n_layers, blk) for blk in range(n_layers)]
        )
        self.MTL_module = nn.Linear(embed_dim, self.MTL_token_num * embed_dim)

    def forward_parallel(self, x, coords):
        B, N, D = x.shape
        x = self.pos_embed(x, coords)

        for block in self.blocks:
            x = block(x)

        x = self.MTL_module(x).view(B, N, self.MTL_token_num, self.embed_dim)
        x = torch.amax(x, dim=1)
        return x

    def forward_chunked(self, x, coords):
        B, N, D = x.shape
        state = [
            torch.zeros(
                (B, self.embed_dim // HEAD_SIZE, HEAD_SIZE, HEAD_SIZE),
                device=x.device,
                dtype=torch.float32,
            )
            for _ in range(len(self.blocks))
        ]

        global_logits = torch.full(
            (B, self.MTL_token_num, self.embed_dim), float("-inf"), device=x.device
        )
        num_chunks = (N + CHUNK_LEN_INFERENCE - 1) // CHUNK_LEN_INFERENCE

        for i in range(num_chunks):
            start_idx = i * CHUNK_LEN_INFERENCE
            end_idx = min((i + 1) * CHUNK_LEN_INFERENCE, N)

            chunk_x = x[:, start_idx:end_idx, :]
            chunk_coords = coords[:, start_idx:end_idx, :]
            chunk_x = self.pos_embed(chunk_x, chunk_coords)

            for block_idx, block in enumerate(self.blocks):
                chunk_x, state[block_idx] = block(chunk_x, state[block_idx])

            chunk_logits = self.MTL_module(chunk_x)
            chunk_logits = chunk_logits.view(B, -1, self.MTL_token_num, self.embed_dim)

            current_chunk_max = torch.amax(chunk_logits, dim=1)
            global_logits = torch.maximum(global_logits, current_chunk_max)

            del chunk_x, chunk_logits

        return global_logits

    def forward(self, x, coords):
        return (
            self.forward_parallel(x, coords)
            if self.training
            else self.forward_chunked(x, coords)
        )
