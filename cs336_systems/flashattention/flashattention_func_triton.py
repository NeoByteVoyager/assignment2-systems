import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config(
            {"Q_TILE_SIZE": 16, "K_TILE_SIZE": 16},
        ),
        triton.Config(
            {"Q_TILE_SIZE": 16, "K_TILE_SIZE": 32},
        ),
        triton.Config(
            {"Q_TILE_SIZE": 16, "K_TILE_SIZE": 64},
        ),
        triton.Config(
            {"Q_TILE_SIZE": 32, "K_TILE_SIZE": 16},
        ),
        triton.Config(
            {"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32},
        ),
        triton.Config(
            {"Q_TILE_SIZE": 32, "K_TILE_SIZE": 64},
        ),
        triton.Config(
            {"Q_TILE_SIZE": 64, "K_TILE_SIZE": 16},
        ),
        triton.Config(
            {"Q_TILE_SIZE": 64, "K_TILE_SIZE": 32},
        ),
    ],
    key=["N_QUERIES", "D"],
)

@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr=False,
):
    # pid
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # block_ptr
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(Q_TILE_SIZE * query_tile_index, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(Q_TILE_SIZE * query_tile_index, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(Q_TILE_SIZE * query_tile_index,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    # load q
    q = tl.load(Q_block_ptr)
    # Initialize
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    q_idx = Q_TILE_SIZE * query_tile_index + tl.arange(0, Q_TILE_SIZE)

    if is_causal:
        loop_end = tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE)
    else:
        loop_end = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(loop_end):
        # load k, v
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # s: (Q_TILE_SIZE, K_TILE_SIZE)
        s = tl.dot(q, k.T) * scale
        if is_causal:
            k_idx = K_TILE_SIZE * j + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] >= k_idx[None, :]
            s += tl.where(mask, 0, -1e6)
        # update
        t_m = tl.max(s, axis=-1)
        new_m = tl.maximum(t_m, m)

        p = tl.exp(s - new_m[:, None])
        l = l * tl.exp(m - new_m) + tl.sum(p, axis=-1)
        p = p.to(v.dtype) # BUG2: precision
        o = o * tl.exp(m[:, None] - new_m[:, None]) + tl.dot(p, v)
        m = new_m

        # BUG1: MOVE
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o / l[:, None]
    l = m + tl.log(l)

    tl.store(O_block_ptr, o.to(q.dtype)) # BUG2: precision
    tl.store(L_block_ptr, l)

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, is_causal=False):

        # output tensor
        o = torch.empty_like(q, device=q.device)
        l = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        ctx.is_causal = is_causal

        grid = lambda META:(
            triton.cdiv(q.shape[1], META['Q_TILE_SIZE']), q.shape[0]
        )

        flash_fwd_kernel[grid](
            q, k, v,
            o, l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            q.shape[1], k.shape[1],
            q.shape[-1] ** (-0.5),
            q.shape[-1],
            is_causal=is_causal
        )

        ctx.save_for_backward(l, q, k, v, o)
        return o


    @staticmethod
    def backward(ctx, grad_o):
        l, q, k, v, o = ctx.saved_tensors
        is_causal = ctx.is_causal
        scaler = q.shape[-1] ** -0.5

        d = torch.sum(o * grad_o, dim=-1)
        s = q @ k.transpose(-2, -1) * scaler

        q_idx = torch.arange(0, q.shape[-2], device=q.device)
        k_idx = torch.arange(0, k.shape[-2], device=q.device)
        mask = q_idx[:, None] >= k_idx[None, :]
        if is_causal:
            s += torch.where(mask, 0, -1e6)

        p = torch.exp(s - l[..., None]).to(grad_o.dtype)
        dv = p.transpose(-2, -1) @ grad_o
        dp = grad_o @ v.transpose(-2, -1)

        ds = p * (dp - d[..., None])
        dq = ds @ k * scaler
        dk = ds.transpose(-2, -1) @ q * scaler


        return dq, dk, dv, None # forward args

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    B = 2
    Nq = 128
    Nk = 128
    d = 256

    Q = torch.randn(B, Nq, d, device=device, requires_grad=True)
    K = torch.randn(B, Nk, d, device=device, requires_grad=True)
    V = torch.randn(B, Nk, d, device=device, requires_grad=True)

    # 你的 FlashAttention
    O_flash = FlashAttentionTriton.apply(Q, K, V, True)
    O_flash.sum().backward()
    q_grad = Q.grad
    k_grad = K.grad
    v_grad = V.grad
    Q.grad = None
    K.grad = None
    V.grad = None

    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    O_ref.sum().backward()
    print(f"O allclose: {(O_ref - O_flash).abs().max()}")
    print(f"Q max diff:{(Q.grad - q_grad).abs().max()}")
    print(f"K max diff:{(K.grad - k_grad).abs().max()}")
    print(f"V max diff:{(V.grad - v_grad).abs().max()}")