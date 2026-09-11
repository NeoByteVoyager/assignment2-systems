import math
from math import sqrt
import torch
from sympy.multipledispatch.dispatcher import RaiseNotImplementedError


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_casual=False):
        '''
        inputs: q, k, v
        output: o, l
        flashattention algorithm by pytorch
        '''
        B = q.shape[0]
        Nq, Nk = q.shape[1], k.shape[1]
        d = q.shape[-1]
        # define tile size
        Bq, Bk = 16, 16
        Tq, Tk = Nq // Bq, Nk // Bk

        # allocate O, L
        o = torch.empty(q.shape, device=q.device)
        l = torch.empty(B, Nq, device=q.device)

        # main algorithm
        for i in range(Tq):
            # load qi
            si = i * Bq
            # allocate qi, li, mi, oi
            qi = q[: ,si: si + Bq, :]
            li = torch.zeros(B, Bq, device=q.device)
            mi = torch.full((B, Bq), float("-inf"), device=q.device, dtype=q.dtype)
            oi = torch.zeros(qi.shape, device=q.device)

            for j in range(Tk):
                # load kj, vj
                sj = j * Bk
                kj = k[:, sj: sj + Bk, :]
                vj = v[:, sj: sj + Bk, :]
                # calcuate s: (Bq, Bk)
                sij = qi @ kj.transpose(-2, -1) / sqrt(d)
                # update mi, li, oi
                t_mi, _ = torch.max(sij, dim=-1)
                new_mi = torch.maximum(mi, t_mi)

                p = torch.exp(sij - new_mi[..., None])
                li = li * torch.exp(mi - new_mi) + torch.sum(p, dim=-1)
                oi = oi * torch.exp(mi[..., None] - new_mi[..., None]) + p @ vj
                mi = new_mi

            o[:, si: si + Bq, :] = oi / li[..., None]
            l[ :,si: si + Bq] = mi + torch.log(li)
        ctx.save_for_backward(l, q, k, v, o)
        return o
    @staticmethod
    def backward(ctx, grad_output):
        RaiseNotImplementedError


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2
    Nq = 32
    Nk = 32
    d = 16

    Q = torch.randn(B, Nq, d, device=device)
    K = torch.randn(B, Nk, d, device=device)
    V = torch.randn(B, Nk, d, device=device)

    # 你的 FlashAttention
    O_flash = FlashAttentionPytorch.apply(Q, K, V, False)

    # 普通 attention，作为标准答案
    S = Q @ K.transpose(-2, -1) / math.sqrt(d)
    P = torch.softmax(S, dim=-1)
    O_ref = P @ V

    print(O_flash)
    print(O_ref)
    print("max error:", (O_flash - O_ref).abs().max())

    print(
        "all close:",
        torch.allclose(O_flash, O_ref, atol=1e-5, rtol=1e-5)
    )