import torch
import torch.nn
import time
import json
from torch.nn.attention import sdpa_kernel, SDPBackend

'''
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
    scaled_attention = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    masked_attention = scaled_attention.masked_fill(~mask, float("-inf"))
    attention_socre = torch.softmax(masked_attention, -1)
    return attention_socre @ v
'''

WARM_UP = 5
LOOPS = 100

BATCH_SIZE = 8
SEQ = [256, 512, 1024, 2048, 4096, 8192]
DIM = [16, 32, 64, 128]

COMPILE = True

attention = torch.nn.functional.scaled_dot_product_attention


def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for s in SEQ:
        for d in DIM:
            q = torch.randn(BATCH_SIZE, 1, s, d, device=device, dtype=torch.float16, requires_grad=True)
            k = torch.rand_like(q, device=device, requires_grad=True)
            v = torch.rand_like(q, device=device, requires_grad=True)
            # mask = torch.tril(torch.ones(s, s)).bool().to(device)
            # warm_up
            for _ in range(WARM_UP):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = attention(query=q, key=k, value=v, is_causal=True)
                out.sum().backward()
                q.grad = None
                k.grad = None
                v.grad = None
            # benchmark
            forward_times = []
            peak_mems = []
            mems = []
            backward_times = []

            for _ in range(LOOPS):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = attention(query=q, key=k, value=v, is_causal=True)

                torch.cuda.synchronize()
                end = time.perf_counter()
                forward_times.append((end - start) * 1000)

                memory_before_backward = torch.cuda.memory_allocated()
                mib = memory_before_backward / (1024 ** 2)
                mems.append(mib)

                peak_mem = torch.cuda.max_memory_allocated()
                mib = peak_mem / (1024 ** 2)
                peak_mems.append(mib)

                torch.cuda.synchronize()
                start = time.perf_counter()

                out.sum().backward()

                torch.cuda.synchronize()
                end = time.perf_counter()
                backward_times.append((end - start) * 1000)

                q.grad = None
                k.grad = None
                v.grad = None

            forward_avg = sum(forward_times) / LOOPS
            backward_avg = sum(backward_times) / LOOPS
            memory_avg = sum(mems) / LOOPS
            peak_memory_avg = sum(peak_mems) / LOOPS

            print(f"seq_len:{s}, dim:{d}")
            print(f"forward_time: {forward_avg} ms")
            print(f"mems_avg: {memory_avg} MiB")
            print(f"peak_mems_avg:{peak_memory_avg} MiB")
            print(f"backward_time: {backward_avg} ms")

            results.append({
                "batch_size": BATCH_SIZE,
                "seq_len": s,
                "dim": d,
                "forward_ms": forward_avg,
                "backward_ms": backward_avg,
                "mem_avg": memory_avg,
                "peak_mem_avg": peak_memory_avg
            })

    with open("attention_benchmark.json", "w") as f:
        json.dump(results, f, indent=4)


benchmark()