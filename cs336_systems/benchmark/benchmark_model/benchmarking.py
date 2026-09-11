import argparse
from statistics import mean, stdev
import time
import torch
from cs336_basics.adamw import AdamW
from cs336_basics.crossentropy import crossEntropy
from cs336_basics.transformer_lm import Model

def benchmark(
    d_model,
    num_layers,
    num_heads,
    d_ff,
    vocab_size=10000,
    batch_size=4,
    context_length=256,
    rope_theta=10000,
    warmup=10,
    steps=20,
    mode="forward",
    compiled = False
):
    model = Model(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.compile(model.to(device)) if compiled else model.to(device)

    input_ids = torch.randint(
        0, vocab_size, (batch_size, context_length), device=device
    )
    targets = torch.randint(
        0, vocab_size, (batch_size, context_length), device=device
    )

    optimizer = AdamW(model.parameters(), 0.003, 1e-4, (0.9, 0.99), 1e-8)

    if mode == "forward":
        # warm up
        for _ in range(warmup):
            model(input_ids)
        torch.cuda.synchronize()

        times = []
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()

            model(input_ids)

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    elif mode == "backward":
        # warm up
        for _ in range(warmup):
            logits = model(input_ids)
            loss = crossEntropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
        torch.cuda.synchronize()

        times = []
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()

            logits = model(input_ids)
            loss = crossEntropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1)
            )
            optimizer.zero_grad()

            loss.backward()

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    elif mode == "optimize":
        # warm up
        for _ in range(warmup):
            logits = model(input_ids)
            loss = crossEntropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        times = []
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()

            logits = model(input_ids)
            loss = crossEntropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    mean_time = mean(times)
    std_time = stdev(times)
    print(times)
    print(f"mean_time: {mean_time:.6f} ms")
    print(f"std_time:  {std_time:.6f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="forward",
        choices=["forward", "forward_backward", "full"],
    )
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()
    benchmark(
        args.d_model, args.num_layers, args.num_heads, args.d_ff, mode=args.mode, compiled=args.compiled
    )