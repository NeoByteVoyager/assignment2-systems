import argparse
from statistics import mean, stdev
import timeit
import torch
from cs336_basics.adamw import AdamW
from cs336_basics.crossentropy import crossEntropy
from cs336_basics.transformer_lm import Model
import torch.cuda.nvtx as nvtx
import cs336_basics.transformer_block as transformer_block


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]

    with nvtx.range("computing attention scores"):
        attention_map = torch.einsum(
            "... seq1 d_k, ... seq2 d_k -> ... seq1 seq2",
            q, k
        ) / (d_k ** 0.5)

        if mask is not None:
            attention_map = attention_map.masked_fill(~mask, float("-inf"))

    with nvtx.range("computing softmax"):
        scaled_attention = torch.softmax(attention_map, dim=-1)

    with nvtx.range("final matmul"):
        output = torch.einsum(
            "... seq1 seq2, ... seq2 d_v -> ... seq1 d_v",
            scaled_attention, v
        )

    return output


transformer_block.scaled_dot_product_attention = annotated_scaled_dot_product_attention


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
    model = model.to(device)
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
            start = timeit.default_timer()

            torch.cuda.nvtx.range_push("forward")
            model(input_ids)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

            end = timeit.default_timer()
            times.append(end - start)

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
            logits = model(input_ids)
            loss = crossEntropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1)
            )
            optimizer.zero_grad()

            torch.cuda.synchronize()
            start = timeit.default_timer()

            torch.cuda.nvtx.range_push("backward")
            loss.backward()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

            end = timeit.default_timer()
            times.append(end - start)

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
            logits = model(input_ids)
            loss = crossEntropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()

            torch.cuda.synchronize()
            start = timeit.default_timer()

            # 显式打上 optimizer 标记
            torch.cuda.nvtx.range_push("optimizer")
            optimizer.step()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

            end = timeit.default_timer()
            times.append(end - start)

    mean_time = mean(times)
    std_time = stdev(times)
    print(times)
    print(f"mean_time: {mean_time:.6f} s")
    print(f"std_time:  {std_time:.6f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="forward",
        choices=["forward", "backward", "optimize"],
    )
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)

    args = parser.parse_args()
    benchmark(
        args.d_model, args.num_layers, args.num_heads, args.d_ff, mode=args.mode
    )