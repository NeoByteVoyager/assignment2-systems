import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import csv

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size, num_elements, warm_up=5, iters=20):
    setup(rank, world_size)

    data = torch.randn(num_elements, device=f"cuda:{rank}")

    for _ in range(warm_up):
        dist.all_reduce(data, async_op=False)

    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    # Perform all-reduce
    start_time = time.time()

    for i in range(iters):
        dist.all_reduce(tensor=data, async_op=False)

    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    end_time = time.time()
    duration = end_time - start_time
    avg_time = duration / iters
    print(
        f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {avg_time}",
        flush=True)
    with open("all_reduce_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([world_size, num_elements * 4 / (1024 ** 2), avg_time])

    dist.destroy_process_group()


if __name__ == "__main__":
    with open("all_reduce_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["world_size", "mem_mb", "time_s"])


    world_sizes = [2, 4, 6]
    mems = [1, 10, 100, 1024]
    for world_size in world_sizes:
        for mem in mems:
            num_elements = mem * (1024 ** 2) // 4
            mp.spawn(fn=distributed_demo, args=(world_size, num_elements), nprocs=world_size)