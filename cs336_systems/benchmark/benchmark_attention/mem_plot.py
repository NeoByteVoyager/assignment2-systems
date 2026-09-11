import json
import matplotlib.pyplot as plt

with open("attention_benchmark.json", "r") as f:
    eager_result = json.load(f)

with open("attention_benchmark_compiled.json", "r") as f:
    compiled_result = json.load(f)

for row in eager_result:
    row['mode'] = 'eager'

for row in compiled_result:
    row['mode'] = 'compiled'

result = eager_result + compiled_result

dims = sorted(set(r["dim"] for r in result))
modes = ['eager', 'compiled']

for d in dims:
    for m in modes:
        xs = []
        ys = []

        for row in result:
            if row['mode'] == m and row['dim'] == d:
                xs.append(row['seq_len'])
                ys.append(row['mem_avg'])
                #ys.append(row['peak_mem_avg'])
        plt.plot(xs, ys, marker="o", label=f"d={d} mode={m}")

plt.xlabel("Sequence Length")
plt.ylabel("Mem")
plt.title("Mem vs Sequence Length")
plt.legend()
plt.xscale("log", base=2)
plt.show()



'''
eager scale with seq_len
with open("attention_benchmark.json", "r") as f:
    results = json.load(f)

dims = sorted(set(row["dim"] for row in results))

for d in dims:
    xs = []
    ys = []
    for row in results:
        if row["dim"] == d:
            xs.append(row["seq_len"])
            ys.append(row["mem_avg"])
            #ys.append(row["peak_mem_avg"])
    plt.plot(xs, ys, marker="o", label=f"d={d}")

plt.xlabel("Sequence Length")
plt.ylabel("Memory Before Backward (MiB)")
plt.title("Memory vs Sequence Length")
plt.legend()
plt.xscale("log", base=2)
plt.show()
'''