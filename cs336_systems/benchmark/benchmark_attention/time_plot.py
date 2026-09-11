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
                #ys.append(row['forward_ms'])
                ys.append(row['backward_ms'])
        plt.plot(xs, ys, marker="o", label=f"d={d} mode={m}")

plt.xlabel("Sequence Length")
plt.ylabel("Forward Time (ms)")
plt.title("Forward Time vs Sequence Length")
plt.legend()
plt.xscale("log", base=2)
plt.show()

'''
uncompiled scale with seq_len

with open("attention_benchmark.json", "r") as f:
    results = json.load(f)

dims = sorted(set(row["dim"] for row in results))
seqs = sorted(set(row["seq_len"] for row in results))

# 画 forward time
for d in dims:
    xs = []
    ys = []
    for row in results:
        if row["dim"] == d:
            xs.append(row["seq_len"])
            ys.append(row["forward_ms"])
    plt.plot(xs, ys, marker="o", label=f"d={d}")

plt.xlabel("Sequence Length")
plt.ylabel("Forward Time (ms)")
plt.title("Forward Time vs Sequence Length")
plt.legend()
plt.xscale("log", base=2)
plt.show()
    
'''