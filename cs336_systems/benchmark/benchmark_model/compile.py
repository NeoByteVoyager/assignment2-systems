import pandas as pd
'''
model:
    dim: 512
    num_layers: 4
    num_heads: 8
    d_ff: 1344
'''
data = {
    "Mode": [
        "Forward",
        "Forward + Backward",
        "Full Step"
    ],
    "Vanilla (ms)": [
        14.520050,
        44.125533,
        55.288512
    ],
    "Compiled (ms)": [
        11.192840,
        35.111603,
        45.529783
    ]
}

df = pd.DataFrame(data)

df["Speedup"] = df["Vanilla (ms)"] / df["Compiled (ms)"]


print(df)