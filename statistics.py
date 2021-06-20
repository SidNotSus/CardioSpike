import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

x = 'x'
y = 'y'

df = pd.read_csv('../train.csv')

print(df[df['y'] == 1].max())
print(df[df['y'] == 1].min())

start, end = [], []
n = df.x.size
middle_len = 0  # 10.328697850821744
for i in range(n - 1):
    if df.iloc[i].y == 0 and df.iloc[i + 1].y == 1:
        start.append(i + 1)
    if df.iloc[i].y == 1 and df.iloc[i + 1].y == 0:
        end.append(i)

# edges = list(zip(start, end))
# middle_len = sum(list(map(lambda x: x[1] - x[0], edges))) / len(edges)
# print(middle_len)

pattern = [0] * 20
p = 0
for st in start:
    p += 1
    for i in range(20):
        pattern[i] += df.loc[st + i - 6].x

pattern = [x / p for x in pattern]
print(pattern)
plt.plot(pattern)
plt.show()

start_f, end_f = [], []
dfc = df.copy()
dfc['y'] = 0

pattern_f = [0] * 20
p2 = 0
for i in range(n - 3):
    a, b, c, d, = (
        df.loc[i + 0].x,
        df.loc[i + 1].x,
        df.loc[i + 2].x,
        df.loc[i + 3].x,
    )
    if a < b > c < d \
            and abs(a - d) < 40 \
            and abs(b - c) > 85 \
            and abs(b - c) < 180 \
            and abs(a - b) > 45 \
            and abs(c - d) > 45:
        p2 += 1
        for k in range(0, 0 + 20):
            if -1 < k < n:
                try:
                    pattern_f[k] += df.loc[i - 8 + k].x
                except:
                    pass

pattern_f = [x / p2 for x in pattern_f]
print(pattern_f)

plt.plot(pattern_f)
plt.show()
