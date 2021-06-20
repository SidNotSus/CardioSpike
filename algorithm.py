import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm


class Algorithm(object):

    def __init__(self, file_name, **kwargs):
        self.data: pd.DataFrame = pd.read_csv(file_name)
        self.gdata = None
        self.clear_gdata = []

    @staticmethod
    def paint(part):
        print(part)
        color__ = []
        for i in part['y']:
            color__.append('r' if i else 'g')
        part['x'].plot.bar(width=1, color=color__)
        plt.show()

    def preprocessing(self):

        x = 'x'
        y = 'y'
        id_max = self.data['id'].max()
        self.data[y] = 0
        min_x_val = 450
        max_x_val = 1050
        self.data = self.data[self.data[x] > min_x_val]
        self.data = self.data[self.data[x] < max_x_val]
        self.gdata = [self.data[self.data['id'] == _]
                      if not self.data[self.data['id'] == _].empty
                      else [] for _ in range(1, id_max + 1)]
        self.gdata = list(filter(([]).__ne__, self.gdata))

        def edge(median, std):
            return median - 150, median + 150

        def recount_median(disturb, mu, std):
            from math import sqrt
            return (1 - sqrt(std / mu)) * mu + sqrt(std / mu) * max(disturb, key=disturb.count)

        def validate_spike(pos, part):
            min_ind_ = pos[2]

            for i in range(4):
                try:
                    if not pos[1][1] < part.loc[min_ind_ - i + pos[0]].x < pos[1][0]:
                        return False
                    if not pos[1][1] < part.loc[min_ind_ + pos[0] + 3 + i].x < pos[1][0]:
                        return False
                except KeyError:
                    pass
            return True

        for part in self.gdata:
            length = len(part.x)
            id_max = len(part[x])

            mu, std = norm.fit(list(part[x]))

            pattern = [mu] * 4 + [mu * 1.125] + [mu * 0.875] + [mu] * 4

            def validate_pattern(pos):
                try:
                    section = part.x[length + pos - 3:  pos + length + 7]
                    import numpy as np
                    if np.correlate(section, pattern)[0] > 0.5:
                        return True
                    return False
                except:
                    return False

            disturb = [e // 7 * 7 for e in part[x]]
            mu = recount_median(disturb=disturb, mu=mu, std=std)

            left, right = edge(mu, std)
            part = part[part[x] > left]
            part = part[part[x] < right]
            part = part.reset_index(drop=True)
            dummy_spikes = []
            spikes = []
            # spike starts with pos index of higher RtoR interval
            # print(part)
            for i in range(0, len(part.x) - 3):
                min_ind = list(part.index)[0]
                a, b, c, d = (
                    part.loc[i + 0 + min_ind].x,
                    part.loc[i + 1 + min_ind].x,
                    part.loc[i + 2 + min_ind].x,
                    part.loc[i + 3 + min_ind].x,
                )
                if a < b > c < d:
                    if 75 < abs(b - c) < 175 and abs(a - d) < 20:
                        if abs(a - b) > 27 and abs(c - d) > 27:
                            dummy_spikes.append((i, (b, c), min_ind))
                            if validate_spike(dummy_spikes[-1], part=part):
                                # if validate_pattern(i):
                                spikes.append(i)
                                for k in range(i - 5, i + 6):
                                    try:
                                        part.loc[min_ind + k].y = 1
                                    except:
                                        pass
                else:
                    continue
            # print(dummy_spikes)
            # print(len(dummy_spikes), len(spikes))

            self.clear_gdata.append(part)
        r = self.clear_gdata[0]

        for i in range(1, len(self.clear_gdata)):
            r = r.append(self.clear_gdata[i])

        return r


p = Algorithm('../test.csv')
result = p.preprocessing()
data = pd.read_csv('../test.csv')
data['y'] = 0
result = result.reset_index(drop=True)
print(result.y.sum())
for i in range(len(result)):
    row = result.loc[i]

    t = row.time
    data.loc[(data['time'] == t) & (data['id'] == row.id), 'y'] = row.y

data.drop('x', inplace=True, axis=1)
data.to_csv('result.csv', index=False)
print(result.y.sum())
print(data.y.sum())
