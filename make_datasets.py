#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 参考：https://stats.stackexchange.com/questions/243392/generate-sample-data-from-gaussian-mixture-model


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


markers = ('s', 'o', 'x')
cmap = ListedColormap(('blue', 'yellow', 'green'))

np.random.seed(0)  # 解析結果を固定させている

n_component = 2
n_class = 3

alpha_c_m = [0.2, 0.1, 0.3, 0.1, 0.2, 0.1]  # 混合度

means = [
    [-3, 1],
    [-2, 0.5],
    [-0.5, -1],
    [0, -2],
    [3, 1],
    [2, 0],
]

covs = [
    [
        [1.5, 0.6],
        [0.6, 0.3],
    ],
    [
        [2, 0],
        [0, 1],
    ],
    [
        [1, 2.7],
        [2.7, 9],
    ],
    [
        [9, 0.63],
        [0.63, 0.09],
    ],
    [
        [3, 1],
        [1, 2.5],
    ],
    [
        [0.8, 0],
        [0, 4],
    ],
]

N_train = 500
N_test = 2000

# 学習データとテストデータを20セット作る
for num in range(20):
    x = np.array([])
    y = np.array([])
    for n in range(N_train):
        idx = np.random.choice(n_component*n_class, p=alpha_c_m)
        sample = np.random.multivariate_normal(means[idx], covs[idx])
        if n == 0:
            x = sample.reshape([1, 2]).copy()
            y = np.array([int(idx/2)])  # 0, 1 -> 0; 2, 3 -> 1; 4, 5 -> 2
            continue
        x = np.concatenate((x, sample.reshape([1, 2])))
        # 0, 1 -> 0; 2, 3 -> 1; 4, 5 -> 2
        y = np.concatenate((y, np.array([int(idx/2)])))

    # 学習データをプロット
    if num == 0:
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=x[y == cl, 0],
                        y=x[y == cl, 1],
                        alpha=0.6,
                        s=40,
                        c=cmap(idx),
                        edgecolors='black',
                        marker=markers[idx])

        plt.gca().set_aspect('equal')
        plt.show()
        # plt.savefig("./graph/生成データのプロット.png")

    data = pd.DataFrame(x)
    data.index = y
    data.to_csv("./data/data_"+str(num+1)+".csv")

    x = np.array([])
    y = np.array([])

    for n in range(N_test):
        idx = np.random.choice(n_component*n_class, p=alpha_c_m)
        sample = np.random.multivariate_normal(means[idx], covs[idx])
        if n == 0:
            x = sample.reshape([1, 2]).copy()
            y = np.array([int(idx/2)])  # 0, 1 -> 0; 2, 3 -> 1; 4, 5 -> 2
            continue
        x = np.concatenate((x, sample.reshape([1, 2])))
        # 0, 1 -> 0; 2, 3 -> 1; 4, 5 -> 2
        y = np.concatenate((y, np.array([int(idx/2)])))
    data = pd.DataFrame(x)
    data.index = y
    data.to_csv("./data/test_"+str(num+1)+".csv")
