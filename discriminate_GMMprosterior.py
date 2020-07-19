#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMMの事後確率を使って識別するプログラム
"""

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


alpha_c_m = [0.2, 0.1, 0.3, 0.1, 0.2, 0.1]

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

n_class = 3
n_component = 2


# 事後確率P(k|x)を返す関数
# x: 2d-array
def prosteriorProbability(x):
    # 遅かったVer.なので，コメントアウトしてる（はず）
    # ret = []
    # for xi in x:
    #     _ = []
    #     bunbo = 0
    #     for idx in range(n_class*n_component):
    #         bunbo += alpha_c_m[idx]*stats.multivariate_normal.pdf(xi, mean=means[idx], cov=covs[idx])
    #     for i in range(n_class):
    #         bunsi = 0
    #         bunsi += alpha_c_m[2*i]*stats.multivariate_normal.pdf(xi, mean=means[2*i], cov=covs[2*i])
    #         bunsi += alpha_c_m[2*i+1]*stats.multivariate_normal.pdf(xi, mean=means[2*i+1], cov=covs[2*i+1])
    #         _.append(bunsi/bunbo)
    #     ret.append(_)
    # return np.array(ret)

    for idx in range(n_class):
        prob = alpha_c_m[2*idx] * \
            stats.multivariate_normal.pdf(
                x, mean=means[2*idx], cov=covs[2*idx])
        prob += alpha_c_m[2*idx+1]*stats.multivariate_normal.pdf(
            x, mean=means[2*idx+1], cov=covs[2*idx+1])
        prob = prob.reshape([prob.shape[0], 1])
        if idx == 0:
            p = prob.copy()
            continue
        p = np.concatenate([p, prob], axis=1)

    s = np.sum(p, axis=1)
    for i in range(x.shape[0]):
        p[i] = p[i] / s[i]
    return p


# 決定境界を作図する関数
def plot_decision_regions(x, y, resolution=0.01):

    # 今回は被説明変数が3クラスのため散布図のマーカータイプと3種類の色を用意
    # クラスの種類数に応じて拡張していくのが良いでしょう
    markers = ('s', 'o', 'x')
    cmap = ListedColormap(('blue', 'yellow', 'green'))

    # 2変数の入力データの最小値から最大値まで引数resolutionの幅でメッシュを描く
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))

    # pd.DataFrame(x1_mesh).to_csv("./mesh1.csv", header=None, index=None)
    # pd.DataFrame(x2_mesh).to_csv("./mesh2.csv", header=None, index=None)

    # メッシュデータ全部を学習モデルで分類
    z = prosteriorProbability(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
    z = np.argmax(z, axis=1)
    z = z.reshape(x1_mesh.shape)
    # pd.DataFrame(z).to_csv("./z.csv", header=None, index=None)

    # メッシュデータと分離クラスを使って決定境界を描いている
    plt.contourf(x1_mesh, x2_mesh, z, alpha=0.3, cmap=cmap)
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0],
                    y=x[y == cl, 1],
                    alpha=0.7,
                    s=40,
                    c=cmap(idx),
                    edgecolors='black',
                    marker=markers[idx])

    plt.gca().set_aspect('equal')

    plt.savefig("./graph/決定境界(GMM).png")
    plt.show()


data = pd.read_csv("./data/data_1.csv", index_col=0)
x = data.values
y = data.index
plot_decision_regions(x, y)

# GMMの事後確率を使って識別を20回行い，平均識別率を算出する
result = []
for i in range(20):
    test = pd.read_csv("./data/test_"+str(i+1)+".csv", index_col=0)
    x = test.values
    y = test.index
    pred = prosteriorProbability(x)
    pred = np.argmax(pred, axis=1)
    result.append(100*np.sum(pred == y)/y.shape[0])

result = np.array(result)
print(result.mean())
print(result.std(ddof=0))
