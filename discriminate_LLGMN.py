#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLGMNで識別するプログラム
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from My_LLGMN import My_LLGMN


# 決定境界をプロットする関数
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
    print("メッシュデータを識別中．．．")
    # z = ll.predict(model, np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
    z = my_llgmn.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
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
    plt.show()
    # plt.savefig("./graph/決定境界(M=2).png")


C = 3

"""
# 決定境界をプロットする部分（計算量が多いのでコメントアウトしている）
data = pd.read_csv("./data/data_1.csv", index_col=0)
x = data.values
y = data.index

my_llgmn = My_LLGMN(n_class=C, input_dim=x.shape[1], n_component=1)
my_llgmn.build_network()
my_llgmn.learn(x_train=x, y_train=y, is_mini_batch=False, n_epoch=50)

plot_decision_regions(x, y, resolution=0.01)
"""

# 各コンポーネント数において，20回の精度評価を行い，平均識別率を算出する
# 結構な時間（数十分）がかかるので実行するときは気をつけてください
result = []
for i in range(20):
    print("data:"+str(i+1))
    _ = []
    for c in [20]:  # 1, 2, 3, 5, 10]:
        data = pd.read_csv("./data/data_"+str(i+1)+".csv", index_col=0)
        x = data.values
        y = data.index

        my_llgmn = My_LLGMN(n_class=C, input_dim=x.shape[1], n_component=c)
        my_llgmn.build_network()
        my_llgmn.learn(x_train=x, y_train=y, is_mini_batch=False, n_epoch=50)

        test = pd.read_csv("./data/test_"+str(i+1)+".csv", index_col=0)
        x = test.values
        y = test.index
        pred = my_llgmn.predict(x_test=x)
        pred = np.argmax(pred, axis=1)
        _.append(100*np.sum(pred == y)/y.shape[0])
    result.append(_)

result = np.array(result)
print(result.mean(axis=0))
print(result.std(axis=0, ddof=0))
