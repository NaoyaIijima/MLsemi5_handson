# -*- coding: utf-8 -*-
"""
2020/1/27 作成
2020/5/14 変更（predict関数の作成）
"""

from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def convert_labels(pos_neg_labels, n_class):
    labels = pos_neg_labels.astype(int)
    return to_categorical(labels, n_class)


def nonlinear_transform(data):
    d = data.copy()
    tmp = np.zeros([1, d.shape[0]])
    for i in range(d.shape[1]):
        for j in range(i, d.shape[1]):
            tmp = np.vstack([tmp, d[:, i] * d[:, j]])
    _ = np.ones([d.shape[0], 1])
    d = np.hstack([_, d])
    d = np.hstack([d, tmp.T[:, 1:]])
    return d


# Define and build a network structure
def build_network(n_class, n_component, h):
    main_input = Input(shape=(h,), dtype="float32", name="main_input")

    layer = Dense(n_class * n_component - 1, use_bias=False)
    middle_layer_ = layer(main_input)

    auxiliary_input = Input(shape=(1,), name="aux_input")
    middle_layer = keras.layers.concatenate([middle_layer_, auxiliary_input])

    acctivate = Activation("softmax")
    middle_layer2 = acctivate(middle_layer)

    layer = Dense(n_class, use_bias=False, name="main_output")
    layer.trainable = False
    main_output = layer(middle_layer2)

    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    model.summary()

    # Weights to sum the each component's outputs
    weights = model.get_weights()
    # weights[0] = np.ones(weights[0].shape)  # for Debug!!
    w2 = np.zeros(weights[1].shape)
    for c in range(n_class):
        for j in range(n_component):
            w2[n_component * c + j, c] = 1
    weights[1] = w2
    model.set_weights(weights=weights)

    return model


def main_llgmn(
    x_train, y_train, n_batch_size=128, n_epoch=30, n_class=2, n_component=1
):
    tmp = x_train.shape[1]
    h = int(1 + tmp * (tmp + 3) / 2)

    nonlinear_x_train = nonlinear_transform(x_train)

    y_train = convert_labels(y_train, n_class)

    model = build_network(n_class, n_component, h)

    # compile the model (multi-class classification)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    if n_class == 2:
        model.compile(
            optimizer=sgd, loss="binary_crossentropy", metrics=[metrics.binary_accuracy]
        )
    else:
        model.compile(
            optimizer=sgd,
            loss="categorical_crossentropy",
            metrics=[metrics.categorical_accuracy],
        )

    # Early-stopping
    # early_stopping = EarlyStopping(monitor='loss', patience=0, verbose=1)#, min_delta=10e-4)

    mc = ModelCheckpoint("./model.h5", monitor="loss", verbose=1,
                         save_best_only=True,
                         save_weights_only=False,
                         mode='min',
                         period=1)

    aux_train = np.zeros([x_train.shape[0], 1])
    fit = model.fit(
        {"main_input": nonlinear_x_train, "aux_input": aux_train},
        {"main_output": y_train},
        epochs=n_epoch,
        batch_size=x_train.shape[1],  # 全サンプルを使って勾配計算・重み更新,
        callbacks=[mc],
        # callbacks=[early_stopping],
        # batch_size=n_batch_size,  # ミニバッチ学習
        # verbose=0,  # 学習過程を表示さない設定
    )

    # plt.plot(fit.history['loss'])
    # plt.yscale("log")
    # plt.show()

    return model


def predict(model, x_test):
    nonlinear_x_test = nonlinear_transform(x_test)
    aux_test = np.zeros([x_test.shape[0], 1])
    pred = model.predict(
        {"main_input": nonlinear_x_test, "aux_input": aux_test})
    return pred


if __name__ == "__main__":
    C = 3
    N = 100
    x_train = np.random.randn(N, 2)
    x_train[0: int(N / C), :] += 2
    x_train[int(N/C): int(2*N/C), :] -= 3

    y_train = np.ones((N, 1))
    y_train[: int(N / C), :] -= 1.0
    y_train[int(N/C): int(2*N/C), :] += 1.0

    x_test = np.random.randn(10, 2) + 2

    model = main_llgmn(x_train=x_train, y_train=y_train,
                       n_epoch=500, n_class=C, n_component=1)
    pred = predict(model, x_test)
    pred_ = np.argmax(pred, axis=1)
    print(pred_)
