# reference: https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188
from datetime import datetime
import math
import os
import random

from tensorflow.keras import layers, models, optimizers
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


IMG_H, IMG_W = 200, 200
MODEL_PATH = '../models'
IMG_PATH = '../images/'


def plot_result(history, model_path):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'accurary.png'))

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'loss.png'))


def build_model(n_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_H, IMG_W, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(n_classes, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=["acc"])

    # モデル構成の確認
    model.summary()
    return model


def read_landing_data():
    land_df = pd.read_csv('../data/landing.csv', encoding='SHIFT_JIS')
    land_df.columns = ['year', *['m{}'.format(str(i).zfill(2)) for i in range(1, 13)], 'count']

    # 1 ~ 5, 11, 12月は台風の上陸数が少ないので削除
    use_months = ['year', 'count', 'm06', 'm07', 'm08', 'm09', 'm10']
    land_df = land_df[use_months]
    land_df.fillna(0, inplace=True)

    # 1982年からしか月平均海面水温のデータがないので、それ以外を削除
    land_df = land_df[land_df['year'] >= 1982]
    n_classes = int(land_df.drop(['year', 'count'], axis=1).max().values.max()) + 1

    return land_df, n_classes


def convert_img(filename):
    img = Image.open(os.path.join(IMG_PATH, filename))
    img = img.convert('RGB')
    img = img.resize((IMG_H, IMG_W))
    return np.asarray(img)


def make_sample(files, land_df):
    X, y = [], []
    for filename in files:
        year = filename.split('_')[0]
        month = filename.split('_')[1][:2]
        if year != '2019':
            X.append(convert_img(filename))
            monthly_count = land_df[land_df['year'] == int(year)]['m{}'.format(month)].values[0]
            y.append(monthly_count)
    return np.array(X), np.array(y)


def make_sample_2019():
    X_2019 = []
    for filename in ['2019_{}.png'.format(month) for month in ['06', '07', '08', '09']]:
        X_2019.append(convert_img(filename))
    return np.array(X_2019)


def main():
    all_files = os.listdir(IMG_PATH)
    land_df, n_classes = read_landing_data()
    random.shuffle(all_files)

    th = math.floor(len(all_files) * 0.8)
    train = all_files[0:th]
    test = all_files[th:]
    X_train, y_train = make_sample(train, land_df)
    X_test, y_test = make_sample(test, land_df)
    X_2019 = make_sample_2019()

    # Regularization
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255
    X_2019 = X_2019.astype("float") / 255
    # Converts a class vector (integers) to binary class matrix.
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    model = build_model(n_classes)
    history = model.fit(X_train, y_train, epochs=20, batch_size=6, validation_data=(X_test, y_test))

    pred = model.predict(X_2019)
    print(pred)
    print('predicted 2019 typhoon counts: ', np.argmax(pred, axis=1).sum())

    model_path = os.path.join(MODEL_PATH, '{0:%Y%m%d}_{0:%H%M}'.format(datetime.now()))
    os.makedirs(model_path)

    plot_result(history, model_path)
    print('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join(model_path, 'cnn_model.json'), 'w').write(json_string)
    print('save weights')
    model.save_weights(os.path.join(model_path, 'cnn_model_weights.hdf5'))


if __name__ == "__main__":
    main()
