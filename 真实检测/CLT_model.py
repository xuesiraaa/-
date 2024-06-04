import pickle
from datetime import datetime

import keras_nlp
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, Layer, Bidirectional, LSTM, Conv1D
import os
import data_generator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# 非必须包↓若用adamw则加，实测用keras自带的adam效果更好
import tensorflow_addons as tfa
import pandas as pd
# 位置编码信息
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def clt():
    projection_dim = 128
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    # Transformer layers的大小
    transformer_layers = 4
    mlp_head_units = [256, 128]  # 输出部分的MLP全连接层的大小
    model_dim = 128

    inputs = layers.Input(shape=(max_len, 4), name="inputs")
    embeddings = Conv1D(filters=32, kernel_size=1, padding='same', activation='relu')(inputs)
    embeddings = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embeddings)
    # embeddings = Conv1D(filters=model_dim, kernel_size=8, padding='same', activation='relu')(embeddings)
    # embeddings = Conv1D(filters=model_dim, kernel_size=3, padding='same', activation='relu')(embeddings)
    # embeddings = LSTM(32, return_sequences=True)(embeddings)
    # embeddings = LSTM(model_dim)(embeddings)
    # 位置编码
    positional_encoding = keras_nlp.layers.SinePositionEncoding()(embeddings)
    embeddings = layers.Add()([embeddings, positional_encoding])
    # 创建多个Transformer encoding 块
    for _ in range(transformer_layers):
        embeddings = LSTM(model_dim, return_sequences=True)(embeddings)
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(embeddings)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=model_dim // num_heads, dropout=0.5
        )(x1, x1)
        # Skip connection.
        x2 = layers.Add()([attention_output, embeddings])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.5)
        # Skip connection 2.
        embeddings = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(embeddings)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # 增加MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # 输出分类.
    logits = layers.Dense(num_classes, activation='softmax')(features)
    # 构建
    model = keras.Model(inputs=inputs, outputs=logits)
    model.summary()
    return model





def show(history):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
batch_size = 64
num_epochs = 100
max_len = 500

print("Data downloading and pre-processing ... ")

x_train, y_train = data_generator.train_data(max_len)

num_classes = len(y_train[0])
# 转换 x_train 到 NumPy 数组（如果它还不是）
x_train_np = [np.array(ts) for ts in x_train]
# 获取时间序列的最大长度（可选，用于检查是否需要填充）
max_seq_len = max(len(ts) for ts in x_train_np)
# 如果需要，定义填充值（默认为0）和填充后的长度（2000）
padding_value = 0
# 由于每个时间序列有五个属性，我们需要一个三维的张量（batch_size, time_steps, features）
# 我们需要转换 x_train_np 以匹配这个形状
x_train_3d = np.array(
    [np.pad(ts, ((0, max_len - len(ts)), (0, 0)), 'constant', constant_values=padding_value) for ts in x_train_np])
# 转换 NumPy 数组到 TensorFlow 张量
x_train_tf = tf.convert_to_tensor(x_train_3d, dtype=tf.float32)
x_train_tf = np.array(x_train_tf)
y_train = np.array(y_train)
#models = [clt(), cnn_lstm(), cnn(), lstm()]
models = [clt()]
X_train, X_test, Y_train, Y_test = train_test_split(x_train_tf, y_train, test_size=0.25, random_state=42)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
# 创建mask
# 由于我们已经填充了数据，mask将基于原始长度和目标长度创建
x_train_masks = tf.sequence_mask([len(ts) for ts in x_train_np], maxlen=max_len, dtype=tf.float32)
# x_train_masks = 1 - x_train_masks
# print(x_train_tf[0], x_train_masks[0])
print('Model building ... ')

#names = ['clt', 'cnnlstm', 'cnn', 'lstm']
names = ['lstm']
lll = 0
    # 模型编译
for model in models:
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    # 获取当前时间并格式化为字符串
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # 定义模型保存路径和文件名
    save_dir = 'models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_path = os.path.join(save_dir, f'model_{timestamp}.h5')

    history = model.fit(
        x=X_train,
        y=Y_train,
        verbose=1,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(X_test, Y_test),
        callbacks=[ModelCheckpoint(result_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=4, min_lr=0.000001),
                   # 当验证集损失函数不再改善时，停止训练
                   #EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto'),
                   ],
    )

    show(history)
