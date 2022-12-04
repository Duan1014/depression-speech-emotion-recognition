import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from attention import attention_3d_block
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
num = 1583


def attention_model():
    input_layer = Input(shape=(1, num))  # 输入层
    lstm_layer = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))(input_layer)  # lstm层
    dropout = Dropout(0.1)(lstm_layer)
    attention_mul = attention_3d_block(dropout)  # attention层
    flatten = Flatten()(attention_mul)
    dense = Dense(32, activation='relu')(flatten)  # 全连接层
    output_layer = Dense(2, activation='softmax')(dense)
    model = Model([input_layer], outputs=[output_layer])
    optimzer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])
    return model


# 读取数据X,Y,考虑性别Y_gen
df = pd.read_csv(r'D:\毕业设计\transfer-learning\dep_features.csv')
Y = df.loc[:, 'depr'].values
Y_gen = df['gender'].values
X = df.iloc[:, 2:].values

# 归一化后添加性别特特征 X_gen, Y
scaler = StandardScaler()
X_new = scaler.fit_transform(X.astype(np.float32).reshape(X.shape[0], X.shape[1]))
X_gen = np.column_stack((Y_gen, X_new))

# 标签进行独热编码，数据转化成三维
X_lan = X_gen.reshape(X_gen.shape[0], 1, X_gen.shape[1])
Y_lan = to_categorical(Y, 2)

# 划分训练集、验证集、测试集 6：2：2
X_train_all, X_test, Y_train_all, Y_test = train_test_split(X_lan, Y_lan, test_size=0.2)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=0.25)

# 有relief的的预训练
model = attention_model()
model.load_weights(r'D:\LSTMAS\model\weights.h5')

# 设置微调哪些层
for layer in model.layers[:12]:
    print(layer.name)
    layer.trainable = False
    # 分类层
    if layer.name == 'dense_2':
        layer.trainable = True
    # 注意力层
    if layer.name == 'dense':
        layer.trainable = True
    # 全连接层
    if layer.name == 'dense_1':
        layer.trainable = False
    # LSTM层
    if layer.name == 'lstm':
        layer.trainable = False

# 打印模型，并查看可训练的参数
model.summary()
print('参与训练的权重：')
for x in model.trainable_weights:
    print(x.name)
print('\n')

optimzer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])
History = model.fit(X_train, Y_train, batch_size=64, epochs=500,
                    validation_data=(X_valid, Y_valid))

model.evaluate(X_test, Y_test)
# model.evaluate(X_lan, Y_lan)

# 画accuracy和loss
plt.figure()
plt.plot(History.epoch, History.history['accuracy'], 'b', label='training')
plt.plot(History.epoch, History.history['val_accuracy'], 'r', label='validation')
plt.legend()
plt.savefig(r'D:\毕业论文\软件提交\微调结果\LSTM+S+A\accuracy')

plt.figure()
plt.plot(History.epoch, History.history['loss'], 'b', label='training')
plt.plot(History.epoch, History.history['val_loss'], 'r', label='validation')
plt.legend()
plt.savefig(r'D:\毕业论文\软件提交\微调结果\LSTM+S+A\loss')

# 绘制混淆矩阵
plt.figure()
pre_label = np.argmax(model.predict(X_test), axis=-1)
true_label = Y_test.argmax(axis=1)
conf_mat = confusion_matrix(y_true=true_label, y_pred=pre_label)
cm = pd.DataFrame(conf_mat, columns=["nor", "depr"], index=["nor", "depr"])
sns.heatmap(cm, cmap=plt.cm.Blues, fmt='d', annot=True)
plt.savefig(r'D:\毕业论文\软件提交\微调结果\LSTM+S+A\confusion_matrix')
plt.show()
