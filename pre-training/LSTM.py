import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 读取数据Data,Label,性别作为标签
df = pd.read_csv(r'D:\毕业设计\test_smile\dep_features.csv')
Data = df.iloc[:, 2:].values
Label = df['depr'] + df['gender'] * 2

# 将没有缺失值的行设为True
# 删除有Nan的行,X,Y
index = ~np.isnan(Data).any(axis=1)
X = Data[index, :]
Y = Label[index]

# 归一化
scaler = StandardScaler()
X_new = scaler.fit_transform(X.astype(np.float32).reshape(X.shape[0], X.shape[1]))

# 标签进行独热编码，数据转化成三维
Y_model = to_categorical(Y, 4)
X_model = X_new.reshape(X_new.shape[0], 1, X_new.shape[1])

# 划分训练集、验证集、测试集 6：2：2
X_train_all, X_test, Y_train_all, Y_test = train_test_split(X_model, Y_model, test_size=0.2)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=0.25)


# 序列法构建模型
def create_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, 1582), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))  # 全连接层
    model.add(Dense(4, activation='softmax'))  # 分类层
    optimzer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])
    return model


# 创建模型并打印
model = create_model()
model.summary()


# 训练、测试
History = model.fit(X_train, Y_train, batch_size=64, epochs=100,
                    validation_data=(X_valid, Y_valid))
model.evaluate(X_test, Y_test)

# 分别保存模型的结构和参数
model.save(r'D:\毕业论文\软件提交\模型和图片\基线四分类\model.h5', save_format="h5")
model.save_weights(r'D:\毕业论文\软件提交\模型和图片\基线四分类\weights.h5')
json_string = model.to_json()
filename = r'D:\毕业论文\软件提交\模型和图片\基线四分类\model_to_json.json'
with open(filename, 'w') as file:
    file.write(json_string)


# 画accuracy和loss随训练次数的变化
plt.figure()
plt.plot(History.epoch, History.history['accuracy'], 'b', label='training')
plt.plot(History.epoch, History.history['val_accuracy'], 'r', label='validation')
plt.legend()
plt.savefig(r'D:\毕业论文\软件提交\模型和图片\基线四分类\accuracy')

plt.figure()
plt.plot(History.epoch, History.history['loss'], 'b', label='training')
plt.plot(History.epoch, History.history['val_loss'], 'r', label='validation')
plt.legend()
plt.savefig(r'D:\毕业论文\软件提交\模型和图片\基线四分类\loss')

# 绘制混淆矩阵
plt.figure()
pre_label = np.argmax(model.predict(X_test), axis=-1)
true_label = Y_test.argmax(axis=1)
conf_mat = confusion_matrix(y_true=true_label, y_pred=pre_label)
cm = pd.DataFrame(conf_mat, columns=["m_nor", "m_depr", "f_nor", "f_depr"],
                  index=["m_nor", "m_depr", "f_nor", "f_depr"])
sns.heatmap(cm, cmap=plt.cm.Blues, fmt='d', annot=True)
plt.savefig(r'D:\毕业论文\软件提交\模型和图片\基线四分类\confusion_matrix')
plt.show()

