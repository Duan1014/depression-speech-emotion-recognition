import seaborn as sns
import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from attention import attention_3d_block
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Input, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 读取数据Data,Label,考虑性别
df = pd.read_csv(r'D:\毕业设计\test_smile\dep_features.csv')
Label = df.loc[:, 'depr'].values
Label_gen = df['gender'].values
Data = df.iloc[:, 2:].values


# 将没有缺失值的行设为True
# 删除有Nan的行 X,Y
index = ~np.isnan(Data).any(axis=1)
X = Data[index, :]
Y = Label[index]
Y_gen = Label_gen[index]

# 归一化后添加性别特特征 X_gen, Y
scaler = StandardScaler()
X_new = scaler.fit_transform(X.astype(np.float32).reshape(X.shape[0], X.shape[1]))
X_gen = np.column_stack((Y_gen, X_new))

# smote过采样X_model, Y_smote
smote_model = SMOTE(k_neighbors=5, random_state=42)
X_smote, Y_smote = smote_model.fit_resample(X_gen, Y)

# 标签进行独热编码，数据转化成三维
Y_model = to_categorical(Y_smote, 2)
X_model = X_smote.reshape(X_smote.shape[0], 1, X_smote.shape[1])

# 划分训练集、验证集、测试集 6：2：2
X_train_all, X_test, Y_train_all, Y_test = train_test_split(X_model, Y_model, test_size=0.2)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=0.25)

# 定义LSTM的输入特征数
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


# 创建模型并打印
model = attention_model()
model.summary()

# 训练、测试
History = model.fit(X_train, Y_train, batch_size=64, epochs=100,
                    validation_data=(X_valid, Y_valid))
model.evaluate(X_test, Y_test)

# 分别保存模型的结构和参数
model.save(r'D:\毕业论文\软件提交\模型和图片\LSTM+S+A\model.h5', save_format="h5")
model.save_weights(r'D:\毕业论文\软件提交\模型和图片\LSTM+S+A\weights.h5')
json_string = model.to_json()
filename = r'D:\毕业论文\软件提交\模型和图片\LSTM+S+A\model_to_json.json'
with open(filename, 'w') as file:
    file.write(json_string)

# 查看attention层的输出
attention_model = Model(model.input, model.get_layer("attention_vec").output)
at_X_test = attention_model.predict(X_test)

# 可视化attention
attention_vector = at_X_test[0].flatten()
pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar', title='Attention Mechanism')
plt.savefig(r'D:\毕业论文\软件提交\模型和图片\LSTM+S+A\attention_show')

# 画accuracy和loss随训练次数的变化
plt.figure()
plt.plot(History.epoch, History.history['accuracy'], 'b', label='training')
plt.plot(History.epoch, History.history['val_accuracy'], 'r', label='validation')
plt.legend()
plt.savefig(r'D:\毕业论文\软件提交\模型和图片\LSTM+S+A\accuracy')

plt.figure()
plt.plot(History.epoch, History.history['loss'], 'b', label='training')
plt.plot(History.epoch, History.history['val_loss'], 'r', label='validation')
plt.legend()
plt.savefig(r'D:\毕业论文\软件提交\模型和图片\LSTM+S+A\loss')

# 绘制混淆矩阵
plt.figure()
pre_label = np.argmax(model.predict(X_test), axis=-1)
true_label = Y_test.argmax(axis=1)
conf_mat = confusion_matrix(y_true=true_label, y_pred=pre_label)
cm = pd.DataFrame(conf_mat, columns=["nor", "depr"], index=["nor", "depr"])
sns.heatmap(cm, cmap=plt.cm.Blues, fmt='d', annot=True)
plt.savefig(r'D:\毕业论文\软件提交\模型和图片\LSTM+S+A\confusion_matrix')
plt.show()
