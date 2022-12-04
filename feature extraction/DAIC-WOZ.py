
import csv
import os

# 用于存放生成特征文件的路径
output_path = r'D:\毕业设计\test_smile\dep_feature'
# 定义特征提取的配置文件
feature_config = 'IS10_paraling.conf -I '
# 定义opensmile 路径
op_path = r'D:\software\opensmile-3.0-win-x64'

# 写表头
filename = r'D:\毕业设计\test_smile\dep_features.csv'
with open(filename, 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    first_row = ['depr', 'gender']
    for i in range(1582):
        first_row.append(i + 1)
    writer.writerow(first_row)

# 遍历文件夹
label_gen = 1
label_dep = 1
for root, dirs, files in os.walk(r'D:\毕业设计\DAIC'):
    for dir_name in dirs:
        if dir_name == 'f_nor':
            label_gen = 1
            label_dep = 0
        if dir_name == 'm_depr':
            label_gen = 0
            label_dep = 1
        if dir_name == 'm_nor':
            label_gen = 0
            label_dep = 0

        audio_path = os.path.join(root, dir_name)
        audio_list = os.listdir(audio_path)
        features = []  # 二维数组，存放的是一个文件夹中所有音频的特征
        for audio in audio_list:
            if audio[-4:] == '.wav':
                this_path_input = os.path.join(audio_path, audio)
                print(this_path_input)
                this_path_output = os.path.join(output_path, audio[:-4] + '.csv')
                print(this_path_output)
                cmd = r'{}\bin\SMILExtract -C {}\config\is09-13\{}"{}" -O "{}"'. \
                    format(op_path, op_path, feature_config, this_path_input, this_path_output)
                print(cmd)
                os.system(cmd)

                with open(this_path_output, 'r') as f:
                    last_line = f.readlines()[-1]
                    feature = last_line.split(',')
                    # 去掉第一个和最后一个元素
                    feature = feature[1:-1]
                    feature.insert(0, label_gen)  # 第二列是gender
                    feature.insert(0, label_dep)  # 第一列是depression
                    features.append(feature)

        with open(filename, 'a+', newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(features)
