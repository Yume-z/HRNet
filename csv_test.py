import os
import csv

# 设置要遍历的文件夹路径和csv文件路径
folder_path = "./txt_test/"
csv_file_path = "./test.csv"

# 打开csv文件，准备写入数据
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["image_name", "x_0", "y_0", "x_1", "y_1"])
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 判断是否为txt文件
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                # 打开txt文件，读取数据
                with open(file_path, mode='r') as txt_file:
                    data = txt_file.read()
                    data = data[1:-2]

                    data_list = data.split(',')

                # 将数据写入csv文件中
                writer.writerow([file[:-4]+'.jpg', data_list[0], data_list[1], data_list[2], data_list[3]])
