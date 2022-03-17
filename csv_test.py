# coding:utf-8
import csv
import codecs
import os
import json
import numpy as np
# 将json文件转化为csv文件
# python tools/train.py --cfg experiments/300w/face_alignment_300w_hrnet_w18.yaml

def csvt(path_json,img_name,list):
    with open(path_json,'r', encoding='gb18030') as path_json:
        jsonx=json.load(path_json)
        location = (img_name, 1, 256, 512)
        for shape in jsonx['shapes']:
            xy=np.array(shape['points'])
            for x,y in xy:
                location = location + (x,y)
        list.append(location)
        return(list)

dir_json = 'trans/'#原json位置,train和val分开各运行
list_json = os.listdir(dir_json)

data =[]
t = ("image_name", "scale", "center_w", "center_h")
for i in range(0, 96):
    x = f"original_{i}_x"
    y = f"original_{i}_y"
    t = t + (x, y)
data.append(t)

for cnt,json_name in enumerate(list_json):
    print('cnt=%d,name=%s'%(cnt,json_name))
    path_json = dir_json + json_name
    # print(path_json, path_txt)
    img_name = json_name[0:5]+".jpg"
    data = csvt(path_json,img_name,data)


# f = codecs.open('face_landmarks_300w_train.csv','w','gbk')
f = codecs.open('face_landmarks_300w_valid.csv','w','gbk')
writer = csv.writer(f)
for i in data:
    writer.writerow(i)
f.close()
