import os
import json
import numpy as np
import cv2
from PIL import Image

# # 单个图像
# image_path = 'img/US627.jpg'
# image = cv2.imread(image_path)  # 读取文件名对应的图片
# path_json = 'testdata/US627.json'
#
# location = []
# with open(path_json, 'r', encoding='gb18030') as path_json:
#     jsonx = json.load(path_json)
#
#     for shape in jsonx['shapes']:
#         if shape["label"] != "L5":
#             xy = np.array(shape['points'])
#             for x, y in xy:
#                 location.append((x,y))
# # # image = np.array(Image.open(image_path).convert('L'), dtype=np.float32)  #数据格式
#
# point_size = 4
# point_color = (0, 0, 255) # BGR
# thickness = 8 # 可以为 0 、4、8
#
# for point in location:
#     p = (int(point[0]),int(point[1]))
#     cv2.circle(image, p, point_size, point_color, thickness)
#
# # cv2.imshow('test',image)
# # cv2.waitKey (10000) # 显示 10000 ms 即 10s 后消失
# # cv2.destroyAllWindows()
# # save_path = "E:/GraduationProject/test"
#
# cv2.imwrite("./US627.png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
#
# # for root, dirs, files in os.walk('E:/GraduationProject/test/trans', True):
# #     for i, file in enumerate(files):
# #         print(file, i)


# # 多个图像

def csvt(path_json,img_name):
    with open(path_json,'r', encoding='gb18030') as path_json:
        jsonx=json.load(path_json)

        image_path = os.path.join('img/',img_name)
        image = cv2.imread(image_path)


        location = []
        for shape in jsonx['shapes']:
            if shape["label"] != "L5":
                xy = np.array(shape['points'])
                for x, y in xy:
                    location.append((x, y))
        point_size = 4
        point_color = (0, 0, 255)  # BGR
        thickness = 8  # 可以为 0 、4、8

        for point in location:
            p = (int(point[0]), int(point[1]))
            cv2.circle(image, p, point_size, point_color, thickness)

        image_path = os.path.join('visual_total/',img_name)

        cv2.imwrite(image_path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


dir_json = 'label4_9/'#原json位置,train和val分开各运行
list_json = os.listdir(dir_json)

for cnt,json_name in enumerate(list_json):
    print('cnt=%d,name=%s'%(cnt,json_name))
    path_json = dir_json + json_name
    # print(path_json, path_txt)
    img_name = json_name[0:5]+".jpg"
    csvt(path_json,img_name)









