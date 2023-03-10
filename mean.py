import cv2
import os


m = [0,0,0]
s = [0,0,0]
j = 0
path_img = 'image/'
for root, dirs, files in os.walk(path_img, True):
    for file in files:  # 文件遍历
        line1 = path_img + file
        img = cv2.imread(line1)



        img = img / 255
        for i in range(3):
            m[i]=img[i, :, :].mean()+m[i]
            s[i]=img[i, :, :].std()+s[i]
        j+=1

print(m[0]/j, m[1]/j,m[2]/j)
print(s[0]/j, s[1]/j,s[2]/j)