import os
import shutil


# Rename jpg and txt
path = r"./data"
image_path = r"./image"
txt_path = r"./txt"
dirs = os.listdir(path)

# path_det = r"./test_gt_img"
#
# dirs = os.listdir(path_det)

for dir in dirs:
    path_dir = os.path.join(path,dir,'sub_image')
    files = os.listdir(path_dir)
    jpg_dir = []
    txt_dir = []
    for file in files:
        if 'txt' in file:
            txt_dir.append(file)
        else:
            jpg_dir.append(file)


    for i,file in enumerate(jpg_dir):
        newname = dir[:2] + dir[2:].zfill(3) + '_' + str(i+1).zfill(2) + file[-4:]
        print(newname)
        oldpath = os.path.join(path_dir,file)
        newpath = os.path.join(path_dir, newname)
        os.rename(oldpath,newpath)

        des_path = os.path.join(image_path, newname)
        shutil.move(newpath,des_path)

    for i,file in enumerate(txt_dir):
        newname = dir[:2] + dir[2:].zfill(3) + '_' + str(i+1).zfill(2) + file[-4:]
        print(newname)
        oldpath = os.path.join(path_dir,file)
        newpath = os.path.join(path_dir, newname)
        os.rename(oldpath,newpath)

        des_path = os.path.join(txt_path, newname)
        shutil.move(newpath, des_path)




#
