import os
import cv2


def ranamesJPG(filepath):
    renames_nums = 0
    images = os.listdir(filepath)
    for name in images:
        if name.find("bmp")==-1:
            renames_nums += 1
            os.rename(filepath+name, filepath+'浙A'+name.split('.')[0]+'.jpg')
            print("修改成功第%d张"%renames_nums)
            continue
         

if __name__ == '__main__':
    filepath = "./bmp/"
    ranamesJPG(filepath)