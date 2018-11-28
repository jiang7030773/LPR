'''写了点小代码，统一图片命名和格式,删掉有重复命名的'''

import os

def rename(filepath):
    renames_nums = 0
    images = os.listdir(filepath)
    for name in images:
        if len(name) != 11:
            os.rename(filepath+name,filepath+name[0:7]+'.jpg')
            renames_nums += 1
    print("修改了%d张图片名"%renames_nums)



def Modify_suffix(filepath):
    renames_nums = 0
    images = os.listdir(filepath)
    for name in images:
        if name.find("bmp")==-1:
            renames_nums += 1
            os.rename(filepath+name, filepath+'浙A'+name.split('.')[0]+'.jpg')
            print("修改成功第%d张"%renames_nums)
            continue

def del_Repeat_name(filepath):
    Repeat_name = 0
    list1 = []
    images = os.listdir(filepath)
    for name in images:
        if name[0:7] not in list1:
            list1.append(name[0:7])
        else:
            os.remove(filepath+name)
            Repeat_name += 1
    print('删掉了%d张'%Repeat_name)   

if __name__ == '__main__':
    filepath = "./car_pic/image/train/"
    rename(filepath)
    del_Repeat_name(filepath)
    rename(filepath)