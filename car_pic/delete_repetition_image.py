import os

list1 = []
def delete_repetition_image(filepath):
    delete_nums = 0
    images = os.listdir(filepath)
    for name in images:
        if name.split('.')[0][0:7] in list1:
            delete_nums += 1
            os.remove(filepath+name)
            print("删掉了第%d张图片"%delete_nums)
        else:
            list1.append(name.split('.')[0][0:7])

if __name__ == '__main__':
    filepath = "./bmp/"
    delete_repetition_image(filepath)