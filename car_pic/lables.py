import os
import codecs

def ListFilesToTxt(dir,file,wildcard,recursion):
    exts = wildcard.split(" ")
    for root, subdirs, files in os.walk(dir):
        for name in files:
            for ext in exts:
                print(root,name,ext)
                if(name.endswith(ext)):
                    file.write(name.split(".")[0]+'\n')
                    # file.write(name + "\n")
                    break
        if(not recursion):
            break
def Test():
  dir="./car_pic/image/test/"
  outfile="./car_pic/image/test_labels.txt"
  wildcard = ".jpg"

  file = open(outfile,'w+')
  file = codecs.open(outfile, 'w', 'utf-8')

  if not file:
    print ("cannot open the file %s for writing" % outfile)
  ListFilesToTxt(dir,file,wildcard, 0)

  file.close()
Test()