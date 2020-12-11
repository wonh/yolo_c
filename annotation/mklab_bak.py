import xml.dom.minidom
import json
import os
import random
import shutil
import imghdr
from PIL import Image
import tqdm
import sys

import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
    else:
        print('File already exit')
    return True

srcPath = r'./xml_3w8r+pinp1+cds/'
# name_path = r'./data_3w6+pinp1+cds/xml/train'
name_path = r'./data_3w8/xml/val'
dstPath = r'./data_3w8r+pinp1+cds_140/'
trainImgPath = os.path.join(dstPath,'images','train')
valImgPath = os.path.join(dstPath,'images','val')
trainLabelPath = os.path.join(dstPath,'labels','train')
valLabelPath = os.path.join(dstPath,'labels','val')
trainXmlPath = os.path.join(dstPath,'xml','train')
valXmlPath = os.path.join(dstPath,'xml','val')
mkdir(trainImgPath)
mkdir(valImgPath)
mkdir(trainLabelPath)
mkdir(valLabelPath)
mkdir(trainXmlPath)
mkdir(valXmlPath)
tmpPath = r'./jpg_all/'

trainTxt = os.path.join(dstPath,'train.txt')
valTxt = os.path.join(dstPath,'val.txt')

f_train = open(trainTxt,'a')
f_val = open(valTxt,'a')
current_path = os.path.abspath(__file__)
global m1,m2
checkSum = []
classify = {}
illegalLabel = []
# illegalLabel = ['\\', '4w', '9w', '10.', '5w', '2d', '54', 'w10', '47', 'dog', 'W', '19', '10w', '33', '8w', '43','9 ']
legalLabel = {'1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'10','a':'11','g':'12','k':'13','l':'14','w':'15'}
# legalLabel = {'1':'1', '9':'2', '11':'3'}
def parseXml(path,flag = False):
    global m1,m2
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    filename = root.getElementsByTagName('filename')
    fileInfo = root.getElementsByTagName('size')
    fileBbox = root.getElementsByTagName('object')
    if len(filename) != 1 or len(fileInfo) != 1:
        print('Debug!!',path,'[%d][%d]'%(len(filename),len(fileInfo)))
    else:
        name = filename[0].firstChild.data
        w = int(fileInfo[0].getElementsByTagName('width')[0].firstChild.data)
        h = int(fileInfo[0].getElementsByTagName('height')[0].firstChild.data)
        if w == 0 or h == 0:
            img = Image.open(os.path.join(tmpPath,name))
            h,w = img.size
        tmp = ''

        for i in range(len(fileBbox)):
            n_class = fileBbox[i].getElementsByTagName('name')[0].firstChild.data
            bndbox = fileBbox[i].getElementsByTagName('bndbox')
            if flag == True:
                checkSum.append(n_class)
                continue
            if n_class in illegalLabel:
                continue
            checkSum.append(n_class)
            if n_class not in legalLabel.keys():
                continue
            n_class = legalLabel[n_class]
            n_class = int(n_class) - 1
            if n_class in classify.keys():
                classify[n_class] += 1
            else:
                classify[n_class] = 1
            if len(bndbox) != 1:
                print('Debug!! bndbox')
            else:
                # print(type(bndbox[0].getElementsByTagName('xmin')[0].firstChild.data))
                x_min = float(bndbox[0].getElementsByTagName('xmin')[0].firstChild.data)
                y_min = float(bndbox[0].getElementsByTagName('ymin')[0].firstChild.data)
                x_max = float(bndbox[0].getElementsByTagName('xmax')[0].firstChild.data)
                y_max = float(bndbox[0].getElementsByTagName('ymax')[0].firstChild.data)
                if x_max > m1:
                    m1 = x_max
                if y_max > m2:
                    m2 = y_max
                # print('y:{}'.format(y_max))

                w_x = (x_max - x_min)   #宽度
                h_y = (y_max - y_min)   #高度
                c_x = (x_min + x_max)/2  #中心点x
                c_y = (y_min + y_max)/2   #中心点y
                p = 1.4
                x_max = min(int(c_x + p*w_x/2), w)
                y_max = min(int(c_y + p*h_y/2), h)
                x_min = max(int(c_x - p*w_x/2), 0)
                y_min = max(int(c_y - p*h_y/2), 0)
                w_x = (x_max - x_min)
                h_y = (y_max - y_min)

            if w_x > 0 and h_y > 0:
                    n_data = [str(round((x_min+w_x/2.0)/w,6)),str(round((y_min+h_y/2.0)/h,6)),str(round(w_x/w,6)),str(round(h_y/h,6))]
                    tmp += str(n_class)+' '+(' ').join(n_data)+'\n'

        if tmp != '':
            rand_num = random.randint(0, 100)
            basename = os.path.basename(path)
            if rand_num >= 200:
                shutil.copyfile(os.path.join(tmpPath,name),os.path.join(trainImgPath,name))  #拷贝图片
                shutil.copyfile(path,os.path.join(trainXmlPath,basename))  #拷贝xml
                f = open(os.path.join(trainLabelPath,name.replace('jpg','txt')),'a')  #在.txt文件中写入坐标
                f.write(tmp)
                f.close()
                # print(os.path.join(trainImgPath,name))
                # f_train.write((os.path.abspath(os.path.join(trainImgPath,name))+'\n').encode('gbk','ignore'))
                try:
                    f_train.write(os.path.abspath(os.path.join(trainImgPath,name))+'\n') #训练文件写入.txt
                except:
                    print(name)
            else:
                shutil.copyfile(os.path.join(tmpPath,name),os.path.join(valImgPath,name))  #拷贝图片
                shutil.copyfile(path, os.path.join(valXmlPath, basename))  #拷贝xml
                f = open(os.path.join(valLabelPath,name.replace('jpg','txt')),'a')  #在.txt文件中写入坐标
                f.write(tmp)
                f.close()
                # print(os.path.join(valImgPath,name))
                # f_val.write((os.path.abspath(os.path.join(valImgPath,name))+'\n').encode('gbk','ignore'))
                try:
                    f_val.write(os.path.abspath(os.path.join(valImgPath,name))+'\n')
                except:
                    print(name)

for filedata in tqdm.tqdm(os.listdir(name_path)):
    path = os.path.join(srcPath,filedata)
    if not os.path.exists(path):
        continue
    if filedata.find('xml') >= 0 and os.path.exists(os.path.join('./jpg',filedata.replace('xml','jpg'))) and imghdr.what(os.path.join('./jpg',filedata.replace('xml','jpg'))) != None:
        parseXml(path,True)

illegalLabel = list(set(checkSum).difference(list(legalLabel.keys())))
print(illegalLabel)

global m1, m2
m1 = 0
m2 = 0
for filedata in tqdm.tqdm(os.listdir(name_path)):
    path = os.path.join(srcPath,filedata)
    if not os.path.exists(path):
        continue
    if filedata.find('xml') >= 0 and os.path.exists(os.path.join(tmpPath,filedata.replace('xml','jpg'))) and imghdr.what(os.path.join(tmpPath,filedata.replace('xml','jpg'))) != None:
        parseXml(path)
print(m1,m2)
f_train.close()
f_val.close()


