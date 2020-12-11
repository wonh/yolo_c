# import xml.dom.minidom
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil

if __name__ == '__main__':

    with open('./my.name', "r", encoding="gbk") as f:
        names = f.read().split("\n")[:-1]
    class_num = []
    labels = np.zeros((1,5))
    print(labels)
    category = [range(15)]
    labels_path = r'./data_3w6+pinp1+cds/labels/train'
    images_path = r'./jpg_all'
    xml_path = r'./xml_3w8_revise'
    dst_path = r'./output1'

    for file in tqdm(os.listdir(labels_path)):
        file_path = os.path.join(labels_path, file)
        label = np.loadtxt(file_path).reshape(-1, 5)
        unique_cls = np.unique(label[:,0])
        for i in unique_cls:
            dst = './output1/{}/images'.format(i)
            xml_dst = './output1/{}/xml'.format(i)
            os.makedirs(dst, exist_ok=True)
            os.makedirs(xml_dst, exist_ok=True)
            src_path = os.path.join(images_path,file.replace(".txt",".jpg"))
            shutil.copy(src_path,os.path.join(dst,file.replace(".txt",".jpg")))
            shutil.copy(os.path.join(xml_path, file.replace('.txt','.xml')),os.path.join(xml_dst,file.replace(".txt",".xml")))
        # labels.append(label)
        labels = np.concatenate((labels,label), axis=0)
        # print(labels)

    for i in range(15):
        print('category{}'.format(i),len(labels[labels[:,0]==i]))
        class_num.append(np.count_nonzero(labels[:,0]==i))
    plt.figure(0)
    # plt.hist(np.asarray(class_num), bins=10, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.bar(names, np.asarray(class_num))
    plt.xlabel("类别")
    # plt.xticks(names)
    # 显示纵轴标签
    plt.ylabel("个数")
    # 显示图标题
    plt.title("数据分布图")
    plt.savefig(r'k.jpg')
    plt.show()

