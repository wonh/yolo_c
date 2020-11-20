import os
import shutil
from tqdm import tqdm

train_path = r'../data/train/Net-Jam-1500'
train_path2 = r'../data/train/polyp-fake-35204'

infer_path4 = r'../data/info/dst/Net-Jam-1500-full/src-fake4'
infer_path5 = r'../data/info/dst/Net-Jam-1500-full/src-fake5'
infer_path6 = r'../data/info/dst/Net-Jam-1500-full/src-fake6'



train_name = []

for root, _, files in tqdm(os.walk(train_path2)):
    for file in files:
        train_name.append(file)
num = 0
for path in (infer_path4, infer_path5, infer_path6):
    for root, dirs, files in tqdm(os.walk(path)):
        for file in files:
            id1 = root.split('/')[-1]
            if 'polyp' not in id1:
                continue
            if file in train_name:
                continue
            else:
                # shutil.copy(os.path.join(root,file), os.path.join(r'./outputz2/',file))
                print(os.path.join(root,file))
                num += 1
print(num)

