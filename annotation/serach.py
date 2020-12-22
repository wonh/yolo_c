import os
import shutil

file_name = []
file_name2 = []
for file in os.listdir(r'E:\whr\Label_1w6\jpg_weibi'):
    basename = file.split('.')[0]
    file_name.append(basename)

for file in os.listdir(r'E:\whr\Label_1w6\xml_weibi'):
    # name = file.replace('.jpg', '.txt')
    basename = file.split('.')[0]
    file_name2.append(basename)
    if basename not in file_name:
        print(basename)
        shutil.move(os.path.join(r'E:\whr\Label_1w6\xml_weibi',file), os.path.join(r'E:\whr\Label_1w6\tmp1',file))

for file in os.listdir(r'E:\whr\Label_1w6\jpg_weibi'):
    basename = file.split('.')[0]
    if basename not in file_name2:
        print(basename)
        shutil.move(os.path.join(r'E:\whr\Label_1w6\jpg_weibi', file), os.path.join(r'E:\whr\Label_1w6\tmp1',file))