import os
file_name = []
for file in os.listdir(r'./labels'):
    file_name.append(file)

for file in os.listdir(r'./images'):
    name = file.replace('.jpg', '.txt')
    if name not in file_name:
        print(name)
