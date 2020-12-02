import os
# get .txt document
rootdir=os.path.join('./bupt_data/')
# read
write_path=open('./bupt_data/paths.txt', 'w')
for (dirpath,dirnames,filenames) in os.walk(rootdir):
    for filename in filenames:
        if filename == 'IMU.txt':
            write_path.write(os.path.join(dirpath, filename) + '$')
            write_path.write(os.path.join(dirpath, 'press_label.txt') + '\n')
write_path.close()
