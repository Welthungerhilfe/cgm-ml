import glob
import os
from shutil import copyfile



for file in glob.iglob('/whhdata/qrcode/**/depth/*.png', recursive=True):
    print(file)

    dst_path = file.replace('whhdata', 'localssd2')

    dst_folder = os.path.dirname(dst_path)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)


    copyfile(file, dst_path)
    print(dst_path)

