#!/usr/bin/env python

import os
import sys
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd

def extract_rgb_image(image_path):
    #print(image_path)
    if len(cv.imread(image_path, cv.IMREAD_UNCHANGED).shape) == 3:
    #f = Image.open(image_path)
    #image = np.asarray(f)
    #f.close()
    #if image.ndim == 3:
        return True
    else:
        return False


def main():
    train = '/data/meta/annotation_train.txt'
    test = '/data/meta/annotation_test.txt'

    #for file_path, image_path in zip([train, test], ['train', 'test']):
    for file_path, image_path in zip([test], ['test']):
        f_list = []
        l_list = []
        for line in open(file_path):
            line = line.split(" ")
            if extract_rgb_image(os.path.join("/data/images", line[0])):
                f_list.append(line[0])
                l_list.append(int(line[1]))
                if(len(f_list) % 10000 == 0):
                    print(len(f_list))
        out_fname = file_path.split(".")[0] + "_rgb.txt"
        pd.DataFrame(data={'file':f_list, 'label':l_list}).to_csv(out_fname, index=False, sep=" ", header=False)
    

if __name__ == '__main__':
    main()
    


    
