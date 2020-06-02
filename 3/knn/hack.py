import numpy as np

from knn import knn
from show_image import show_image
from extract_image import extract_image

def hack(img_name, k):
    '''
    HACK Recognize a CAPTCHA image
      Inputs:
          img_name: filename of image
      Outputs:
          digits: 1x5 matrix, 5 digits in the input CAPTCHA image.
    '''
    data = np.load('hack_data.npz')

    # YOUR CODE HERE (you can delete the following code as you wish)
    x_train = data['x_train']
    y_train = data['y_train']

    img = extract_image(img_name)

    digits = knn(img, x_train, y_train, k)

    return digits


from bs4 import BeautifulSoup
import urllib.request
from urllib.request import urlopen
import cv2
import os


def download_CAPTCHA(file):
    urllib.request.urlretrieve("http://cwcx.zju.edu.cn/WFManager/loginAction_getCheckCodeImg.action", file)

def get_raw_data(dirname, n=120):
    if not os.path.exists(dirname):
        for i in range(n):
            download_CAPTCHA(dirname + '/' + str(i) + '.jpg')
    else:
        print('Directory "' + dirname + '" has already exists!')

def generate_dataset(dirname, n, filename):
    labels = open(dirname + '/labels.txt', 'r')
    dataset = {}
    x, y = [], []
    for i in range(n):
        line = labels.readline()
        img = extract_image(dirname + '/' + str(i) + '.jpg')
        x.append(img)
        for j in range(img.shape[0]):
            y.append(int(line[j]))
    x, y = np.vstack(x), np.array(y)
    n_train = n
    np.savez(filename, x_train=x[:n_train], y_train=y[:n_train], x_test=x[n_train:], y_test=y[n_train:])


download_CAPTCHA('hack_data/test.jpg')
dirname = 'hack_data/raw'
generate_dataset(dirname, 120, 'hack_data')
# data = np.load('hack_data.npz')
# print(data['y_train'].shape)