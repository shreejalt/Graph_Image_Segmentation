from algo import *
import matplotlib.pyplot as plt
from cv2 import resize,imwrite, imread
import os

sigma = 0.5
min_val = 50
thresh = 500
path_read = './Data'
path_write = './Results'
for image in os.listdir(path):
    img = cv2.imread(os.path.join(path,image))
    img = resize(img, (256,256))
    out = ans_out(thresh, img, min_val, sigma)
    imwrite(os.path.join(path_write,str(image)+"_out.jpg"), out) 
    
