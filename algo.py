import numpy as np
from disjoint_set import *
from scipy.signal import convolve2d
from math import ceil

def make_filter(sigma, alpha=4):
    sig = max(sigma, 0.01)
    length = int(ceil(sig * alpha)) + 1
    m = length / 2
    n = m + 1
    x, y = np.mgrid[-m:n,-m:n]
    g = np.exp(m ** 2) * np.exp(-0.5 * (x**2 + y**2))
    return g / g.sum()

def smooth_filter(img, filterg):
    img_smooth_red = convolve2d(img[:, :, 0],filterg, mode='same')
    img_smooth_green = convolve2d(img[:, :, 1],filterg, mode='same')
    img_smooth_blue = convolve2d(img[:, :, 2],filterg, mode='same')
  
    return (img_smooth_red, img_smooth_green, img_smooth_blue)

def L2_norm(img_smooth_red, img_smooth_green, img_smooth_blue, x1, y1, x2, y2):

    L2_norm_array = np.array([img_smooth_red[y1, x1] - img_smooth_red[y2, x2],img_smooth_green[y1, x1] - img_smooth_green[y2, x2],
            img_smooth_blue[y1, x1] - img_smooth_blue[y2, x2]])
    return np.linalg.norm(L2_norm_array, ord = 2)

def define_graph(img_smooth_red, img_smooth_green, img_smooth_blue):
    
    img_height = img_smooth_red.shape[0]
    img_width = img_smooth_red.shape[1]
    
    edges = np.zeros(shape = (img_width*img_height*4,3))
    cnt=0
    for y in range(img_height):
        for x in range(img_width):
            if x < img_width-1:
                edges[cnt,0] = int(y*img_width + x)
                edges[cnt,1] = int(y*img_width + (x+1))
                edges[cnt,2] =  L2_norm(img_smooth_red, img_smooth_green, img_smooth_blue, x,y,x+1,y)
                cnt+=1
            if y < img_height-1:
                edges[cnt,0] = int(y*img_width + x)
                edges[cnt,1] = int((y+1)*img_width + x)
                edges[cnt,2] =  L2_norm(img_smooth_red, img_smooth_green, img_smooth_blue, x,y,x,y+1)
                cnt+=1
            if x < img_width -1 and y < img_height - 2:
                edges[cnt,0] = int(y*img_width + x)
                edges[cnt,1] = int((y+1)*img_width + (x+1))
                edges[cnt,2] =  L2_norm(img_smooth_red, img_smooth_green, img_smooth_blue, x,y,x+1,y+1)
                cnt+=1
            if x < img_width-1 and y>0:
                edges[cnt,0] = int(y*img_width + x)
                edges[cnt,1] = int((y-1)*img_width + (x+1))
                edges[cnt,2] =  L2_norm(img_smooth_red, img_smooth_green, img_smooth_blue, x,y,x+1,y-1)
                cnt+=1
    return edges[~(edges==0.0).all(1)]
    
def change_thresh(size, thresh):
    return thresh/size

def segment_graph(edges, img_dims, thresh):
    sorted_edges = edges[edges[:,2].argsort()]
    thresh_array = np.zeros(img_dims[0]*img_dims[1], dtype=float)
    ds = disjoint_set_2(img_dims[0]*img_dims[1])
    
    for i in range(len(thresh_array)):
        thresh_array[i] = change_thresh(1, thresh)
        
    for i in range(len(sorted_edges)):
        inp_arr = sorted_edges[i,:]

        inp1 = ds.find_set(int(inp_arr[0]))
        inp2 = ds.find_set(int(inp_arr[1]))

        if inp1 != inp2:
            if inp_arr[2] <= thresh_array[inp1] and inp_arr[2]<=thresh_array[inp2]:
                ds.join_set(inp1, inp2)
                inp1 = ds.find_set(inp1)
                thresh_array[inp1] = inp_arr[2]+change_thresh(ds.size(inp1), thresh)
    return ds

def join_small_segments(ds, min_val, edges):
    
    for i in range(len(edges)):
        
        inp1 = ds.find_set(int(edges[i,0]))
        inp2 = ds.find_set(int(edges[i,1]))
        
        if inp1!=inp2 and ((ds.size(inp1)<=min_val) or (ds.size(inp2)<=min_val)):
            ds.join_set(inp1, inp2)
    return ds

def generate_image(img_dims, ds):
    
    out = np.zeros(shape=(img_dims[0], img_dims[1],3))
    colors = np.zeros(shape=(img_dims[0]*img_dims[1],3))
    for i in range(img_dims[0]*img_dims[1]):
        colors[i,0] = np.random.randint(0,255)
        colors[i,1] = np.random.randint(0,255)
        colors[i,2] = np.random.randint(0,255)
    for i in range(img_dims[0]):
        for j in range(img_dims[1]):
            parent = ds.find_set((i*img_dims[1] + j))
            out[i,j,:] = colors[parent,:]
    return out

def ans_out(thresh, img, min_val, sigma):
    filterg = make_filter(sigma)
    img_smooth_red, img_smooth_green, img_smooth_blue = smooth_filter(img, filterg)
    img_dims = (img.shape[0],img.shape[1])
    edges = define_graph(img_smooth_red, img_smooth_green, img_smooth_blue)
    ds = segment_graph(edges, img_dims, thresh)
    ds = join_small_segments(ds, min_val, edges)
    out = generate_image(img_dims, ds)
    return out