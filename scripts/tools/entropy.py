import cv2
import numpy as np
import math
import time

import torch
import torch.nn.functional as F
from PIL import Image

color_dict = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "blue": (5, 39, 175),
}

def get_entropy(img_):
    x, y = img_.shape[0:2]
    # img_ = cv2.resize(img_, (100, 100)) # 缩小的目的是加快计算速度
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(img_)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res
# for path_ in open('list.txt'):
#     t1 = time.time()
#     path = path_[:-1]
#     image = cv2.imread(path,0)
#     t2 = time.time()
#     res = get_entropy(image)

if __name__ == "__main__":
    patch_size = 64
    hw = 256 // patch_size
    
    path = "tests/image_5.png"
    image = cv2.imread(path, 0)
    # print(image.shape[0], image.shape[1], image.shape[2])
    
    entropy_list = []
    for i in range(hw):
        for j in range(hw):
            E = get_entropy(image[patch_size*i: patch_size*(i+1), patch_size*j:patch_size*(j+1)])
            print("entropy: ", E, i, j)
            entropy_list.append(E)
    entropy_tensor = torch.from_numpy(np.array(entropy_list)).view(1, hw, hw)
    print(entropy_tensor.size())

    entropy_tensor = entropy_tensor.sub(entropy_tensor.min()).div(max(entropy_tensor.max() - entropy_tensor.min(), 1e-5)).unsqueeze(0)
    print(entropy_tensor.size())
    entropy_tensor = F.interpolate(entropy_tensor, scale_factor=patch_size, mode="nearest").squeeze(0)
    print(entropy_tensor.size())
    
    low = Image.new('RGB', (256, 256), color_dict["blue"])
    high = Image.new('RGB', (256, 256), color_dict["red"])
    
    image = Image.open(path)
    score_map_i_np = entropy_tensor.view(256,256,1).cpu().detach().numpy()
    score_map_i_blend = Image.fromarray(np.uint8(high * score_map_i_np + low * (1.0 - score_map_i_np)))
    
    image_i_blend = Image.blend(image, score_map_i_blend, 0.5)
    image_i_blend.save("entropy.png")