import numpy as np


def img_tensor_to_np(img_tensor):
    img_np = np.zeros_like(img_tensor).reshape(img_tensor.shape[1], -1, 3)
    for i in range(img_tensor.shape[1]):
        for j in range(img_tensor.shape[2]):
            arr = np.array([img_tensor[0, i, j], img_tensor[1, i, j], img_tensor[2, i, j]])
            img_np[i, j] = arr
    return img_np
