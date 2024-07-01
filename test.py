import numpy as np
import cv2

arr = np.load('logger/openai-2024-05-31-15-50-56-386247/samples_100x256x256x3/arr_1.npy')  # load cby.npy

MIN_DEPTH = 0   # minimum depth
MAX_DEPTH = min(300, np.percentile(arr, 99)) # maximum depth
arr = np.clip(arr, MIN_DEPTH, MAX_DEPTH)  # clip

MIN_DEPTH = np.min(arr)  # min depth
MAX_DEPTH = np.max(arr)  # max depth
arr = (arr - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)  # 归一化
arr *= 255  #  转化成8bit整数格式
arr_gray = arr.astype(np.uint8)  # gray
arr_bgr = cv2.cvtColor(arr_gray, cv2.COLOR_GRAY2BGR)  # gray color
cv2.imwrite('arrm5_1.png', arr_bgr)   # save gray 
# cv2.imshow('arr', arr_gray)
# cv2.waitKey(0)np.load('logger/openai-2024-05-20-19-21-42-295408/samples_100x256x256x3/arr_0.npy') # load cby.npy