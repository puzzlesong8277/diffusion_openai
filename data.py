# -*- coding: utf-8 -*-
 
import cv2
import numpy as np
import os.path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
import random
import xml.dom.minidom as DOC
from skimage import exposure
 
 
# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
 
    return coords
 
# 将bounding box信息写入xml文件中, bouding box格式为[[x_min, y_min, x_max, y_max, name]]
def generate_xml(img_name, coords, img_size, out_root_path, images_path):
    '''
    输入：
        img_name：图片名称，如a.jpg
        coords:坐标list，格式为[[x_min, y_min, x_max, y_max, name]]，name为概况的标注
        img_size：图像的大小,格式为[h,w,c]
        out_root_path: xml文件输出的根路径
    '''
    doc = DOC.Document()  # 创建DOM文档对象
 
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
 
    title = doc.createElement('folder')
    title_text = doc.createTextNode('VOC2007')
    title.appendChild(title_text)
    annotation.appendChild(title)
 
    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)
 
    title = doc.createElement('path')
    title_text = doc.createTextNode(images_path + img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)
 
    source = doc.createElement('source')
    annotation.appendChild(source)
 
    title = doc.createElement('database')
    title_text = doc.createTextNode('The VOC2007 Database')
    title.appendChild(title_text)
    source.appendChild(title)
 
    title = doc.createElement('annotation')
    title_text = doc.createTextNode('PASCAL VOC2007')
    title.appendChild(title_text)
    source.appendChild(title)
 
    size = doc.createElement('size')
    annotation.appendChild(size)
 
    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)
 
    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)
 
    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)
 
    for coord in coords:
        object = doc.createElement('object')
        annotation.appendChild(object)
 
        title = doc.createElement('name')
        title_text = doc.createTextNode(coord[4])
        title.appendChild(title_text)
        object.appendChild(title)
 
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)
 
        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(float(coord[0]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(coord[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(coord[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(coord[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
 
    # 将DOM对象doc写入文件
    f = open(os.path.join(out_root_path, img_name[:-4] + '.xml'), 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()
 
 
# 为旋转提供的函数--按照旋转角度计算新生成的图片中boxs位置
def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # get width and heigh of changed image
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # rot_mat: the final rot matrix
    # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
    point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
    point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
    point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
    # concat np.array
    concat = np.vstack((point1, point2, point3, point4))
    # change type
    concat = concat.astype(np.int32)
    # print(concat)
    rx, ry, rw, rh = cv2.boundingRect(concat)
    return rx, ry, rw, rh
 
# 椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg
 
 
# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg
 
 
# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy
 
 
# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy
 
 
# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
 
    return rotated
 
# 平移
def translation(img, M):
    rows, cols = img.shape[:2]
 
    # 用仿射变换实现平移，第三个参数为输出的图像大小，值得注意的是该参数形式为(width, height)。
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst
 
 
# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image
 
 
# 图片文件夹路径
file_dir = r'path/to'
source_pic_root_path = file_dir  # your image folders
source_xml_root_path = file_dir  # your xml path
save_images = r'/home/lwf/nanting/guided-diffusion-main/path/to copy'  # image save path
save_labels = r'/home/lwf/nanting/guided-diffusion-main/path/to copy'  # label save path
need_aug_num = 6
 
for parent, _, files in os.walk(source_pic_root_path):
    for file in files:
        if(file[-4:]=='.xml'):
            continue
        cnt = 0
        while cnt < need_aug_num:
            try:
                img_path = os.path.join(parent, file)
                xml_path = os.path.join(source_xml_root_path, file[:-4] + '.xml')
 
                img = cv2.imread(img_path)
                coords = parse_xml(source_xml_root_path + file[:-4] + ".xml")  # 读xml文件
                names = [coord[4] for coord in coords]
                bboxes = [coord[:4] for coord in coords]
 
                w = img.shape[1]
                h = img.shape[0]
 
                # #旋转90 180 270
                rotated_item = rotate(img, 90)
                cv2.imwrite(save_images + file[0:-4] + '_r90.jpg', rotated_item)
                # rotated_item = rotate(img, 180)
                # cv2.imwrite(save_images + file[0:-4] + '_r180.jpg', rotated_item)
                # rotated_item = rotate(img, 270)
                # cv2.imwrite(save_images + file[0:-4] + '_r270.jpg', rotated_item)
                # # 旋转任意角度
                # # 第一步先旋转随机的一个角度
                # item = random.randint(1, 360)  # 随机生成最小值为1，最大值为360的整数（可以等于上下限）
                # rotated_item = rotate(img, item)
                # cv2.imwrite(save_images + file[0:-4] + '_r' + str(item) + '.jpg', rotated_item)
                #
                # # # 第二步生成对应图片的xml标注文件，重点是重置标注框
                # # new_name = img_name[0:-4] + '_r' + str(item)
                # # tree = ET.parse(xml_path)
                # # tree.find('filename').text = new_name + '.jpg'
                # # root = tree.getroot()
                # #
                # # if item in [90, 270]: # 如果旋转90或者270度，那直接宽高交换，其实没什么意义
                # #     d = tree.find('size')
                # #     width = int(d.find('width').text)
                # #     height = int(d.find('height').text)
                # #     # swap width and height
                # #     d.find('width').text = str(height)
                # #     d.find('height').text = str(width)
                # # for box in root.iter('bndbox'):
                # #     xmin = float(box.find('xmin').text)
                # #     ymin = float(box.find('ymin').text)
                # #     xmax = float(box.find('xmax').text)
                # #     ymax = float(box.find('ymax').text)
                # #     x, y, w, h = rotate_xml(img, xmin, ymin, xmax, ymax, item)
                # #     # change the coord
                # #     box.find('xmin').text = str(x)
                # #     box.find('ymin').text = str(y)
                # #     box.find('xmax').text = str(x + w)
                # #     box.find('ymax').text = str(y + h)
                # #     box.set('updated', 'yes')
                # # # write into new xml
                # # tree.write(save_labels + new_name + ".xml")
                # print("----------------")
 
                #平移任意位置
                # 第一步先平移
                x_min = w  # 裁剪后的包含所有目标框的最小的框
                x_max = 0
                y_min = h
                y_max = 0
                for bbox in bboxes:
                    x_min = min(x_min, bbox[0])
                    y_min = min(y_min, bbox[1])
                    x_max = max(x_max, bbox[2])
                    y_max = max(y_max, bbox[3])
 
                d_to_left = x_min  # 包含所有目标框的最大左移动距离
                d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
                d_to_top = y_min  # 包含所有目标框的最大上移动距离
                d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离
 
                x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
                y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)
 
                M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
                shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                new_name = file[0:-4] + '_t' + str(int(x)) + '_' + str(int(y))
                cv2.imwrite(save_images + new_name + '.jpg', shift_img)
                #第二步 生成对应的xml
                shift_bboxes = list()
                for bbox in bboxes:
                    i = 0
                    shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y, names[i]])
                    i += 1
 
                auged_img = shift_img
                auged_bboxes = shift_bboxes
                generate_xml(new_name + ".jpg", auged_bboxes, list(auged_img.shape), save_labels, save_images)
 
 
                # # # 镜像
                flipped_img = flip(img)
                cv2.imwrite(save_images + img_name[0:-4] + '_fli.jpg', flipped_img)
                #
                # # 增加噪声
                # # img_salt = SaltAndPepper(img, 0.3)
                # # cv2.imwrite(file_dir + img_name[0:7] + '_salt.jpg', img_salt)
                # img_gauss = addGaussianNoise(img, 0.3)
                # gaussian_noise_name = img_name[0:-4] + '_noise'
                # cv2.imwrite(save_images + gaussian_noise_name + '.jpg', img_gauss)
                # tree = ET.parse(xml_path)
                # tree.find('filename').text = gaussian_noise_name + '.jpg'
                # tree.find('path').text = save_images + gaussian_noise_name + '.xml'
                # tree.write(save_labels + gaussian_noise_name + ".xml")
                #
                # # 变亮、变暗
                # img_darker = darker(img)
                # darker_name = img_name[0:-4] + '_darker'
                # cv2.imwrite(save_images + darker_name + '.jpg', img_darker)
                # tree.find('filename').text = darker_name + '.jpg'
                # tree.find('path').text = save_images + darker_name + '.xml'
                # tree.write(save_labels + darker_name + ".xml")
                # img_brighter = brighter(img)
                # brighter_name =img_name[0:-4] + '_brighter'
                # cv2.imwrite(save_images + brighter_name + '.jpg', img_brighter)
                # tree.find('filename').text = brighter_name + '.jpg'
                # tree.find('path').text = save_images + brighter_name + '.xml'
                # tree.write(save_labels + brighter_name + ".xml")
                #
                # blur = cv2.GaussianBlur(img, (7, 7), 1.5)
                # blur_name = img_name[0:-4] + '_blur'
                # cv2.imwrite(save_images + blur_name + '.jpg', blur)
                # tree.find('filename').text = blur_name + '.jpg'
                # tree.find('path').text = save_images + blur_name + '.xml'
                # tree.write(save_labels + blur_name + ".xml")
                cnt += 1
            except:
                cnt += 1
                continue
 
 
