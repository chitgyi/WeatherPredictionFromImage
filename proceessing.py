import cv2
import numpy as np
from PIL import Image
import random
import Features

def describe(image_path,cropped_image_path,cont=True,bright=True,haze=True,sharpness=True,color_hist=True,intensity_hist=True,white_thresh=175):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cropped_image_rgb = cv2.imread(cropped_image_path, cv2.IMREAD_COLOR)
    description_array = np.reshape(cropped_image_rgb,(np.size(cropped_image_rgb))) / 255
    (norm_cont,c,max_b,avg_d,avg_b) = Features.contrast(image)
    if(cont == True):
        description_array = np.append(description_array,[norm_cont],axis=0)
    if (bright == True):
        bright_value = Features.brightness(image)
        description_array = np.append(description_array, [bright_value], axis=0)
    if (haze == True):
        haze_value = Features.haze(c,max_b,avg_d,avg_b)
        description_array = np.append(description_array, [haze_value], axis=0)
    if (sharpness == True):
        sharp_value = Features.sharpness(image)
        description_array = np.append(description_array, [sharp_value], axis=0)
    if (color_hist == True):
        color_h = Features.color_hist(image)
        description_array = np.concatenate((description_array, color_h), axis=0)
    if (intensity_hist == True):
        intensity = Features.intensity_hist(image,white_thresh)
        description_array = np.append(description_array, [intensity], axis=0)
    return description_array

def shuffle(data, label):
    temp = list(zip(data, label))
    random.shuffle(temp)
    return zip(*temp)

def __find_sky_area(path_of_image):
    read_image = cv2.imread(path_of_image, 50)
    edges = cv2.Canny(read_image, 150, 300)

    shape = np.shape(edges)
    left = np.sum(edges[0:shape[0] // 2, 0:shape[1] // 2])
    right = np.sum(edges[0:shape[0] // 2, shape[1] // 2:])

    if right > left:
        return 0  # if right side of image includes more building etc. return 0 to define left side(0 side) is sky area
    else:
        return 1  # if left side of image includes more building etc. return 1 to define right side(1 side) is sky area

def resize_image(base_size, path_of_image, destination, new_image_name):
    img = Image.open(path_of_image)

    if img.size[0] >= img.size[1]:
        sky_side = __find_sky_area(path_of_image)
        base_height = base_size
        wpercent = (base_height / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(wpercent)))
        img = img.resize((wsize, base_height), Image.ANTIALIAS)
        if sky_side == 0:  # Left side is sky side, so keep it and crop right side
            img = img.crop((0, 0, base_size, img.size[1]))  # Keeps sky area in image, crops from other non-sky side
        else:  # Right side is sky side, so keep it and crop left side
            img = img.crop((img.size[0] - base_size, 0, img.size[0],
                            img.size[1]))  # Keeps sky area in image, crops from other non-sky side
        img.save(destination + '/' + new_image_name)
    else:
        base_width = base_size
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), Image.ANTIALIAS)
        img = img.crop((0, 0, img.size[0], base_size))  # Keeps sky area in image, crops from lower part
        img.save(destination + '/' + new_image_name)