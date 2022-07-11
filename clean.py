import cv2
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor, ceil
from PIL import Image
import math
from skimage import data
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

image_scale = 0

def read_image(path):

    img = cv2.imread(path)  # path , cv2.IMREAD_GRAYSCALE
    
    '''
    cv2.imwrite("./asda.png", img)
    print(img) 
    dimg = Image.fromarray(img)
    dimg.save("./asda.png")
    dimg.show()
    '''
    
    size = img.shape
    dimension = (size[0], size[1])

    return img, size, dimension

def image_change_scale(img, dimension, scale=100, interpolation=cv2.INTER_LINEAR):
    scale /= 100
    new_dimension = (int(dimension[1]*scale), int(dimension[0]*scale))
    resized_img = cv2.resize(img, new_dimension, interpolation=interpolation)
    return resized_img

def show_result(images_list):
    global image_scale

    titles = list(images_list.keys())
    images = list(images_list.values())

    fig, axs = plt.subplots(2, 3)
    fig.suptitle(f'{image_scale} Percent of the original size', fontsize=16)

    axs[0, 0].set_title(titles[0])
    axs[0, 0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))

    axs[0, 1].set_title(titles[1])
    axs[0, 1].imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))

    axs[0, 2].set_title(titles[2])
    axs[0, 2].imshow(cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))

    axs[1, 0].set_title(titles[3])
    axs[1, 0].imshow(cv2.cvtColor(images[3], cv2.COLOR_BGR2RGB))

    axs[1, 1].set_title(titles[4])
    axs[1, 1].imshow(cv2.cvtColor(images[4], cv2.COLOR_BGR2RGB))

    axs[1, 2].set_title(titles[5])
    axs[1, 2].imshow(cv2.cvtColor(images[5], cv2.COLOR_BGR2RGB))

def result_comparison(error_list, error_type):

    interpolation_methods = ["Nearest Neighbor",
                             "Bilinear", "Cubiclinear", "Lanczos"]

    print(f"\n........................{error_type} error calculation between the smalled image and the original image............................\n")
    print(f"{interpolation_methods[0]} Error Rate: {error_list[0]}")
    print(f"{interpolation_methods[1]} Error Rate: {error_list[1]}")
    print(f"{interpolation_methods[2]} Error Rate: {error_list[2]}")
    print(f"{interpolation_methods[3]} Error Rate: {error_list[3]}")

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]*1.1, round(y[i], 4), ha = 'center', Bbox = dict(facecolor='none', edgecolor='blue', alpha =.8))

def main():
    global image_scale
    images_list = {}

    # Read Image
    img, size, dimension = read_image("./dog.png")
    print(f"Image size is: {size}")
    images_list['Original Image'] = img

    # Change Image Size
    scale_percent = 20 # percent of original image size
    image_scale = scale_percent
    resized_img = image_change_scale(img, dimension, scale_percent)
    print(f"Smalled Image size is: {resized_img.shape}")
    images_list['Smalled Image'] = resized_img

    # Change image to original size using nearest neighbor interpolation
    nn_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_NEAREST)
    images_list['Nearest Neighbor Interpolation'] = nn_img

    # Change image to original size using bilinear interpolation
    bil_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_LINEAR)
    images_list['Bilinear Interpolation'] = bil_img

    # Change image to original size using cubiclinear interpolation (4*4 pixel neighborhood)
    cubic_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_CUBIC)
    images_list['CubicLinear Interpolation'] = cubic_img

    # Change image to original size using lanczos interpolation (8*8 pixel neighborhood)
    czos_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_LANCZOS4)
    images_list['Lanczos Interpolation'] = czos_img

    # error calculate between the smalled image and the original image
    error_list_psnr = []
    error_list_psnr.append(peak_signal_noise_ratio(nn_img, img))
    error_list_psnr.append(peak_signal_noise_ratio(bil_img, img))
    error_list_psnr.append(peak_signal_noise_ratio(cubic_img, img))
    error_list_psnr.append(peak_signal_noise_ratio(czos_img, img))

    error_list_ssim = []
    error_list_ssim.append(structural_similarity(nn_img, img, channel_axis=2))
    error_list_ssim.append(structural_similarity(bil_img, img, channel_axis=2))
    error_list_ssim.append(structural_similarity(cubic_img, img, channel_axis=2))
    error_list_ssim.append(structural_similarity(czos_img, img, channel_axis=2)) 

    # Show Result
    show_result(images_list)

    # Result Comparison
    result_comparison(error_list_psnr, "PSNR")
    result_comparison(error_list_ssim, "SSIM")

    interpolation_methods = ["Nearest Neighbor",
                             "Bilinear", "Cubiclinear", "Lanczos"]

    
    plt.figure()
    plt.bar(interpolation_methods, error_list_psnr, color=[
            'red', 'blue', 'purple', 'green'])
        
    plt.title("PSNR Values")
    addlabels(interpolation_methods, error_list_psnr)
    plt.ylim(0,100)
    plt.figure()
    plt.bar(interpolation_methods, error_list_ssim, color=[
            'red', 'blue', 'purple', 'green'])

    plt.title("SSIM Values")
    addlabels(interpolation_methods, error_list_ssim)
    plt.ylim(0, 1) 

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()