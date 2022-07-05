import cv2
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor, ceil
from PIL import Image
import math

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

def image_change_scale(img, dimension, scale=100, interpolation=cv2.INTER_CUBIC):
    scale /= 100
    new_dimension = (int(dimension[1]*scale), int(dimension[0]*scale))
    resized_img = cv2.resize(img, new_dimension, interpolation=interpolation)
    return resized_img
    
def nearest_interpolation(image, dimension):

    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    enlarge_time = int(
        sqrt((dimension[0] * dimension[1]) / (image.shape[0]*image.shape[1])))
    
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            row = floor(i / enlarge_time)
            column = floor(j / enlarge_time)

            new_image[i, j] = image[row, column]

    return new_image

def bilinear_interpolation(image, dimension):

    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width)/(dimension[1])
    scale_y = (height)/(dimension[0])

    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    for k in range(3):
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                x = (j+0.5) * (scale_x) - 0.5
                y = (i+0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                # Prevent crossing
                x_int = min(x_int, width-2)
                y_int = min(y_int, height-2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = image[y_int, x_int, k]
                b = image[y_int, x_int+1, k]
                c = image[y_int+1, x_int, k]
                d = image[y_int+1, x_int+1, k]

                pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                    (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff

                new_image[i, j, k] = pixel.astype(np.uint8)

    return new_image

def W(x):
    '''Weight function that return weight for each distance point
    Parameters:
    x (float): Distance from destination point

    Returns:
    float: Weight
    '''
    a = -0.5
    pos_x = abs(x)
    if -1 <= abs(x) <= 1:
        return ((a+2)*(pos_x**3)) - ((a+3)*(pos_x**2)) + 1
    elif 1 < abs(x) < 2 or -2 < x < -1:
        return ((a * (pos_x**3)) - (5*a*(pos_x**2)) + (8 * a * pos_x) - 4*a)
    else:
        return 0

def bicubic_interpolation(img, dimension):

    nrows = dimension[0]
    ncols = dimension[1]

    output = np.zeros((nrows, ncols, img.shape[2]), np.uint8)
    for c in range(img.shape[2]):
        for i in range(nrows):
            for j in range(ncols):
                xm = (i + 0.5) * (img.shape[0]/dimension[0]) - 0.5
                ym = (j + 0.5) * (img.shape[1]/dimension[1]) - 0.5

                xi = floor(xm)
                yi = floor(ym)

                u = xm - xi
                v = ym - yi

                # -------------- Using this make ignore some points and increase the value of black in image border
                # x = [(xi - 1), xi, (xi + 1), (xi + 2)]
                # y = [(yi - 1), yi, (yi + 1), (yi + 2)]
                # if ((x[0] >= 0) and (x[3] < img.shape[1]) and (y[0] >= 0) and (y[3] < img.shape[0])):
                #     dist_x0 = W(x[0] - xm)
                #     dist_x1 = W(x[1] - xm)
                #     dist_x2 = W(x[2] - xm)
                #     dist_x3 = W(x[3] - xm)
                #     dist_y0 = W(y[0] - ym)
                #     dist_y1 = W(y[1] - ym)
                #     dist_y2 = W(y[2] - ym)
                #     dist_y3 = W(y[3] - ym)

                #     out = (img[x[0], y[0], c] * (dist_x0 * dist_y0) +
                #            img[x[0], y[1], c] * (dist_x0 * dist_y1) +
                #            img[x[0], y[2], c] * (dist_x0 * dist_y2) +
                #            img[x[0], y[3], c] * (dist_x0 * dist_y3) +
                #            img[x[1], y[0], c] * (dist_x1 * dist_y0) +
                #            img[x[1], y[1], c] * (dist_x1 * dist_y1) +
                #            img[x[1], y[2], c] * (dist_x1 * dist_y2) +
                #            img[x[1], y[3], c] * (dist_x1 * dist_y3) +
                #            img[x[2], y[0], c] * (dist_x2 * dist_y0) +
                #            img[x[2], y[1], c] * (dist_x2 * dist_y1) +
                #            img[x[2], y[2], c] * (dist_x2 * dist_y2) +
                #            img[x[2], y[3], c] * (dist_x2 * dist_y3) +
                #            img[x[3], y[0], c] * (dist_x3 * dist_y0) +
                #            img[x[3], y[1], c] * (dist_x3 * dist_y1) +
                #            img[x[3], y[2], c] * (dist_x3 * dist_y2) +
                #            img[x[3], y[3], c] * (dist_x3 * dist_y3))

                #     output[i, j, c] = np.clip(out, 0, 255)
                # ---------------------------

                out = 0
                for n in range(-1, 3):
                    for m in range(-1, 3):
                        if ((xi + n < 0) or (xi + n >= img.shape[1]) or (yi + m < 0) or (yi + m >= img.shape[0])):
                            continue

                        out += (img[xi+n, yi+m, c] * (W(u - n) * W(v - m)))

                output[i, j, c] = np.clip(out, 0, 255)

    return output

def psnr_calculator(img1, img2):

    mse = np.mean( (img1 - img2) ** 2 )

    if mse == 0:
        return 100

    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim_calculator(img1, img2):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []

            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    
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

    interpolation_methods = ["Nearest Opencv", "Nearest Neighbor",
                             "Bilinear Opencv", "Bilinear", "Cubiclinear Opencv", "Cubiclinear", "Lanczos"]

    print(f"\n........................{error_type} error calculation between the smalled image and the original image............................\n")
    print(f"{interpolation_methods[0]} Error Rate: {error_list[0]}")
    print(f"{interpolation_methods[1]} Error Rate: {error_list[1]}")
    print(f"{interpolation_methods[2]} Error Rate: {error_list[2]}")
    print(f"{interpolation_methods[3]} Error Rate: {error_list[3]}")
    print(f"{interpolation_methods[4]} Error Rate: {error_list[4]}")
    print(f"{interpolation_methods[5]} Error Rate: {error_list[5]}")
    print(f"{interpolation_methods[6]} Error Rate: {error_list[6]}\n")

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]*1.1, round(y[i], 4), ha = 'center', Bbox = dict(facecolor='none', edgecolor='blue', alpha =.8))

def main():
    global image_scale
    images_list = {}

    # Read Image
    img, size, dimension = read_image("./image.png")
    print(f"Image size is: {size}")
    images_list['Original Image'] = img

    # Change Image Size
    scale_percent = 25  # percent of original image size
    image_scale = scale_percent
    resized_img = image_change_scale(img, dimension, scale_percent)
    print(f"Smalled Image size is: {resized_img.shape}")
    images_list['Smalled Image'] = resized_img

    # Change image to original size using nearest neighbor interpolation
    nn_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_NEAREST)
    images_list['Nearest Neighbor Interpolation'] = nn_img
    
    nn_img_algo = nearest_interpolation(resized_img, dimension)
    nn_img_algo = np.array(Image.fromarray(nn_img_algo.astype('uint8')).convert('RGB'))

    # Change image to original size using bilinear interpolation
    bil_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_LINEAR)
    images_list['Bilinear Interpolation'] = bil_img

    bil_img_algo = bilinear_interpolation(resized_img, dimension)
    bil_img_algo = np.array(Image.fromarray(bil_img_algo.astype('uint8')).convert('RGB'))

    # Change image to original size using cubiclinear interpolation (4*4 pixel neighborhood)
    cubic_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_CUBIC)
    images_list['CubicLinear Interpolation'] = cubic_img

    # cubic_img_algo = BiCubic_interpolation(
    #     resized_img, dimension[0], dimension[1])
    cubic_img_algo = bicubic_interpolation(resized_img, dimension)
    cubic_img_algo = np.array(Image.fromarray(
        cubic_img_algo.astype('uint8')).convert('RGB'))

    # Change image to original size using lanczos interpolation (8*8 pixel neighborhood)
    czos_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_LANCZOS4)
    images_list['Lanczos Interpolation'] = czos_img

    # error calculate between the smalled image and the original image
    error_list_psnr = []
    error_list_psnr.append(psnr_calculator(nn_img, img))
    error_list_psnr.append(psnr_calculator(nn_img_algo, img))
    error_list_psnr.append(psnr_calculator(bil_img, img))
    error_list_psnr.append(psnr_calculator(bil_img_algo, img))
    error_list_psnr.append(psnr_calculator(cubic_img, img))
    error_list_psnr.append(psnr_calculator(cubic_img_algo, img))
    error_list_psnr.append(psnr_calculator(czos_img, img))

    error_list_ssim = []
    error_list_ssim.append(ssim_calculator(nn_img, img))
    error_list_ssim.append(ssim_calculator(nn_img_algo, img))
    error_list_ssim.append(ssim_calculator(bil_img, img))
    error_list_ssim.append(ssim_calculator(bil_img_algo, img))
    error_list_ssim.append(ssim_calculator(cubic_img, img))
    error_list_ssim.append(ssim_calculator(cubic_img_algo, img))
    error_list_ssim.append(ssim_calculator(czos_img, img)) 

    # Show Result
    show_result(images_list)

    # Result Comparison
    result_comparison(error_list_psnr, "PSNR")
    result_comparison(error_list_ssim, "SSIM")

    interpolation_methods = ["Nearest Opencv", "Nearest Neighbor",
                             "Bilinear Opencv", "Bilinear", "Cubiclinear Opencv", "Cubiclinear", "Lanczos"]

    
    plt.figure()
    plt.bar(interpolation_methods, error_list_psnr, color=[
            'red', 'blue', 'purple', 'green', 'fuchsia', 'yellow', 'black'])
        
    plt.title("PSNR Values")
    addlabels(interpolation_methods, error_list_psnr)
    plt.ylim(0,100)
    plt.figure()
    plt.bar(interpolation_methods, error_list_ssim, color=[
            'red', 'blue', 'purple', 'green', 'fuchsia', 'yellow', 'black'])

    plt.title("SSIM Values")
    addlabels(interpolation_methods, error_list_ssim)
    plt.ylim(0, 1) 

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()