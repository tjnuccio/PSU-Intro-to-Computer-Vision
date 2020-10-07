"""Author: TJ Nuccio
Image Blending Project: The purpose of this project is to take two images of similar shape
and color scheme, 'spline them', blending them together to produce an image that looks more
natural than just copying pixels from half of one and half of the other. The method uses
pyramid blending described in the paper: Burt, P. J. and Adelson, E. H. (1983b). A multiresolution spline with applications to image mosaics. ACM Transactions on Graphics, 2(4):217–236.
This program supports multiple filter options whereas Burt and Adelson suggested the Gaussian.
Usage: python3 main.py img_a img_b filter_choice

Filter Choice:

gaussian: Gaussian kernal convolved over image using OpenCv filter2D function
box: Box kernal convolved over image using OpenCv filter2D function
slow_gaussian: gaussian: Gaussian kernal convolved over image using my own convolution function
slow_box: Box kernal convolved over image using my own convolution function
"""

import sys
import cv2
import numpy as np

kernel_box = 1 / 25 * np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1]], dtype=np.float32)

kernel_gaussian = 1 / 256 * np.array([[1, 4, 6, 4, 1],
                                      [4, 16, 24, 16, 4],
                                      [6, 24, 36, 24, 6],
                                      [4, 16, 24, 16, 4],
                                      [1, 4, 6, 4, 1]], dtype=np.float32)


def blur_image(image, option):
    if option == "gaussian":                            # Gaussian filter with OpenCV
        image = cv2.filter2D(image, -1, kernel_gaussian)
    elif option == "noblur":                            # No filter
        pass
    elif option == "box":                                # Box filter with OpenCV
        image = cv2.filter2D(image, -1, kernel_box)
    elif option == "slow_gaussian":                     # My implementation of Gaussian filter + Convolution
        image = my_conv(image, kernel_gaussian)
    elif option == "slow_box":                          # My implementation of Box fitler + Convolution
        image = my_conv(image, kernel_box)

    return image


def my_conv(img_tmp, kernel):
    # Hard-coded convolution implementation
    def conv_patch(kernel, patch):
        sum = 0
        for intY in range(5):
            for intX in range(5):
                sum = sum + kernel[intY, intX] * patch[intY, intX]
        return sum

    (h, w, c) = img_tmp.shape
    img_dst = np.zeros((h, w, c))
    for x in range(c):
        for a in range(h):
            for b in range(w):
                if b + 4 < w and a + 4 < h:
                    img_dst[a + 2, b + 2, x] = conv_patch(kernel, img_tmp[a:a + 5, b:b + 5, x])

    return img_dst


def pyrdown(img_tmp, option):
    (height, width, c) = img_tmp.shape

    nextlevel = np.zeros(shape=(int(height / 2), int(width / 2), c))

    img_tmp = blur_image(img_tmp, option)

    for y in range(int(height / 2)):
        for x in range(int(width / 2)):
            nextlevel[y, x] = img_tmp[2 * y, 2 * x]

    return nextlevel


def pyrup(img_tmp, option):
    (height, width, c) = img_tmp.shape

    nextlevel = np.zeros(shape=(int(height * 2), int(width * 2), c))

    for y in range(int(height * 2)):
        for x in range(int(width * 2)):
            nextlevel[y, x] = img_tmp[int(y / 2), int(x / 2)]

    nextlevel = blur_image(nextlevel, option)  # blur image after upsample

    return nextlevel

def reconstruct(laplacian_pyr):                 #Newer function
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst

if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2020, Project: Image Blending')
    print('==================================================')

    print("Usage: python3 file.py img1 img2 filter_option img_result_name")
    print("Filter Options: gaussian | box | slow_gaussian | slow_box | noblur")

    img_a = sys.argv[1]
    img_a = cv2.imread(img_a)  # Read source img

    img_b = sys.argv[2]
    img_b = cv2.imread(img_b)  # Read source img

    img_a = cv2.resize(img_a, (512, 512))
    img_b = cv2.resize(img_b, (512, 512))

    mask = np.zeros((512, 512, 3), dtype='float32')
    mask[:512, 256:512, :] = (1, 1, 1)

    opt = sys.argv[3]
    print("Filter Option: ", opt)

    result = sys.argv[4]

    rows, cols, dpt = img_a.shape
    level = np.math.ceil(np.math.log(min(rows, cols) / 16, 2))
    level = int(level)

    # generate Gaussian pyramid for A
    G = img_a.copy()
    gpA = [G]
    for i in range(level):
        G = pyrdown(G, opt)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = img_b.copy()
    gpB = [G]
    for i in range(level):
        G = pyrdown(G, opt)
        gpB.append(G)

    # generate Gaussian pyramid for mask
    G = mask.copy()
    gpM = [G]
    for i in range(level):
        G = pyrdown(G, opt)
        gpM.append(G)

    maskPyr = []  # Reverses the gaussian pyramid for the mask to be correct side up
    for i in range(level - 1, 0, -1):
        maskPyr.append(gpM[i])

    maskPyr.append(mask)

    # generate Laplacian Pyramid for A
    lpA = [gpA[level - 1]]
    for i in range(level - 1, 0, -1):
        GE = pyrup(gpA[i], opt)
        L = np.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[level - 1]]
    for i in range(level - 1, 0, -1):
        GE = pyrup(gpB[i], opt)
        L = np.subtract(gpB[i - 1], GE)
        lpB.append(L)

    #Blend                  #This is where the original begins
    LS = []
    for la, lb, mask in zip(lpA, lpB, maskPyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, level):
        ls_ = pyrup(ls_, opt)
        ls_ = cv2.add(ls_, LS[i])

    print("Blend File Created: " + result + ".png")
    cv2.imwrite(result + ".png", ls_)
