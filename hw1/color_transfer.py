import cv2
import sys
import numpy as np


def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR, dtype=np.float32)

    img_RGB = img_BGR.copy()

    img_RGB = img_RGB[:, :, ::-1]

    return img_RGB


def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB, dtype=np.float32)

    img_BGR = img_RGB.copy()

    img_BGR = img_BGR[:, :, ::-1]

    return img_BGR


def convert_color_space_RGB_to_Lab(img_RGB, img_tar):
    '''
    convert image color space RGB to Lab
    '''
    # RGB -> LMS
    img_LMS = np.zeros_like(img_RGB, dtype=np.float32)  # Initialize lms img
    img_LMS_tar = np.zeros_like(img_tar, dtype=np.float32)  # Initialize lms img

    img_LMS[:, :, 0] = np.log10(0.3811 * img_RGB[:, :, 0] + 0.5783 * img_RGB[:, :, 1] + 0.0402 * img_RGB[:, :, 2])  # Equation 4 + 5 on source
    img_LMS[:, :, 1] = np.log10(0.1967 * img_RGB[:, :, 0] + 0.7244 * img_RGB[:, :, 1] + 0.0782 * img_RGB[:, :, 2])
    img_LMS[:, :, 2] = np.log10(0.0241 * img_RGB[:, :, 0] + 0.1288 * img_RGB[:, :, 1] + 0.8444 * img_RGB[:, :, 2])

    img_LMS_tar[:, :, 0] = np.log10(0.3811 * img_tar[:, :, 0] + 0.5783 * img_tar[:, :, 1] + 0.0402 * img_tar[:, :, 2])  # Equation 4 + 5 on target
    img_LMS_tar[:, :, 1] = np.log10(0.1967 * img_tar[:, :, 0] + 0.7244 * img_tar[:, :, 1] + 0.0782 * img_tar[:, :, 2])
    img_LMS_tar[:, :, 2] = np.log10(0.0241 * img_tar[:, :, 0] + 0.1288 * img_tar[:, :, 1] + 0.8444 * img_tar[:, :, 2])

    # LMS -> Lab
    img_Lab_src = np.zeros_like(img_RGB, dtype=np.float32)  # Initialize lab img
    img_Lab_tar = np.zeros_like(img_tar, dtype=np.float32)  # Initialize lab img

    img_Lab_src[:, :, 0] = 0.5773 * img_LMS[:, :, 0] + 0.5773 * img_LMS[:, :, 1] + 0.5773 * img_LMS[:, :,2]  # Equation 6 on source
    img_Lab_src[:, :, 1] = 0.4082 * img_LMS[:, :, 0] + 0.4082 * img_LMS[:, :, 1] + -0.8164 * img_LMS[:, :, 2]
    img_Lab_src[:, :, 2] = 0.7071 * img_LMS[:, :, 0] + -0.7071 * img_LMS[:, :, 1] + 0.0 * img_LMS[:, :, 2]

    img_Lab_tar[:, :, 0] = 0.5773 * img_LMS_tar[:, :, 0] + 0.5773 * img_LMS_tar[:, :, 1] + 0.5773 * img_LMS_tar[:, :,2]  # Equation 6 on target
    img_Lab_tar[:, :, 1] = 0.4082 * img_LMS_tar[:, :, 0] + 0.4082 * img_LMS_tar[:, :, 1] + -0.8164 * img_LMS_tar[:, :, 2]
    img_Lab_tar[:, :, 2] = 0.7071 * img_LMS_tar[:, :, 0] + -0.7071 * img_LMS_tar[:, :, 1] + 0.0 * img_LMS_tar[:, :, 2]

    # Color transfer
    l_mean_src = np.mean(img_Lab_src[:, :, 0])  # Compute mean of source
    a_mean_src = np.mean(img_Lab_src[:, :, 1])
    b_mean_src = np.mean(img_Lab_src[:, :, 2])

    l_std_src = np.std(img_Lab_src[:, :, 0])  # Compute standard deviation of source
    a_std_src = np.std(img_Lab_src[:, :, 1])
    b_std_src = np.std(img_Lab_src[:, :, 2])

    l_mean_tar = np.mean(img_Lab_tar[:, :, 0])  # Compute mean of target
    a_mean_tar = np.mean(img_Lab_tar[:, :, 1])
    b_mean_tar = np.mean(img_Lab_tar[:, :, 2])

    l_std_tar = np.std(img_Lab_tar[:, :, 0])  # Compute standard deviation of source
    a_std_tar = np.std(img_Lab_tar[:, :, 1])
    b_std_tar = np.std(img_Lab_tar[:, :, 2])

    img_Lab_src[:, :, 0] -= l_mean_src  # Equation 10
    img_Lab_src[:, :, 1] -= a_mean_src
    img_Lab_src[:, :, 2] -= b_mean_src

    img_Lab_src[:, :, 0] = (l_std_tar / l_std_src) * img_Lab_src[:, :, 0]  # Equation 11
    img_Lab_src[:, :, 1] = (a_std_tar / a_std_src) * img_Lab_src[:, :, 1]
    img_Lab_src[:, :, 2] = (b_std_tar / b_std_src) * img_Lab_src[:, :, 2]

    img_Lab_src[:, :, 0] += l_mean_tar  # Add averages for source
    img_Lab_src[:, :, 1] += a_mean_tar
    img_Lab_src[:, :, 2] += b_mean_tar

    result_RGB = convert_color_space_Lab_to_RGB(img_Lab_src)

    return result_RGB


def convert_color_space_Lab_to_RGB(img_Lab):
    #Equation 8
    img_LMS = np.zeros_like(img_Lab, dtype=np.float32)

    matrix_one = np.array([[1, 1, 1],
                            [1, 1, -1],
                            [1, -2, 0]])

    matrix_two = np.array([[np.sqrt(3) / 3, 0, 0],
                             [0, np.sqrt(6) / 6, 0],
                             [0, 0, np.sqrt(2) / 2]])

    lab_to_lms_mat = np.matmul(matrix_one, matrix_two)

    img_LMS = matrixmul(lab_to_lms_mat, img_Lab)

    img_LMS = np.power(10, img_LMS)

    img_RGB = np.zeros_like(img_Lab, dtype=np.float32)

    # Equation 9
    img_RGB_matrix = np.array([[4.4679, -3.5873, 0.1193],
                               [-1.2186, 2.3809, -0.1624],
                               [0.0497, -0.2439, 1.2045]])

    img_RGB = matrixmul(img_RGB_matrix, img_LMS)

    return img_RGB


def matrixmul(matrix, img):
    # Apply matrix to img

    result_img = np.zeros_like(img, dtype=np.float32)
    result_img[:, :, 0] = img[:, :, 0] * matrix[0][0] + img[:, :, 1] * matrix[0][1] + img[:, :, 2] * matrix[0][2]
    result_img[:, :, 1] = img[:, :, 0] * matrix[1][0] + img[:, :, 1] * matrix[1][1] + img[:, :, 2] * matrix[1][2]
    result_img[:, :, 2] = img[:, :, 0] * matrix[2][0] + img[:, :, 1] * matrix[2][1] + img[:, :, 2] * matrix[2][2]

    return result_img


def convert_color_space_RGB_to_CIECAM97s(img_RGB, img_tar):


    img_CIECAM97s_src = np.zeros_like(img_RGB, dtype=np.float32)
    img_CIECAM97s_tar = np.zeros_like(img_RGB, dtype=np.float32)
    img_LMS = np.zeros_like(img_RGB, dtype=np.float32)  # Initialize lms img
    img_LMS_tar = np.zeros_like(img_tar, dtype=np.float32)  # Initialize lms img

    img_LMS[:, :, 0] = np.log10(0.3811 * img_RGB[:, :, 0] + 0.5783 * img_RGB[:, :, 1] + 0.0402 * img_RGB[:, :, 2])  # Equation 4 + 5 on source
    img_LMS[:, :, 1] = np.log10(0.1967 * img_RGB[:, :, 0] + 0.7244 * img_RGB[:, :, 1] + 0.0782 * img_RGB[:, :, 2])
    img_LMS[:, :, 2] = np.log10(0.0241 * img_RGB[:, :, 0] + 0.1288 * img_RGB[:, :, 1] + 0.8444 * img_RGB[:, :, 2])

    img_LMS_tar[:, :, 0] = np.log10(0.3811 * img_tar[:, :, 0] + 0.5783 * img_tar[:, :, 1] + 0.0402 * img_tar[:, :, 2])  # Equation 4 + 5 on target
    img_LMS_tar[:, :, 1] = np.log10(0.1967 * img_tar[:, :, 0] + 0.7244 * img_tar[:, :, 1] + 0.0782 * img_tar[:, :, 2])
    img_LMS_tar[:, :, 2] = np.log10(0.0241 * img_tar[:, :, 0] + 0.1288 * img_tar[:, :, 1] + 0.8444 * img_tar[:, :, 2])

    mat = np.array([[2.00, 1.00, 0.05],
                   [1.00, -1.09, 0.09],
                   [0.11, 0.11, -0.22]])

    img_CIECAM97s_src = matrixmul(mat, img_LMS)
    img_CIECAM97s_tar = matrixmul(mat, img_LMS_tar)

    mean_src_0 = np.mean(img_CIECAM97s_src[:, :, 0])  # Compute mean of source
    mean_src_1 = np.mean(img_CIECAM97s_src[:, :, 1])
    mean_src_2 = np.mean(img_CIECAM97s_src[:, :, 2])

    std_src_0 = np.std(img_CIECAM97s_src[:, :, 0])  # Compute standard deviation of source
    std_src_1 = np.std(img_CIECAM97s_src[:, :, 1])
    std_src_2 = np.std(img_CIECAM97s_src[:, :, 2])

    mean_tar_0 = np.mean(img_CIECAM97s_tar[:, :, 0])  # Compute mean of target
    mean_tar_1 = np.mean(img_CIECAM97s_tar[:, :, 1])
    mean_tar_2 = np.mean(img_CIECAM97s_tar[:, :, 2])

    std_tar_0 = np.std(img_CIECAM97s_tar[:, :, 0])  # Compute standard deviation of source
    std_tar_1 = np.std(img_CIECAM97s_tar[:, :, 1])
    std_tar_2 = np.std(img_CIECAM97s_tar[:, :, 2])

    img_CIECAM97s_src[:, :, 0] -= mean_src_0  # Equation 10
    img_CIECAM97s_src[:, :, 1] -= mean_src_1
    img_CIECAM97s_src[:, :, 2] -= mean_src_2

    img_CIECAM97s_src[:, :, 0] = (std_tar_0 / std_src_0) * img_CIECAM97s_src[:, :, 0]  # Equation 11
    img_CIECAM97s_src[:, :, 1] = (std_tar_1 / std_src_1) * img_CIECAM97s_src[:, :, 1]
    img_CIECAM97s_src[:, :, 2] = (std_tar_2 / std_src_2) * img_CIECAM97s_src[:, :, 2]

    img_CIECAM97s_src[:, :, 0] += mean_tar_0  # Add averages for source
    img_CIECAM97s_src[:, :, 1] += mean_tar_1
    img_CIECAM97s_src[:, :, 2] += mean_tar_2

    result_CIECAM97s = convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s_src)

    return result_CIECAM97s


def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    img_LMS = np.zeros_like(img_CIECAM97s, dtype=np.float32)

    mat = np.array([[2.00, 1.00, 0.05],
                    [1.00, -1.09, 0.09],
                    [0.11, 0.11, -0.22]])

    mat = np.linalg.inv(mat)

    img_LMS = matrixmul(mat, img_CIECAM97s)

    img_LMS = np.power(10, img_LMS)

    img_RGB = np.zeros_like(img_CIECAM97s, dtype=np.float32)

    img_RGB_matrix = np.array([[4.4679, -3.5873, 0.1193],
                               [-1.2186, 2.3809, -0.1624],
                               [0.0497, -0.2439, 1.2045]])

    img_RGB = matrixmul(img_RGB_matrix, img_LMS)

    return img_RGB


def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')

    src = convert_color_space_BGR_to_RGB(img_RGB_source)
    target = convert_color_space_BGR_to_RGB(img_RGB_target)

    src = convert_color_space_RGB_to_Lab(src, target)

    img_final = convert_color_space_RGB_to_BGR(src).clip(0.0, 255.0)

    return img_final


def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')

    img_RGB_src = convert_color_space_BGR_to_RGB(img_RGB_source).astype(np.float32)
    img_RGB_tar = convert_color_space_BGR_to_RGB(img_RGB_target).astype(np.float32)

    mean_src_0 = np.mean(img_RGB_src[:, :, 0])  # Compute mean of source
    mean_src_1 = np.mean(img_RGB_src[:, :, 1])
    mean_src_2 = np.mean(img_RGB_src[:, :, 2])

    std_src_0 = np.std(img_RGB_src[:, :, 0])  # Compute standard deviation of source
    std_src_1 = np.std(img_RGB_src[:, :, 1])
    std_src_2 = np.std(img_RGB_src[:, :, 2])

    mean_tar_0 = np.mean(img_RGB_tar[:, :, 0])  # Compute mean of target
    mean_tar_1 = np.mean(img_RGB_tar[:, :, 1])
    mean_tar_2 = np.mean(img_RGB_tar[:, :, 2])

    std_tar_0 = np.std(img_RGB_tar[:, :, 0])  # Compute standard deviation of source
    std_tar_1 = np.std(img_RGB_tar[:, :, 1])
    std_tar_2 = np.std(img_RGB_tar[:, :, 2])

    img_RGB_src[:, :, 0] -= mean_src_0  # Equation 10
    img_RGB_src[:, :, 1] -= mean_src_1
    img_RGB_src[:, :, 2] -= mean_src_2

    img_RGB_src[:, :, 0] = (std_tar_0 / std_src_0) * img_RGB_src[:, :, 0]  # Equation 11
    img_RGB_src[:, :, 1] = (std_tar_1 / std_src_1) * img_RGB_src[:, :, 1]
    img_RGB_src[:, :, 2] = (std_tar_2 / std_src_2) * img_RGB_src[:, :, 2]

    img_RGB_src[:, :, 0] += mean_tar_0  # Add averages for source
    img_RGB_src[:, :, 1] += mean_tar_1
    img_RGB_src[:, :, 2] += mean_tar_2

    img_RGB_src = convert_color_space_RGB_to_BGR(img_RGB_src).clip(0.0, 255.0)

    return img_RGB_src


def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    src = convert_color_space_BGR_to_RGB(img_RGB_source)
    target = convert_color_space_BGR_to_RGB(img_RGB_target)

    src = convert_color_space_RGB_to_CIECAM97s(src, target)

    img_final = convert_color_space_RGB_to_BGR(src).clip(0.0, 255.0)

    return img_final


def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new


if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2020, HW1: color transfer')
    print('==================================================')

    img_RGB_source = sys.argv[1]
    img_RGB_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5]

    img_RGB_source = cv2.imread(img_RGB_source)     #Read source img
    img_RGB_target = cv2.imread(img_RGB_target)     #Read target img

    img_RGB_new_Lab = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')       #Apply color transfer to img in Lab space
    cv2.imwrite(path_file_image_result_in_Lab, img_RGB_new_Lab)

    img_RGB_new_RGB = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')       #Apply color transfer to img in RGB space
    cv2.imwrite(path_file_image_result_in_RGB, img_RGB_new_RGB)

    img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')       #Apply color transfer to img in CIECAM97s space
    cv2.imwrite(path_file_image_result_in_CIECAM97s, img_RGB_new_CIECAM97s)

# def gaussian_pyramid(img, num_levels):
#     lower = img.copy()
#     gaussian_pyr = [lower]
#     for i in range(num_levels):
#         lower = cv2.pyrDown(lower)
#         gaussian_pyr.append(np.float32(lower))
#     return gaussian_pyr
#
# def laplacian_pyramid(gaussian_pyr):
#     laplacian_top = gaussian_pyr[-1]
#     num_levels = len(gaussian_pyr) - 1
#
#     laplacian_pyr = [laplacian_top]
#     for i in range(num_levels, 0, -1):
#         size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
#         gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
#         laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
#         laplacian_pyr.append(laplacian)
#     return laplacian_pyr