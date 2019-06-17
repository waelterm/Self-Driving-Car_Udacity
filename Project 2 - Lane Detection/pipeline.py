import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from camera_calibration import camera_calibration
from thresholds import thresholds
from perspective_transform import perspective_transform
from locate_lines import fit_polynomial
from search_prior import search_around_poly
from curvature import measure_curvature_real
from draw_image import draw_image
from numpy.linalg import inv
from curvature import lane_center_offset
import copy

LANE_PIXEL_WIDTH = 1105-217
LANE_PIXEL_LENGTH = 705 - 440
YM_PER_PIX = 30/LANE_PIXEL_LENGTH
XM_PER_PIX = 3.7/LANE_PIXEL_WIDTH

def poly_result(fit, y):
    """
    This function evaluates a polynomial of 2nd order
    :param fit: polyomial coefficients
    :param y: independent variable
    :return: dependent variable
    """
    x = fit[0] * y ** 2 + fit[1] * y + fit[2]
    return x

def compare_images(image_1, image_2, gray_1=False, gray_2=False, title_1='Original Image',
                   title_2='Modified Image'):
    """
    This function displays image_1 and image_2
    :param image_1:
    :param image_2:
    :param gray_1:
    :param gray_2:
    :param title_1:
    :param title_2:
    :return:
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title(title_1)
    if gray_1:
        ax1.imshow(image_1, cmap='gray')
    else:
        ax1.imshow(image_1)
    ax2.set_title(title_2)
    if gray_2:
        ax2.imshow(image_2, cmap='gray')
    else:
        ax2.imshow(image_2)
    plt.show()


def pipeline(write_images, show_images, show_images2, write_images2, show_images3, write_images3, show_images4, write_images4):
    """
    This is the pipeline used to develop the image annotation algoorithm used in main.py
    See main.py for mor information regarding the code
    :return: None
    """

    # Best guess initial polynomial
    left_fit = np.array([0, 0, 217])
    right_fit = np.array([0, 0, 950])

    # Calibration
    do_calibration = False
    if do_calibration:
        calibration_images = "camera_cal/"
        dimension_list = [[5, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9],
                          [6, 9], [6, 9], [6, 9], [6, 8], [6, 7], [6, 9], [6, 9], [6, 9], [6, 9]]
        mtx, dist = camera_calibration(calibration_images, dimension_list, display_corners=False)
    else:
        mtx = np.array(
            [[1.16067642 * 1000, 0.00000000, 6.65724920 * 100], [0.00000000, 1.15796966 * 1000, 3.88971790 * 100],
             [0.00000000, 0.00000000, 1.00000000]])
        dist = np.array([[-0.25427498, 0.04987807, -0.00043039, 0.00027334, -0.1336389]])

    # Load Test Images
    test_images = "test_images/"
    all_files = os.listdir(test_images)
    image_names = [file for file in all_files if file[-4:] == '.jpg']
    init = True # init flag decides if box algorithm or search near polynomial is used to find lane lines
    for image in image_names:
        print(test_images + image)
        img = cv2.imread(test_images + image)
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        line_image = thresholds(undistorted, s_thresh=(150, 255), sx_thresh=(30, 100))
        if write_images:
            cv2.imwrite(test_images + "filtered/" + image, line_image * 255);
        if show_images:
            compare_images(undistorted, line_image, False, True, 'Undistorted Image', 'Filtered Image')
        warped, M = perspective_transform(line_image)
        if show_images2:
            compare_images(line_image, warped, True, True, 'Filtered Image', 'Warped Image')
        if write_images2:
            cv2.imwrite(test_images + "warped/" + image, warped * 255);
        if init:
            out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, lefty, leftx, righty, rightx = fit_polynomial(warped)
            if show_images3:
                #print(out_img)
                compare_images(undistorted, out_img, True, True, 'Undistorted Image', 'Annotated Image')
            if write_images3:
                cv2.imwrite(test_images + "fitted/" + image, out_img);
        else:
            out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, lefty, leftx, righty, rightx = \
                search_around_poly(warped, left_fit, right_fit)
            if show_images3:
                #print(out_img)
                compare_images(undistorted, out_img, True, True, 'Undistorted Image', 'Annotated Image')
            if write_images3:
                cv2.imwrite(test_images + "fitted/" + image, out_img);
        left_curv, right_curv, left_fit_cr, right_fit_cr = measure_curvature_real(warped, copy.copy(left_fit), copy.copy(right_fit))
        #print("Left Fit: {}".format(left_fit))
        #print("Right Fit: {}".format(right_fit))

        print("Left curvature: {}".format(left_curv))
        print("Right curvature: {}".format(right_curv))
        right_base = poly_result(right_fit, 720) * XM_PER_PIX
        left_base = poly_result(left_fit, 720) * XM_PER_PIX
        print("Left Base: {}".format(left_base))
        print("Right Base: {}".format(right_base))
        offset = lane_center_offset(left_base, right_base)
        print("Lane Center Offset: {}".format(offset))
        # Checking that they have similar curvature
        # Checking that they are separated by approximately the right distance horizontally
        # Checking that they are roughly parallel ???????
        if (left_curv / right_curv) < 1.5 and (left_curv / right_curv) > 0.66 \
                and right_fit_cr[0] - left_fit_cr[0] > 3 and right_fit_cr[0] - left_fit_cr[0] < 4.5:
            valid = True
        output = draw_image(warped, left_fitx, right_fitx, ploty, img, inv(M), undistorted)
        if show_images4:
            compare_images(line_image, output, True, False, 'Line', 'Annotated Image')
        if write_images4:
            cv2.imwrite(test_images + "final/" + image, output);


if __name__ == '__main__':
    pipeline(write_images=True, write_images2=True, write_images3=True, show_images=False, show_images2=False,
             show_images3=False, show_images4= False, write_images4= True)
