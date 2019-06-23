import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def camera_calibration(calibration_images, dimension_list, display_corners):
    """
    Selects all images in calibration_images as well as the dimensions in each picture and calculates the calibration
    coefficients
    :param calibration_images: directory with calibration images
    :param dimension_list: list of the number of corners found in each image
    :param display_corners: boolean to display image with corners during calibration
    :return: Calibration coefficients
    """
    all_files = os.listdir(calibration_images)
    image_names = [file for file in all_files if file[-4:] == '.jpg']
    if len(image_names) != len(dimension_list):
        raise ValueError('Number of images ({}) and number of dimensions ({}) do not match.'.format(len(image_names),
                                                                                         len(dimension_list)))
    image_points = []
    object_points = []
    for i in range(len(image_names)):
        image = image_names[i]
        nx = dimension_list[i][1]
        ny = dimension_list[i][0]
        print(calibration_images + image)
        img = cv2.imread(calibration_images + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if not ret:
            print("Warning: Image {} was not used because not all corners could be found".format(image))
            continue
        else:
            objp = np.zeros((ny * nx, 3), np.float32)
            objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
            image_points.append(corners)
            object_points.append(objp)
        if display_corners:
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imshow("Display Corners", img)
            cv2.waitKey(0)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return mtx, dist

def undistort_images(calibration_images, mtx, dist, display_image = False, write_image = True):
    """
    Uses calibration coefficients to undistort images and display and or save them.
    :param calibration_images: directory of calibration images
    :param mtx: calibration coefficients
    :param dist: calibration coefficients
    :param display_image: boolean to display undistorted images
    :param write_image: boolean to save undistorted images
    :return: True
    """
    all_files = os.listdir(calibration_images)
    image_names = [file for file in all_files if file[-4:] == '.jpg']
    for image in image_names:
        print(image)
        img = cv2.imread(calibration_images + image)
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        if write_image:
            cv2.imwrite(calibration_images + "undistorted/" + image, undistorted);
        if display_image:
            cv2.imshow("Display Corners", undistorted)
            cv2.waitKey(0)
    return True



if __name__ == '__main__':
    calibration_images = "camera_cal/"
    dimension_list = [[5, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9],
                      [6, 9], [6, 9], [6, 8], [6, 7], [6, 9], [6, 9], [6, 9], [6, 9]]
    mtx, dist = camera_calibration(calibration_images, dimension_list, display_corners = False)
    undistort_images(calibration_images, mtx, dist)



