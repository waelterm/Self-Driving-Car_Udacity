import numpy as np
import cv2

def find_polygon(straight_line_image):
    """
    This function was used to find the polygon fitting on the straight road image.
    To see, run main.
    :param straight_line_image: image of a straight line
    :return: None
    """
    pts = np.array([[217, 705], [606, 440], [676, 440], [1105, 705]], np.int32)
    # I have lowered the upper corners of the polynomial from the originally found corners above to avoid having parts
    # of the lane cut off after the perspective transform
    pts = np.array([[217, 705], [606-int(39/2), 440+13], [676+int(43/2), 440+13], [1105, 705]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(straight_line_image, [pts], True, (0, 0, 255), thickness = 2)
    cv2.imshow("Show Polygon", straight_line_image)
    cv2.waitKey(0)

def perspective_transform(undistort):
    """
    Transforms the polygon found with find_polygon, to a rectangular section
    :param undistorted image
    :return: warped image and transformation matrix
    """
    # I have lowered the upper corners of the polynomial from the originally found corners above to avoid having parts
    # of the lane cut off after the perspective transform
    #src = np.float32([[217, 705],[606, 440] , [676, 440], [1105, 705]])
    src = np.float32([[217, 705], [606-int(39/2), 440+13], [676+int(43/2), 440+13], [1105, 705]])
    dst = np.float32([[250, 720], [250 , 0], [1280-250, 0], [1290 - 250, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undistort, M, (undistort.shape[1], undistort.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, M

if __name__ == '__main__':
    write_images = True
    mtx = np.array(
        [[1.16067642 * 1000, 0.00000000, 6.65724920 * 100], [0.00000000, 1.15796966 * 1000, 3.88971790 * 100],
         [0.00000000, 0.00000000, 1.00000000]])
    dist = np.array([[-0.25427498, 0.04987807, -0.00043039, 0.00027334, -0.1336389]])
    for image in ['test_images/straight_lines1.jpg', 'test_images/straight_lines2.jpg']:
        img = cv2.imread(image)
        #print(img.shape)
        undistorted_straight_line_image = cv2.undistort(img, mtx, dist, None, mtx)
        find_polygon(undistorted_straight_line_image)
        if write_images:
            cv2.imwrite("test_images/" + "polygon/" + image[-19:], undistorted_straight_line_image);