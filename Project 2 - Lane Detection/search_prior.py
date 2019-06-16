import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt



def fit_poly(img_shape, left_fit, right_fit):
    """
    This function discretizes a polynomial
    """
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return left_fitx, right_fitx, ploty, left_fit, right_fit


def poly_result(fit, y):
    """
    This function evaluates a polynomial of 2nd order
    :param fit: polyomial coefficients
    :param y: independent variable
    :return: dependent variable
    """
    x = fit[0] * y ** 2 + fit[1] * y + fit[2]
    return x


def search_around_poly(binary_warped, left_fit, right_fit):
    margin = 100

    # Find non-zero items in binary warped image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Filter those near the previously detected lane lines, and add them to a list
    mask_left = ((nonzerox + 100) >= poly_result(left_fit, nonzeroy)) \
                & ((nonzerox - 100) < poly_result(left_fit, nonzeroy))
    mask_right = ((nonzerox + 100) >= poly_result(right_fit, nonzeroy)) \
                 & ((nonzerox - 100) < poly_result(right_fit, nonzeroy))

    left_lane_inds = mask_left.nonzero()[0]
    right_lane_inds = mask_right.nonzero()[0]

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    try:
        left_fitx, right_fitx, ploty, new_left_fit, new_right_fit = fit_poly(binary_warped.shape, left_fit, right_fit)
    except TypeError:
        return None, None, None, None, None, None, None, None, None, None
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Use found points to calculate new polynomials
    left_fit = np.polyfit(x=lefty, y=leftx, deg=2)
    right_fit = np.polyfit(x=righty, y=rightx, deg=2)
    print(left_fit)
    print(right_fit)
    try:
        left_fitx, right_fitx, ploty, new_left_fit, new_right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty, left_fit, right_fit)
    except TypeError:
        return None, None, None, None, None, None, None, None, None, None
    return out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, lefty, leftx, righty, rightx

if __name__ == '__main__':
    # Load our image - this should be a new frame since last time!
    binary_warped = mpimg.imread('warped_example.jpg')

    # Polynomial fit values from the previous frame
    # Make sure to grab the actual values from the previous step in your project!
    left_fit = np.array([2.13935315e-04, -3.77507980e-01, 4.76902175e+02])
    right_fit = np.array([4.17622148e-04, -4.93848953e-01, 1.11806170e+03])
    result = search_around_poly(binary_warped, left_fit, right_fit)