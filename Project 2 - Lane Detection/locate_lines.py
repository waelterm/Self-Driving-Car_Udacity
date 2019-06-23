import numpy as np
import cv2
import matplotlib.pyplot as plt

def hist(warped):
    """
    Creates a histogram to find location of initial box to look for lane lines
    :param warped:
    :return:
    """
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return midpoint, leftx_base, rightx_base

def locate_lines(warped):
    """
    Locates Lane lines using the moving boxes algorithm
    :param warped: warped image
    :return: annotated image and lane line information
    """
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    out_img = np.dstack((warped, warped, warped))
    nwindows = 10
    # Set the width of the windows +/- margin
    left_margin = 100
    right_margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = warped.nonzero()

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    prev_leftx = leftx_current
    prev_rightx = rightx_current

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - left_margin
        win_xleft_high = leftx_current + left_margin
        win_xright_low = rightx_current - right_margin
        win_xright_high = rightx_current + right_margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Find points within window
        mask_left = (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & (nonzeroy < win_y_high) \
                    & (nonzeroy >= win_y_low)
        mask_right = (nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & (nonzeroy < win_y_high) \
                     & (nonzeroy >= win_y_low)
        good_left_inds = mask_left.nonzero()[0]
        good_right_inds = mask_right.nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            # Valid points for left lane line were found and can be added to list
            left_margin = 100
            prev_leftx = leftx_current
            leftx_current = np.int(np.mean([nonzerox[i] for i in good_left_inds]))
        else:
            left_margin += 25 # Increase margin if number of detected points was not sufficient
            leftx_current += leftx_current - prev_leftx # Move in the same direction as previously

        if len(good_right_inds) > minpix:
            # Valid points for right lane line were found and can be added to list
            prev_rightx = rightx_current
            rightx_current = np.int(np.mean([nonzerox[i] for i in good_right_inds]))
            right_margin = 100
        else:
            right_margin += 25 # Increase margin if number of detected points was not sufficient
            rightx_current += rightx_current - prev_rightx # Move in the same direction as previously

    # Combine Lane line indices into one list
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    """
    Find polynomial on binary warped image
    :param binary_warped:
    :return: Annotated image and Lane line information
    """
    leftx, lefty, rightx, righty, out_img = locate_lines(binary_warped)
    # Move in the same direction as previously
    left_fit = np.polyfit(x=lefty, y=leftx, deg=2)
    right_fit = np.polyfit(x=righty, y=rightx, deg=2)
    #print("Left Fit: {}".format(left_fit))
    #print("Right Fit: {}".format(right_fit))
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        #print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    show_polynomial = False
    if show_polynomial:
        y_left = []
        x_left = []
        y_right = []
        x_right = []
        for i in range(len(left_fitx)):
            if int(left_fitx[i]) < 1279 and int(left_fitx[i]) > 1:
                for j in range(3):
                    y_left.append(int(ploty[i]))
                    x_left.append(int(left_fitx[i])+j -1)
            if int(right_fitx[i]) < 1279 and int(right_fitx[i]) > 1:
                for j in range(3):
                    y_right.append(int(ploty[i]))
                    x_right.append(int(right_fitx[i])+j -1)

        out_img[y_left, x_left] = [255, 255, 255]
        out_img[y_right, x_right] = [255, 255, 255]


    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, ploty , left_fitx, right_fitx, lefty, leftx, righty, rightx


if __name__ == '__main__':
    pass
