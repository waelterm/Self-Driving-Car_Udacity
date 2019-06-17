import numpy as np

LANE_PIXEL_WIDTH = 1105-217
LANE_PIXEL_LENGTH = 705 - 440
YM_PER_PIX = 30/LANE_PIXEL_LENGTH
XM_PER_PIX = 3.7/LANE_PIXEL_WIDTH

def lane_center_offset(left_base_m, right_base_m):
    """
    Calculates the lane center offset using the base positions of the lane lines.
    :param left_base_m:
    :param right_base_m:
    :return:
    """
    return (1280/2*XM_PER_PIX) - (left_base_m + right_base_m) / 2


def pixel_to_meters(left_fit, right_fit, YM_PER_PIX, XM_PER_PIX):
    """
    This function turns a second order polynomial from pixel units to meters
    :param left_fit: polynomial of left lane line
    :param right_fit: polynomial of left lane line
    :param YM_PER_PIX: ratio of meters per pixel in y direction
    :param XM_PER_PIX: ratio of meters per pixel in x direction
    :return: Scaled left and right polynomials
    """
    left_fit[0] *= XM_PER_PIX/YM_PER_PIX**2
    left_fit[1] *= XM_PER_PIX / YM_PER_PIX
    left_fit[2] *= XM_PER_PIX
    right_fit[0] *= XM_PER_PIX / YM_PER_PIX ** 2
    right_fit[1] *= XM_PER_PIX / YM_PER_PIX
    right_fit[2] *= XM_PER_PIX
    return left_fit, right_fit

def measure_curvature_real(warped, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/ 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    left_fit_cr, right_fit_cr = pixel_to_meters(left_fit, right_fit, YM_PER_PIX, XM_PER_PIX)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = 720 *ym_per_pix

    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = (1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** (3 / 2) / abs(
        2 * left_fit_cr[0])  ## Implement the calculation of the left line here
    right_curverad = (1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** (3 / 2) \
                     / abs(2 * right_fit_cr[0])  ## Implement the calculation of the right line here

    return left_curverad, right_curverad, left_fit_cr, right_fit_cr


if __name__ == '__main__':
    LANE_PIXEL_WIDTH = 1105-217
    LANE_PIXEL_LENGTH = 705 - 440
    YM_PER_PIX = 30/LANE_PIXEL_LENGTH
    XM_PER_PIX = 3.7/LANE_PIXEL_WIDTH