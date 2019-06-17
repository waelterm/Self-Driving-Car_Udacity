import numpy as np
import cv2
from camera_calibration import camera_calibration
from thresholds import thresholds
from perspective_transform import perspective_transform
from locate_lines import fit_polynomial
from search_prior import search_around_poly
from curvature import measure_curvature_real
from curvature import lane_center_offset
from curvature import measure_lane_curvature_real
from draw_image import draw_image
from numpy.linalg import inv
from moviepy.editor import VideoFileClip
from Lines_new import Line
import copy


# Constants to calculate distances in m from pixel values
LANE_PIXEL_WIDTH = 1105-217
LANE_PIXEL_LENGTH = 705 - 440
YM_PER_PIX = 30/LANE_PIXEL_LENGTH
XM_PER_PIX = 3.7/LANE_PIXEL_WIDTH


# Global Line object describing the adjacent lane lines
global left_line
global right_line
left_line = Line(current_fit = np.array([0,0,217]), n = 3)
right_line = Line(current_fit = np.array([0,0,1105]), n = 3)



def poly_result(fit, y):
    """
    This function evaluates a polynomial of 2nd order
    :param fit: polyomial coefficients
    :param y: independent variable
    :return: dependent variable
    """
    x = fit[0] * y ** 2 + fit[1] * y + fit[2]
    return x


def process_image(img):
    """
    This function finds the lane lines in each image and stores their information in the global line objects
    :param img: One frame of Highway video
    :return: Annotated Image
    """
    # Correct for distortion based on calibration parameters
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    # Apply filter to detect lines in image
    line_image = thresholds(undistorted, s_thresh=(150, 255), sx_thresh=(30, 100))
    # Warp the image to achieve a birds-eye view
    warped, M = perspective_transform(line_image)
    # Locate Lines using moving boxes if lane lines not detected in previous frame
    if not left_line.detected or len(left_line.all_fit)< left_line.n:
        print('Attempting Box algorithm')
        out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, lefty, leftx, righty, rightx  = \
            fit_polynomial(warped)
    # Locate Lines using search near previously found lane lines
    else:
        print('Attempting Search around Polynomial')
        out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, lefty, leftx, righty, rightx = \
            search_around_poly(warped, left_line.best_fit, right_line.best_fit)
        # If this algorithm fails, use Box algorithm instead
        if out_img is None:
            print('Search around Polynomial was NOT successful.')
            out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, lefty, leftx, righty, rightx = \
                fit_polynomial(warped)
    # Calculate curvature for found lane line polynomials, at the bottom of the image
    left_curv, right_curv, left_fit_cr, right_fit_cr = measure_curvature_real(warped, copy.copy(left_fit), copy.copy(right_fit))
    # This following if statements checks if the found curves are reasonable by checking:
        # 1. That the distance between the two curves is reasonable
        # 2. That both polynomials exist
        # 3. That their curvatures are similar (curvatures are allowed to vary by 50%)
        # 4. That the curves are similar (c1 and c2 are allowed to vary by 50%)
    valid = True
    right_base = poly_result(right_fit, 720)
    left_base = poly_result(left_fit, 720)
    if not ((right_base - left_base) > 2.5/XM_PER_PIX and (right_base - left_base) < 4.5/XM_PER_PIX):
        print(right_base)
        print(left_base)
        print("Invalid Distance")
        print("Distance: {}".format((right_base - left_base)*XM_PER_PIX))
        valid = False
    if not (left_fit is not None and right_fit is not None):
        print("Invalid Data")
        valid = False
    if not ((left_curv / right_curv) < 3 and (left_curv / right_curv) > 0.33 and left_curv > 800 and right_curv > 800):
        print("Invalid Curvature")
        print("Left Line Curvature {}".format(left_curv))
        print("Right Line Curvature {}".format(right_curv))
        valid = False
    if not ((left_fit[0] / right_fit[0]) < 3 and (left_fit[0] / right_fit[0]) > 0.33 \
            and (left_fit[1] / right_fit[1]) < 3 and (left_fit[1] / right_fit[1]) > 0.33) :
        print("Lines are not parallel.")
        valid = False
    if valid or left_line.cnt > 3 or left_line.recent_xfitted == []:
        # The counter avoids that the algorithm gets stuck for too long

        # Update Lane Lines
        left_line.update(detected = True, radius_of_curvature = left_curv, allx = None,
                         ally = None, current_fit = left_fit, fitx = left_fitx)
        right_line.update(detected=True, radius_of_curvature=right_curv, allx=None,
                         ally=None, current_fit=right_fit, fitx = right_fitx)
        #print("Curvature of left lane line: {}".format(left_curv))
        print("Curvature of right lane line: {}".format(right_curv))
    else:
        print("Warning: Lines were not detected!")
        left_line.detected = False
        right_line.detected = False
        left_line.cnt += 1
        right_line.cnt += 1
    print("Curvature of right lane line: {}".format(right_curv))

    lane_curvature = measure_lane_curvature_real(warped, copy.copy(left_line.best_fit), copy.copy(right_line.best_fit))
    offset = lane_center_offset(left_line.line_base_pos, right_line.line_base_pos)
    print("Lane Center Offset: {}".format(offset))
    output = draw_image(warped, left_line.recent_xfitted, right_line.recent_xfitted, ploty, img, inv(M), undistorted)
    # I used this as an example to understand how to use cv2.putText:
    # https: // www.programcreek.com / python / example / 83399 / cv2.putText
    cv2.putText(output, "Radius of curvature: {:.2f}m".format(lane_curvature), org = (200,75), \
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color = (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    if offset < 0:
        offset_string = "Vehicle is {:.2f}m left of center.".format(abs(offset))
    else:
        offset_string = "Vehicle is {:.2f}m right of center.".format(offset)
    cv2.putText(output, offset_string, org=(200, 150), \
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2,
                lineType=cv2.LINE_AA)
    #cv2.imshow("Display Corners", output)
    #cv2.waitKey(0)
    return output


if __name__ == '__main__':
    # Running calibration on calibration images
    do_calibration = False
    if do_calibration:
        calibration_images = "camera_cal/"
        dimension_list = [[5, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9], [6, 9],
                          [6, 9], [6, 9], [6, 9], [6, 8], [6, 7], [6, 9], [6, 9], [6, 9], [6, 9]]
        mtx, dist = camera_calibration(calibration_images, dimension_list, display_corners=False)
    # Using previously created calibration coefficients
    else:
        mtx = np.array(
            [[1.16067642 * 1000, 0.00000000, 6.65724920 * 100], [0.00000000, 1.15796966 * 1000, 3.88971790 * 100],
             [0.00000000, 0.00000000, 1.00000000]])
        dist = np.array([[-0.25427498, 0.04987807, -0.00043039, 0.00027334, -0.1336389]])
    # Loading Test images
    #test_images = "test_images/"
    #all_files = os.listdir(test_images)
    #image_names = [file for file in all_files if file[-4:] == '.jpg']
    #init = True

    # Loading Video and running process_image function on each frame
    video = 'project_video.mp4'
    challenge_output = 'project_video_output.mp4'
    clip3 = VideoFileClip(video)
    challenge_clip = clip3.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)