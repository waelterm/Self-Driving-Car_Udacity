# Define a class to receive the characteristics of each line detection
import numpy as np

LANE_PIXEL_WIDTH = 1105-217
LANE_PIXEL_LENGTH = 705 - 440
YM_PER_PIX = 30/LANE_PIXEL_LENGTH
XM_PER_PIX = 3.7/LANE_PIXEL_WIDTH

class Line():
    def __init__(self,current_fit,  n=3):
        """
        This class holds and updates the lane line information
        :param current_fit: first polynomial estimate to avoid initiation issues
        :param n: how many polynomials are saved and averaged
        """
        self.cnt = 0
        self.n = n
        # was the line detected in the last iteration?
        self.detected = False
        self.recent_xfitted_list = []
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None

        self.all_fit = [current_fit]
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = current_fit
        #polynomial coefficients for the most recent fit
        self.current_fit = current_fit
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def update(self, detected, radius_of_curvature, allx, ally, current_fit, fitx):
        """
        This function updates the Lane line class
        :return: NOne
        """
        #print("Current_fit received: {}".format(current_fit))
        self.cnt = 0
        self.detected = detected
        self.recent_xfitted_list.append(fitx)
        if len(self.recent_xfitted_list) > self.n:
            self.recent_xfitted_list = self.recent_xfitted_list[1:]
        self.recent_xfitted = [np.mean([self.recent_xfitted_list[j][i] for j in range(len(self.recent_xfitted_list))]) \
                               for i in range(len(self.recent_xfitted_list[0]))]
        self.bestx = np.mean(self.recent_xfitted)
        self.current_fit = current_fit
        self.all_fit.append(current_fit)
        if len(self.all_fit) > self.n:
            self.all_fit = self.all_fit[1:]
        self.best_fit = np.array([np.mean([fit[0] for fit in self.all_fit]),np.mean([fit[1] for fit in self.all_fit]), np.mean([fit[2] for fit in self.all_fit])])
        self.radius_of_curvature = radius_of_curvature
        self.line_base_pos = (self.best_fit[0] * 720 ** 2 + self.best_fit[1] * 720 + self.best_fit[2]) * XM_PER_PIX
        self.diffs = self.current_fit - self.all_fit[-2]
        self.allx = allx
        self.ally = ally



