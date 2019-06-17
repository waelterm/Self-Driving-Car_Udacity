## Project 2 - Lane Finding - Writeup 

### This project describes the developed pipeline for the Advanced Lane Finding Project of Udacities Self Driving Car Nanodegree. This pipeline uses parts of script during the course, but the specific pipeline was implemented by Philipp Waeltermann


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_cal/calibration_example.png "Undistorted"
[image2]: ./output_images/undistorted/test2.jpg "Road Transformed"
[image3]: ./output_images/filtered/test2.jpg "Binary Example"
[image4]: ./output_images/warped/test2.jpg "Warp Example"
[image5]: ./output_images/fitted/test2.jpg "Fit Visual"
[image6]: ./output_images/final/test2.jpg "Output"
[image7]: ./test_images/test2.jpg "Original Image"
[image8]: ./output_images/polygon/straight_lines1.jpg "Polygon 1"
[image9]: ./output_images/polygon/straight_lines2.jpg "Polygon 2"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the camera_calibration.py file located in "./camera_calibration.py".

There are two relevant functions in this file. The first function is camera_calibration. It takees the folder with the calibration images and a list of the dimensions. This list includes the number of corners in each image.
For each image, the ChessboardCorners will be found  using the cv2.findChessboardCorners function. Their locations are added to the image point list.
In addition to that, the object points will aslo be created by creating a grid based on the dimensions/ number of corners for each image. The object and image points are then used to calculate the transformation matrix used to undistort the image. 
The second function takes a directory of images and the transformation matrix and undistorts the images using the cv2.undistort() function.

The result can be seen below:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

For all of the pipeline related images I will use test image number 2. Which originally looks like this:
![alt text][image7]

After applying the distortion matrix as described above, the image looks like this:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The functions for this step can be found in the thresholds.py file. However, the specific thresholds used are defined in the files that utilize those functions. Those functions are pipeline.py for test images and main.py for annotation of the image.
I used a set of two filteres to find pixels related to lane lines. For the first filter I applied the sobel in x direction and used a lower threshold of 30 and an upper threshold of 100. For the second filter I transformed the image from BGR to hls. I used lower threshold of 150 and an upper threshold of 255. The relevant points are then shown in a binary image. The result can be seen below

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for the perspective transform can be found in the perspective_transform.py function.
The first function called find_polygon, was used ot find the polygon describing a straight line.
The polygons I found can be seen below:

![alt text][image8]
![alt text][image9]

The second function perspective_transform takes and undistorted image and returns the image with a changed perspective to bird's eye view. The source points for this transformation are the ones found using the polygon function described above. The destination points were chosen to turn the polygon into a retangle that fits into the image with enough space on the side to allow for slightly wider/narrower lanes:
```python
src = np.float32([[217, 705],[606, 440] , [676, 440], [1105, 705]])
dst = np.float32([[250, 720], [250 , 0], [1280-250, 0], [1290 - 250, 720]])
```

The result of the perspective transform can be seen below:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are two algorithms that are being used to fit the lane-line pixels.
Algorithm number one can be found in the locate_lines.py file, and the second algorithm will be found in the search_prior.py file.

The first algorithm is used when the line detection in the previous frame was not valid. It uses a histogram on the lower half of the image to find the base position on the bottom of the image. At that position, a box with margin of 100 pixels on each side will be created and within that box, I search for lane ine pizels using the binary warped image. The average lateral position of those pizels will then be used for the position of the next image. If not sufficient points are found within the box, the box will be moved by the same offset as the two previous boxes, and its margins will be increased to allow for a larger search area to make sure the lines will be found again. All of the detected image points will then be stored in a list and fitted to a second order polynomial

The second order polynomial, can also be calculated using the polynomial from the previous frame. In this algorithm the margin will be applied around the previously detected polynomial instead of discrete boxes. This algorithm searches a smaller space and is therefore faster.

The arbitration between the two images is done by the main.py and depends on a number of sanity checks done on the previous image.
A result of this can be seen below:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature calculation is done on the curvature.py file. This file turns the polynomial from pixel coordinates into meters.
This conversation is based on the data found for the perspective transform. The curvature is then calculated at the bottom of the image (position of the vehicle) using the following formula:
```python
y_eval = 720 *ym_per_pix
left_curverad = (1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** (3 / 2) / abs(2 * left_fit_cr[0])  
right_curverad = (1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** (3 / 2) / abs(2 * right_fit_cr[0])  
```

The offset is calculated by taking the average of the base positions from the left and right polynomial, this lane center is then subtracted from the vehicle position which is the half the image width converted from pixel units into meters.

For the example images shown, the following values have been calculated:

Left curvature: 1766.0671092527023
Right curvature: 3674.3906658329925
Left Base: 1.3938096638387134
Right Base: 4.6673667259167235
Lane Center Offset: -0.3639215282110517

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final result looks like this:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

For the complete pipeline, a line class has been developed in Lines_new.py
This class is updated at each frame, and holds important information. It also applies some filtering to ensure a more stable lane detection.
Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have found that the sanity checks sometimes reject valid lane lines even though they were correctly found.
This fairly often happens when the street seems to have a slight incline or decline. In those situations, the perspective transform does not work effectively. It looks as if the lines are not parallel even though they actually are.
A gradient detection algorithm could help making the perspective transform more optimized for inclining or declining streets
  
Additionally, it can sometimes be seen that the last part of the line is not found properly. This is not a bad failure mode, because it only slightly changes the polynomial at a large distance ahead. 

Lastly, I was able to see from the challenge videos, that my algorithm does not perform well in situations where there seem to be other lines in the center of the lane. This could be fixed by adding a region of interest and not regarding any points outside this region.
