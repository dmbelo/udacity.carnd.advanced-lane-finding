import numpy as np
import cv2
import json
import glob
import matplotlib.pyplot as plt


class Camera():
    def __init__(self, img_size):
        self.mtx = []
        self.dtx = []
        self.img_size = tuple(img_size)
        self.M_plan = []  # perspective transform
        self.M_front = []  # inverse perspective transform

    def calibrate(self, img_files, nx, ny, save=False):
        """Calibrate camera with a set of chessboard images"""
        obj_pts = []
        img_pts = []
        for idx, file in enumerate(img_files):
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...(6,5,0)
            if ret:
                obj_p = np.zeros((nx * ny, 3), np.float32)
                obj_p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
                obj_pts.append(obj_p)
                img_pts.append(corners)

                if save:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    write_name = 'corners_found' + str(idx) + '.jpg'
                    cv2.imwrite(write_name, img)

        img_size = (img.shape[1], img.shape[0])
        out = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
        # ret, mtx, dist, rvecs, tvecs
        self.mtx = out[1]
        self.dtx = out[2]

    def perspective_setup(self, src, dst):
        # Argumnets: source and destination points
        self.M_plan = cv2.getPerspectiveTransform(src, dst)
        self.M_front = cv2.getPerspectiveTransform(dst, src)

    def undistort(self, img):
        """Undistort image"""
        return cv2.undistort(img, self.mtx, self.dtx, None, self.mtx)

    def plan_view(self, img):
        f = cv2.INTER_LINEAR
        return cv2.warpPerspective(img, self.M_plan, self.img_size, flags=f)

    def front_view(self, img):
        f = cv2.INTER_LINEAR
        return cv2.warpPerspective(img, self.M_front, self.img_size, flags=f)


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last 5 fits of the line
        self.recent_x_pixels = [[], [], [], [], []]
        self.recent_y_pixels = [[], [], [], [], []]
        # average x values of the fitted line over the last n iterations
        # self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        # self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        self.x = []
        self.y = []
        # radius of curvature of the line in some units
        # self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        # self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        # self.diffs = np.array([0, 0, 0], dtype=np.float)
        # x values for detected line pixels
        # self.allx = None
        # y values for detected line pixels
        # self.ally = None

    def update_xy(self, x, y):
        self.recent_x_pixels[:-1] = self.recent_x_pixels[1:]
        self.recent_x_pixels[-1] = x
        self.recent_y_pixels[:-1] = self.recent_y_pixels[1:]
        self.recent_y_pixels[-1] = y

    def fit(self):
        n = 5e3
        # List of lists into one numpy array
        x = np.array([item for sublist in self.recent_x_pixels for item in sublist])
        y = np.array([item for sublist in self.recent_y_pixels for item in sublist])
        # print(x.shape[0])
        if x.shape[0] > n:
            return np.polyfit(y, x, 2)
        else:
            p = np.polyfit(y, x, 1)
            return np.concatenate([[0], p])


class Lane():
    def __init__(self, width):
        self.left_line = Line()
        self.right_line = Line()
        self.width = width

    def blind_search(self, binary):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary, binary, binary))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary.shape[0] - (window+1)*window_height
            win_y_high = binary.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on mean pos.
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Update the line objects
        self.left_line.update_xy(leftx, lefty)
        self.right_line.update_xy(rightx, righty)

        # Fit a polynomial to each
        left_fit = self.left_line.fit()
        right_fit = self.right_line.fit()

        # PLOTTING
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        self.left_line.x = left_fitx
        self.left_line.y = ploty
        self.right_line.x = right_fitx
        self.right_line.y = ploty

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

        # Record findings
        self.left_line.detected = True
        self.left_line.current_fit = left_fit

        self.right_line.detected = True
        self.right_line.current_fit = right_fit

    def targeted_search(self, binary):
        # Skip the sliding windows step once you know where the lines are
        # Now you know where the lines are you have a fit! In the next frame
        # of video you don't need to do a blind search again, but instead you
        # can just search in a margin around the previous line position
        # like this:
        left_fit = self.left_line.current_fit
        right_fit = self.right_line.current_fit
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Update the line objects
        self.left_line.update_xy(leftx, lefty)
        self.right_line.update_xy(rightx, righty)

        # Fit a polynomial to each
        left_fit = self.left_line.fit()
        right_fit = self.right_line.fit()

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        self.left_line.x = left_fitx
        self.left_line.y = ploty
        self.right_line.x = right_fitx
        self.right_line.y = ploty

        # And you're done! But let's visualize the result here as well
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary, binary, binary)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.figure()
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    def search(self, binary):
        if (self.left_line.detected | self.right_line.detected):
            self.targeted_search(binary)
        else:
            self.blind_search(binary)

    def sanity_check(self):
        # Checking that they have similar curvature
        # Checking that they are separated by approximately the right
        # distance horizontally
        # Checking that they are roughly parallel
        raise NotImplementedError

    def center_offset(self, img):
        raise NotImplemented

    def overlay(self, img, camera):
        # Create an image to draw the lines on
        # warp_zero = np.zeros_like(warped).astype(np.uint8)
        # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        left_x = self.left_line.x
        left_y = self.left_line.y
        right_x = self.right_line.x
        right_y = self.right_line.y
        pts_left = np.array([np.transpose(np.vstack([left_x, left_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, right_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, camera.M_front, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        return cv2.addWeighted(img, 1, newwarp, 0.3, 0)


def gaussian_blur(img, kernel):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def abs_sobel_thresh(gray, dx, dy, kernel=3, thresh=(0, 255)):
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=kernel)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled, dtype=np.uint8)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary


def mag_sobel_thresh(gray, kernel=3, thresh=(0, 255)):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    scaled = np.uint8(255*mag/np.max(mag))
    binary = np.zeros_like(scaled, dtype=np.uint8)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary


def dir_sobel_thresh(gray, kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    sobel_x_abs = np.absolute(sobel_x)
    sobel_y_abs = np.absolute(sobel_y)
    grad = np.arctan2(sobel_x_abs, sobel_y_abs)
    binary = np.zeros_like(grad, dtype=np.uint8)
    binary[(grad >= thresh[0]) & (grad <= thresh[1])] = 1
    return binary


def hls_s_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:, :, 2]
    binary = np.zeros_like(S, dtype=np.uint8)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary


def gradient_threshold(img):
    smooth = gaussian_blur(img, kernel=9)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    abs_bin = abs_sobel_thresh(gray, dx=1, dy=0, kernel=7, thresh=(30, 70))
    mag_bin = mag_sobel_thresh(gray, kernel=9, thresh=(25, 100))
    dir_bin = dir_sobel_thresh(gray, kernel=9, thresh=(np.pi/2*0.8, np.pi/2))
    hls_bin = hls_s_thresh(img, thresh=(170, 255))
    # Combine the two binary thresholds
    binary = np.zeros_like(abs_bin, dtype=np.uint8)
    binary[((mag_bin == 1) & (dir_bin == 1)) | (hls_bin == 1)] = 1
    return binary, (abs_bin, hls_bin, mag_bin, dir_bin)


def setup(config_file='config.json'):
    with open(config_file) as f:
        config = json.load(f)
    img_size = config['Image resolution']
    cal_glob = config['Calibration image search pattern']
    nx = config['Number of corners - X']
    ny = config['Number of corners - Y']
    src = np.array(config['Source perspective points'], dtype=np.float32)
    dst = np.array(config['Destination perspective points'], dtype=np.float32)
    lane_width = config['Lane width']

    camera = Camera(img_size)
    cal_images = glob.glob(cal_glob)
    camera.calibrate(cal_images, nx=nx, ny=ny, save=False)
    camera.perspective_setup(src, dst)

    lane = Lane(lane_width)

    return camera, lane


def pipeline(img, camera, lane):
    undist = camera.undistort(img)
    plan = camera.plan_view(undist)
    binary, info = gradient_threshold(plan)
    lane.search(binary)
    # Smoothing ... last 5 frames...
    # lane.sanity_check()
    # x = lane.center_offset(img)
    overlay = lane.overlay(undist, camera)
    plt.figure()
    plt.imshow(overlay)
    plt.show()


def main():
    camera, lane = setup()
    img = cv2.cvtColor(cv2.imread('test6.jpg'), cv2.COLOR_BGR2RGB)
    pipeline(img, camera, lane)


def draw_lane(img, pts):
    pts = pts.astype(np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 255), 3)


def test_threshold(img, abs_bin, hls_bin, mag_bin, dir_bin):
    test = np.zeros_like(mag_bin)
    test[(mag_bin == 1) & (dir_bin == 1)] = 1
    print(mag_bin.dtype)
    print(dir_bin.dtype)
    stacks = np.dstack((np.zeros_like(test), test*255, hls_bin*255))
    # stacks = np.dstack((mag_bin*0, dir_bin*0, test*255))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(img)
    ax2.imshow(stacks)
    plt.show()


def test_perspective(img, camera):
    with open('config.json') as f:
        config = json.load(f)
    undist = camera.undistort(img)
    plan = camera.plan_view(undist)
    draw_lane(undist, np.array(config['Source perspective points']))
    draw_lane(undist, np.array(config['Destination perspective points']))
    plt.subplot(211)
    plt.imshow(undist)
    plt.subplot(212)
    plt.imshow(plan)
    plt.show()


if __name__ == '__main__':
    main()
