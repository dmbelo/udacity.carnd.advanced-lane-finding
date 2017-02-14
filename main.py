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
        flag = cv2.INTER_LINEAR
        return cv2.warpPerspective(img, self.M_plan, self.img_size, flags=flag)

    def front_view(self, img):
        flag = cv2.INTER_LINEAR
        return cv2.warpPerspective(img, self.M_front, self.img_size, flags=flag)


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype=np.float)
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


class Lane():
    def __init__(self, left, right, width):
        self.left_line = left
        self.right_line = right
        self.width = width

    def blind_search(self, binary):
        raise NotImplementedError

    def targeted_search(self, binary):
        raise NotImplementedError

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

    def overlay(self, img):
        raise NotImplemented


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
    grad = np.arctan2(sobel_y_abs, sobel_x_abs)
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
    abs_bin = abs_sobel_thresh(gray, dx=1, dy=0, kernel=3, thresh=(20, 100))
    # mag_bin = mag_sobel_thresh(gray, sobel_kernel=7, thresh=(5, 40))
    hls_bin = hls_s_thresh(img, thresh=(170, 255))
    # Combine the two binary thresholds
    binary = np.zeros_like(abs_bin, dtype=np.uint8)
    binary[(abs_bin == 1) | (hls_bin == 1)] = 1
    return binary, abs_bin, hls_bin


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
    camera.calibrate(cal_images, nx=nx, ny=ny, save=True)
    camera.perspective_setup(src, dst)

    lane = Lane(Line(), Line(), lane_width)

    return camera, lane


def pipeline(img, camera, lane):
    undist = camera.undistort(img)
    # plan = camera.plan_view(undist)
    binary, abs_bin, hsl_bin = gradient_threshold(undist)
    lane.search(binary)
    # Smoothing ... last 5 frames...
    # lane.sanity_check()
    # x = lane.center_offset(img)
    # lane.overlay(img)
    plt.subplot(121)
    plt.imshow(undist)
    plt.subplot(122)
    plt.imshow(binary, cmap='gray')
    plt.show()


def main():
    camera, lane = setup()
    img = cv2.cvtColor(cv2.imread('test1.jpg'), cv2.COLOR_BGR2RGB)

    pipeline(img, camera, lane)


if __name__ == '__main__':
    main()
