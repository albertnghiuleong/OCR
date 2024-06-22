import cv2
import pytesseract
import numpy as np
import pandas as pd
import os
import shutil
import time
#import easyocr #https://github.com/JaidedAI/EasyOCR
import ctypes
import ctypes.util
import tkinter as tk
import argparse
import threading
import yaml
import socket
import json
from tkinter import filedialog
from datetime import datetime

canny_parameters = []

class TesseractError(Exception):
    pass

class Tesseract(object):
    _lib = None
    _api = None

    class TessBaseAPI(ctypes._Pointer):
        _type_ = type('_TessBaseAPI', (ctypes.Structure,), {})

    @classmethod
    def setup_lib(cls, lib_path=None):
        if cls._lib is not None:
            return
        if lib_path is None:
            lib_path = "/usr/local/lib/libtesseract.so.4"
        cls._lib = lib = ctypes.CDLL(lib_path)

        # source:
        # https://github.com/tesseract-ocr/tesseract/blob/95ea778745edd1cdf6ee22f9fe653b9e061d5708/src/api/capi.h

        lib.TessBaseAPICreate.restype = cls.TessBaseAPI

        lib.TessBaseAPIDelete.restype = None # void
        lib.TessBaseAPIDelete.argtypes = (
            cls.TessBaseAPI,) # handle

        lib.TessBaseAPIInit3.argtypes = (cls.TessBaseAPI, ctypes.c_char_p, ctypes.c_char_p) 

        lib.TessBaseAPISetImage.restype = None
        lib.TessBaseAPISetImage.argtypes = (cls.TessBaseAPI, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)    

        lib.TessBaseAPISetVariable.argtypes = (cls.TessBaseAPI, ctypes.c_char_p, ctypes.c_char_p)

        lib.TessBaseAPIGetUTF8Text.restype = ctypes.c_char_p
        lib.TessBaseAPIGetUTF8Text.argtypes = (
            cls.TessBaseAPI,)

    def __init__(self, language='eng_best', datapath=None, lib_path=None):
        if self._lib is None:
            self.setup_lib(lib_path)
        self._api = self._lib.TessBaseAPICreate()
        print ("initializing tesseract!!!!")
        if self._lib.TessBaseAPIInit3(self._api, datapath, language):
            print ("Tesseract initialization failed!!")
            raise TesseractError('initialization failed')

    def __del__(self):
        if not self._lib or not self._api:
            return
        if not getattr(self, 'closed', False):
            self._lib.TessBaseAPIDelete(self._api)
            self.closed = True

    def _check_setup(self):
        if not self._lib:
            raise TesseractError('lib not configured')
        if not self._api:
            raise TesseractError('api not created')

    def set_image(self, imagedata, width, height,
                  bytes_per_pixel, bytes_per_line=None):
        self._check_setup()
        if bytes_per_line is None:
            bytes_per_line = width * bytes_per_pixel
        print ("bytes per line={}".format(bytes_per_line))
        self._lib.TessBaseAPISetImage(self._api,
                                      imagedata, width, height,
                                      bytes_per_pixel, bytes_per_line)

    def set_variable(self, key, val):
        self._check_setup()
        self._lib.TessBaseAPISetVariable(self._api, key, val)

    def get_utf8_text(self):
        self._check_setup()
        return self._lib.TessBaseAPIGetUTF8Text(self._api)

    def get_text(self):
        self._check_setup()
        result = self._lib.TessBaseAPIGetUTF8Text(self._api)
        if result:
            return result.decode('utf-8')

def find_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3-q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]

    return outlier_indices

def convert_to_grayscale(image_data):
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

# a method to make it look similar to tesslib.py
def tesseract_process_image2(tess, frame_piece):
    grayscaled = len(frame_piece.frame.shape) == 2
    if not grayscaled:
        image_data = convert_to_grayscale(frame_piece.frame)

    height, width = frame_piece.frame.shape
    tess.set_variable("tesseract_char_whitelist", frame_piece.whitelist)
    tess.set_variable("tessedit_pageseg_mode", str(frame_piece.psm))
    # tess.set_variable("user_words_suffix", "user-data")
    # tess.set_variable("user_pattern_suffix", "user-pattern")
    tess.set_variable("image_default_resolution", "70")
    tess.set_image(frame_piece.frame.ctypes, width, height, 1)
    text = tess.get_utf8_text()
    return text.strip()

class FramePiece(object):
  def __init__(self, img, whitelist):
    self.frame = img
    self.whitelist = whitelist if whitelist else "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
    self.psm = 6

# overloaded method for view page
def tesseract_process_image(tess, frame, whitelist=None):
    frame_piece = FramePiece(frame, whitelist)
    return tesseract_process_image2(tess, frame_piece)

#import video

# If you don't have tesseract executable in your PATH, include the following:
#pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Albert-HL.Ng\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
def show_image(mat,window_name='Image'):
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(window_name, on_EVENT_LBUTTONDOWN)
    cv2.imshow(window_name, mat)
    cv2.waitKey(0)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def obj_detection(mat_image):
    
    #eye detection
    cascade = cv2.CascadeClassifier(r'C:\Users\Albert-HL.Ng\Documents\Python\opencv-master\data\haarcascades\haarcascade_eye.xml')
    gray = cv2.cvtColor(mat_image, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in objects:
        cv2.rectangle(mat_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Object Detection', mat_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_contours(mat_image, tree_ind=0):

    global configFile

    if(tree_ind == 1):
        level = 3

        h, w = mat_image.shape[:2]
        contours0, hierarchy = cv2.findContours( mat_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        '''
        cv2.RETR_EXTERNAL: Retrieves only the external or outermost contours, ignoring any inner contours or holes within objects.
        cv2.RETR_LIST: Retrieves all contours without any hierarchical relationships.
        cv2.RETR_CCOMP: Retrieves all contours and organizes them into a two-level hierarchy. The first level contains external contours, and the second level contains contours of holes within objects.
        cv2.RETR_TREE: Retrieves all contours and reconstructs a full hierarchy of contours, including parent-child relationships.
        '''

        contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
        def update(levels):
            vis = np.zeros((h, w, 3), np.uint8)
            levels = levels - 3
            cv2.drawContours( vis, contours, (-1, 2)[levels <= 0], (128,255,255),
                3, cv2.LINE_AA, hierarchy, abs(levels) )
            cv2.imshow('contours', vis)
            global level
            level = levels
            return vis
        
        update(3)
        cv2.createTrackbar( "levels+3", "contours", 3, 7, update )
        cv2.imshow('image', mat_image)
        cv2.waitKey()

        output = update(level)        
    

    #breakpoint()

    if configFile['edge_detection']['connected_contour_ind'] == 1:    
        contours, hierarchy = cv2.findContours(mat_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        conditions = (hierarchy[0, :, 3] != -1)
        contours = tuple(contour for contour, cond in zip(contours, conditions) if cond)
    else:
        contours, _ = cv2.findContours(mat_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
    
    rects = []
    for contour in contours:  
     
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is rectangular
        if len(approx) == 4:
            # Draw a rectangle around the contour
            x, y, w, h = cv2.boundingRect(approx)
            rects.append((x, y, w, h))
            #cv2.rectangle(mat_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(mat_image, (x, y), (x + w, y + h), (128,255,255), 2)
    if(tree_ind == 0):
        output = contours     

    return output, rects
    

def find_edge(mat_image, thrshold = []):
    '''
    Canny parameters
    Low Threshold: 
    This threshold value is used to identify potential weak edges. Any edge with a gradient magnitude below the low threshold is suppressed and considered as non-edge pixels. Typically, this value is set to a fraction (e.g., 0.4 or 0.5) of the high threshold.
    
    High Threshold: 
    This threshold value is used to identify strong edges. Any edge with a gradient magnitude above the high threshold is considered as strong edges. Typically, this value is set based on some statistical analysis or by experimenting with different values
    
    Aperture Size: 
    The aperture size determines the size of the Sobel kernel used to compute the image gradients. It affects the level of noise filtering and the thickness of detected edges. A larger aperture size can capture larger-scale edges but may also result in thicker edges. Common values for the aperture size are 3, 5, or 7.    
    '''
    global configFile
    apertureSize = configFile['edge_detection']['apertureSize']

    def nothing(*arg):
        pass
    
    if len(thrshold) == 0 and configFile['edge_detection']['canny_threshold_calc_ind'] == 0:
        cv2.namedWindow('edge')
        cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
        cv2.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

        #cap = video.create_capture(fn)
        while True:
            #_flag, img = cap.read()
            gray = cv2.cvtColor(mat_image, cv2.COLOR_BGR2GRAY)
            thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
            thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
            edge = cv2.Canny(gray, thrs1, thrs2, apertureSize)
            vis = mat_image.copy()
            vis = np.uint8(vis/2.)
            vis[edge != 0] = (0, 255, 0)
            cv2.imshow('edge', vis)
            ch = cv2.waitKey(5)
            if ch == 27:           
                cv2.destroyAllWindows() 
                break
    else:
        if configFile['edge_detection']['canny_threshold_calc_ind'] == 1:
            # Calculate the median gradient magnitude
            sobelx = cv2.Sobel(mat_image, cv2.CV_64F, 1, 0, ksize=apertureSize)
            sobely = cv2.Sobel(mat_image, cv2.CV_64F, 0, 1, ksize=apertureSize)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            median_gradient = np.median(gradient_magnitude)

            # Set the threshold values based on a fraction of the median gradient
            thrs1 = 0.7 * median_gradient
            thrs2 = 1.3 * median_gradient

        else:
            thrs1, thrs2 = thrshold
        gray = cv2.cvtColor(mat_image, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize)
        vis = mat_image.copy()
        vis = np.uint8(vis/2.)
        vis[edge != 0] = (0, 255, 0)
    
    global canny_parameters
    canny_parameters = [thrs1, thrs2, apertureSize]
    
    return vis, edge
def cv_imread(filepath):
    cv_img = cv2.imdecode(np.fromfile(filepath,dtype=np.int8), -1) #cv2.imdecode(np.fromfile(filepath,dtype=np.unit8), -1)
    return cv_img
def number_detect(mat_image):
    tesseract_ind = 1
    tesseract_api_ind = 0

    if tesseract_ind == 1:
        if tesseract_api_ind == 0:
            # Example of adding any additional options
            
            custom_oem_psm_config = r'--oem 3 --psm 6 --tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'
            result = pytesseract.image_to_string(mat_image, config=custom_oem_psm_config)
        else:
            #height, width, depth = mat_imge.shape
            print(datetime.utcnow())
            tess = Tesseract()
            print("ocr image Start:{}".format(datetime.utcnow()))
            frame_piece = FramePiece(mat_image)
            result = tesseract_process_image2(tess, frame_piece)
    else:        
        reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
        result = reader.readtext(mat_image)
    return result
def circles_detect(mat_image, param = []):

    '''
    method	Detection method, see HoughModes. The available methods are HOUGH_GRADIENT and HOUGH_GRADIENT_ALT.
    dp	Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height. For HOUGH_GRADIENT_ALT the recommended value is dp=1.5, unless some small very circles need to be detected.
    minDist	Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    param1	First method-specific parameter. In case of HOUGH_GRADIENT and HOUGH_GRADIENT_ALT, it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller). Note that HOUGH_GRADIENT_ALT uses Scharr algorithm to compute image derivatives, so the threshold value should normally be higher, such as 300 or normally exposed and contrasty images.
    param2	Second method-specific parameter. In case of HOUGH_GRADIENT, it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first. In the case of HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure. The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine. If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less. But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.
    minRadius	Minimum circle radius.
    maxRadius	Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, HOUGH_GRADIENT returns centers without finding the radius. HOUGH_GRADIENT_ALT always computes circle radiuses.
    '''
    #circles = cv2.HoughCircles(mat_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=50)
    #reducing param2 until 9 circles are found
    global canny_parameters, configFile

    if configFile['circle_detection']['9points_calibration_ind'] == 1:
        maxRadiue = int(mat_image.shape[0]/8)        
    else:
        maxRadiue = 0
    
    if len(param) != 0:
        circles = cv2.HoughCircles(mat_image, cv2.HOUGH_GRADIENT, dp=1, minDist=int(mat_image.shape[0]/8), param1=param[0], param2=param[1], maxRadius=maxRadiue)
    elif len(canny_parameters) == 0:
        circles = cv2.HoughCircles(mat_image, cv2.HOUGH_GRADIENT, dp=1, minDist=int(mat_image.shape[0]/8), param1=50, param2=10, maxRadius=maxRadiue)
    else:
        circles = cv2.HoughCircles(mat_image, cv2.HOUGH_GRADIENT, dp=1, minDist=int(mat_image.shape[0]/8), param1=max(canny_parameters[0],canny_parameters[1]), param2=10, maxRadius=maxRadiue)

    #// smooth it, otherwise a lot of false circles may be detected
    #GaussianBlur( gray, gray, Size(9, 9), 2, 2 );
    #ksize = 0
    #sigmaX = 0.1
    #blurred_image = cv2.GaussianBlur(mat_image, (ksize, ksize), sigmaX)    
    #breakpoint()
    if circles is not None:
        if configFile['circle_detection']['outliers_exclude_ind'] == 1:
            outliers_index = find_outliers(circles[0,:,2])
            mask = np.ones(circles.shape, dtype=bool)
            mask[0,outliers_index,:] = False
            circles = circles[mask].reshape(circles.shape[0], circles.shape[1]- len(outliers_index), circles.shape[2])
        
        '''
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(mat_image, (x, y), r, (255, 255, 255), 2)
            cv2.circle(mat_image, (x, y), 2, (255, 255, 255), 3)
            cv2.putText(mat_image, f"Center: ({x}, {y})", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        '''
        

        if configFile['circle_detection']['9points_calibration_ind'] == 0:
            mat_image = draw_circles(mat_image, circles, (255, 255, 255))
            cv2.imshow("Circle Detection", mat_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return circles, mat_image

def draw_circles(mat_image, circles, color):
    circles = np.round(circles[0, :]).astype(int)
    for (x, y, r) in circles:
        cv2.circle(mat_image, (x, y), r, color, 2)
        cv2.circle(mat_image, (x, y), 2, color, 3)
        cv2.putText(mat_image, f"Center: ({x}, {y})", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return mat_image
STC_points_camera = np.array([[207,160],[311,159],[207,160],[311,159],[207,160],[311,159],[207,160],[311,159],[207,160]])
STC_points_robot = np.array([[207,160],[311,159],[207,160],[311,159],[207,160],[311,159],[207,160],[311,159],[207,160]])

class HandInEyeCalibration():

    def __init__(self, calibrate_ind=1, calibrated_m=None):
        if calibrate_ind == 1:
            self.m = self.get_m(np.asarray(STC_points_camera, dtype=np.int64), np.asarray(STC_points_robot, dtype=np.int64))
        else:
            self.m = calibrated_m

    def get_m(self, points_camrea, points_robot):
        m, _ = cv2.estimateAffine2D(points_camrea, points_robot)
        return m
    
    def get_points_robot(self, x_camera, y_camera):
        m = self.m #self.get_m(STC_points_camera, STC_points_robot)
        robot_x = (m[0][0] * x_camera) + (m[0][1] * y_camera) + m[0][2]
        robot_y = (m[1][0] * x_camera) + (m[1][1] * y_camera) + m[1][2]
        return robot_x, robot_y

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x,y)
        cv2.circle(mat_image, (x,y), 1, (255,0,0), thickness=-1)
        cv2.putText(mat_image, xy, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), thickness=1)
        cv2.imshow("image", mat_image)

def leave_roi_only(mat_image, rect):
    (x, y, w, h) = rect
    mask = np.zeros_like(mat_image)
    mask[y:y+h, x:x+w] = 255
    result = cv2.bitwise_and(mat_image, mask)
    return result

def get_robot_points(mat_image, camera_points):
    def display_image():
        cv2.imshow("Image", mat_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def process_input():
        global user_inputs
        user_inputs = []
        for entry in entries:
            #user_inputs.append(entry.get())

            user_inputs.append([float(entry[0].get()),float(entry[1].get())])
        #user_input = entry.get()
        #print("User input:", user_input);
        window.destroy()
        window.quit()
        #window.destroy()

    # Create a thread for displaying the image
    image_thread = threading.Thread(target=display_image)
    image_thread.start()
    #let user input corresponding robot coordinates
    # Create the window with text box using Tkinter
    window = tk.Tk()

    entries = []
    for idx, cam_point in enumerate(camera_points):
        #breakpoint()
        cam_point = np.round(cam_point).astype(int)
        # Create a label
        label = tk.Label(window, text="%d,%d" % (cam_point[0],cam_point[1]))        
        #label = tk.Label(window, text="testing")
        #label.pack()
        label.grid(row=idx,column=0)

        # Create a text box
        entry_x = tk.Entry(window)
        entry_x.grid(row=idx,column=1)
        entry_y = tk.Entry(window)
        entry_y.grid(row=idx,column=2)
        #entry.pack()

        # Add the Entry widget to the list
        #entries.append(entry)
        entries.append([entry_x,entry_y])

    '''
    # Create a label for instructions
    label = tk.Label(window, text="Enter your input:")
    label.pack()

    # Create a text box for user input
    entry = tk.Entry(window)
    entry.pack()
    '''
    

    # Create a button to process the input
    button = tk.Button(window, text="Submit", command=process_input)
    button.grid(row = len(camera_points), column=0,columnspan=3)
    #button.pack()

    # Start the GUI event loop
    window.mainloop()
    # Wait for the image thread to finish
    image_thread.join()
    #breakpoint()
    return user_inputs
'''
def camera_calibration(mat_image):

    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(window_name, on_EVENT_LBUTTONDOWN)
    cv2.imshow(window_name, mat)
    cv2.waitKey(0)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
'''
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
def threshold(mat_image, lower_range=[], upper_range =[]):

    if len(lower_range) == 0:
        

        def on_low_H_thresh_trackbar(val):
            global low_H
            global high_H
            low_H = val
            low_H = min(high_H-1, low_H)
            cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
        def on_high_H_thresh_trackbar(val):
            global low_H
            global high_H
            high_H = val
            high_H = max(high_H, low_H+1)
            cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
        def on_low_S_thresh_trackbar(val):
            global low_S
            global high_S
            low_S = val
            low_S = min(high_S-1, low_S)
            cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
        def on_high_S_thresh_trackbar(val):
            global low_S
            global high_S
            high_S = val
            high_S = max(high_S, low_S+1)
            cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
        def on_low_V_thresh_trackbar(val):
            global low_V
            global high_V
            low_V = val
            low_V = min(high_V-1, low_V)
            cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
        def on_high_V_thresh_trackbar(val):
            global low_V
            global high_V
            high_V = val
            high_V = max(high_V, low_V+1)
            cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)
        parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
        parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
        args = parser.parse_args()
        #cap = cv2.VideoCapture(args.camera)
        cv2.namedWindow(window_capture_name)
        cv2.namedWindow(window_detection_name)
        cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
        cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
        cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
        cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
        cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
        cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
        while True:
            
            #ret, frame = cap.read()
            #if frame is None:
                #break
            frame = mat_image
            HSV_ind = 1
            if HSV_ind == 1:
                frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            else:
                frame_HSV = frame
            frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            
            
            cv2.imshow(window_capture_name, frame)
            cv2.imshow(window_detection_name, frame_threshold)
            
            key = cv2.waitKey(30)
            if key == ord('q') or key == 27:
                lower_range = np.asarray((low_H, low_S, low_V))
                upper_range = np.asarray((high_H, high_S, high_V))
                mask = frame_threshold
                result = cv2.bitwise_and(mat_image,mat_image,mask = mask)
                cv2.destroyAllWindows() 
                break
    else:
        lower_range = np.array([0,0,0])  # Set the Lower range value of color in BGR
        upper_range = np.array([100,70,255])   # Set the Upper range value of color in BGR
        mask = cv2.inRange(mat_image,lower_range,upper_range) # Create a mask with range
        result = cv2.bitwise_and(mat_image,mat_image,mask = mask)  # Performing bitwise and operation with mask in img variable
    #breakpoint()
    return result, lower_range, upper_range
    '''
    -oem
    OCR Engine modes:
　　　　0 Legacy engine only.
　　　　1 Neural nets LSTM engine only.
　　　　2 Legacy + LSTM engines.
　　　　3 Default, based on what is available.

    -psm
    Page segmentation modes:
　　　　0 Orientation and script detection (OSD) only.
　　　　1 Automatic page segmentation with OSD.
　　　　2 Automatic page segmentation, but no OSD, or OCR.
　　　　3 Fully automatic page segmentation, but no OSD. (Default)
　　　　4 Assume a single column of text of variable sizes.
　　　　5 Assume a single uniform block of vertically aligned text.
　　　　6 Assume a single uniform block of text.
　　　　7 Treat the image as a single text line.
　　　　8 Treat the image as a single word.
　　　　9 Treat the image as a single word in a circle.
　　　　10 Treat the image as a single character.
　　　　11 Sparse text. Find as much text as possible in no particular order.
　　　　12 Sparse text with OSD.
　　　　13 Raw line. Treat the image as a single text line,
　　　　 bypassing hacks that are Tesseract-specific.
    '''
def image_preprocess(mat_image, config):
    for process_step in config['order']:
        match process_step:
            case 'BLUR':
                if config['blur_ind'] == 1:
                    ksize = config['ksize']
                    sigmaX = config['sigmaX']
                    mat_image = cv2.GaussianBlur(mat_image, (ksize, ksize), sigmaX)
            case 'MASK':
                if config['mask_ind'] == 1:
                    image_hsv = cv2.cvtColor(mat_image, cv2.COLOR_BGR2HSV)    
                    lower_range = np.asarray(config['mask_lower_range']) #np.array([0,0,0])  # Set the Lower range value of color in BGR
                    upper_range = np.asarray(config['mask_upper_range']) #np.array([100,70,255])   # Set the Upper range value of color in BGR
                    mask = cv2.inRange(mat_image,lower_range,upper_range) # Create a mask with range
                    mat_image = cv2.bitwise_and(mat_image,mat_image,mask = mask)  # Performing bitwise and operation with mask in img variable                    
            case 'GRAY':
                if config['gray_ind'] == 1:
                    mat_image = cv2.cvtColor(mat_image, cv2.COLOR_BGR2GRAY)
                    # seems no need to further trim but can further clean up
                    mat_image = cv2.threshold(mat_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]    
            case 'THRESHOLD':
                if config['threshold_ind'] == 1:
                    mat_image = threshold(mat_image)[0]
            case _:
                mat_image = mat_image
    
    return mat_image

def connect_client_socket(ip, port):
     # Create a TCP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the server
    client_socket.connect((ip, port))
    return client_socket

def connect_server_socket(ip, port):
     # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the server
    server_socket.bind((ip, port))
    # Listen for incoming connections
    server_socket.listen()

    while True:
        # Accept a client connection
        client_socket, client_address = server_socket.accept()
        print("Accepted connection from", client_address)

        # Create a new thread to handle the client
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()
    return server_socket

def handle_client(client_socket):
    # Receive data from the client
    received_data = client_socket.recv(1024).decode()
    print("Received from client:", received_data)

    # Send a response back to the client
    response = "Hello, client!"
    client_socket.sendall(response.encode())

    # Close the client socket
    client_socket.close()

def send_data_socket(client_socket, data, encoding="ascii"):    
    json_data = json.dumps(data, indent=4)
    client_socket.sendall(json_data.encode(encoding))

def receive_data_socket(client_socket, encoding="ascii"):
    json_data = client_socket.recv(1024).decode(encoding)
    received_data = json.loads(json_data)
    return received_data

def calc_distance(x,y):
    return np.linalg.norm(x - y)

def dist_to_line(line_st, line_end, point):

    line = line_end - line_st
    point_vector = point - line_st
    projection = np.dot(point_vector, line) / np.dot(line,line)
    projection_point = line_st + projection * line
    distance = calc_distance(point, projection_point)
    return distance

def sort_coordinates_by_position(coordinates, tol=0):

    sorted_coordinates = []
    remaining_points = coordinates

    while len(sorted_coordinates) != len(coordinates):
        
        #find the first row
        topleft = remaining_points[np.argmin(remaining_points[:,0]+remaining_points[:,1])]
        topright = remaining_points[np.argmax(remaining_points[:,0]-remaining_points[:,1])]
        f = lambda x, line_st=topleft, line_end=topright: dist_to_line(line_st, line_end, x)
        distance = np.apply_along_axis(f, axis=1, arr=remaining_points)
        row_points = np.sort(remaining_points[np.where(distance<=tol)], axis=0)
        sorted_coordinates.extend(row_points.tolist())
        remaining_points = remaining_points[np.where(distance>tol)]

    # Sort the coordinates based on both x-axis and y-axis
    #sorted_indices = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    #sorted_coordinates = coordinates[sorted_indices]
    return np.asarray(sorted_coordinates)

def update_yaml(yaml_data, json_data):
    for key, value in json_data.items():
        if isinstance(value, dict) and key in yaml_data and isinstance(yaml_data[key], dict):
            update_yaml(yaml_data[key], value)
        else:
            yaml_data[key] = value

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--config', default=None) 
    args = parser.parse_args()

    config_input = vars(args)['config']

    program_ts = time.time()
    program_ts_prev = program_ts

    global configFile 
    configFile = yaml.safe_load(open("config.yml"))
    if config_input != None:
        json_data = json.loads(config_input.replace("'", "\""))
        update_yaml(configFile, json_data)
        #configFile.update(json_data)    
    #Reading configs
    number_detect_ind = configFile['runsetting']['number_detect_ind']
    pytesseract.pytesseract.tesseract_cmd = 'r'+configFile['tesseract']['tesseract_exe_path'] #r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    testdata_dir_config = '--tessdata-dir ' + configFile['tesseract']['testdata_dir'] #'--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'
    source_type = configFile['runsetting']['source_type']

    log = []
    resultFolder = configFile['runsetting']['run_no']
    if os.path.isdir(resultFolder):
        shutil.rmtree(resultFolder)
    if not os.path.isdir(resultFolder):
        os.makedirs(resultFolder)

    if source_type == 1:
        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename()
        #breakpoint()
        #filename = 'Image2.jpg'
        #filename = 'v1sample.png'
    else:
        filename = configFile['runsetting']['source_path']
    mat_image = cv_imread(filename)
    #mat_image = cv2.imread(filename)

    mat_image = image_preprocess(mat_image, configFile['image_processing']) 

    if configFile['runsetting']['edge_detection_ind'] == 1:
        # number_detect(gray)
        #find_contours(gray)   
        if configFile['edge_detection']['calibration_ind'] == 1:            
            thrshold = []
        else:
            thrshold = np.asarray(configFile['edge_detection']['canny_threshold']) #[2000, 4000]
        mat_edge = find_edge(mat_image, thrshold=thrshold)[1]
        mat_contour, rects = find_contours(mat_edge)

    #breakpoint()
    if configFile['roi_detection']['largest_rect_ind'] == 1:
        rects_np = np.asarray(rects)
        # find latest ROI
        (x, y, w, h) = rects[np.argmax(rects_np[:,3])]
        roi = mat_edge[y:y+h, x:x+w]
        result = leave_roi_only(mat_edge, (x, y, w, h))
    
    #roi = mat_edge[y:y+h, x:x+w]
    #breakpoint()
    if configFile['runsetting']['circle_detection_ind'] == 1:
        trial_no = 0
        no_circles = 0
        param = []
        param_trial = []
        no_circle_trial = []

        if configFile['circle_detection']['blur_ind'] == 1:
            ksize = configFile['circle_detection']['blur_kernel_size']
            sigmaX = configFile['circle_detection']['blur_sigmaX']
            roi = cv2.GaussianBlur(roi, (ksize, ksize), sigmaX)
            
        while True:            
            circles, image_w_circles = circles_detect(roi, param)
            if circles is None:
                break
            if configFile['circle_detection']['9points_calibration_ind'] == 0 or len(circles[0]) == 9 or trial_no > configFile['circle_detection']['9points_calibration_max_trial_no']:
                break
            else:
                trial_no += 1
                              

                no_circles_last = no_circles    
                            
                no_circles = len(circles[0])

                
                if trial_no == 1:
                    if len(canny_parameters) != 0:
                        param = [max(canny_parameters[0],canny_parameters[1]), configFile['circle_detection']['param2']]                        
                    else:
                        param = [configFile['circle_detection']['param1'], configFile['circle_detection']['param2']]
                else:                                                            
                    param_trial.append(param[1])
                    no_circle_trial.append(no_circles)
                    if trial_no > 2:
                        if no_circles_last != no_circles:
                            slope = (param[1] - param_last)  / (no_circles - no_circles_last)
                        param_next = param[1] - slope * (no_circles - 9)
                        if param_next < 0:
                            breakpoint()
                    else:
                        if no_circles > 9: #too much circles recognised, Lowering param1 allows more circles to be detected, but can also result in false detections., Higher values of param2 will result in fewer circles being detected but with higher confidence.
                            param_next = param[1] * 2 # 1.1
                        else:
                            param_next = param[1] * 0.5 #0.9
                            
                    param_last = param[1]
                    param[1] = param_next
                    
                
            
        circles[:,:,0] = circles[:,:,0]+x
        circles[:,:,1] = circles[:,:,1]+y
        #mat_image[y:y+h, x:x+w] = cv2.cvtColor(image_w_circles, cv2.COLOR_GRAY2BGR)
        mat_image = draw_circles(mat_image, circles, (255,255,255))

        if configFile['circle_detection']['show_detected_circle_ind'] == 1:
            show_image(mat_image)
        cv2.imwrite(f"{resultFolder}/detected_circles.jpg", mat_image)
        #breakpoint()

    if configFile['runsetting']['robot_connection_ind'] == 1:
        ip = configFile['robot']['ip']
        port = configFile['robot']['port']
        tcp_type = configFile['robot']['tcp_type']

        if tcp_type  == "CLIENT":
            robot_socket = connect_client_socket(ip, port)
            #send data
            data = {"cmdType": "query", "dsID": "HCRemoteMonitor", "queryAddrr": ["Version"]}
            send_data_socket(robot_socket, data)            
            #receive data
            received_data = receive_data_socket(robot_socket, encoding="ascii")
            print(received_data)
        else:
            connect_server_socket(ip, port)
                        
    if configFile['runsetting']['camera_calibration_ind'] == 1:
        #9 points detected from circles

        coordinates = circles[0,:,:2]  
        # Sort the coordinates based on both x-axis and y-axis
        sorted_indices = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
        sorted_coordinates = coordinates[sorted_indices]        

        STC_points_camera = sort_coordinates_by_position(circles[0,:,:2], tol=np.average(circles[0,:,2])) #sorted_coordinates #circles[0,:,:2]

        df = pd.DataFrame(STC_points_camera)
        df.to_csv(resultFolder+'/camera_points.csv', index=False)

        #breakpoint()
        if configFile['camera_calibration']['robot_coordinates_input_type'] == 1:
            source = configFile['camera_calibration']['robot_coordinates_path'] 
            df = pd.read_csv(source, index_col=False, header=None)
            # Convert the pandas DataFrame to a NumPy array
            STC_points_robot = df.to_numpy()#sort_coordinates_by_position(df.to_numpy())       
        else:
            STC_points_robot = np.asarray(get_robot_points(mat_image, STC_points_camera))    
        
        HECalibration = HandInEyeCalibration()
        df = pd.DataFrame(HECalibration.m)
        df.to_csv(resultFolder+'/robot_m.csv', index=False)        

    if configFile['runsetting']['robot_coordination_ind'] == 1:        
        if configFile['runsetting']['camera_calibration_ind'] == 0:           
            m = pd.read_csv(configFile['camera_calibration']['transform_parameters_path'] + '/robot_m.csv')
            HECalibration = HandInEyeCalibration(calibrate_ind=0,calibrated_m=m.values)

        if configFile['robot_coordination_ind']['source_type'] == 0: #manual input
            coordinate = configFile['robot_coordination_ind']['coordinate']
            robot_coordinate = HECalibration.get_points_robot(coordinate[0],coordinate[1])
            print(robot_coordinate)
    #circles_detect(result)
    
    
    if number_detect_ind == 1:
        gray = cv2.cvtColor(mat_image, cv2.COLOR_BGR2GRAY)
        # seems no need to further trim but can further clean up
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]   
        #gray = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX)
        #gray  = cv2.medianBlur(gray, ksize)

        number_detect_array = []
        for idx_rect, rect in enumerate(rects):
            (x, y, w, h) = rect
            roi = mat_image[y:y+h, x:x+w]
            #cv2.imwrite(f"output/cropped_contour_{idx_rect}.jpg", roi)
            text = number_detect(gray[y:y+h, x:x+w])
            #text = number_detect(mat_edge[y:y+h, x:x+w])
            #text = number_detect(cv2.cvtColor(mat_edge[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY))

            if len(text) > 0:
                cv2.imwrite(f"output/cropped_contour_{idx_rect}.jpg", roi)
                number_detect_array.append(text)
                cv2.imwrite(f"output/cropped_contour_gray_{idx_rect}.jpg", gray[y:y+h, x:x+w])
                cv2.imwrite(f"output/cropped_edge_{idx_rect}.jpg", mat_edge[y:y+h, x:x+w])
            #number_detect_array.append(number_detect(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
            #cv2.imwrite(f"output/cropped_contour_gray2_{idx_rect}.jpg", cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

            program_ts = time.time()        
            log.append(program_ts - program_ts_prev)
            program_ts_prev = program_ts

    program_ts = time.time()
    log.append(program_ts - program_ts_prev)
    program_ts_prev = program_ts

    with open(resultFolder+'/runlog.txt', 'w') as f:
        for no, line in enumerate(log):
            if no > 0:
                # Construct the line content:
                line_content = line
                f.write(f"{line_content}\n")    
        