#!/usr/bin/env python
# -*- coding: utf-8 -*-

from audioop import reverse
import numpy as np
import cv2, random, math, copy

Width = 640
Height = 480
#버드아이 뷰를 이용하면 차선이 보이지 않음
cap = cv2.VideoCapture("../subProject.avi")
#cap = cv2.VideoCapture("xycar_track1.mp4")
window_title = 'camera'

warp_img_w = 320
warp_img_h = 240

warpx_margin = 45
warpy_margin = 3

nwindows = 20
margin = 50
minpix = 5

lane_bin_th = 2

warp_src  = np.array([
    [230-warpx_margin, 300-warpy_margin],  
    [45-warpx_margin,  450+warpy_margin],
    [445+warpx_margin, 300-warpy_margin],
    [610+warpx_margin, 450+warpy_margin]
], dtype=np.float32)

warp_dist = np.array([
    [0,0],
    [0,warp_img_h],
    [warp_img_w,0],
    [warp_img_w, warp_img_h]
], dtype=np.float32)

calibrated = True
if calibrated:
    mtx = np.array([
        [422.037858, 0.0, 245.895397], 
        [0.0, 435.589734, 163.625535], 
        [0.0, 0.0, 1.0]
    ])
    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))

def calibrate_image(frame):
    global Width, Height
    global mtx, dist
    global cal_mtx, cal_roi
    
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]

    return cv2.resize(tf_image, (Width, Height))

#warp img 가 너무 작음
def warp_image(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv

def warp_process_image(img):
    global nwindows
    global margin
    global minpix
    global lane_bin_th

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    m = cv2.mean(gray)[0]
    gray = cv2.add(gray,(20 - m)) # 평균 밝기 50으로 고정
    gray = cv2.GaussianBlur(gray,(5, 5), 0)

    # 차체 보이는거 제외하는 원 추가
    cv2.circle(gray, (155, 250),60, 255, -1)
    cv2.circle(gray, (10, 250),40, 255, -1)
    cv2.circle(gray, (315, 250),50, 255, -1)
    _, lane = cv2.threshold(gray, lane_bin_th, 255, cv2.THRESH_BINARY)

    lane = 255-lane # 영상 반전

    histogram = np.sum(lane[180:220,:], axis=0) 

    midpoint = np.int(histogram.shape[0]/2)

    print(max(histogram[:midpoint]), "  ///  ",max(histogram[midpoint:]))

    hist_threshold = 2200

    if max(histogram[:midpoint]) < hist_threshold:
        leftx_current = 0 
    else:
        leftx_current = np.argmax(histogram[:midpoint])

    if max(histogram[midpoint:]) < hist_threshold:
        rightx_current = 320
    else:
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint


    print(leftx_current,rightx_current )

    window_height = np.int(lane.shape[0]/nwindows)
    nz = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []
    
    lx, ly, rx, ry = [], [], [], []

    #out_img = np.dstack((lane, lane, lane))*255

    for window in range(nwindows):

        win_yl = lane.shape[0] - (window+1)*window_height
        win_yh = lane.shape[0] - window*window_height

        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin

        cv2.rectangle(img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2) 
        cv2.rectangle(img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2) 

        good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nz[1][good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nz[1][good_right_inds]))

        lx.append(leftx_current)
        ly.append((win_yl + win_yh)/2)

        rx.append(rightx_current)
        ry.append((win_yl + win_yh)/2)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    #left_fit = np.polyfit(nz[0][left_lane_inds], nz[1][left_lane_inds], 2)
    #right_fit = np.polyfit(nz[0][right_lane_inds] , nz[1][right_lane_inds], 2)
    
    lfit = np.polyfit(np.array(ly),np.array(lx),2)
    rfit = np.polyfit(np.array(ry),np.array(rx),2)

    img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
    img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
    cv2.imshow("img", img)
    cv2.imshow("lane", lane)
    
    #return left_fit, right_fit
    return lfit, rfit

def draw_lane(image, warp_img, Minv, left_fit, right_fit):
    global Width, Height
    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(warp_img).astype(np.uint8)
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) 
    pts = np.hstack((pts_left, pts_right))
    
    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))

    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)

def start():
    global Width, Height, cap

    _, frame = cap.read()
    while not frame.size == (Width*Height*3):
        _, frame = cap.read()
        continue

    print("start")

    while cap.isOpened():
        
        _, frame = cap.read()

        #image = calibrate_image(frame)
        image = frame
        warp_img, M, Minv = warp_image(image, warp_src, warp_dist, (warp_img_w, warp_img_h))
        left_fit, right_fit = warp_process_image(warp_img)
        lane_img = draw_lane(image, warp_img, Minv, left_fit, right_fit)
        cv2.imshow(window_title, lane_img)

        cv2.waitKey(1)
        # if cv2.waitKey(0) != 33:
        #     pass

if __name__ == '__main__':
    start()
