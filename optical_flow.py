import math
import os

import numpy as np
import cv2 as cv

#Camera matrix
K = np.identity(3)
K[0, 2] = 1280 / 2.0
K[1, 2] = 960 / 2.0
K[0, 0] = K[1, 1] = 1280 / (2.0 * np.tan(60 * np.pi / 360.0))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors for the optical flow tracking
color = np.random.randint(0, 255, (100, 3))

#from the first frame choose the features
old_frame = cv.imread('D:\cv2feature\sztaki\images\\train\\00001.png')
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask = np.zeros_like(old_frame)

#going through all the pictures in pairs, enumerating to only calculate R, t for the xth picture
picfolder="D:\cv2feature\sztaki\images\\train"
#rot and trans calculation FPS
div = 10

for u, (old_picname, curr_picname) in enumerate(zip(os.listdir(picfolder)[::], os.listdir(picfolder)[1::])):
    #first frame is the old frame
    old_frame = cv.imread(picfolder+'/'+old_picname)
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    #secound frame is the new frame
    frame = cv.imread(picfolder+'/'+curr_picname)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Calculating points, and status bits
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    #not using feautres that are missing from the screen
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        #deleting older points from Rt_clac which is decleared in the last iterations
        if 'Rt_calc' in locals():
            for number, error in enumerate(st):
                if error == 0:
                    Rt_calc = np.concatenate((Rt_calc[:number], Rt_calc[number+1:]))

    #calculating essential matrix
    if u % div == div-1:
        E = cv.findEssentialMat(good_new[:], Rt_calc[:], cameraMatrix=K)[0]
        R1, R2, t = cv.decomposeEssentialMat(E)
        R, _ =cv.Rodrigues(R1)
        theta = math.sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2])
        v = R/theta
        #print(v, theta*180/3.14)
        P1 = np.concatenate((K, [[0],[0],[0]]), axis=1)
        P2 = np.dot(K, np.concatenate((R1, t), axis=1))


        X = cv.triangulatePoints(P1, P2, good_new[0], Rt_calc[0])
        X /= X[3]
        print(X)
    #drawing lines on the iamge
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)

    img = cv.add(frame, mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    #saving coordinates to use for R and t
    if u % div == 0:
        Rt_calc = good_old

    #making old good_new into new p0
    p0 = good_new.reshape(-1,1,2)