#!/usr/bin/env python

'''
RESOURCES: 
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

'''


import numpy as np
import os
import glob
import random
import collections

from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import float_info

class CropUtil(object):
   
    def r(self, image):
        #image = cv2.imread('2.jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # ret3,otsu = cv2.threshold(blurred,130,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        adap_th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        # invert image to get img
        # img_binary = cv2.bitwise_not(adap_th)
        
        # show the images
        return adap_th
    
    def transform(self, pos):
    # This function is used to find the corners of the object
    # and the dimensions of the object
        pts=[]
        n=len(pos)
        for i in range(n):
            pts.append(list(pos[i][0]))
            
        sums={}
        diffs={}
        tl=tr=bl=br=0
        for i in pts:
            x=i[0]
            y=i[1]
            sum=x+y
            diff=y-x
            sums[sum]=i
            diffs[diff]=i
        sums=sorted(sums.items())
        diffs=sorted(diffs.items())
        n=len(sums)
        rect=[sums[0][1],diffs[0][1],diffs[n-1][1],sums[n-1][1]]
        #	   top-left   top-right   bottom-left   bottom-right
        
        h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2)		#height of left side
        h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2)		#height of right side
        h=max(h1,h2)
        
        w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)		#width of upper side
        w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)		#width of lower side
        w=max(w1,w2)
        return int(w),int(h),rect

    def rotate(self, img, contours):
        n=len(contours)
        print('In rotate, total contours: {0}'.format(n))
        max_area = 0
        min_area = 0
        minC = 0
        maxC = 0
        i = 0
        for c in contours:
            i = i + 1
            area=cv2.contourArea(c)
            if n == 1:
                pos = c
                break
            # print('{0}: area: {1}'.format(i, area))
            if area > max_area:
                min_area = max_area
                max_area = area
                maxC = c                
            else:
                min_area=area
                minC = c
            
            perc = min_area * 100 / max_area
            if perc < 20 or perc >= 80:                               
                pos = maxC
            if perc > 50 and perc < 80:
                pos = minC
            
                
        peri=cv2.arcLength(pos,True)
        approx=cv2.approxPolyDP(pos,0.02*peri,True)
        area = cv2.contourArea(pos)
        if n > 1:
            print('min_area: {0}, max_area: {1}, PERC: {2}, AREA: {3} '.format(min_area, max_area, perc, area)) 
        # size=img.shape
        w,h,arr=self.transform(approx)
        # print('Width: {0}, height: {1}'.format(int(w), int(h)))
        pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts1=np.float32(arr)
        M=cv2.getPerspectiveTransform(pts1,pts2)
        image=cv2.warpPerspective(img,M,(w,h))
        image = cv2.resize(image,(w,h),interpolation = cv2.INTER_AREA)
        return image

    
    def crop(self, img):
        # make copy of image
        original_image = np.copy(img)
        image = self.r(img)
        # find contours
        # cv2.RETR_LIST – retrieves all the contours.
        # cv2.RETR_EXTERNAL – retrieves external or outer contours only.
        # cv2.RETR_CCOMP – retrieves all in a 2-level hierarchy.
        # cv2.RETR_TREE – retrieves all in a full hierarchy.
        image, img_contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # find contours of large enough area
        large_contours = sorted(img_contours, key = cv2.contourArea, reverse = True)[:5]
        if len(large_contours) == 0:
            print('CONTOURS COUNT: {0}'.format(len(large_contours)))
            return img
            
        min_id_area = 20000
        large_contours = [cnt for cnt in large_contours if cv2.contourArea(cnt) > min_id_area]

        idContours = []
        # loop over our contours
        for cnt in large_contours:
            accuracy=0.020*cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,accuracy,True)
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(box)
            leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
         
            if leftmost[0] > 3 and leftmost[1] > 3 and topmost[0] > 3 and topmost[1] > 3:
                idContours.append(box)
            else:
                if leftmost[0] <= 3 and leftmost[1] <= 3 and topmost[0] <= 3 and topmost[1] <= 3 and len(large_contours) > 1:
                    # print('SKIPPED, IMAGE SHAPE: {0}, left: {1}, right: {2}, top: {3}, bottom: {4} '.format(img.shape, leftmost, rightmost, topmost, bottommost))                                    
                    i = 0
                else:
                    # print('APPLIED2, IMAGE SHAPE: {0}, left: {1}, right: {2}, top: {3}, bottom: {4} '.format(img.shape, leftmost, rightmost, topmost, bottommost))                
                    idContours.append(box)
                
        large_id_contours = sorted(idContours, key = cv2.contourArea, reverse = True)[:2]

        result = self.rotate(original_image, large_id_contours)
        # cv2.drawContours(img_and_contours, large_id_contours, -1, (0,255,0), 3)
        result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        return result

    
    # @staticmethod
    def cropImagesAtPath(self, path_to_data):
        print("In CropUtil, cropImagesAtPath: {0}".format(path_to_data))
        result = []
        folders = []
        files = []
        for f in glob.glob(path_to_data):
            if os.path.isdir(f):
                folders.append(os.path.abspath(f))
            else:
                files.append(os.path.abspath(f))

        print("Folders (classes) found: %s " % [os.path.split(x)[1] for x in folders])
        for folder in folders:
            print("Folder %s:" % (folder))
            for f in glob.glob(folder+"/*"):
                if os.path.isfile(f):
                    files.append(os.path.abspath(f))
            print("\n----------------------------\n")

        print('Total Images: >> {0} '.format(len(files)))
        # print("Files found: %s " % [os.path.split(x)[1] for x in files])
        
        try:
            for f in files:
                img = cv2.imread(f)
                img = self.crop(img)
                result.append(img)                                                      
        except Exception as e:
            print("ERROR in cropImagesAtPath: {0}".format(e))
        
        return result

    # @staticmethod
    def cropImages(self, images):
        print("In CropUtil, cropImages, total count: {0} ".format(len(images)))
        result = []
        try:
            for img in images:
                img = self.crop(img)
                result.append(img)
                # cv2.imwrite('./results/test.jpg', img)
        except Exception as e:
            print("ERROR in cropImages: {0}".format(e))
        
        return result
    

        