#!/usr/bin/env python

import Augmentor
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

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.datasets import mnist

class ImageUtil(object):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        print(CONFIG)

    # @staticmethod
    def augmentation(self, path_to_data):
        print("In ImageClassifier, augmentation: {0}".format(path_to_data))
        folders = []
        for f in glob.glob(path_to_data):
            if os.path.isdir(f):
                folders.append(os.path.abspath(f))

        print("Folders (classes) found: %s " % [os.path.split(x)[1] for x in folders])

        pipelines = {}
        for folder in folders:
            print("Folder %s:" % (folder))
            pipelines[os.path.split(folder)[1]] = (Augmentor.Pipeline(folder))
            print("\n----------------------------\n")
        
        for p in pipelines.values():
            print("Class %s has %s samples." % (p.augmentor_images[0].class_label, len(p.augmentor_images)))
            

        try:
            num_of_samples = int(1000)
            print(folders[0])
            path_to_data = folders[0]
            p = Augmentor.Pipeline(path_to_data)
            # Add some operations to an existing pipeline.
            # First, we add a horizontal flip operation to the pipeline:
            p.flip_left_right(probability=0.4)
            # Now we add a vertical flip operation to the pipeline:
            p.flip_top_bottom(probability=0.8)
            # Add a rotate90 operation to the pipeline:
            p.rotate90(probability=0.1)
            p.zoom_random(probability=0.5, percentage_area=0.8)
            # Now we can sample from the pipeline:
            # p.sample(num_of_samples)
            p.flip_top_bottom(probability=0.5)
            p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
            # Now we can sample from the pipeline:
            # p.sample(num_of_samples)

            p2 = Augmentor.Pipeline(path_to_data)
            p2.rotate90(probability=0.5)
            p2.rotate270(probability=0.5)
            p2.flip_left_right(probability=0.8)
            p2.flip_top_bottom(probability=0.3)
            p2.crop_random(probability=1, percentage_area=0.5)
            p2.resize(probability=1.0, width=320, height=320)
            # p2.sample(num_of_samples)
            
        except Exception as e:
            print("Unable to run augmentation: {0}".format(e))


    def get_angle(self, calibration_image):
        """
        :param calibration_image: The HSV-image to use for calculation
        :return: Rotation angle of the field in image
        """
        # TODO: correct return value comment?
        rgb = cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR)
        angle = 0
        count = 0

        gray = cv2.cvtColor(cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi/180, 110)

        if lines.shape[0]:
            line_count = lines.shape[0]
        else:
            raise Exception('field not detected')

        for x in range(line_count):

            for rho, theta in lines[x]:
                a = np.cos(theta)
                b = np.sin(theta)
                # print(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*a)
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*a)

                corr_angle = np.degrees(b)
                if corr_angle < 5:
                    # print(CorrAngle)
                    angle = angle + corr_angle
                    count = count + 1
                    cv2.line(rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
        print(angle)
        if isinstance(angle, int) and isinstance(count, int):
            angle = angle / count
            self.angle = angle
            return angle
        else:
            self.angle = 0.1
            return False 

    def auto_canny(self, image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged


    def r(self, image):
        #image = cv2.imread('2.jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # blurred = cv2.GaussianBlur(gray, (1,1), 1)
        # edged = cv2.Canny(img_preprocessed, 10, 250)        
        # median = cv2.medianBlur(img,5)
        # blurred = cv2.bilateralFilter(gray,9,75,75)
        # wide = cv2.Canny(blurred, 10, 50)
        # tight = cv2.Canny(blurred, 225, 250)
        # auto = self.auto_canny(blurred)

        ret3,otsu = cv2.threshold(blurred,130,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        adap_th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        # invert image to get img
        img_binary = cv2.bitwise_not(adap_th)
        
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
        # image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # image = cv2.adaptiveThreshold(dst,255,1,0,11,2)
        image = cv2.resize(image,(w,h),interpolation = cv2.INTER_AREA)
        return image

    
    def crop(self, img):
        # coins_gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
        # coins_preprocessed = cv2.GaussianBlur(coins_gray, (5, 5), 0)
        # img = cv2.resize(img,(640,320),interpolation = cv2.INTER_AREA)
        image = self.r(img)
        # find contours
        # cv2.RETR_LIST – retrieves all the contours.
        # cv2.RETR_EXTERNAL – retrieves external or outer contours only.
        # cv2.RETR_CCOMP – retrieves all in a 2-level hierarchy.
        # cv2.RETR_TREE – retrieves all in a full hierarchy.
        image, img_contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # make copy of image
        img_and_contours = np.copy(img)

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
            # print(len(approx))
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(box)
            # hull = cv2.convexHull(box)
            # hull_area = cv2.contourArea(hull)
            # solidity = float(area)/hull_area
            # print("CONTOUR SOLIDITY: {0} ".format(solidity))
            # contour_perc = (area * 100) / img.size            
            # (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
            # print('CONTOUR ANGLE: {0} '.format(angle))
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
            
            # if leftmost[0] > 3 and leftmost[1] > 3 and topmost[0] > 3 and topmost[1] > 3:
            #     # print('APPLIED1, IMAGE SHAPE: {0}, left: {1}, right: {2}, top: {3}, bottom: {4} '.format(img.shape, leftmost, rightmost, topmost, bottommost))                
            #     idContours.append(box)
            # else:
            #     if len(large_contours) == 1:
            #         idContours.append(box)
            #         # print('SKIPPED, IMAGE SHAPE: {0}, left: {1}, right: {2}, top: {3}, bottom: {4} '.format(img.shape, leftmost, rightmost, topmost, bottommost))                                    
                
        # print('ID CONTOURS: >>  {0}'.format(len(idContours)))
        large_id_contours = sorted(idContours, key = cv2.contourArea, reverse = True)[:2]

        result = self.rotate(img, large_id_contours)
        
        # draw contours
        # mask = np.zeros_like(img_and_contours)
        cv2.drawContours(img_and_contours, large_id_contours, -1, (0,255,0), 3)
        # self.transform(img_and_contours[0])
        # out = np.zeros_like(img_and_contours) # Extract out the object and place into output image
        # out[mask == 255] = img[mask == 255]
        
        # print number of contours
        # print('number of ID countours: %d' % len(large_id_contours))
        return result

    
    # @staticmethod
    def cropImagesAtPath(self, path_to_data):
        print("In ImageUtil, cropImagesAtPath: {0}".format(path_to_data))
        result = []
        folders = []
        for f in glob.glob(path_to_data):
            if os.path.isdir(f):
                folders.append(os.path.abspath(f))

        print("Folders (classes) found: %s " % [os.path.split(x)[1] for x in folders])
        pipelines = {}
        for folder in folders:
            print("Folder %s:" % (folder))
            if os.path.split(folder)[1] == 'pan':
                pipelines[os.path.split(folder)[1]] = (Augmentor.Pipeline(folder))
                print("\n----------------------------\n")
        
        for p in pipelines.values():
            try:
                for aug_img in p.augmentor_images:
                    img = cv2.imread(aug_img.image_path)
                    img = self.crop(img)
                    result.append(img)
                    # print("Class %s has %s samples." % (p.augmentor_images[0].class_label, len(p.augmentor_images)))
                    # print(p.augmentor_images[0].image_path)                                        
            except Exception as e:
                print("Unable to run cropImage: {0}".format(e))
        
        return result

    # @staticmethod
    def cropImages(self, images):
        print("In ImageUtil, cropImages, total count: {0} ".format(len(images)))
        result = []
        try:
            print('PROCESS START....') 
            for img in images:
                img = self.crop(img)
                result.append(img)
                '''
                img = cv2.imread(p.augmentor_images[0].image_path)
                ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
                ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
                ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
                ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
                titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
                images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

                for i in range(6):
                    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
                    plt.title(titles[i])
                    plt.xticks([]),plt.yticks([])

                plt.show()
                '''
                # cv2.imwrite('./results/test.jpg', img)
        except Exception as e:
            print("Unable to run cropImage: {0}".format(e))
        
        return result
    

        