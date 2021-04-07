import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time


MIN_AREA1=600
MAX_AREA1 = 2000

MIN_AREA2=600
MAX_AREA2 = 2000


GRADE_BOX_MIN_AREA=100000
GRADE_BOX_MAX_AREA = 300000
GRADE_BOX_MIN_RATIO = 8
DEBUG = False
DEBUG_LVL2 = False  
path = 'pallet pics'
a=0

def find_defects(gray,grade_box_min_area=GRADE_BOX_MIN_AREA,
                     grade_box_max_area=GRADE_BOX_MAX_AREA,
                     grade_box_min_ration=GRADE_BOX_MIN_RATIO,
                     min_area1=MIN_AREA1,
                     max_area1=MAX_AREA1,
                     min_area2=MIN_AREA2,
                     max_area2=MAX_AREA2,
                     min_pellet_box_count=4 ):
    global a
    gray = np.array(gray).astype(np.uint8)
    
    blur = cv2.blur(gray, (21,21))


    if DEBUG:
        cv2.imshow('gray', cv2.resize(gray, None, fx=0.2, fy=0.2) )
        cv2.waitKey(0)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    if DEBUG:
        cv2.imshow('gray', cv2.resize(gray, None, fx=0.2, fy=0.2) )
        cv2.waitKey(0)




    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)
    if DEBUG:
        cv2.imshow('thresh1_', cv2.resize(thresh1 , None, fx=0.2, fy=0.2) )
        cv2.waitKey(0)

    thresh1 = cv2.erode(thresh1, np.ones((1,31)) , iterations=3 )
    thresh1 = cv2.dilate(thresh1, np.ones((3,3)) , iterations=30 )
    thresh1 = cv2.erode(thresh1, np.ones((3,3)) , iterations=10 )
    

    if DEBUG:
        cv2.imshow('thresh1_', cv2.resize(thresh1 , None, fx=0.2, fy=0.2) )
        cv2.waitKey(0)

    h,w = thresh1.shape[:2]
    sum_cols = np.sum(thresh1, axis=1)
    sum_cols = sum_cols/255
    thresh_rows = np.where(sum_cols>w*0.6)[0]
    thresh1[thresh_rows,:]=255

    if DEBUG:
        cv2.imshow('thresh1_', cv2.resize(thresh1 , None, fx=0.2, fy=0.2) )
        cv2.waitKey(0)

    h,w = thresh1.shape[:2]
    sum_rows = np.sum(thresh1, axis=0)
    sum_rows = sum_rows/255
    thresh_cols = np.where(sum_rows>w*0.7)[0]
    thresh1[:,thresh_cols]=255



    
    if DEBUG:
        cv2.imshow('thresh1_', cv2.resize(thresh1 , None, fx=0.2, fy=0.2) )
        cv2.waitKey(0)

    cnts,_= cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    grade_box_mask = np.zeros_like(gray)
    grade_box=[]
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        _,(w,h),_= rect
        if DEBUG_LVL2:
            debug_mask = np.zeros_like(gray)
            cv2.drawContours(debug_mask, [box], 0, 255, thickness=-1)
            print('area',area)
            print('aspect_ratio(h/w)=', h/w,'   aspect_ratio(h/w)=', w/h)
            print('-'*20)
            cv2.imshow('debug_GradeBox_mask', cv2.resize(debug_mask, None, fx=0.2,fy=0.2))
            cv2.waitKey(0)
        
        if grade_box_min_area<area<grade_box_max_area and (h/w>grade_box_min_ration or w/h>grade_box_min_ration):
            cv2.drawContours(grade_box_mask, [box], 0, 255, thickness=-1)
            grade_box.append(box)
    
    #CROP------------------------------------
    border=100
    grade_box_ = np.array(grade_box)
    min_x_= grade_box_[:,:,0].min()
    min_y_= grade_box_[:,:,1].min()
    max_x_= grade_box_[:,:,0].max()
    max_y_= grade_box_[:,:,1].max()

    max_x_ = min( max_x_ + border , gray.shape[1])
    max_y_ = min( max_y_ + border , gray.shape[0])
    min_x_ = max( min_x_ - border ,0)
    min_y_ = max( min_y_ - border ,0)
    #CROP------------------------------------
    
    valid = 0
    if len(grade_box)>=min_pellet_box_count:
        valid = 1

            
            
    es = cv2.erode(grade_box_mask, np.ones((3,3)) , iterations=1 )      



    if DEBUG:
        cv2.imshow('grade_box_mask', cv2.resize(grade_box_mask , None, fx=0.2, fy=0.2) )
        cv2.waitKey(0)


    pellet_car_gray = cv2.bitwise_and(gray, gray, mask = grade_box_mask)
    pellet_car_gray = cv2.equalizeHist(pellet_car_gray)

    thresh2 = cv2.adaptiveThreshold(pellet_car_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 11)
    res = cv2.bitwise_and(thresh2, cv2.bitwise_not(thresh1))
    res = cv2.erode(res, np.ones((5,1)) , iterations=1 )
    res = cv2.dilate(res, np.ones((5,1)) , iterations=2 )
    res = cv2.erode(res, np.ones((5,1)) , iterations=1 )

    if DEBUG:
        cv2.imshow('res', cv2.resize(res , None, fx=0.2, fy=0.2) )
        cv2.imshow('pellet_car_gray', cv2.resize(pellet_car_gray , None, fx=0.2, fy=0.2) )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #res = np.zeros_like(gray)
    cnts,_ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_defects=[[min_x_, min_y_, max_y_ - min_y_, max_x_ - min_x_ ,valid]]
    for cnt in cnts:
        _,(w,h),_ = cv2.minAreaRect(cnt)
        area = w*h
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        cx = int(M['m10']/(M['m00']+0.001))
        cy = int(M['m01']/(M['m00']+0.001))
        if DEBUG_LVL2:
            debug_grade = np.copy(gray)
            cv2.drawContours(debug_grade, [cnt], 0, 255, thickness=-1)
            cv2.imshow('debug_grade_hofre', cv2.resize(debug_grade, None, fx=0.4, fy=0.4) )
            cv2.waitKey(0)
        if min_area1<=area<max_area1:
            result_defects.append([cx-min_x_, cy-min_y_ ,int(min(h,w)),1,valid])
        
        elif min_area2<=area<max_area2:
            result_defects.append([cx-min_x_, cy-min_y_ ,int(min(h,w)),2,valid])
    
    return np.array(result_defects, dtype=np.int32)
                     










for i,file in enumerate (os.listdir(path)):
    print('the image is ', file)
    #if '183' in file:
    #    DEBUG = False  

        #DEBUG_LVL2=True
    if i<2:
        continue
    img = cv2.imread(os.path.join(path,file))

    t = time.time()
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    defects = find_defects(gr)
    x,y,h,w,_ = defects[0]
    img = img[y:y+h, x:x+w,:]
    

    #print('validation:', defects[0,-1])
    
    for defect in defects[1:]:
            cv2.circle(img, (defect[0],defect[1]), 5, (255,0,0), thickness=-1)
            
    t = time.time() - t
    #print(t)
    #ccv2.imshow('res', cv2.resize(res, None, fx=0.4, fy=0.4) )
    cv2.imshow('img', cv2.resize(img, None, fx=0.4, fy=0.4) )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

