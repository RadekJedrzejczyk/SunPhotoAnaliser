import os
import cv2 #opencv -
import numpy as np
import random
import shutil
import concurrent.futures
"""General settings """
NUM_THREADS = 8
MIN_ELIPSE_ECCENTRICITY = 0.99
MIN_ELIPSE_AXIS = 20
MIN_SUNSPOT= 0.00001 #in relation to the Sun Area
MAX_SUNSPOT= 0.5
MIN_OVERALL_SUNSPOT_AREA = 0.01


"""Paths"""
folder_path = 'sdo_data'
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
save_to='sdo_data/filtered'



def file_filtering(folder_path,save_to,file_name):
    """szukanie dysku"""
    image_original = cv2.imread(folder_path+'/'+file_name)
    image = cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)
    height, width = image_original.shape[:2]

    
    edges = cv2.Canny(image, 10, 50)
    kernel = np.ones((3, 3), np.uint8) 
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        elipses =[]
        areas = []
        for cnt in contours:
            if len(cnt)>=5:
                sun_ellipse = cv2.fitEllipse(cnt)
                if sun_ellipse:
                    (cx, cy), (major_axis, minor_axis), angle = sun_ellipse
                    sun_area=np.pi*major_axis/2*minor_axis/2
                    if not (np.isnan(sun_area) or major_axis<=MIN_ELIPSE_AXIS or minor_axis<=MIN_ELIPSE_AXIS or major_axis/minor_axis <MIN_ELIPSE_ECCENTRICITY):
                        elipses.append(sun_ellipse)
                        areas.append(sun_area)
        if len(areas)>0:
            max_index = np.argmax(areas)
            sun_ellipse=elipses[max_index]
            (cx, cy), (major_axis, minor_axis), angle = sun_ellipse
        
            """szukanie plam"""
            if sun_ellipse:  
                    #maska
                mask = np.zeros_like(image) #black mask
                cv2.ellipse(mask, sun_ellipse, 255, -1)  #paint the elipse on mask (full inside, that's why -1)
                masked_binary = cv2.bitwise_and(image, mask)  
                new_binary=cv2.adaptiveThreshold(masked_binary,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,301,21)#THRESH_BINARY
                new_binary =  cv2.bitwise_and(new_binary, mask)  
                sun_area=np.pi*major_axis/2*minor_axis/2
                 #kontury
                inner_contours, _ = cv2.findContours(new_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   #Find contours inside
                min_area=sun_area*MIN_SUNSPOT
                max_area = sun_area *MAX_SUNSPOT
                    
                    # Filtrowanie małych konturów
                filtered_contours = [cnt for cnt in inner_contours if (cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < max_area)]
                if len(filtered_contours) >0:
                    filtered_contours.pop(0)

                    spots_area=0
                    for contour in filtered_contours:
                        spots_area+=cv2.contourArea(contour)
                    if spots_area/sun_area*100>MIN_OVERALL_SUNSPOT_AREA:
                        shutil.copyfile(folder_path+'/'+file_name, save_to+'/'+file_name)
                    else:
                        print(f"Sun spot area {spots_area/sun_area*100:.3f}% of sun - example rejected")
                else:
                    print("No filtered contours!")
            else:
                print("No elipse found!")
    else:
        print("No contours found!")




with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(file_filtering, folder_path, save_to, file) for file in all_files]
    
