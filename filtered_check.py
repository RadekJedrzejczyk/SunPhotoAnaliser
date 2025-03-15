import os
import random
import cv2 #opencv -
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
"""General settings """
NUM_THREADS = 8
MIN_ELIPSE_ECCENTRICITY = 0.99
MIN_ELIPSE_AXIS = 20
MIN_SUNSPOT= 0.00001 #in relation to the Sun Area
MAX_SUNSPOT= 0.5


"""4096 quality settings """
FONTSCALE4096 = 6
THICKNESS4096 = 10
EDGEKERNEL4096 = np.ones((3, 3), np.uint8) 
CONTOURSSIZE4096 =5 

"""1024 quality settings """
FONTSCALE1024 = 2
THICKNESS1024 = 3
CONTOURSSIZE1024 =2 

"""PATHS"""
folder_path = 'sdo_data/filtered'
save_to = 'sdo_sample'
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
random_files = random.sample(all_files, 500)

def describe_picture(folder_path,save_to,file_name):
    """szukanie dysku"""
    image_original = cv2.imread(folder_path+'/'+file_name)
    image = cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)
    height, width = image_original.shape[:2]

        #APPLY OPTIONS
    if height == 4096:
        font_scale =FONTSCALE4096
        thickness = THICKNESS4096
        contour_size = CONTOURSSIZE4096
    elif height ==1024:
        font_scale = FONTSCALE1024
        thickness = THICKNESS1024
        contour_size = CONTOURSSIZE1024
    else:
        print("Uknown size")
        return False
    
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
            image_original_copy=image_original.copy()
            image_elipse=cv2.ellipse(image_original_copy, sun_ellipse, (255, 0, 0), contour_size)
        
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
                    
                    print(f'Sun spots are on {spots_area/sun_area*100:.3f}% of sun area')
                    image_contours=cv2.drawContours(image_elipse, filtered_contours, -1, (0, 255, 0), contour_size)  
                            
                        
                    """dodawanie napisu"""
                    image_with_text = cv2.copyMakeBorder(image_contours, 0, int(height/10), 0, 0, cv2.BORDER_CONSTANT, None, value = (255,255,255)) 
                    original_image_rescaled = cv2.copyMakeBorder(image_original, 0, int(height/10), 0, 0, cv2.BORDER_CONSTANT, None, value = (255,255,255)) #przygotowanie do połączenia

                    height, width = image_with_text.shape[:2]
                    font = cv2.FONT_HERSHEY_SIMPLEX 
                    position = (int(width/100), int(height-height/100))
                    line_type = cv2.LINE_AA
                    cv2.putText(image_with_text, f'Sun spots are on {spots_area/sun_area*100:.3f}% of the Sun.', position , font, font_scale, 0, thickness, line_type)

                    """Łączenie obrazów"""
                    combined_image = np.hstack((image_with_text, original_image_rescaled))

                        # Zapisanie obrazu z napisem
                    cv2.imwrite(f'{save_to}/{file_name}_text.jpg', combined_image)
                else:
                    print("No filtered contours!")
            else:
                print("No elipse found!")
    else:
        print("No contours found!")





with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(describe_picture, folder_path, save_to, file) for file in random_files]
 #   for future in futures:
  #      future.result() 