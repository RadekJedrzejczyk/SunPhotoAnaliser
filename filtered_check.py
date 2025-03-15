import os
import random
import cv2 #opencv -
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures


def describe_picture(folder_path,save_to,file_name):
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
                    if not (np.isnan(sun_area) or major_axis<=20 or minor_axis<=20 or major_axis/minor_axis < 0.95):
                        elipses.append(sun_ellipse)
                        areas.append(sun_area)
        if len(areas)>0:
            max_index = np.argmax(areas)
            sun_ellipse=elipses[max_index]
            (cx, cy), (major_axis, minor_axis), angle = sun_ellipse
            image_elipse=cv2.ellipse(image_original, sun_ellipse, (255, 0, 0), 10)
        
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
                min_area=sun_area/100000
                max_area = sun_area *0.5
                    
                    # Filtrowanie małych konturów
                filtered_contours = [cnt for cnt in inner_contours if (cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < max_area)]
                if len(filtered_contours) >0:
                    filtered_contours.pop(0)
                        
                    image_contours = image_elipse.copy()
                    cv2.drawContours(image_contours, filtered_contours, -1, (0, 255, 0), 5)  # Czerwone kontury
                    
                
                    spots_area=0
                    for contour in filtered_contours:
                        spots_area+=cv2.contourArea(contour)
                    print(f'Sun spots are on {spots_area/sun_area*100:.3f}% of sun area')
                    image_contours=cv2.drawContours(image_elipse, filtered_contours, -1, (0, 255, 0), 5)  # Czerwone kontury
                            
                        
                    """dodawanie napisu"""
                    image_with_text = cv2.copyMakeBorder(image_contours, 0, int(height/10), 0, 0, cv2.BORDER_CONSTANT, None, value = (255,255,255)) 
                    height, width = image_with_text.shape[:2]
                    font = cv2.FONT_HERSHEY_SIMPLEX 
                    font_scale = 6
                    position = (int(width/100), int(height-height/100))
                    thickness = 10
                    line_type = cv2.LINE_AA
                    cv2.putText(image_with_text, f'Sun spots are on {spots_area/sun_area*100:.3f}% of the Sun.', position , font, font_scale, 0, thickness, line_type)
                        
                        # Zapisanie obrazu z napisem
                    cv2.imwrite(f'sdo_sample/{file_name}_text.jpg', image_with_text)
                else:
                    print("No filtered contours!")
            else:
                print("No elipse found!")
    else:
        print("No contours found!")


NUM_THREADS = 16 
folder_path = 'sdo_data/filtered'
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
random_files = random.sample(all_files, 500)


with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(describe_picture, folder_path, save_to, file) for file in random_files]
 #   for future in futures:
  #      future.result() 