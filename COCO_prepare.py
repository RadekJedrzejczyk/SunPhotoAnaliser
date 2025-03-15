import cv2 #opencv -
import numpy as np
import json
import os
import concurrent.futures
import random
"""General settings """
NUM_THREADS = 8
MIN_ELIPSE_ECCENTRICITY = 0.99
MIN_ELIPSE_AXIS = 20
MIN_SUNSPOT= 0.00001 #in relation to the Sun Area
MAX_SUNSPOT= 0.5


"""4096 quality settings """
CONTOURSSIZE4096 =5 

"""1024 quality settings """
CONTOURSSIZE1024 =2 

"""PATHS"""
folder_path = 'sdo_data/filtered'
json_output_path = 'datasets/'
training_name='train.json'
validation_name='valid.json'
test_name='test.json'

"""TRAINING/VALIDATION SETS """
PERCENT_TRAINING = 0.75
PERCENT_VALID=0.15

"""AUTOMATIC"""
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
train_size = int(PERCENT_TRAINING * len(all_files))
val_size = int(PERCENT_VALID * len(all_files))

"""Datasets preparation"""
all_files.sort()
random.seed(42)
random.shuffle(all_files)
train_files = all_files[:train_size]
val_files = all_files[train_size:train_size + val_size]
test_files = all_files[train_size + val_size:]
datasets = [
    (train_files, json_output_path + training_name),
    (val_files, json_output_path + validation_name),
    (test_files, json_output_path + test_name)
]

def move_files(file_list, destination_folder):
    for file_name in file_list:
        src = os.path.join(folder_path, file_name)
        dst = os.path.join(destination_folder, file_name)
        shutil.copy(src, dst)

move_files(train_files, json_output_path+'train')
move_files(val_files, json_output_path+'val')
move_files(test_files, json_output_path+'test')


def save_coco_annotations(output_json, class_name, images_data):
    """
    Tworzy plik JSON w formacie COCO dla wielu obrazów.
    :param output_json: Ścieżka do pliku wynikowego JSON.
    :param class_name: Nazwa klasy obiektu (np. "sunspot").
    :param images_data: Lista słowników zawierających 'image_path' i 'approx_list'.
    """
    categories = [{"id": 1, "name": class_name, "supercategory": "object"}]
    images = []
    annotations = []
    annotation_id = 1  # starting id
    image_id = 1  

    for img_data in images_data:
        image_path = img_data["image_path"]
        approx_list = img_data["approx_list"]

        # Read height and width
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading file {image_path}")
            continue
        h, w = image.shape[:2]

        # Picture data
        images.append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": w,
            "height": h
        })

        # Contours description
        for approx in approx_list:
            polygon = approx.reshape(-1, 2).tolist()  # Convert to points
            segmentation = [sum(polygon, [])]  # flatten
            x, y, w_box, h_box = cv2.boundingRect(approx)  # Bounding box

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": cv2.contourArea(approx),
                "bbox": [x, y, w_box, h_box],
                "iscrowd": 0
            })
            annotation_id += 1  # Sunspot IDs

        image_id += 1  # Images IDs

    # FInal JSON
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"COCO JSON saved as {output_json}")


#####################################################################

def generate_list(folder_path,file_name):
    """szukanie dysku"""
    image_original = cv2.imread(folder_path+'/'+file_name)
    image = cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)
    height, width = image_original.shape[:2]

        #APPLY OPTIONS
    if height == 4096:
        contour_size = CONTOURSSIZE4096
    elif height ==1024:
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
                                   
                    """Data preparation """
                    image_contours2 = image_elipse.copy()
                    approx_list=[]
                    for i, cnt in enumerate(filtered_contours):
                        epsilon = 0.01 * cv2.arcLength(cnt, True)  # precision
                        approx = cv2.approxPolyDP(cnt, epsilon, True) #shape approximation
                        cv2.drawContours(image_contours2, [approx], -1, (0, 255, 0), contour_size)
                        approx_list.append(approx) 
                    print("RETURNED!")
                    return {"image_path": folder_path+'/'+file_name, "approx_list": approx_list}
                else:
                    print("No filtered contours!")
            else:
                print("No elipse found!")
    else:
        print("No contours found!")


########################################################################################################
for file_list, output_path in datasets:
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(generate_list, folder_path, file) for file in file_list]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    save_coco_annotations(output_path, 'sunspot', results)