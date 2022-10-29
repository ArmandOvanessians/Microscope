import cv2
import numpy as np
import os,sys
from matplotlib import pyplot as plt

def process(img_gray):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_canny = cv2.Canny(img_gray, 12, 54)
    img_canny = cv2.Canny(img_gray, 50, 150)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=10)
    img_erode = cv2.erode(img_dilate, kernel, iterations=8)
    return img_erode

path = r'C:\Users\Armand Ovanessians\Microscope\ORB\Data\data_s_us\s'
os.chdir(path)

i = 0
j = 0
for file in os.listdir(path):
    if file.endswith('jpg'):
        i = i + 1
        if '_pap_' in file or '_tol_' in file:
            image1 = os.path.join(path, file)
            image2_name = file.replace('_pap_', '_unstained_'). \
                replace('_tol_', '_unstained_')

            img = cv2.imread(filename=image1, flags=cv2.IMREAD_GRAYSCALE)
            img1_h, img1_w = img.shape
            print(img.shape)
            contours, _ = cv2.findContours(process(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            largest_contour = max(contours, key=cv2.contourArea)
            print(largest_contour.shape)
            cv2.drawContours(img, largest_contour, -1, (0, 255, 0), 2)

            point = []
            for i in range(5):
                # coor = (np.random.randint(1, img1_h, 1)[0], np.random.randint(1, img1_w, 1)[0])
                while True:
                    # coor = (height -- y in cv2, width -- x in cv2)
                    coor = (np.random.randint(1, img1_h, 1)[0], np.random.randint(1, img1_w, 1)[0])
                    coor = tuple(int(x) for x in coor)
                    if cv2.pointPolygonTest(largest_contour, (coor[1],coor[0]), False) == 1:
                        break

                print(f'{image2_name}:',img.shape,'->',coor,cv2.pointPolygonTest(largest_contour, (coor[1],coor[0]), False))
                cv2.circle(img, (coor[1],coor[0]), 20, (0, 0, 255), -1)
                point.append(coor)
            point = np.array(point)

            image1 = file
            # # save_path = f"/mnt/data/xiangyucs/Data_Sep_22/1024_480/Test_Y_1/{image2_name}"
            # os.chdir(save_path)
            save_path = f"C:/Users/Armand Ovanessians/Microscope/ORB/result/{image2_name}/point_fixed.jpg"
            # isExist = os.path.exists(os.path.dirname(save_path))
            # if not isExist:
            #     # Create a new directory because it does not exist
            #     os.makedirs(os.path.dirname(save_path))
            np.save(f'C:/Users/Armand Ovanessians/Microscope/ORB/result/{image2_name}/point.npy', point)
            cv2.imwrite(save_path, img)



