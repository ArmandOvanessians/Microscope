import cv2
import os
import numpy as np

def project(homography, point):
    scale = lambda x: np.array((x[0],x[1],1)).reshape((3,1))
    point = list(map(scale, point))
    sum_v = lambda x: np.sum(homography.dot(x),1)
    sum = list(map(sum_v,point))
    scale_back = lambda x: (int(round(x[0] / x[2])),int(round(x[1] / x[2])))
    point = list(map(scale_back,sum))
    print(f'Tran->{point}')
    return point


def load_data(path =r'C:\Users\Armand Ovanessians\Microscope\ORB\Data\data_s_us\s'):

    for file in os.listdir(path):

        if file.endswith('jpg'):
            if '_pap_' in file or '_tol_' in file:
                image1 = os.path.join(path, file)
                image1_name = file
                image2_name = file.replace('_pap_', '_unstained_'). \
                    replace('_tol_', '_unstained_')
                path = f"C:/Users/Armand Ovanessians/Microscope/ORB/result/{image2_name}"
                point = np.load(os.path.join(path,'point.npy'))
                homo = np.load(os.path.join(path,'homo.npy'))
                # point(height, weight)
                print(f'----{image2_name}----')
                point = tuple([x[1],x[0]]for x in point)
                print(f'Given -> {point}')
                point = project(homography=homo,point=point)
                path_moving = os.path.join(path, 'moving-unstained.jpg')
                img_moving = cv2.imread(filename=path_moving, flags=cv2.IMREAD_GRAYSCALE)

                for i in point:

                    cv2.circle(img_moving, (i[0], i[1]), 20, (0, 0, 255), -1)
                save_path = f"{path}/point_moved.jpg"
                cv2.imwrite(save_path, img_moving)


if __name__ == '__main__':
    load_data()