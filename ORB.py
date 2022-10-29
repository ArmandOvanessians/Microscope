import cv2
import os,sys
import numpy as np
# import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.colors
import cv2 as cv
from skimage.transform import resize
from PIL import Image, ImageOps


# Resize the image
def img_resize(data,new_shape = [480, 1024]):
    '''
    The order of interpolation. The order has to be in the range 0-5:
    0: Nearest-neighbor
    1: Bi-linear (default)
    2: Bi-quadratic
    3: Bi-cubic
    4: Bi-quartic
    5: Bi-quintic
    '''
    img = cv.imread(filename=data, flags=cv.IMREAD_GRAYSCALE)
    img = resize(img, new_shape, order=3, mode='constant', cval=0, preserve_range=bool)
    return img

# Calculate the center of a bounding box
def center_cal(cor):
    x, y = (cor[0][0] + cor[1][0]) // 2, (cor[0][1] + cor[1][1]) // 2
    return (y,x)

### Run ORB method
### Input (path of stained images, path of unstained images)
def read_data(path_stained = r'C:\Users\Armand Ovanessians\Microscope\ORB\Data\data_s_us\s',
              path_unstained = r'C:\Users\Armand Ovanessians\Microscope\ORB\Data\data_s_us\us',
              #save_path = '/mnt/data/xiangyucs/Data_Sep_22/1024_480/Test_Y_O',
              resize = False):

    stained_data = []
    unstained_data = []
    i = 0
    j = 0
    for file in os.listdir(path_stained):
        if file.endswith('jpg'):
            i= i + 1
            ### image1 -> path of stained image, image2 -> path of unstained image
            if '_pap_' in file or '_tol_' in file:
                image1 = os.path.join(path_stained, file)
                image2_name = file.replace('_pap_', '_unstained_').\
                        replace('_tol_', '_unstained_')
                try:
                    if image2_name in os.listdir(path_unstained):
                        j = j + 1
                        image2 = os.path.join(path_unstained, image2_name)
                        print(f'Read data(i = {i}, j = {j}): {file} and {image2_name}')

                        # If we need to resize the image and store them into a new folder
                        if resize == True:
                            save_path_s = '/mnt/data/xiangyucs/data_resize/1024_480/s'
                            isExist = os.path.exists(os.path.dirname(save_path_s))
                            if not isExist:
                                # Create a new directory because it does not exist
                                os.makedirs(os.path.dirname(save_path_s))
                            save_path_s = os.path.join(save_path_s,file)
                            img1_s = img_resize(data=image1)
                            cv2.imwrite(save_path_s, img1_s)
                            stained_data.append(img1_s)

                            save_path_us = '/mnt/data/xiangyucs/data_resize/1024_480/us'
                            isExist = os.path.exists(os.path.dirname(save_path_us))
                            if not isExist:
                                # Create a new directory because it does not exist
                                os.makedirs(os.path.dirname(save_path_us))
                            save_path_us = os.path.join(save_path_us,image2_name)
                            img2_us = img_resize(data=image2)
                            cv2.imwrite(save_path_us, img2_us)
                            unstained_data.append(img2_us)

                        # Open the image files.
                        img1_color = cv2.imread(image1)  # Image to be aligned.
                        img2_color = cv2.imread(image2)  # Reference image.

                        # Convert to grayscale.
                        img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
                        img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

                        height, width = img2.shape

                        # Create ORB detector with 5000 features.
                        orb_detector = cv2.ORB_create(5000)
                        # orb_detector = cv2.ORB_create(1000)

                        # Find keypoints and descriptors.
                        # The first arg is the image, second arg is the mask
                        # (which is not required in this case).
                        kp1, d1 = orb_detector.detectAndCompute(img1, None)
                        kp2, d2 = orb_detector.detectAndCompute(img2, None)

                        # Match features between the two images.
                        # We create a Brute Force matcher with
                        # Hamming distance as measurement mode.
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                        # Match the two sets of descriptors.
                        matches = matcher.match(d1, d2)

                        # Sort matches on the basis of their Hamming distance.
                        matches = list(matches)
                        matches.sort(key=lambda x: x.distance)
                        matches = tuple(matches)

                        # Take the top 50 % matches forward.
                        matches = matches[:int(len(matches) * 0.5)]
                        # matches = matches[:int(len(matches)*0.9)]
                        no_of_matches = len(matches)

                        # Define empty matrices of shape no_of_matches * 2.
                        p1 = np.zeros((no_of_matches, 2))
                        p2 = np.zeros((no_of_matches, 2))

                        for i in range(len(matches)):
                            p1[i, :] = kp1[matches[i].queryIdx].pt
                            p2[i, :] = kp2[matches[i].trainIdx].pt

                        ### Save the result
                        save_path = f"C:/Users/Armand Ovanessians/Microscope/ORB/result/{image2_name}"
                        # save_path = f"{save_path}/{image2_name}"
                        isExist = os.path.exists(save_path)
                        if not isExist:
                            # Create a new directory because it does not exist
                            os.makedirs(save_path)

                        output = cv2.drawMatches(img1=img1,
                                                 keypoints1=kp1,
                                                 img2=img2,
                                                 keypoints2=kp2,
                                                 matches1to2=matches,
                                                 outImg=None,
                                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        # save the matching result between paired images
                        plt.imshow(output, cmap='gray')
                        plt.title('matching result')
                        image2_name = 'Match_' + image2_name
                        path_name = os.path.join(save_path,image2_name)
                        cv2.imwrite(path_name, output)
                        plt.show()

                        # save paired input images - stained and unstained
                        path_name = os.path.join(save_path,'fixed-stained.jpg')
                        cv2.imwrite(path_name, img1)
                        path_name = os.path.join(save_path, 'moving-unstained.jpg')
                        cv2.imwrite(path_name, img2)

                        # Find and save the homography matrix.
                        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
                        np.save(f'{save_path}/homo.npy',homography)


                        # Use this matrix to transform the
                        # colored image wrt the reference image.
                        # Transform stained image
                        transformed_img = cv2.warpPerspective(img1_color,
                                                              homography, (width, height))

                        #save moved image
                        plt.imshow(transformed_img, cmap='gray')
                        plt.title('transformed')
                        path_name = os.path.join(save_path,'moved.jpg')
                        cv2.imwrite(path_name, transformed_img)
                        plt.show()
                        # save overlay images
                        background = Image.open(image2)
                        overlay = Image.open(path_name)
                        overlay = ImageOps.grayscale(overlay)
                        overlay = ImageOps.colorize(overlay, black="yellow", white="white")
                        background = background.convert("RGBA")
                        overlay = overlay.convert("RGBA")
                        new_img = Image.blend(background, overlay, 0.9)
                        # overlay_name = 'OL' + image2_name + '.png'
                        path_name = os.path.join(save_path,'overlay.png')
                        new_img.save(path_name, "PNG")

                        print('next')
                except FileNotFoundError:
                    print(f'File not found: {image2_name}')

    return np.array(stained_data), np.array(unstained_data)


if __name__ == '__main__':
    read_data()