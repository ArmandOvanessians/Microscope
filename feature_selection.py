import numpy as np
import cv2 as cv
import json
import os
import sys
import matplotlib.pyplot as plt

class akazeml:
    def __init__(self, filename="feature_selection.py"):
        self.filename = filename
        self.path = os.path.abspath(os.path.join(self.filename, "../../Data"))

    # read images from ../../Data
    def read_data(self):
        data = []
        for file in os.listdir(self.path):
            if file.endswith('jpg'):
                if 'HD' in file or 'Rap' in file:
                    image1 = os.path.join(self.path,file)
                    image2_name = file.replace('HD','unstained').replace('Rap','Uns')
                    try:
                        if image2_name in os.listdir(self.path):
                            image2 = os.path.join(self.path,image2_name)
                            data.append((image1, image2))
                    except FileNotFoundError:
                        print(f'File not found: {image2_name}')


        return data

    # def label_features(self, img1, img2, label=True):
    #     data = self.read_data()
    #     if label == True:
    #         for i in data:
    #             # Image 1 feature - stained
    #             if img1 in i and img2 in i:
    #                 f1 = open(f"{i[0].split('.')[0]}.json")
    #                 img1 = json.load(f1)
    #                 st_loc = np.array(list(img1.values()))
    #                 st_loc = st_loc.reshape(st_loc.shape[0], 1, st_loc.shape[1])
    #
    #                 # Image2 feature - unstained
    #                 f2 = open(f"{i[1].split('.')[0]}.json")
    #                 img2 = json.load(f2)
    #                 ust_loc = np.array(list(img2.values()))
    #
    #     else:
    #         return data

    def label_features(self, img1, img2):
        image1_data = os.path.join(self.path, img1)
        image2_data = os.path.join(self.path, img2)
        # Image1 features
        f1 = open(f"{image1_data.split('.')[0]}.json")
        img1 = json.load(f1)
        st_loc = np.array(list(img1.values()))
        st_loc = st_loc.reshape(st_loc.shape[0], 1, st_loc.shape[1])
        # Image2 features
        f2 = open(f"{image2_data.split('.')[0]}.json")
        img2 = json.load(f2)
        ust_loc =np.array(list(img2.values()))

        return st_loc,ust_loc

    def akaze(self):
        print(f"OpenCV Version: {cv.__version__}")
        # Load images (grayscale)
        images = self.read_data()
        for data in images:
            image1_name = os.path.basename(data[0])
            image2_name = os.path.basename(data[1])
            image1_data = data[0]
            image2_data = data[1]
            print(f" Stained Image is - {image1_data} \n Unstained Image is - {image2_data}")
            image1=cv.imread(filename = image1_data,flags=cv.IMREAD_GRAYSCALE)
            image2=cv.imread(filename = image2_data,flags=cv.IMREAD_GRAYSCALE)
            if image1 is None or image2 is None:
                print('Could not open or find the images!')
                exit(0)

            # Show Image
            plt.imshow(image1,cmap='gray')
            plt.title(image1_name)
            plt.show()
            plt.imshow(image2,cmap='gray')
            plt.title(image2_name)
            plt.show()

            # # Add noise or transforamtion
            # # image2 = noise.transforamtion(image2).affine()
            # image2 = noise.add_noise(image2).Gaussian()
            # plt.imshow(image2,cmap='gray')
            # plt.show()

            # Find the keypoints and compute the descriptors
            # for input and training-set image using AKAZE
            AKAZE = cv.AKAZE_create()
            keypoints1, descriptors1 = AKAZE.detectAndCompute(image1, None)
            keypoints2, descriptors2 = AKAZE.detectAndCompute(image2, None)
            assert type(descriptors1) == type(descriptors2)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            # Convert to float32
            descriptors1 = np.float32(descriptors1)
            descriptors2 = np.float32(descriptors2)

            # Create FLANN object
            FLANN = cv.FlannBasedMatcher(indexParams=index_params,searchParams=search_params)

            # Matching descriptor vectors using FLANN Matcher
            matches = FLANN.knnMatch(queryDescriptors=descriptors1,trainDescriptors=descriptors2, k=2)

            # Lowe's ratio test
            # Use 2-nn matches and ratio criterion to find correct keypoint matches
            ratio_thresh = 0.7
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            # # Showing the matching results
            # # Draw only "good" matches
            # output = cv.drawMatches(img1=image1,
            #                         keypoints1=keypoints1,
            #                         img2=image2,
            #                         keypoints2=keypoints2,
            #                         matches1to2=good_matches,
            #                         outImg=None,
            #                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # plt.imshow(output,cmap='gray')
            # plt.show()

            # Homography Result and Transform
            MIN_MATCH_COUNT = 30
            path_save = os.path.abspath(os.path.join(self.filename, f"../../Result"))
            isExist = os.path.exists(path_save)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(path_save)
            res1_path = os.path.abspath(os.path.join(path_save, f"{image1_name}.txt"))
            if len(good_matches) > MIN_MATCH_COUNT:
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0) # Find Inliers Points
                matchesMask = mask.ravel().tolist()
                h, w = image1.shape
                # print(f"Size of Stained Image: {h} * {w}")
                # print(f"Size of Unstained Image: {image2.shape[0]} * {image2.shape[1]}")
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                #### Calculate the results
                st_loc,ust_loc = self.label_features(image1_data, image2_data)
                tran_loc = cv.perspectiveTransform(st_loc, M).reshape(ust_loc.shape)
                mse = ((tran_loc-ust_loc)**2).mean(axis=0)

                with open(res1_path, 'a+') as f:
                    print('*******************************',file=f)
                    print(f"A-KAZE Matching Results for {image1_name} and {image2_name}",file=f)
                    print(f"Size of Stained Image: {h} * {w}",file=f)
                    print(f"Size of Unstained Image: {image2.shape[0]} * {image2.shape[1]}",file=f)
                    print(f"# MSE of X={mse[0]} and Y={mse[1]}",file=f)
                    print('# Keypoints 1:                        \t', len(keypoints1),file=f)
                    print('# Keypoints 2:                        \t', len(keypoints2),file=f)
                    print('# Matches:                            \t', len(good_matches),file=f)
                    print('# Inliers:                            \t', matchesMask.count(1),file=f)
                    print('# Inliers Ratio:                      \t', matchesMask.count(1)/float(len(good_matches)),file=f)
                    print('*******************************',file=f)
                image2 = cv.polylines(image2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            else:
                with open(res1_path, 'a+') as f:
                    print('*******************************',file=f)
                    print(f"A-KAZE Matching Results for {image1_name} and {image2_name}",file=f)
                    print(f"Size of Stained Image: {h} * {w}",file=f)
                    print(f"Size of Unstained Image: {image2.shape[0]} * {image2.shape[1]}",file=f)
                    print('# Keypoints 1:                        \t', len(keypoints1), file=f)
                    print('# Keypoints 2:                        \t', len(keypoints2), file=f)
                    print('# Matches:                            \t', len(good_matches), file=f)
                    print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT),file=f)
                    print('*******************************', file=f)
                print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
                matchesMask = None

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            img3 = cv.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, **draw_params)
            plt.imshow(img3, 'gray')
            plt.title(image1_name)
            # plt.savefig("img3.pdf", format="pdf", bbox_inches="tight")
            fig_path = os.path.abspath(os.path.join(path_save, f"{image1_name}.jpg"))
            plt.savefig(fig_path)
            plt.show()

            # # Homography Result
            # MIN_MATCHES = 50
            # if len(good_matches) > MIN_MATCHES:
            #     src_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            #     dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            #     m, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
            #     corrected_img = cv.warpPerspective(image1, m, (image2.shape[1], image2.shape[0]))
            #     print(m)
            #     return corrected_img
            # return image2



if __name__ == '__main__':
    # image1_data = "CT001-035B_HD.jpg"
    # image2_data = "CT001-035B_unstained_HD.jpg"
    #
    # akaze(image1_data, image2_data)
    data = akazeml().akaze()

