import numpy as np
import cv2
import skimage
import imutils
import random
import copy

class transforamtion:
    def __init__(self, image):
        self.image = image

    # Rotation
    def rotate0(self,degree=60):
        rows, cols = self.image.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree, 1)
        dst = cv2.warpAffine(self.image, M, (cols, rows))
        return dst
    def rotate1(self,degree=60):
        return skimage.transform.rotate(self.image, degree)

    # Affine Transformation
    def affine(self):
        rows, cols = self.image.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(self.image, M, (cols, rows))
        return  dst

    # Perspective/Homography Transforamtion
    def perspective(self):
        rows, cols = self.image.shape
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[50, 30], [200, 50], [100, 200], [100, 80]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(self.image, M, (300, 300))
        return dst

class add_noise:
    def __init__(self, image):
        self.image = image

    #  Gaussian Noise
    def Gaussian(self,var=0.5):
        noisy_gau_1 = skimage.util.random_noise(self.image, mode='gaussian', var=var)
        img_gauss = np.array(255*noisy_gau_1).astype('uint8')
        # mu = 0
        # sigma = var**0.5
        # gauss = np.random.normal(mu,sigma,self.image.size)
        # gauss = gauss.reshape(self.image.shape).astype('uint8')
        # img_gauss = cv2.add(self.image, gauss)
        return img_gauss

    # Salt&Pepper noise 1
    def s_p(self):
        noisy_sp = skimage.util.random_noise(self.image, mode='s&p',amount=0.2)
        output = np.array(255*noisy_sp,dtype='uint8')
        # plt.imshow(noisy_sp, cmap='gray')
        # plt.axis('off')
        # plt.savefig('s_p.pdf', bbox_inches='tight')
        # plt.show()
        return output

    # Salt&Pepper noise2 (add diffent amount of noise into data)
    def s_p_amount (self,amount):
        img_x, img_y = self.image.shape
        s_vs_p = 0.5
        out = np.copy(self.image)
        # Salt mode
        num_salt = np.ceil(amount * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in self.image.shape]
        out[tuple(coords)] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in self.image.shape]
        out[tuple(coords)] = 0

        return out

    # Block noise
    def box (self, length = 50, width= 60):
        img = copy.deepcopy(self.image)
        img_x, img_y = img.shape
        seed = [0,255]
        box = []
        for i in range(length*width):
            random_num = random.choice(seed)
            box.append(random_num)
        box = np.array(box).reshape(length,width)
        x= np.random.randint(low=0, high=img_x-length, size=1)[0]
        y= np.random.randint(low=0, high=img_y-width, size=1)[0]
        img[x:x+length, y:y+width] = box
        return img

    def box_black (self, length = 50, width= 60):
        img = copy.deepcopy(self.image)
        img_x, img_y = img.shape
        seed = [0,255]
        box = np.full((length, width), 100)
        # for i in range(length*width):
        #     random_num = random.choice(seed)
        #     box.append(random_num)
        # box = np.array(box).reshape(length,width)
        x= np.random.randint(low=0, high=img_x-length, size=1)[0]
        y= np.random.randint(low=0, high=img_y-width, size=1)[0]
        img[x:x+length, y:y+width] = box
        return img

    def box_white (self, length = 50, width= 60):
        img = copy.deepcopy(self.image)
        img_x, img_y = img.shape
        seed = [0,255]
        box = np.full((length, width), 255)
        # for i in range(length*width):
        #     random_num = random.choice(seed)
        #     box.append(random_num)
        # box = np.array(box).reshape(length,width)
        x= np.random.randint(low=0, high=img_x-length, size=1)[0]
        y= np.random.randint(low=0, high=img_y-width, size=1)[0]
        img[x:x+length, y:y+width] = box
        return img
