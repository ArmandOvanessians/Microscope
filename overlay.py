from PIL import Image, ImageOps
import os
import cv2

path = '/mnt/data/xiangyucs/data_resize/1024_480/overlay/0'
os.chdir(path)
#
# background = Image.open("us.jpg")
# overlay = Image.open("moved.jpg")
#
# background = background.convert("RGBA",colors='red')
# overlay = overlay.convert("RGBA")
#
# new_img = Image.blend(background, overlay, 1.0)
# new_img.save("overlay.png","PNG")

background = Image.open("s.jpg")
overlay = Image.open("moved.jpg")

overlay = ImageOps.grayscale(overlay)
overlay = ImageOps.colorize(overlay, black="red", white="white")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.9)
new_img.save("overlay.png","PNG")

# def mask_color_img(img, mask, color=[0, 255, 255], alpha=0.3):
#     '''
#     img: cv2 image
#     mask: bool or np.where
#     color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
#     alpha: float [0, 1].
#
#     Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
#     '''
#     out = img.copy()
#     img_layer = img.copy()
#     img_layer[mask] = color
#     out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
#     cv2.imwrite("overlay.png", out)
#     return(out)

#
# img = cv2.imread('us.jpg')  # Image to be aligned.
# mask = cv2.imread('moved.jpg')  # Reference image.
# # mask_color_img(img=img,mask=mask)
# output = img.copy()
# overlay = mask.copy()
#
# #Adding the transparency parameter
# alpha = 1
#
# #Performing image overlay
# cv2.addWeighted(mask, alpha, output, 1 - alpha,0, output)
# cv2.imwrite("overlay.png", output)

