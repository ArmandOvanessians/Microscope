import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil

# coor = (h-left, h--right, w-left, w-right)
def box(center, len=300):
    coor = list([int(i[0]-len/2), int(i[0]+len/2), int(i[1]-len/2), int(i[1]+len/2)] for i in center)

    return coor

# def center_cal(cor):
#     x, y = (cor[0][0] + cor[1][0]) // 2, (cor[0][1] + cor[1][1]) // 2
#     return (y,x)
#
# def save_img(input):
#     img = input.astype(int)
#     rescaled = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
#     im = Image.fromarray(rescaled)
#     var = f'{input=}'.partition('=')[0]
#     im.save(os.path.join(path,f'{var}.png'))

def load_data(path =r'C:\Users\Armand Ovanessians\Microscope\ORB\Data\data_s_us\s'):
    for file in os.listdir(path):
        if file.endswith('jpg'):
            if '_pap_' in file or '_tol_' in file:
                image1 = os.path.join(path, file)
                image1_name = file
                image2_name = file.replace('_pap_', '_unstained_'). \
                    replace('_tol_', '_unstained_')
                path = f"C:/Users/Armand Ovanessians/Microscope/ORB/result/{image2_name}"
                stained_point = np.load(os.path.join(path,'point.npy'))
                unstained_point = np.load(os.path.join(path, 'moving_point.npy'))
                img_s= cv2.imread(f'{path}/fixed-stained.jpg')  # Image to be aligned.
                img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
                img_us= cv2.imread(f'{path}/moving-unstained.jpg')  # Reference image.
                img_us = cv2.cvtColor(img_us, cv2.COLOR_BGR2GRAY)

                print(f'{image2_name} \n shape: stained->{img_s.shape}, unstained->{img_us.shape}')
                box_stained = box(stained_point)
                box_unstained = box(unstained_point)
                print(f'box on stained img: {box_stained}')
                print(f'box on unstained img: {box_unstained}')
                name = ['box_stained','box_unstained']
                path_patch = f'{path}/patch'
                shutil.rmtree(path_patch, ignore_errors=True)
                isExist = os.path.exists(path_patch)
                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(path_patch)
                for i in range(5):
                    patch_s = img_s[box_stained[i][0]:box_stained[i][1],box_stained[i][2]:box_stained[i][3]]

                    patch_us = img_us[box_unstained[i][0]:box_unstained[i][1], box_unstained[i][2]:box_unstained[i][3]]
                    #
                    print(box_stained[i], patch_s.shape, '---', box_unstained[i], patch_us.shape)
                    if 0< box_stained[i][0] <= img_s.shape[0] and 0< box_stained[i][1] <= img_s.shape[0] and \
                            0<box_stained[i][2] <= img_s.shape[1] and 0<box_stained[i][3] <= img_s.shape[1] and \
                            0<box_unstained[i][0] <= img_us.shape[0] and 0<box_unstained[i][1] <= img_us.shape[0] and \
                            0<box_unstained[i][2] <= img_us.shape[1] and 0<box_unstained[i][3] <= img_us.shape[1]:

                                save_s= f"{path_patch}/s-{i}.jpg"
                                save_us = f"{path_patch}/us-{i}.jpg"
                                cv2.imwrite(save_s, patch_s)
                                cv2.imwrite(save_us, patch_us)
                    else:
                        f = open(f"C:/Users/Armand Ovanessians/Microscope/ORB/result/attention.txt", "a")
                        print(f'-----Attention:{image2_name} - {i} !!-----',file=f)
                        print(f'{image2_name} \n shape: stained->{img_s.shape}, unstained->{img_us.shape}',file=f)
                        print(box_stained[i], patch_s.shape, '---', box_unstained[i], patch_us.shape,file=f)
                        f.close()





if __name__ == '__main__':
    load_data()


# spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')
# moving_image = moving
# disp_tensor = flow
# # warp the moving image with the transformer
# moved_image_tensor = spatial_transformer([moving_image, disp_tensor])[0,:,:,0]
# print(f'moved_image_tensor.shape -> {moved_image_tensor.shape}')
# plt.imshow(moved_image_tensor, cmap='gray')
# plt.title('Moved with dots')
# plt.show()
#
# moving_mask = moving[0,:,:,0]
# save_img(moving_mask)
# # Draw rectangles
#
# # im = Image.open('input.png')
# #
# # # Create figure and axes
# # fig, ax = plt.subplots()
# #
# # # Display the image
# # ax.imshow(im)
# #
# # # # Create a Rectangle patch
# # # rect = patches.Rectangle((200, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
# # # # Add the patch to the Axes
# # # ax.add_patch(rect)
# #
# # plt.show()
#
# # rectangle
# # print(moving_mask.shape)
#
# ### Img_3 ####
# # r1 = [(200, 50),  (250, 100)]
# # r2 = [(400, 250), (450, 300)]
# # r3 = [(600, 150), (650, 200)]
# # r4 = [(200, 350), (250, 400)]
# # r5 = [(680, 320), (730, 370)]
# # r6 = [(50, 320),  (100, 370)]
# ### ## ###
#
# # ### img1 ###
# # r1 = [(410, 130),  (410+64, 130+64)]
# # r2 = [(500, 140), (500+64, 140+64)]
# # r3 = [(200, 150), (200+64, 150+64)]
# # r4 = [(180, 50), (180+64, 50+64)]
# # r5 = [(310, 180), (310+64, 180+64)]
# # r6 = [(80, 200),  (80+64, 200+64)]
#
# # ## Img 2 ###
# # r1 = [(410, 130),  (410+64, 130+64)]
# # r2 = [(600, 240), (600+64, 240+64)]
# # r3 = [(200, 150), (200+64, 150+64)]
# # r4 = [(180, 250), (180+64, 250+64)]
# # r5 = [(310, 180), (310+64, 180+64)]
# # r6 = [(520, 200),  (520+64, 200+64)]
# # ### ## ###
#
# ## h3 - img1 ###
# r1 = [(380, 380),  (380+64, 380+64)]
# r2 = [(470, 340), (470+64, 340+64)]
# r3 = [(570, 350), (570+64, 350+64)]
# r4 = [(740, 330), (740+64, 330+64)]
# r5 = [(660, 390), (660+64, 390+64)]
# r6 = [(820, 400),  (820+64, 400+64)]
# ### ## ###
#
# rectangle_group= {'1':r1,'2':r2,'3':r3,'4':r4,'5':r5,'6':r6}
# print('rectangle', rectangle_group)
#
# for i in rectangle_group:
#     coord = rectangle_group[i]
#     moving_patch = moving_mask[coord[0][1]:coord[1][1],coord[0][0]:coord[1][0]]
#     cv.imwrite(f'/mnt/data/xiangyucs/data_resize/patch/us/{int(i)+6*3}.jpg', moving_patch)
#
# # rectangle_group= {'1':r1,'2':r2,'4':r4}
#
# cv.rectangle(moving_mask, r1[0], r1[1], (0, 255), 2)
# cv.rectangle(moving_mask, r2[0], r2[1], (0, 255), 2)
# cv.rectangle(moving_mask, r3[0], r3[1], (0, 255), 2)
# cv.rectangle(moving_mask, r4[0], r4[1], (0, 255), 2)
# cv.rectangle(moving_mask, r5[0], r5[1], (0, 255), 2)
# cv.rectangle(moving_mask, r6[0], r6[1], (0, 255), 2)
# cv.imwrite('moving_mask.jpg', moving_mask)
#
# gap = abs(center_cal(r1)[0]-r1[0][1])
#
# # moving_image = moving[:,center_cal(r3)[0]-gap:center_cal(r3)[0]+gap,center_cal(r3)[1]-gap:center_cal(r3)[1]+gap,:]
# # disp_tensor = flow[:,center_cal(r3)[0]-gap:center_cal(r3)[0]+gap,center_cal(r3)[1]-gap:center_cal(r3)[1]+gap,:]
# # output = sp_trans(moving_image,disp_tensor)
#
# # moving_image = moving[:,center_cal(r3)[0]-gap:center_cal(r3)[0]+gap,center_cal(r3)[1]-gap:center_cal(r3)[1]+gap,:]
# disp_tensor = flow[:,center_cal(r1)[0]:center_cal(r1)[0]+1,center_cal(r1)[1]:center_cal(r1)[1]+1,:]
# y_move = disp_tensor[0,0,0,:][0]
# x_move = disp_tensor[0,0,0,:][1]
# moved_center = (center_cal(r1)[0] + y_move, center_cal(r1)[1] +x_move)
# print(moving_image.shape)
# print(disp_tensor)
# print(disp_tensor[0,0,0,:])
# fixed_rec = {}
# fixed_dict = {}
# for i in rectangle_group:
#     center = center_cal(rectangle_group[i])
#     disp_tensor = flow[:, center[0]:center[0] + 1, center[1]:center[1] + 1, :]
#     y_move = disp_tensor[0, 0, 0, :][0]
#     x_move = disp_tensor[0, 0, 0, :][1]
#     moved_center = (int(center[0] + y_move), int(center[1] + x_move))
#     fixed_rec[i] = moved_center
#
#     coord = [(fixed_rec[i][1] - gap, fixed_rec[i][0] - gap), (fixed_rec[i][1] + gap, fixed_rec[i][0] + gap)]
#     fixed_dict[i] = coord
# print(fixed_rec)
# print('fixed_dict', fixed_dict)
# fixed_mask = fixed[0,:,:,0]
# for i in fixed_rec:
#     coord = [(fixed_rec[i][1]-gap,fixed_rec[i][0]-gap),(fixed_rec[i][1]+gap,fixed_rec[i][0]+gap)]
#     cv.rectangle(fixed_mask, coord[0], coord[1], (0, 255), 2)
#     fix_patch = fixed_mask[coord[0][1]:coord[1][1],coord[0][0]:coord[1][0]]
#     print(i,'->' ,fix_patch.shape, fixed_mask.shape)
#     cv.imwrite(f'/mnt/data/xiangyucs/data_resize/patch/s/{int(i)+6*3}.jpg', fix_patch)
# cv.imwrite('fixed_mask.jpg', fixed_mask)
