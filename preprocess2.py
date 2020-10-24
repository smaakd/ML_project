import cv2
import os
import os.path
import glob
import imutils
import numpy as np
import matplotlib.pyplot as plt

CAPTCHA_IMAGE_FOLDER = "./train"
OUTPUT_FOLDER = "./generated_captcha_images"
TRAIN_DATA_FOLDER = "./extracted_letter_images"

captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

true_match_count = 0
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    print(captcha_image_file)
    # img = cv2.imread(captcha_image_file , cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(captcha_image_file)
# img = cv2.imread('./train/HEO.png',0)
    val2 = img2[0,0]
    img2[(img2[:,:,0] == val2[0]) & (img2[:,:,1] == val2[1]) & (img2[:,:,2] == val2[2]) ] = [0,0,0]
    img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((7,7),np.uint8)
    erosion = cv2.erode(thresh1,kernel,iterations = 1)
    inv_erosion = cv2.bitwise_not(erosion)
    rbg_img = cv2.cvtColor(inv_erosion,cv2.COLOR_GRAY2RGB)

    newimg, contours,hierarchy = cv2.findContours(inv_erosion, 1, 2)
    count = 0
    store = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        # store.append((x,y,w,h))
        if(110>=w>=30 and 110>=h>=30 ):
            flag = 0
            for num,(x1,y1,w1,h1) in enumerate(store):
                if(x >= x1 and y >= y1 and x+w <= x1 + w1 and y + h <= y1 + h1):
                    flag = 1
                    break
                if(x <= x1 and y <= y1 and x+w >= x1 + w1 and y + h >= y1 + h1):
                    store[num] = (x,y,w,h)
                    flag = 1
                    break
            if(flag == 0):
                count += 1
                store.append((x,y,w,h))   
        if(30>=w>=15 and 70>=h>=65):
            count += 1
            store.append((x,y,w,h))  
    store = sorted(store , key = lambda x: x[0]) 
    for num,char in enumerate(captcha_correct_text):
        save_path1 = os.path.join(TRAIN_DATA_FOLDER, char)
        save_path = os.path.join(save_path1,filename)
        if not os.path.exists(save_path1):
            os.makedirs(save_path1)
        x,y,w,h = store[num]
        letter_image = rbg_img[y - 2:y + h + 2,x - 2:x + w + 2]
        cv2.imwrite(save_path, letter_image)
    # for x,y,w,h in store:
    #     rbg_img = cv2.rectangle(rbg_img,(x,y),(x+w,y+h),(0,150,0),2)
    # cv2.imshow("preview", rbg_img);
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()     
    if(count == len(captcha_correct_text)):
        true_match_count += 1
print("true match count = {}".format(str(true_match_count)))
print("total count = {}".format(str(len(captcha_image_files))))       
    








    # xdim,ydim = img.shape
    # val = img[0,0]
    # img[img == val] = 254
    # unique, counts = np.unique(img, return_counts=True)
    # no = len(unique)
    # ucount = 0
    # if(no > 1):
    #     no = no - 2
    #     while(ucount < 4 and no > -1):
    #         if(counts[no] > 1500):
    #             # print(no,counts[no])
    #             no -= 1
    #             ucount += 1
    #             continue
    #         img[img == unique[no]] = 254
    #         no -= 1
    #     while(no > -1):
    #         # print(no)
    #         img[img == unique[no]] = 254
    #         no -= 1  
    

    # save_path = os.path.join(OUTPUT_FOLDER, filename)
    # cv.imwrite(save_path, img)


# unique, counts = np.unique(img, return_counts=True)
# minunique = [0 for _ in range(len(unique))]
# maxunique = [0 for _ in range(len(unique))]
# for x1 in range(xdim):
#     for y1 in range(ydim):
#         if(img[x1,y1] != 254):
#             for x in range(len(unique)):
#                 if(unique[x] == img[x1,y1]):
#                     if(minunique[x] == 0):
#                         minunique[x] = (x1,y1)
#                     else:
#                         maxunique[x] = (x1,y1)
#                 break
# print(minunique)
# print(maxunique)                        

# print(pos[0].shape)
# print(unique)
# print(counts)        
# cv.imshow('preview',img)
# cv.waitKey(0)
# cv.destroyAllWindows()


# exit(0)


# kernel = np.ones((5,5),np.uint8)
# erosion = cv.erode(img,kernel,iterations = 1)



# img1 = cv.imread('./reference/A.png',0)          # queryImage
# print(img1.shape)
# # img2 = cv.imread('box_in_scene.png',0) # trainImage
# img2 = img
# # plt.imshow(img1,),plt.show()
# # Initiate SIFT detector
# # sift = cv.SIFT()
# sift = cv.xfeatures2d.SIFT_create()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

# # FLANN parameters
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary

# flann = cv.FlannBasedMatcher(index_params,search_params)

# matches = flann.knnMatch(des1,des2,k=2)

# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]

# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.6*n.distance:
#         matchesMask[i]=[1,0]

# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)

# m = sorted(matches , key = lambda x : x[0].distance)
# max1 = max(matches, key = lambda x:x[0].distance)
# max2 = max(matches, key = lambda x:x[1].distance)


# prin

# count = 1
# while(1):

#     while(count < len(m)):
#         # print(m[count - 1][1].distance)
#         if(m[count][1].distance < m[count-1][1].distance):
#             print(m[count-1][0].distance)
#             img[:,int(m[count-1][0].distance)] = 0
#             cv.imshow('preview', img)
#             cv.waitKey(0)
#             cv.destroyAllWindows()
#             break
#         count += 1
#     count += 2

# exit(0)



# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

# plt.imshow(img3,),plt.show()

