# import numpy as np

# # DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# # INPUT CONVENTION
# # filenames: a list of strings containing filenames of images

# # OUTPUT CONVENTION
# # The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# # Make sure that the length of the array and the list is the same as the number of filenames that 
# # were given. The evaluation code may give unexpected results if this convention is not followed.

# def decaptcha( filenames ):
#     numChars = 3 * np.ones( (len( filenames ),) )
#     # The use of a model file is just for sake of illustration
#     file = open( "model.txt", "r" )
#     codes = file.read().splitlines()
#     file.close()
#     return (numChars, codes)
import glob
import os
import string
from threading import Thread

from keras.models import load_model
from helpers import resize_to_fit
import numpy as np
import cv2
import pickle

import tensorflow as tf

# graph = tf.get_default_graph()
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that 
# were given. The evaluation code may give unexpected results if this convention is not followed.


model = ''
lb = ''

def decaptcha( filenames ):
    global model
    global lb
    MODEL_FILENAME = "captcha_model.hdf5"
    MODEL_LABELS_FILENAME = "model_labels.dat"
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)
    model = load_model(MODEL_FILENAME)
    (numChars,codes) = run(1,filenames,{})

    return (numChars, codes)



def run(procnum,filenames,return_dict):
    numChars = np.ones((len(filenames),))
    codes = []
    for i, file in enumerate(filenames):
        filename = os.path.basename(file)
        img2 = cv2.imread(file)
        val2 = img2[0, 0]
        img2[(img2[:, :, 0] == val2[0]) & (img2[:, :, 1] == val2[1]) & (img2[:, :, 2] == val2[2])] = [0, 0, 0]
        img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        erosion = cv2.erode(thresh1, kernel, iterations=1)
        inv_erosion = cv2.bitwise_not(erosion)
        # rbg_img = cv2.cvtColor(inv_erosion, cv2.COLOR_GRAY2RGB)
        newimg,contours, hierarchy = cv2.findContours(inv_erosion, 1, 2)
        count = 0
        store = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (110 >= w >= 30 and 110 >= h >= 30):
                flag = 0
                for num, (x1, y1, w1, h1) in enumerate(store):
                    if (x >= x1 and y >= y1 and x + w <= x1 + w1 and y + h <= y1 + h1):
                        flag = 1
                        break
                    if (x <= x1 and y <= y1 and x + w >= x1 + w1 and y + h >= y1 + h1):
                        store[num] = (x, y, w, h)
                        flag = 1
                        break
                if (flag == 0):
                    count += 1
                    store.append((x, y, w, h))
            if (30 >= w >= 15 and 70 >= h >= 65):
                count += 1
                store.append((x, y, w, h))
        store = sorted(store, key=lambda x: x[0])
        pred = ""
        for x, y, w, h in store:
            letter_image = inv_erosion[y - 2:y + h + 2, x - 2:x + w + 2]
            # size = 20
            # rot_img2 = resize_to_fit(letter_image, size, size)
            # cv2.imshow("image",letter_image)
            # image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)


            image = resize_to_fit(letter_image, 20, 20)
            image = np.expand_dims(image, axis=2)
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)
            letter = lb.inverse_transform(prediction)[0]


            # alphas = list(string.ascii_uppercase)
            # CAPTCHA_IMAGE_FOLDER = "./reference_new"
            # result = ['',1]
            # for alpha in alphas:
            #     folder = CAPTCHA_IMAGE_FOLDER + '/' + alpha
            #     if (os.path.isdir(folder)):
            #         captcha_image_files = glob.glob(os.path.join(folder, "*"))
            #         # min_err = 1
            #         for (j, captcha_image_file) in enumerate(captcha_image_files):
            #             path = captcha_image_file
            #             img = cv2.imread(path, 0)
            #             # img = resize_to_fit(img, size, size)
            #             img3 = cv2.bitwise_xor(img, rot_img2)
            #             err = np.count_nonzero(img3) / (size * size)
            #             if(err < result[1]):
            #                 result[0] = alpha
            #                 result[1] = err
            # pred += result[0]
            pred += letter
        # if((pred) != (captcha_correct_text)):
        #     print(" {} and {} ".format(pred,file))
        # print(pred)
        # print(" {} and {} ".format(pred,file))
        codes.append(pred)
        numChars[i] = len(pred)
    # print("done")
    if not return_dict.get(str(procnum),None):
        return_dict[str(procnum)]= {}
    return_dict[str(procnum)]['0'] = codes
    return_dict[str(procnum)]['1'] = numChars
    # print("len = ", numChars.shape)
    return (numChars,codes)