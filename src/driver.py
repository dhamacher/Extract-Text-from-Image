from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import cv2
import pytesseract
import os

tess_env = str(os.getenv("TESSERACT_HOME"))

pytesseract.pytesseract.tesseract_cmd = tess_env

# decided to give tesseract a whitelist of acceptable character, since we preferred to have only the capital letters in
# other to avoid small text and strange characters that are sometimes found by tesseract.
custom_config = r"--oem 3 --psm 11 -c tessedit_char_whitelist= 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '"


def preprocess_final(image):
    """ Returns the preprocessed image."""
    image = cv2.bilateralFilter(image, 5, 55, 60)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 240, 255, 1)
    return image


if __name__ == '__main__':
    img = np.array(Image.open('..\\images\\clock-test.jpg'))
    im = preprocess_final(img)
    text = pytesseract.image_to_string(im, lang='eng', config=custom_config)
    print(text.replace('\n', ''))

# ----------------------------------------------------------------------------------------------------------------------

# SOURCE::[https://towardsdatascience.com/extract-text-from-memes-with-python-opencv-tesseract-ocr-63c2ccd72b69]

# ----------------------------------------------------------------------------------------------------------------------
# im = np.array(Image.open('..\\images\\clock-test.jpg'))
# plt.figure(figsize=(10,10))
# plt.title('PLAIN IMAGE')
# plt.imshow(im); plt.xticks([]); plt.yticks([])
# plt.savefig('..\\images\\01_plain_img.png')
#
#
# text = pytesseract.image_to_string(im)
#
# print(text.replace('\n', ' '))
#
#
# # Image cleaning
# #
# # The first function that we applied to our image is bilateral filtering. If you want to understand deeply how it
# # works, there is a nice tutorial on OpenCV site, and you can find the description of the parameters here.
# #
# # In a nutshell, this filter helps to remove the noise, but, in contrast with other filters, preserves edges instead
# # of blurring them. This operation is performed by excluding from the blurring of a point the neighbors that do
# # not present similar intensities. With the chosen parameters, the difference from the other image is not
# # strongly perceptible, however, it led to a better final performance.
# im= cv2.bilateralFilter(im,5, 55,60)
# plt.figure(figsize=(10,10))
# plt.title('BILATERAL FILTER')
# plt.imshow(im); plt.xticks([]); plt.yticks([])
# plt.savefig('..\\images\\02_filtered_img.png', bbox_inches='tight')
#
#
# # The second operation it’s pretty clear: we project our RGB images in grayscale.
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# plt.figure(figsize=(10,10))
# plt.title('GRAYSCALE IMAGE')
# plt.imshow(im, cmap='gray'); plt.xticks([]); plt.yticks([])
# plt.savefig('..\\images\\03_grayscale_img.png', bbox_inches='tight')
#
#
# # The last transformation is binarization. For every pixel, the same threshold value is applied. If the pixel value
# # is smaller than the threshold, it is set to 0, otherwise, it is set to 255. Since we have white text, we want
# # to blackout everything is not almost perfectly white (not exactly perfect since usually text is not “255-white”.
# # We found that 240 was a threshold that could do the work. Since tesseract is trained to recognize black text,
# # we also need to invert the colors. The function threshold from OpenCV can do the two operations jointly,
# # by selecting the inverted binarization.
# _, im = cv2.threshold(im, 240, 255, 1)
# plt.figure(figsize=(10,10))
# plt.title('IMMAGINE BINARIA')
# plt.imshow(im, cmap='gray'); plt.xticks([]); plt.yticks([])
# plt.savefig('..\\images\\04_binarization_img.png', bbox_inches='tight')
#
# # Final run.
# text = pytesseract.image_to_string(im, lang='eng', config=custom_config)
#
# print(text.replace('\n', ''))
#
# print('done')
