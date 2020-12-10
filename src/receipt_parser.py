from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import os
import logging
logger = logging.getLogger(__name__)


def logging_setup():
    log_path = '..\\log\\ocr.log'
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.FileHandler(log_path, mode='a', encoding=None, delay=False)
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger


def bilateral_filter(img):
    try:
        im = cv2.bilateralFilter(img, 5, 55, 60)
        plt.figure(figsize=(10, 10))
        plt.title('BILATERAL FILTER')
        plt.imshow(im)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('..\\images\\02_filtered_img.png', bbox_inches='tight')
        return im
    except Exception as e:
        logger.exception(str(e))
    return None


def apply_grayscale(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(10,10))
    plt.title('GRAYSCALE IMAGE')
    plt.imshow(im, cmap='gray'); plt.xticks([]); plt.yticks([])
    plt.savefig('..\\images\\03_grayscale_img.png', bbox_inches='tight')
    return im


def binarization(img):
    _, im = cv2.threshold(img, 190, 255, 1)  # 2nd param affects outcome of text recognition
    plt.figure(figsize=(10, 10))
    plt.title('IMMAGINE BINARIA')
    plt.imshow(im, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('..\\images\\04_binarization_img.png', bbox_inches='tight')
    return im


def parse():
    try:
        log = logging_setup()
        tess_env = str(os.getenv("TESSERACT_HOME"))
        pytesseract.pytesseract.tesseract_cmd = tess_env
        image_file = Image.open('..\\images\\clock-test.jpg')
        image_array = np.array(image_file)
        # Transformation
        im = bilateral_filter(image_array)
        im = apply_grayscale(im)
        im = binarization(im)
        log.info('Image loaded.')
        custom_config = r"--oem 3 --psm 1 -c tessedit_char_whitelist= 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.$/ '"
        text = pytesseract.image_to_string(im, config=custom_config)
        log.info('pytesseract.image_to_string() executed.')
        with open('..\\images\\text_output.txt', 'w') as f:
            f.write(text)
            log.info('Text output written.')
        log.info('Parsing complete!')
    except Exception as e:
        log.exception(f'{str(e)}')


parse()
