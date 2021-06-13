import requests
import time
import json
import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
from pytesseract import image_to_string, image_to_data, image_to_boxes, image_to_osd, Output
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ocr_result(image_path):
    image = Image.open(image_path)
    enh_con = ImageEnhance.Contrast(image)
    contrast = 2.0
    image = enh_con.enhance(contrast)
    # image = image.convert('L')
    # image = image.resize((800, 800))
    # image.save('OCR_temp.png')
    # image_data = open('../OCR_temp.png', "rb").read()
    return image_to_string(image, lang='eng', config='--psm 0')


def test(image_path):
    # Simple image to string
    print(image_to_string(Image.open(image_path)))

    # Get bounding box estimates
    print(image_to_boxes(Image.open(image_path)))

    # Get verbose data including boxes, confidences, line and page numbers
    print(image_to_data(Image.open(image_path), output_type=Output.STRING))

    # Get information about orientation and script detection
    print(image_to_osd(Image.open(image_path)))


def show_ocr_image(image_path, lang='eng', config=''):
    image = cv2.imread(image_path)
    results = image_to_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), output_type=Output.DICT, lang=lang, config=config)
    for i in range(len(results["text"])):
        # extract the bounding box coordinates of the text region from the current result
        tmp_tl_x = results["left"][i]
        tmp_tl_y = results["top"][i]
        tmp_br_x = tmp_tl_x + results["width"][i]
        tmp_br_y = tmp_tl_y + results["height"][i]
        tmp_level = results["level"][i]
        conf = results["conf"][i]
        text = results["text"][i]

        if tmp_level == 5:
            cv2.putText(image, text, (tmp_tl_x, tmp_tl_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(image, (tmp_tl_x, tmp_tl_y), (tmp_br_x, tmp_br_y), (0, 0, 255), 1)

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    im_shape = np.shape(image)
    print(im_shape)
    plt.figure(figsize=(im_shape[1] / 100, im_shape[0] / 100), dpi=300)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.imsave('test.png',image)
    plt.show()


def show_ocr_pil(image_path, lang='eng', config=''):
    image = Image.open(image_path)
    results = image_to_data(image, output_type=Output.DICT, lang=lang, config=config)
    font = ImageFont.truetype('../font/sarasa-monoT-sc-regular.ttf')
    draw = ImageDraw.Draw(image)
    for i in range(len(results["text"])):
        # extract the bounding box coordinates of the text region from the current result
        tmp_tl_x = results["left"][i]
        tmp_tl_y = results["top"][i]
        tmp_br_x = tmp_tl_x + results["width"][i]
        tmp_br_y = tmp_tl_y + results["height"][i]
        tmp_level = results["level"][i]
        conf = results["conf"][i]
        text = results["text"][i]

        if tmp_level == 5:
            draw.text(xy=(tmp_tl_x, tmp_tl_y - 5), text=text, font=font, fill=(255, 0, 0))
            draw.rectangle(xy=[(tmp_tl_x, tmp_tl_y), (tmp_br_x, tmp_br_y)], outline=(255, 0, 0))

    im_shape = np.shape(image)
    print(im_shape)
    plt.figure(figsize=(im_shape[1] / 100, im_shape[0] / 100), dpi=300)
    plt.imshow(image)
    # plt.imsave('test.png',image)
    plt.show()


if __name__ == '__main__':
    # image_path = '../test_image/OCR_temp.png'

    # res = ocr_result(image_path)
    # print(res)
    # test(image_path)
    # image_path = '../test_image/eng-text.png'
    # show_ocr_image(image_path)

    # image_path = '../test_image/chi_text.png'
    # show_ocr_pil(image_path, lang='chi_sim')

    image_path = '../test_image/OCR_temp.png'
    show_ocr_pil(image_path, lang='eng')
