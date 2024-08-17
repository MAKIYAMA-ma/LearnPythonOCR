from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import cv2
import easyocr
import numpy as np
import sys


def detect_text_easyocr(image_path):
    reader = easyocr.Reader(['ja'])
    img = cv2.imread(image_path)
    prepre_img = pre_process(img)
    results = reader.readtext(prepre_img)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype("msgothic.ttc", 40)

    for result in results:
        if result[2] > 0.20:
            # for tuning up
            print(result)
        if result[2] > 0.60:
            points = np.array(result[0], np.int32)
            p0 = result[0][0]
            text = result[1]

            draw.polygon([tuple(points[0]), tuple(points[1]), tuple(points[2]), tuple(points[3])], outline=(0, 0, 255), width=2)
            draw.text((p0[0], p0[1] - 40), text, font=font, fill=(0, 0, 255))

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    plt.imshow(img)
    plt.axis('off')
    plt.show()


def pre_process(img):
    # グレースケール化
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # バイラテラルフィルタでノイズ除去
    # denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)
    # return denoised_image

    # ガウシアンフィルタでノイズ除去
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image

    # # 大津の方法で二値化
    # _, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # return binary_image


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    image_path = sys.argv[1]
    detect_text_easyocr(image_path)
