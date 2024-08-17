from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pytesseract
import sys


def detect_text_tesseract(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prepre_img = pre_process(img_rgb)

    # results = pytesseract.image_to_string(prepre_img, lang='jpn', output_type=pytesseract.Output.DICT)
    results = pytesseract.image_to_data(prepre_img, lang='jpn', output_type=pytesseract.Output.DICT)
    print(results)

    img_pil = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype("msgothic.ttc", 40)

    for i in range(len(results["text"])):
        if int(results["conf"][i]) > 60:  # 信頼度のフィルタリング
            x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
            points = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], np.int32)
            text = results["text"][i]

            draw.polygon([tuple(points[0]), tuple(points[1]), tuple(points[2]), tuple(points[3])], outline=(0, 0, 255), width=2)
            draw.text((x, y - 40), text, font=font, fill=(0, 0, 255))
            # print(str(x) + "," + str(y) + "," + str(w) + "," + str(h))
            # cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.putText(img_rgb, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    img_rgb = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    plt.imshow(img_rgb)
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
    detect_text_tesseract(image_path)
