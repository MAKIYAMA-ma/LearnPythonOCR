from matplotlib import pyplot as plt
import cv2
import pytesseract
import sys


def detect_text_tesseract(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prepre_img = pre_process(img)

    results = pytesseract.image_to_data(prepre_img, output_type=pytesseract.Output.DICT)

    for i in range(len(results["text"])):
        if int(results["conf"][i]) > 60:
            x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
            text = results["text"][i]

            print(str(x) + "," + str(y) + "," + str(w) + "," + str(h))
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img_rgb, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


def pre_process(img):
    # グレースケール化
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # バイラテラルフィルタでノイズ除去
    denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)
    return denoised_image

    # # 大津の方法で二値化
    # _, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # return binary_image


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    image_path = sys.argv[1]
    detect_text_tesseract(image_path)
