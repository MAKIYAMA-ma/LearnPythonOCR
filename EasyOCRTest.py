from matplotlib import pyplot as plt
import cv2
import easyocr
import sys


def detect_text_easyocr(image_path):
    reader = easyocr.Reader(['ja'])
    img = cv2.imread(image_path)
    results = reader.readtext(img)

    for i in range(len(results["text"])):
        if int(results["conf"][i]) > 60:
            x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
            text = results["text"][i]

            print(str(x) + "," + str(y) + "," + str(w) + "," + str(h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.imshow(img)
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
