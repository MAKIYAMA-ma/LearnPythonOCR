from matplotlib import pyplot as plt
import abc
import cv2


class OCRTest(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def detect_text_ocr(self, prepro_img, original_img):
        pass

    def pre_process(self, img):
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

    def run(self, image_path: str):
        img = cv2.imread(image_path)
        prepro_img = self.pre_process(img)
        img = self.detect_text_ocr(prepro_img, img)

        plt.imshow(img)
        plt.axis('off')
        plt.show()
