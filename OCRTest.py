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
        temp_img = img.copy()

        # 黒っぽい色以外は白とする
        # threshold = 200
        # mask = (temp_img[:, :, 0] >= threshold) | (temp_img[:, :, 1] >= threshold) | (temp_img[:, :, 2] >= threshold)
        # temp_img[mask] = [255, 255, 255]

        # グレースケール化
        if False:
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        else:
            # HSV色空間に変換し、Vチャネルでグレースケール化
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2HSV)
            temp_img = temp_img[:, :, 2]

        # ノイズ除去
        if True:
            # ガウシアンフィルタでノイズ除去
            temp_img = cv2.GaussianBlur(temp_img, (5, 5), 0)
        else:
            # バイラテラルフィルタでノイズ除去
            temp_img = cv2.bilateralFilter(temp_img, 9, 75, 75)

        # 大津法で二値化
        # 本の表紙の場合、背景にも絵があるので二値化すると判読不可となる。
        # 白地に文字を書いたようなドキュメントのOCRなら有効。
        # _, temp_img = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 傾き補正
        # coords = np.column_stack(np.where(temp_img > 0))
        # angle = cv2.minAreaRect(coords)[-1]
        # if angle < -45:
        #     angle = -(90 + angle)
        # else:
        #     angle = -angle
        # (h, w) = temp_img.shape[:2]
        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # temp_img = cv2.warpAffine(temp_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # コントラスト調整
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        temp_img = clahe.apply(temp_img)

        if False:
            # for Debug
            # 前処理後の画像確認
            plt.imshow(temp_img, cmap='gray')
            plt.axis('off')
            plt.show()

        return temp_img

    def resize_image(self, image, target_size=1000):
        # 短辺をtarget_sizeとする倍率を計算
        height, width = image.shape[:2]
        if height < width:
            scale = target_size / height
        else:
            scale = target_size / width

        # 画像のリサイズ
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))

        return resized_image

    def run(self, image_path: str):
        img = cv2.imread(image_path)
        img = self.resize_image(img)
        prepro_img = self.pre_process(img)
        img = self.detect_text_ocr(prepro_img, img)

        plt.imshow(img)
        plt.axis('off')
        plt.show()
