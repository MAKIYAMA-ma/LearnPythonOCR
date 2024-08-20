from OCRTest import OCRTest
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pytesseract
import sys


class TesseractOCRTest(OCRTest):
    def detect_text_ocr(self, prepro_img, original_img):
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # results = pytesseract.image_to_string(prepro_img, lang='jpn', output_type=pytesseract.Output.DICT)
        results = pytesseract.image_to_data(prepro_img, lang='jpn', output_type=pytesseract.Output.DICT)
        print(results)

        img_pil = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        font = ImageFont.truetype("msgothic.ttc", 40)

        for i in range(len(results["text"])):
            if int(results["conf"][i]) > 60:  # 信頼度のフィルタリング
                x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
                points = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], np.int32)
                text = results["text"][i]

                draw.polygon([tuple(points[0]), tuple(points[1]), tuple(points[2]), tuple(points[3])],
                             outline=(0, 0, 255), width=2)
                draw.text((x, y - 40), text, font=font, fill=(0, 0, 255))
                # print(str(x) + "," + str(y) + "," + str(w) + "," + str(h))
                # cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.putText(img_rgb, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        img_rgb = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return img_rgb


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    tester = TesseractOCRTest()
    image_path = sys.argv[1]
    tester.run(image_path)
