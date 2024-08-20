from OCRTest import OCRTest
from PIL import Image, ImageDraw, ImageFont
import cv2
import easyocr
import numpy as np
import sys


class EasyOCRTest(OCRTest):
    def detect_text_ocr(self, prepro_img, original_img):
        img = original_img
        reader = easyocr.Reader(['ja'])
        results = reader.readtext(prepro_img)

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

                draw.polygon([tuple(points[0]), tuple(points[1]), tuple(points[2]), tuple(points[3])],
                             outline=(0, 0, 255), width=2)
                draw.text((p0[0], p0[1] - 40), text, font=font, fill=(0, 0, 255))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    tester = EasyOCRTest()
    image_path = sys.argv[1]
    tester.run(image_path)
