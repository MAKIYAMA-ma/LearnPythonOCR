import cv2
import fiftyone as fo
import fiftyone.core.labels as fol
import pytesseract
import sys


def detect_text_tesseract(image_path):
    # 画像を読み込む
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # テキスト検出
    results = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)

    detections = []
    for i in range(len(results["text"])):
        if int(results["conf"][i]) > 60:  # 信頼度のフィルタリング
            x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
            label = results["text"][i]
            bounding_box = [x/img.shape[1], y/img.shape[0], w/img.shape[1], h/img.shape[0]]
            detections.append(fol.Detection(label=label, bounding_box=bounding_box))

    return detections


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    image_path = sys.argv[1]

    # FiftyOne用のサンプルデータセットを作成
    dataset = fo.Dataset(name="book_covers")

    detections = detect_text_tesseract(image_path)

    # サンプル画像に検出されたテキストを追加
    sample = fo.Sample(filepath=image_path)
    sample["detections"] = fol.Detections(detections=detections)
    dataset.add_sample(sample)

    # FiftyOne GUIで結果を確認
    session = fo.launch_app(dataset)
