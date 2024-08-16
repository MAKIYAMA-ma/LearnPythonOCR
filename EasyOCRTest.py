import cv2
import easyocr
import fiftyone as fo
import fiftyone.core.labels as fol
import sys


def detect_text_easyocr(image_path):
    reader = easyocr.Reader(['en'])  # 言語を指定
    img = cv2.imread(image_path)
    results = reader.readtext(img)

    detections = []
    for (bbox, text, prob) in results:
        if prob > 0.6:  # 信頼度のフィルタリング
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x_min = min(top_left[0], bottom_left[0]) / img.shape[1]
            y_min = min(top_left[1], top_right[1]) / img.shape[0]
            x_max = max(top_right[0], bottom_right[0]) / img.shape[1]
            y_max = max(bottom_left[1], bottom_right[1]) / img.shape[0]
            bounding_box = [x_min, y_min, x_max - x_min, y_max - y_min]
            detections.append(fol.Detection(label=text, bounding_box=bounding_box))

    return detections


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    image_path = sys.argv[1]

    # FiftyOne用のサンプルデータセットを作成
    dataset = fo.Dataset(name="book_covers")

    detections = detect_text_easyocr(image_path)

    # サンプル画像に検出されたテキストを追加
    sample = fo.Sample(filepath=image_path)
    sample["detections"] = fol.Detections(detections=detections)
    dataset.add_sample(sample)

    # FiftyOne GUIで結果を確認
    session = fo.launch_app(dataset)
