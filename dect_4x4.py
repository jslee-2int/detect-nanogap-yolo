from ultralytics import YOLO
import cv2
import numpy as np

# YOLO 모델 로드
model = YOLO(r'runs\detect\train22\weights\best.pt')  # YOLOv8 모델 로드 (적절한 모델 경로로 수정)


# 이미지 분할 함수
def split_image(image, grid_size):
    """이미지를 grid_size x grid_size로 분할합니다."""
    h, w, _ = image.shape
    h_step, w_step = h // grid_size, w // grid_size
    sub_images = []
    for i in range(grid_size):
        for j in range(grid_size):
            x1, y1 = j * w_step, i * h_step
            x2, y2 = x1 + w_step, y1 + h_step
            sub_images.append((image[y1:y2, x1:x2], (x1, y1, x2, y2)))
    return sub_images


# 예측 결과 원본 좌표로 변환
def adjust_boxes(boxes, offset, scale):
    """박스 좌표를 원본 이미지 기준으로 조정합니다."""
    x_offset, y_offset, _, _ = offset
    for box in boxes:
        box[0] += x_offset  # x_min
        box[1] += y_offset  # y_min
        box[2] += x_offset  # x_max
        box[3] += y_offset  # y_max
    return boxes


# 이미지 병합 및 시각화
def draw_boxes(image, boxes, labels, confidences):
    """이미지에 바운딩 박스를 그립니다."""
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        text = f"{label} {conf:.2f}"
        # cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image


# 메인 코드
def predict_and_merge(image_path, grid_size=4):
    """이미지를 분할하여 예측하고 결과를 합칩니다."""
    # 이미지 로드
    image = cv2.imread(image_path)
    original_image = image.copy()
    sub_images = split_image(image, grid_size)

    all_boxes, all_labels, all_confs = [], [], []

    # 분할된 이미지 각각 예측
    for sub_image, offset in sub_images:
        results = model.predict(sub_image, verbose=False)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x_min, y_min, x_max, y_max]
            confidences = result.boxes.conf.cpu().numpy()
            labels = [model.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]

            # 좌표를 원본 이미지 기준으로 변환
            adjusted_boxes = adjust_boxes(boxes, offset, scale=1.0)

            # 결과 저장
            all_boxes.extend(adjusted_boxes)
            all_labels.extend(labels)
            all_confs.extend(confidences)

    # 원본 이미지에 결과 그리기
    result_image = draw_boxes(original_image, all_boxes, all_labels, all_confs)

    # 결과 표시
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 이미지 경로 및 실행
image_path = r"raw_img/10-14-6 Top Pt_q001.jpg"  # 처리할 이미지 경로
predict_and_merge(image_path, grid_size=4)
