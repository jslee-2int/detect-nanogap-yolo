import os
import cv2
import numpy as np


def split_image_and_labels(image_path, label_path, output_dir, grid_size=(4, 4)):
    """
    이미지와 YOLO 형식의 라벨 파일을 grid_size에 맞게 분할합니다.

    Args:
        image_path (str): 입력 이미지 경로
        label_path (str): 입력 라벨 파일 경로 (.txt)
        output_dir (str): 출력 디렉토리 경로
        grid_size (tuple): 분할할 그리드 크기 (행, 열)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 로드
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # 각 분할 영역의 크기 계산
    tile_height = height // grid_size[0]
    tile_width = width // grid_size[1]

    # YOLO 라벨 로드
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # 각 그리드에 대해 처리
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 분할 영역 좌표 계산
            y1 = i * tile_height
            y2 = (i + 1) * tile_height
            x1 = j * tile_width
            x2 = (j + 1) * tile_width

            # 이미지 분할
            tile_img = image[y1:y2, x1:x2]

            # 분할된 이미지의 새 좌표계에서의 라벨 계산
            tile_labels = []
            for label in labels:
                class_id, x_center, y_center, w, h = map(float, label.strip().split())

                # YOLO 좌표를 픽셀 좌표로 변환
                abs_x = x_center * width
                abs_y = y_center * height
                abs_w = w * width
                abs_h = h * height

                # 박스가 현재 타일과 겹치는지 확인
                box_x1 = abs_x - abs_w / 2
                box_y1 = abs_y - abs_h / 2
                box_x2 = abs_x + abs_w / 2
                box_y2 = abs_y + abs_h / 2

                if (box_x1 < x2 and box_x2 > x1 and
                        box_y1 < y2 and box_y2 > y1):
                    # 겹치는 영역 계산
                    intersect_x1 = max(box_x1, x1)
                    intersect_y1 = max(box_y1, y1)
                    intersect_x2 = min(box_x2, x2)
                    intersect_y2 = min(box_y2, y2)

                    # 새로운 박스 중심점과 크기 계산
                    new_x_center = (intersect_x1 + intersect_x2) / 2 - x1
                    new_y_center = (intersect_y1 + intersect_y2) / 2 - y1
                    new_w = intersect_x2 - intersect_x1
                    new_h = intersect_y2 - intersect_y1

                    # YOLO 형식으로 변환 (0~1 범위)
                    new_x_center /= tile_width
                    new_y_center /= tile_height
                    new_w /= tile_width
                    new_h /= tile_height

                    # 유효한 박스인지 확인
                    if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and new_w > 0 and new_h > 0:
                        tile_labels.append(f"{int(class_id)} {new_x_center} {new_y_center} {new_w} {new_h}\n")

            # 분할된 이미지와 라벨 저장
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(f"{output_dir}/{base_name}_tile_{i}_{j}.jpg", tile_img)

            if tile_labels:  # 라벨이 있는 경우에만 파일 생성
                with open(f"{output_dir}/{base_name}_tile_{i}_{j}.txt", 'w') as f:
                    f.writelines(tile_labels)


# 사용 예시
def main():
    image_path = "datasets/dataset/raw/image4.jpg"
    label_path = "datasets/dataset/labels/image4.txt"
    output_dir = "datasets"

    split_image_and_labels(image_path, label_path, output_dir)


if __name__ == "__main__":
    main()