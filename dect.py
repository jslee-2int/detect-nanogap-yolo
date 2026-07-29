import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.gridspec import GridSpec


def draw_boxes_by_size(image, results):
    # 이미지 복사
    annotated_image = image.copy()

    # 모든 박스의 크기 계산을 위한 리스트
    boxes_with_size = []

    # 각 감지된 객체에 대해
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 박스 좌표 가져오기
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # 정수로 변환
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 박스 크기 계산 (면적)
            box_size = (x2 - x1) * (y2 - y1)

            boxes_with_size.append({
                'coords': (x1, y1, x2, y2),
                'size': box_size
            })

    # 크기별로 정렬
    boxes_with_size.sort(key=lambda x: x['size'], reverse=True)

    # 전체 박스 수
    total_boxes = len(boxes_with_size)

    # 크기만 추출
    sizes = [box['size'] for box in boxes_with_size]

    # 크기 그룹화
    bins = 7
    size_groups, bin_edges = np.histogram(sizes, bins=bins)

    # 크기 분포 그래프 및 테이블 생성
    fig = plt.figure(figsize=(12, 6))
    spec = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)

    # 히스토그램
    ax_hist = fig.add_subplot(spec[0, 0])
    colors = cm.rainbow(np.linspace(0, 1, bins))
    ax_hist.bar(range(1, bins + 1), size_groups, color=colors, edgecolor="black")
    ax_hist.set_xlabel("Size Groups")
    ax_hist.set_ylabel("Number of Boxes")
    ax_hist.set_title("Size Distribution of Detected Boxes")
    ax_hist.set_xticks(range(1, bins + 1))
    ax_hist.set_xticklabels([f"{int(bin_edges[i])}~{int(bin_edges[i + 1])}" for i in range(bins)], rotation=45)

    # 테이블
    ax_table = fig.add_subplot(spec[0, 1])
    ax_table.axis("off")  # 축 숨기기
    table_data = [[f"{int(bin_edges[i])}~{int(bin_edges[i + 1])}", size_groups[i]] for i in range(bins)]
    col_labels = ["Size Range", "Count"]
    ax_table.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center").scale(1, 2)

    plt.tight_layout()
    plt.show()

    # 레인보우 색상 생성 (BGR 형식)
    rainbow_colors = [tuple(map(int, np.array(cm.rainbow(i / 7)[:3]) * 255)) for i in range(7)]

    # 박스 그리기
    for i, box_info in enumerate(boxes_with_size):
        x1, y1, x2, y2 = box_info['coords']

        # 크기 그룹에 따라 색상 결정
        group_index = np.digitize(box_info['size'], bin_edges) - 1
        group_index = min(group_index, 6)  # 최대 6 (인덱스 0-6)
        color = rainbow_colors[group_index]

        # 박스 그리기
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, -1)

    # 투명도 설정
    alpha = 0.2
    annotated_image = cv2.addWeighted(annotated_image, alpha, image, 1 - alpha, 0)

    print(f"Total detected objects: {total_boxes}")
    return annotated_image


def predict_image(model_path, image_path, save_path=None, conf=0.5, iou=0.4):
    # 모델 로드
    model = YOLO(model_path)

    # 이미지 읽기
    image = cv2.imread(image_path)

    # 예측 수행
    results = model(image, conf=conf, iou=iou, max_det=5000)

    # 크기별로 색상이 다른 바운딩 박스 그리기
    annotated_image = draw_boxes_by_size(image, results)

    # 결과 이미지 저장
    if save_path:
        cv2.imwrite(save_path, annotated_image)

    # 결과 표시
    cv2.imshow('Prediction', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 사용 예시
predict_image(
    model_path=r'runs\detect\train22\weights\best.pt',
    image_path=r'datasets/split_images/10-8-8_q001_tile_0_0.jpg',
    save_path='output.jpg',
    max_det=5000,
    conf=0.5,
    iou=0.3
)
