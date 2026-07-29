from ultralytics import YOLO

# 1. 사전 훈련된 모델 불러오기
model = YOLO(r'runs\detect\train21\weights\best.pt')  # 또는 다른 크기의 모델 선택 가능 (n, s, m, l, x)

# 2. Auto-annotation 실행
# - source: 이미지나 비디오가 있는 디렉토리 경로
# - save: 결과 저장 여부
# - conf: confidence threshold (신뢰도 임계값)
# - save_txt: YOLO 형식의 txt 파일로 결과 저장
results = model.predict(
    source='datasets/split_images',  # 이미지 경로
    save=True,                     # 결과 이미지 저장
    conf=0.5,                      # 신뢰도 임계값 설정
    iou=0.1,
    max_det=5000,
    save_txt=True                  # txt 파일로 레이블 저장
)
