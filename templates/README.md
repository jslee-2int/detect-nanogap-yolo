# 로컬 데이터 폴더 안내

보안상 SEM 이미지와 실제 데이터셋은 Git에 포함되지 않습니다.
클론 후 아래처럼 폴더를 만들고 데이터를 넣으세요.

```text
raw_img/                  # SEM 원본
resized_img/              # 리사이즈 결과
datasets/
  split_images/           # 타일 분할 결과
  moved_images/
  dataset/
    images/train|val/
    labels/train|val/
runs/                     # 학습 후 자동 생성
yolo11m.pt 등             # ultralytics가 다운로드하거나 로컬에 준비
```

	emplates/datasets/... 는 레이아웃 참고용 빈 구조입니다.
실제 작업은 저장소 루트의 datasets/, aw_img/ 를 사용합니다.
