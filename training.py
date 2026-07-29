from ultralytics import YOLO
import torch
torch.cuda.empty_cache()

if __name__ == '__main__':
    # 모델 초기화
    model = YOLO('yolo11m.pt')  # 사전 훈련된 모델 로드
    # model = YOLO(r'runs\detect\train9\weights\last.pt')  # 사전 훈련된 모델 로드

    # 트레이닝 실행
    results = model.train(
        data='data.yaml',
        epochs=500,
        imgsz=704,
        batch=4,
        device=0,  # GPU 사용
        workers=2  # 워커 수 줄이기
    )
