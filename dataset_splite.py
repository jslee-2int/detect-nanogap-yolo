import os
import shutil
from sklearn.model_selection import train_test_split

# 이미지와 라벨 경로
images = "dataset/images"
labels = "dataset/labels"

# 모든 파일 리스트 가져오기
image_files = [f for f in os.listdir(images) if f.endswith(".jpg")]
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# 폴더 생성
os.makedirs("train/images", exist_ok=True)
os.makedirs("train/labels", exist_ok=True)
os.makedirs("val/images", exist_ok=True)
os.makedirs("val/labels", exist_ok=True)

# 파일 이동
for file in train_files:
    shutil.move(os.path.join(images, file), "train/images/")
    shutil.move(os.path.join(labels, file.replace(".jpg", ".txt")), "train/labels/")

for file in val_files:
    shutil.move(os.path.join(images, file), "val/images/")
    shutil.move(os.path.join(labels, file.replace(".jpg", ".txt")), "val/labels/")
