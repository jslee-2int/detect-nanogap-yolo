import os
import shutil
from sklearn.model_selection import train_test_split

# 원본 경로
image_dir = r'D:\yolo\datasets\one\images'
label_dir = r'D:\yolo\datasets\one\labels'

# 대상 경로
dataset_dir = r'D:\Py_Codes\label_studio\datasets\dataset'
train_image_dir = os.path.join(dataset_dir, 'images', 'train')
train_label_dir = os.path.join(dataset_dir, 'labels', 'train')
val_image_dir = os.path.join(dataset_dir, 'images', 'val')
val_label_dir = os.path.join(dataset_dir, 'labels', 'val')

# 폴더 생성
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
label_files = [f.replace('.jpg', '.txt') for f in image_files]

# train/val 분할 (8:2 비율)
train_images, val_images, train_labels, val_labels = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42
)

# 파일 복사 함수
def copy_files(files, src_dir, dst_dir):
    for file in files:
        shutil.copy(os.path.join(src_dir, file), dst_dir)

# 파일 복사 실행
copy_files(train_images, image_dir, train_image_dir)
copy_files(train_labels, label_dir, train_label_dir)
copy_files(val_images, image_dir, val_image_dir)
copy_files(val_labels, label_dir, val_label_dir)

print(f"Total files: {len(image_files)}")
print(f"Train files: {len(train_images)}")
print(f"Validation files: {len(val_images)}")
