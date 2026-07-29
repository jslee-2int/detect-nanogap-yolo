import os
import cv2
import random
import shutil


def split_image(image_path, output_dir, grid_size=(3, 3)):
    """
    이미지를 전처리하고 grid_size에 맞게 분할합니다.

    Args:
        image_path (str): 입력 이미지 경로
        output_dir (str): 출력 디렉토리 경로
        grid_size (tuple): 분할할 그리드 크기 (행, 열)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return

    # 1. bottom 130px crop
    height, width = image.shape[:2]
    image = image[:height-130, :]

    # 2. 1:1 비율로 만들기 위해 양옆 crop
    height, width = image.shape[:2]
    if width > height:
        # 너비가 더 큰 경우, 양옆을 자름
        diff = width - height
        left_crop = diff // 2
        image = image[:, left_crop:left_crop+height]
    elif height > width:
        # 높이가 더 큰 경우, 위아래를 자름
        diff = height - width
        top_crop = diff // 2
        image = image[top_crop:top_crop+width, :]

    height, width = image.shape[:2]

    # 각 분할 영역의 크기 계산
    tile_height = height // grid_size[0]
    tile_width = width // grid_size[1]

    # 각 그리드에 대해 이미지 분할
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 분할 영역 좌표 계산
            y1 = i * tile_height
            y2 = (i + 1) * tile_height
            x1 = j * tile_width
            x2 = (j + 1) * tile_width

            # 이미지 분할
            tile_img = image[y1:y2, x1:x2]

            # 분할된 이미지 저장
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(f"{output_dir}/{base_name}_tile_{i}_{j}.jpg", tile_img)


def move_random_images(input_folder, output_folder, ratio=0.5):
    """
    입력 폴더의 이미지 파일 중 일부를 무작위로 선택하여 새 폴더로 이동합니다.
    
    Args:
        input_folder (str): 원본 이미지 폴더 경로
        output_folder (str): 이동할 이미지 폴더 경로
        ratio (float): 이동할 이미지 비율 (0.0 ~ 1.0)
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    # 이동할 이미지 수 계산
    num_to_move = int(len(image_files) * ratio)
    
    # 무작위로 이미지 선택
    selected_images = random.sample(image_files, num_to_move)
    
    # 선택된 이미지 이동
    for img in selected_images:
        src = os.path.join(input_folder, img)
        dst = os.path.join(output_folder, img)
        shutil.move(src, dst)
        print(f"Moved: {src} -> {dst}")


def process_folder(input_folder, output_dir, grid_size=(4, 4)):
    """
    폴더 내 모든 이미지 파일을 분할합니다.

    Args:
        input_folder (str): 입력 폴더 경로
        output_dir (str): 출력 디렉토리 경로
        grid_size (tuple): 분할할 그리드 크기 (행, 열)
    """
    # 폴더 내 모든 파일에 대해 처리
    for filename in os.listdir(input_folder):
        # 파일 경로
        file_path = os.path.join(input_folder, filename)

        # 파일이 이미지인지 확인 (JPEG, PNG 등)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            try:
                # 이미지 분할 함수 호출
                split_image(file_path, output_dir, grid_size)
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"Skipping non-image file: {filename}")


# 사용 예시
def main():
    input_folder = "./raw_img"  # 이미지가 있는 폴더
    output_dir = "datasets/split_images"  # 분할된 이미지를 저장할 폴더
    moved_images_dir = "datasets/moved_images"  # 이동된 이미지를 저장할 폴더
    
    # 1. 이미지 파일 중 일부를 새 폴더로 이동
    move_random_images(input_folder, moved_images_dir, ratio=0.75)
    
    # 2. 남은 이미지 파일 처리
    process_folder(input_folder, output_dir, grid_size=(3, 3))


if __name__ == "__main__":
    main()