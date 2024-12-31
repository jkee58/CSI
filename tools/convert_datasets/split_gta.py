import os
import shutil
import scipy.io
from tqdm import tqdm

def load_split_info(split_file):
    data = scipy.io.loadmat(split_file)
    return data

def create_directories(base_path):
    # train 및 val 디렉토리 생성
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    return os.path.join(train_dir, 'images'), os.path.join(val_dir, 'images'), os.path.join(train_dir, 'labels'), os.path.join(val_dir, 'labels')

def split_and_save_images(image_dir, label_dir, split_file, output_dir):
    split_info = load_split_info(split_file)
    train_indices = list(map(int, split_info['trainIds']))
    val_indices = list(map(int, split_info['valIds']))
    
    train_image_dir, val_image_dir, train_label_dir, val_label_dir = create_directories(output_dir)

    # train 이미지 이동
    for idx in tqdm(train_indices):
        src = os.path.join(image_dir, str(idx).zfill(5) + '.png')
        dst = os.path.join(train_image_dir, str(idx).zfill(5) + '.png')
        shutil.copy(src, dst)

    # val 이미지 이동
    for idx in tqdm(val_indices):
        src = os.path.join(image_dir, str(idx).zfill(5) + '.png')
        dst = os.path.join(val_image_dir, str(idx).zfill(5) + '.png')
        shutil.copy(src, dst)
    
    # train 이미지 이동
    for idx in tqdm(train_indices):
        src = os.path.join(label_dir, str(idx).zfill(5) + '.png')
        dst = os.path.join(train_label_dir, str(idx).zfill(5) + '.png')
        shutil.copy(src, dst)

    # val 이미지 이동
    for idx in tqdm(val_indices):
        src = os.path.join(label_dir, str(idx).zfill(5) + '.png')
        dst = os.path.join(val_label_dir, str(idx).zfill(5) + '.png')
        shutil.copy(src, dst)

if __name__ == '__main__':
    image_dir = 'data/gta/images'  # 이미지가 저장된 폴더 경로
    label_dir = 'data/gta/labels'  # 이미지가 저장된 폴더 경로
    split_file = 'data/gta/split.mat'  # split.mat 파일 경로
    output_dir = 'data/splited_gta'  # train 및 val 폴더를 생성할 경로

    split_and_save_images(image_dir, label_dir, split_file, output_dir)
