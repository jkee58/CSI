import argparse
import os
from PIL import Image
import mmengine

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('gta_path', help='gta data path')
    parser.add_argument('--img-dir', default='images', type=str)
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    # parser.add_argument('-o', '--out-dir', help='output path')
    # parser.add_argument(
    #     '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args

def remove_data_pair(data_pair_path):
    img_path, lbl_path = data_pair_path
    img_filename = os.path.basename(img_path)
    lbl_filename = os.path.basename(lbl_path)

    if img_filename != lbl_filename:
        return
    
    img = Image.open(img_path)
    lbl = Image.open(lbl_path)
    img_width, img_height = img.size
    lbl_width, lbl_height = lbl.size

    if img_width != lbl_width or img_height != lbl_height:
        os.remove(img_path)
        os.remove(lbl_path)
        return f"{img_filename} -> {img.size} is not matched to {lbl.size}."


def main():
    args = parse_args()
    gta_path = args.gta_path
    img_dir = os.path.join(gta_path, args.img_dir)
    gt_dir = os.path.join(gta_path, args.gt_dir)

    img_files = []
    for img in mmengine.utils.scandir(
            img_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        img_file = os.path.join(img_dir, img)
        img_files.append(img_file)
    img_files = sorted(img_files)

    lbl_files = []
    for lbl in mmengine.utils.scandir(
            gt_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        lbl_file = os.path.join(gt_dir, lbl)
        lbl_files.append(lbl_file)
    lbl_files = sorted(lbl_files)

    if args.nproc > 1:
        logs = mmengine.utils.track_parallel_progress(remove_data_pair, list(zip(img_files, lbl_files)), args.nproc)
    else:
        logs = mmengine.utils.track_progress(remove_data_pair, list(zip(img_files, lbl_files)))
    
    print("Removed files")
    for log in logs:
        if log is not None:
            print(log)

if __name__ == '__main__':
    main()