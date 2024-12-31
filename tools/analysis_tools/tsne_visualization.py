from typing import Tuple, List
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
from cityscapesscripts.helpers.labels import labels as cityscapes_labels
from mmseg.models.utils import resize

# import warnings
# # 모든 경고를 예외로 발생시키도록 설정
# warnings.filterwarnings("error")


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(file_path)

    feats = X['fused_feats']
    gt_sem_segs = X['gt']

    # TODO: Remove and Support on CacheFeatureHook
    gt_sem_segs = resize(
        torch.from_numpy(gt_sem_segs).type(torch.float32),
        size=(512, 1024)).type(torch.long).numpy()

    gt_sem_segs = np.squeeze(gt_sem_segs, axis=1)

    return feats, gt_sem_segs


def filter_feats(
        feats: np.ndarray,
        gt_sem_segs: np.ndarray,
        ignore_indices: List[int] = [-1, 255]) -> Tuple[np.ndarray, List[int]]:
    num_of_feats, _, height_of_feats, width_of_feats = feats.shape
    num_of_gt_sem_segs, height_of_gt_sem_segs, width_of_gt_sem_segs = gt_sem_segs.shape

    assert num_of_feats == num_of_gt_sem_segs, "The number of features and the number of ground truths must be the same."

    height_scale_factor = height_of_gt_sem_segs // height_of_feats
    width_scale_factor = width_of_gt_sem_segs // width_of_feats

    filtered_feats = []
    class_of_feats = []
    for i in range(num_of_feats):
        for x in range(height_of_feats):
            for y in range(width_of_feats):
                start_row, start_col = x * height_scale_factor, y * width_scale_factor
                feat_pixel_region = gt_sem_segs[i, start_row:start_row +
                                                height_scale_factor,
                                                start_col:start_col +
                                                width_scale_factor]
                class_of_feat = feat_pixel_region[0][0]
                if class_of_feat not in ignore_indices and np.all(
                        feat_pixel_region == class_of_feat):
                    filtered_feats.append(feats[i, :, x, y])
                    class_of_feats.append(class_of_feat)

    return np.stack(filtered_feats, axis=0), class_of_feats


def plot_tsne(embeddings: np.ndarray,
              class_of_feats: List[int],
              save_path: str,
              ignore_indices: List[int] = [-1, 255],
              s: int = 2) -> None:

    x = embeddings[:, 0]
    y = embeddings[:, 1]

    fig, ax = plt.subplots()

    for label in cityscapes_labels:
        if label.trainId in ignore_indices:
            continue

        color = '#{:02x}{:02x}{:02x}'.format(*label.color)
        indices = [
            i for i, class_of_feat in enumerate(class_of_feats)
            if class_of_feat == label.trainId
        ]

        ax.scatter(
            np.take(x, indices),
            np.take(y, indices),
            label=label,
            color=color,
            s=s)

    ax.axis('off')

    fig.savefig(save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Features using T-SNE')
    parser.add_argument(
        '--load-path',
        type=str,
        help='path to load features and ground truths')
    parser.add_argument(
        '--save-path', type=str, help='path to save the t-sne plot')
    parser.add_argument(
        '--n-jobs', type=int, default=int(os.cpu_count() * 0.95))
    parser.add_argument('--verbose', action="store_const", const=1, default=0)
    parser.add_argument('--n-components', type=int, default=2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument(
        '--ignore-indices',
        nargs='+',
        type=int,
        help='classes to be excluded from visualization',
        default=[-1, 255])
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    feats, gt_sem_segs = load_data(args.load_path)
    print(f'Data loading complete!')

    filtered_feats, class_of_feats = filter_feats(
        feats, gt_sem_segs, ignore_indices=args.ignore_indices)

    embeddings = TSNE(
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        n_components=args.n_components,
        random_state=args.random_state).fit_transform(filtered_feats)

    plot_tsne(
        embeddings,
        class_of_feats,
        args.save_path,
        ignore_indices=args.ignore_indices)
    print(f'Plot saved to {args.save_path}')


if __name__ == '__main__':
    main()
