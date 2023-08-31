import cv2
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import argparse

IN_MASK = 255
OUT_MASK = 0


def poisson_blend(im_src, im_tgt, im_mask, center):
    scaled_src, scaled_mask = scale(im_src, im_tgt, im_mask, center)
    set_del_omega(scaled_mask)

    im_tgt = np.array(im_tgt, dtype='float')
    A, b, ij_to_mask_index, mask_indices = calc_A_and_b_matrices(scaled_src, im_tgt, scaled_mask)
    A = csr_matrix(A)
    x = spsolve(A, b)
    im_blend = np.array(im_tgt)
    for m in range(len(mask_indices[0])):
        im_blend[mask_indices[0][m], mask_indices[1][m]] = x[m]
    im_blend[im_blend > 255] = 255
    im_blend[im_blend < 0] = 0
    im_blend = np.array(im_blend, dtype=np.uint8)
    return im_blend


def set_del_omega(mask):
    delete_flag = -1
    directions = np.array([(-1, 0), (1, 0), (0, 1), (0, -1)])
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == IN_MASK:
                for ni, nj in (i, j) + directions:
                    if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj] == OUT_MASK:
                        mask[i, j] = delete_flag
                        break
    mask[mask == delete_flag] = OUT_MASK


def calc_A_and_b_matrices(src, tgt, mask):
    mask_indices = np.nonzero(mask)
    mask_pixel_count = len(mask_indices[0])
    A = lil_matrix((mask_pixel_count, mask_pixel_count), dtype='float')
    b = np.zeros((mask_pixel_count, 3), dtype='float')
    neighbors_dir = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])
    ij_to_mask_index = np.zeros_like(mask, dtype=np.int32)
    for m in range(mask_pixel_count):
        i, j = mask_indices[0][m], mask_indices[1][m]
        ij_to_mask_index[i, j] = m

    for m in range(mask_pixel_count):
        i, j = mask_indices[0][m], mask_indices[1][m]
        A[m, m] = -4
        b[m] = -4 * src[i, j]
        for n in (i, j) + neighbors_dir:
            ni, nj = n[0], n[1]
            if (not 0 <= ni < mask.shape[0]) or not (0 <= nj < mask.shape[1]):
                continue
            b[m] += src[ni, nj]
            if mask[ni, nj] == 0:
                b[m] -= tgt[ni, nj]
            else:
                nij_mask_index = ij_to_mask_index[ni, nj]
                A[m, nij_mask_index] = 1
    return A, b, ij_to_mask_index, mask_indices


def calc_pixel_index(i, j, img):
    return i * img.shape[1] + j
def scale(im_src, im_tgt, im_mask, center):
    scaled_src = np.zeros(im_tgt.shape, dtype='float')
    scaled_mask = np.zeros((im_tgt.shape[0], im_tgt.shape[1]), dtype='float')
    min_col, max_col, min_row, max_row = float('inf'), -1, float('inf'), -1
    for i in range(im_mask.shape[0]):
        for j in range(im_mask.shape[1]):
            if im_mask[i, j] == IN_MASK:
                if i < min_row:
                    min_row = i
                if i > max_row:
                    max_row = i
                if j < min_col:
                    min_col = j
                if j > max_col:
                    max_col = j
    mask_center = ((max_row + min_row) // 2), ((max_col + min_col) // 2)
    for i in range(im_mask.shape[0]):
        for j in range(im_mask.shape[1]):
            if im_mask[i, j] == IN_MASK:
                delta_i, delta_j = mask_center[0] - i, mask_center[1] - j
                scaled_src[center[0] - delta_i, center[1] - delta_j] = im_src[i, j]
                scaled_mask[center[0] - delta_i, center[1] - delta_j] = IN_MASK

    return scaled_src, scaled_mask


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default=f'./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default=f'./banana2/mask.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)

        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]
    center = (int(im_tgt.shape[0] / 2), int(im_tgt.shape[1] / 2))
    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
