import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
import igraph as ig

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel
BETA = None
K_Weight = 500
graph = None
WEIGHTS = []

bg_node = -1
fg_node = -1

RATIO_CONVERGENCE_THRESHOLD = 0.9995
DIFF_CONVERGENCE_THRESHOLD = 400


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iters=20):
    img = np.asarray(img, dtype=np.float64)
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    # Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask, 5)

    prev_energy = None
    for i in range(n_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if i > 0 and check_convergence_by_ratio(energy, prev_energy):
            break

        prev_energy = energy
    mask = mask % 2
    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


class __gaussian:
    def __init__(self, id, pixels, gmm_size, mean=None):
        self.id = id
        self.empty = False
        if len(pixels) == 0:
            self.empty = True
            self.weight = 0
            return
        if mean is None:
            self.mean = np.mean(pixels, axis=0)
        else:
            self.mean = mean
        if len(pixels) == 1:
            self.cov_mat = np.eye(3) * (10 * (10 ** (-10)))
        else:
            self.cov_mat = (np.cov(np.array(pixels).T)) + np.eye(3) * (10 * (10 ** (-10)))
        self.weight = len(pixels) / gmm_size
        self.inverse_cov_mat = np.linalg.inv(self.cov_mat)
        self.det_cov_mat = np.linalg.det(self.cov_mat)

    def prob(self, pixel):
        diff_vector = pixel - self.mean
        exp = np.exp(-0.5 * (diff_vector.T @ self.inverse_cov_mat @ diff_vector))
        alpha = np.sqrt(1 / (((2 * np.pi) ** 3) * self.det_cov_mat))
        return alpha * exp

    def prob_batch(self, pixels_array):
        if self.empty:
            return np.zeros((pixels_array.shape[0]))
        diff_vector = pixels_array - self.mean
        dot_vector = np.einsum('ij,ij->i', diff_vector @ self.inverse_cov_mat,
                               diff_vector)
        exp = np.exp(-0.5 * dot_vector)
        alpha = np.sqrt(1 / (((2 * np.pi) ** 3) * self.det_cov_mat))
        return alpha * exp


def initalize_GMMs(img, mask, n_components=10):
    bg_pixels = []
    fg_pixels = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (mask[i, j] % 2) == 0:
                bg_pixels.append(img[i, j])
            else:
                fg_pixels.append(img[i, j])
    bg_pixels = np.array(bg_pixels)
    fg_pixels = np.array(fg_pixels)
    bg_labels = KMeans(n_clusters=n_components, random_state=0).fit(bg_pixels).labels_
    fg_labels = KMeans(n_clusters=n_components, random_state=0).fit(fg_pixels).labels_

    bgGMM = [__gaussian(i, bg_pixels[bg_labels == i], len(bg_pixels)) for i in range(n_components)]

    fg_pixels = np.array(fg_pixels)
    fg_labels = np.array(fg_labels)
    fgGMM = [__gaussian(i, fg_pixels[fg_labels == i], len(fg_pixels)) for i in range(n_components)]

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    bg_gaussians, fg_gaussians = bgGMM, fgGMM
    bg_pixels = img[mask % 2 == 0]
    fg_pixels = img[mask % 2 == 1]
    bg_probabilities = np.array([gaussian.prob_batch(bg_pixels) for gaussian in bgGMM])
    fg_probabilities = np.array([gaussian.prob_batch(fg_pixels) for gaussian in fgGMM])
    bg_labels = np.argmax(bg_probabilities, axis=0)
    fg_labels = np.argmax(fg_probabilities, axis=0)
    bg_gaussians = [__gaussian(i, bg_pixels[bg_labels == i], len(bg_pixels)) for i in range(len(bg_gaussians))]
    fg_gaussians = [__gaussian(i, fg_pixels[fg_labels == i], len(fg_pixels)) for i in range(len(fg_gaussians))]
    return bg_gaussians, fg_gaussians


def init_graph(img, mask, bgGMM, fgGMM):
    global bg_node
    global fg_node
    global K_Weight
    graph = ig.Graph(img.shape[0] * img.shape[1] + 2)
    bg_node = img.shape[0] * img.shape[1]
    fg_node = img.shape[0] * img.shape[1] + 1
    vertices = np.arange(img.shape[0] * img.shape[1]).reshape(img.shape[0], img.shape[1])
    left_n = zip(vertices[:, 1:].reshape(-1), vertices[:, :-1].reshape(-1))
    left_up_n = zip(vertices[1:, 1:].reshape(-1), vertices[:-1, :-1].reshape(-1))
    up_n = zip(vertices[1:, :].reshape(-1), vertices[:-1, :].reshape(-1))
    right_up_n = zip(vertices[1:, :-1].reshape(-1), vertices[:-1, 1:].reshape(-1))
    left_weights = calc_energy_batch(img[:, 1:] - img[:, :-1], 1)
    left_up_weights = calc_energy_batch(img[1:, 1:] - img[:-1, :-1], np.sqrt(2))
    up_weights = calc_energy_batch(img[1:, :] - img[:-1, :], 1)
    right_up_weights = calc_energy_batch(img[1:, :-1] - img[:-1, 1:], np.sqrt(2))

    soft_vertices = vertices[np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD)]
    graph.add_edges(list(zip(np.full(len(soft_vertices), fill_value=bg_node), soft_vertices)))
    graph.add_edges(list(zip(np.full(len(soft_vertices), fill_value=fg_node), soft_vertices)))
    hard_bg_vertices = vertices[mask == GC_BGD]
    hard_fg_vertices = vertices[mask == GC_FGD]
    graph.add_edges(list(zip(np.full(len(hard_bg_vertices), fill_value=bg_node), hard_bg_vertices)))
    graph.add_edges(list(zip(np.full(len(hard_fg_vertices), fill_value=fg_node), hard_fg_vertices)))
    WEIGHTS.extend(np.full(len(hard_bg_vertices) + len(hard_fg_vertices), fill_value=K_Weight))

    graph.add_edges(list(left_n))
    graph.add_edges(list(left_up_n))
    graph.add_edges(list(up_n))
    graph.add_edges(list(right_up_n))
    WEIGHTS.extend(left_weights)
    WEIGHTS.extend(left_up_weights)
    WEIGHTS.extend(up_weights)
    WEIGHTS.extend(right_up_weights)
    return graph


def calculate_mincut(img, mask, bgGMM, fgGMM):
    global graph
    global BETA
    global WEIGHTS
    if graph is None:
        BETA = calc_beta(img)
        graph = init_graph(img, mask, bgGMM, fgGMM)

    soft_pixels = img[np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD)]
    D_bg = D_batch(soft_pixels, fgGMM)
    D_fg = D_batch(soft_pixels, bgGMM)
    w = list(D_bg)
    w.extend(D_fg)
    w.extend(WEIGHTS)
    graph_cut = graph.st_mincut(bg_node, fg_node, w)
    bg_nodes, fg_nodes = graph_cut.partition[0], graph_cut.partition[1]

    min_cut = [[calc_pixel_from_node(node, img) for node in bg_nodes if node != bg_node],
               [calc_pixel_from_node(node, img) for node in fg_nodes if node != fg_node]]
    energy = graph_cut.value
    return min_cut, energy


def update_mask(mincut_sets, mask):
    for v in mincut_sets[0]:
        i, j = v[0], v[1]
        if mask[i, j] != GC_BGD and mask[i, j] != GC_FGD:
            mask[i, j] = GC_PR_BGD
    for v in mincut_sets[1]:
        i, j = v[0], v[1]
        if mask[i, j] != GC_BGD and mask[i, j] != GC_FGD:
            mask[i, j] = GC_PR_FGD
    return mask


def check_convergence_by_ratio(curr_energy, prev_energy):
    global RATIO_CONVERGENCE_THRESHOLD
    ratio = (curr_energy / prev_energy)
    return ratio > RATIO_CONVERGENCE_THRESHOLD


def check_convergence_by_diff(curr_energy, prev_energy):
    global DIFF_CONVERGENCE_THRESHOLD
    diff = (prev_energy - curr_energy)
    return diff < DIFF_CONVERGENCE_THRESHOLD


def cal_metric(predicted_mask, gt_mask):
    temp = np.full(predicted_mask.shape, fill_value=1)
    interjection = np.sum(temp[np.logical_or(predicted_mask == 1, gt_mask == 1)])
    intersection = np.sum(temp[np.logical_and(predicted_mask == 1, gt_mask == 1)])
    jaccard = intersection / interjection

    accuracy = np.sum(temp[predicted_mask == gt_mask]) / (predicted_mask.shape[0] * predicted_mask.shape[1])

    return accuracy, jaccard
def calc_pixel_from_node(n, img):
    i = n // img.shape[1]
    j = n % img.shape[1]
    return i, j
def calc_energy(v1, v2, img):
    pixel1 = img[v1[0], v1[1]]
    pixel2 = img[v2[0], v2[1]]

    sqr_euclidean_dist = np.linalg.norm(pixel1 - pixel2) ** 2
    physical_dist = 1
    if abs(np.sum((v1[0] - v2[0]) + (v1[1] - v2[1]))) != 1:
        physical_dist = np.sqrt(2)
    return (50 / physical_dist) * np.exp((-BETA) * sqr_euclidean_dist)


def calc_energy_batch(diff_mat, physical_dist):
    dot_vector = np.einsum('ijk,ijk->ij', diff_mat, diff_mat)
    return (50 / physical_dist) * np.exp((-BETA) * dot_vector).reshape(-1)


def D_batch(pixels, GMM):
    s = np.zeros(pixels.shape[0])
    for gaussian in GMM:
        probs = gaussian.prob_batch(pixels)
        s += gaussian.weight * probs
    return -np.log(s)


def D(v, img, GMM):
    gaussian_set = GMM
    i = v[0]
    j = v[1]
    pixel = np.array(img[i][j], dtype='float')
    s = 0
    for gaussian in gaussian_set:
        diff_vector = pixel - gaussian.mean

        exp = np.exp(-0.5 * (diff_vector.T @ gaussian.inverse_cov_mat @ diff_vector))

        alpha = gaussian.weight / ((((2 * np.pi) ** 3) * gaussian.det_cov_mat) ** 0.5)

        s += alpha * exp

    return -np.log(s)


def calc_beta(img):
    sum = 0
    neighbors_num = 0
    left_n = img[:, 1:] - img[:, :-1]
    up_n = img[1:, :] - img[:-1, :]
    left_up_n = img[1:, 1:] - img[:-1, :-1]
    right_up_n = img[1:, :-1] - img[:-1, 1:]
    neighbors_diffs = [left_n, up_n, left_up_n, right_up_n]
    for diff in neighbors_diffs:
        sum += np.sum(np.square(diff))
        neighbors_num += diff.shape[0] * diff.shape[1]
    beta = 1 / (2 * (sum / neighbors_num))
    return beta


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='grave', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)
    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])

    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
