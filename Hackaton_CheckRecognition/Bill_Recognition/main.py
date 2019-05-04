from copy import deepcopy

import cv2
import numpy as np
from math import atan2
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans
from scipy.ndimage import label, generate_binary_structure


def rotate(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def get_array_connectiv(img, labels):
    labels2d = labels.reshape(img.shape[:2])

    labeled_array, num_features = label(labels2d)

    plain_labels = labeled_array.reshape((-1,))
    bins = np.bincount(plain_labels)
    bins[0] = 0
    most_frequent = np.argmax(bins)

    return bins, most_frequent, labeled_array, num_features


from scipy.spatial import ConvexHull

from scipy.ndimage.interpolation import rotate


def minimum_bounding_rectangle(points):
    pi2 = np.pi / 2.
    hull_points = points[ConvexHull(points).vertices]

    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    #     rotations = np.vstack([
    #         np.cos(angles),
    #         -np.sin(angles),
    #         np.sin(angles),
    #         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def cluster(img, clusters=4, treshhold=500):
    img_raw = img.reshape((-1, 3))

    # kmeans = KMeans(n_clusters=clusters).fit(img_raw)
    # centroids = kmeans.cluster_centers_
    # labels = kmeans.predict(img_raw)
    #
    # for i in range(len(labels)):
    #     if (centroids[labels[i]][0] + centroids[labels[i]][1] + centroids[labels[i]][2] < treshhold):
    #         labels[i] = 0
    #     else:
    #         labels[i] = 1

    labels = np.asarray([(int(x[0]) + x[1] + x[2] >= treshhold) for x in img_raw])
    raw = [np.asarray([255, 255, 255]) * x for x in labels]

    raw = np.asarray(raw, dtype=np.uint8)

    new_img = raw.reshape(img.shape)
    kernel = np.ones((3, 3))

    bins, most_frequent, labeled_array, num_features = get_array_connectiv(img , labels)
    mask = np.where(labeled_array == most_frequent)

    labeled_array = np.zeros(labeled_array.shape)
    labeled_array[mask] = 1

    cv2.waitKey(0)
    return labeled_array


def draw_rect(img, rect, show=False):
    rect = list(rect)
    rect.append(rect[0])
    for i in range(len(rect) - 1):
        p1 = rect[i]
        p2 = rect[i + 1]
        cv2.line(img, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), (255, 0, 0), 3)

    if show:
        cv2.imshow("rectangle", img)


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result, rot_mat


def normalize(angle):
    while (angle > 90):
        angle -= 180
    while (angle < -90):
        angle += 180

    return angle


def get_rect_angle_image(receipt):
    points = np.where(receipt > 0)
    points = np.asarray(list(zip(points[0], points[1])), dtype=np.float)

    rect = minimum_bounding_rectangle(points)

    angle1 = atan2((rect[0] - rect[1])[0], (rect[0] - rect[1])[1]) * 180 / np.pi
    angle2 = atan2((rect[0] - rect[3])[0], (rect[0] - rect[3])[1]) * 180 / np.pi

    angle1 = normalize(angle1)
    angle2 = normalize(angle2)

    if (np.abs(angle1) < np.abs(angle2)):
        angle = angle1
    else:
        angle = angle2

    return rect, angle


def rotate_image(img, rect, angle):
    rotated, rot_mat = rotateImage(img, angle)

    rot_mat = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), -angle, 1)
    rect = np.asarray(rect)
    ones = np.ones(shape=(len(rect), 1))
    points_ones = np.hstack([rect, ones])
    new_rect = rot_mat.dot(points_ones.T).T
    new_rect = new_rect.astype(int)
    return rotated, new_rect


def get_cut_from_rotated_img(rotated, new_rect):
    SHIFT = 5

    minx, miny, maxx, maxy = np.min(new_rect[:, 0]), np.min(new_rect[:, 1]), np.max(new_rect[:, 0]), np.max(
        new_rect[:, 1])

    minx -= SHIFT
    maxx += SHIFT
    miny -= SHIFT
    maxy += SHIFT

    minx = max(minx,0)
    miny = max(miny,0)
    maxx = min(maxx, rotated.shape[0])
    maxy = min(maxy,rotated.shape[1])

    res = rotated[minx:maxx, miny:maxy, :]
    return res


def process(img):
    reciept = cluster(img,-1,400)
    tmp = deepcopy(img)
    rect, angle = get_rect_angle_image(reciept)
    # draw_rect(tmp, rect)
    rotated, rect = rotate_image(img, rect, angle)
    res = get_cut_from_rotated_img(rotated, rect)
    #cv2.waitKey(0)
    # cv2.imwrite('./data/out.',res)
    # cv2.imshow("ddd",res)
    # cv2.waitKey(0)
    print(res.shape)
    return res