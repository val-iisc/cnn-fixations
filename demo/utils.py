from skimage import transform, io, img_as_float
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np


def imgPreprocess(img_path, size=224):
    """
    Process the image by removing the mean and reszing so that the smallest
    edge is 256. Then take crops and average to predict.
    """
    mean = np.array([103.939, 116.779, 123.68])
    img = img_as_float(io.imread(img_path)).astype(np.float32)
    # Resize factor, smallest edge to 256.
    resFac = 256.0/min(img.shape[:2])
    newSize = list(map(int, (img.shape[0]*resFac, img.shape[1]*resFac)))
    img = transform.resize(img, newSize, mode='constant')
    # Calculate offsets of oversampling from caffe's oversample.
    if (size == 224):
        offset = [[0, 0], [0, newSize[1]-224], [newSize[0]-224, 0], [
                 newSize[0]-224, newSize[1]-224], [
                 newSize[0]/2.0-112, newSize[1]/2.0-112]]
    else:
        offset = [[0, 0], [0, newSize[1]-227], [newSize[0]-227, 0], [
                 newSize[0]-227, newSize[1]-227], [
                 newSize[0]/2.0-114, newSize[1]/2.0-114]]
    img = img*255.0
    img[:, :, 0] -= mean[2]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[0]
    return img, offset, resFac,  newSize


def pred(net, img):
    synsets = open('ilsvrc_synsets.txt').readlines()
    net.predict([img], oversample=True)
    p = net.blobs['prob'].data
    # mean label
    p_mean = np.argmax(np.mean(p, axis=0))
    image_label = synsets[p_mean].split(',')[0].strip()
    image_label = image_label[image_label.index(' ')+1:]
    # Individual labels of 5 crops (- ignore mirrored samples)
    crop_labels = [np.argmax(p[i]) for i in range(5)]
    # To check if mean label matches any crop otherwise assign label of center
    if p_mean in crop_labels:
        p_m = p_mean
    else:
        p_m = crop_labels[4]
    # Assign labels to crops
    points = []
    for i in range(5):
        if np.argmax(p[i, :]) == p_m:
            points.append([int(p_m)])
        else:
            points.append(0)
    return points, image_label


def outlier_removal(points, diag):
    neighbors = np.zeros((points.shape[0]))
    selPoints = np.empty((1, 2))
    for i in range(points.shape[0]):
        diff = np.sqrt(np.sum(np.square(points-points[i]), axis=1))
        neighbors[i] = np.sum(diff < diag)
    for i in range(points.shape[0]):
        if neighbors[i] > 0.05*points.shape[0]:
            selPoints = np.append(selPoints, points[i:i+1, :], axis=0)
    selPoints = selPoints[1:, :]
    selPoints = selPoints.astype(int)
    return selPoints


def heatmap(img, points, sigma=20):
    k = (np.min(img.shape[:2])) if (
        np.min(img.shape[:2]) % 2 == 1) else (np.min(img.shape[:2])-1)
    mask = np.zeros(img.shape[:2])
    shape = mask.shape
    for i in range(points.shape[0]):
        # Check if inside the image
        if points[i, 0] < shape[0] and points[i, 1] < shape[1]:
            mask[points[i, 0], points[i, 1]] += 1
    # Gaussian blur the points to get a nice heatmap
    blur = cv2.GaussianBlur(mask, (k, k), sigma)
    blur = blur*255/np.max(blur)
    return blur


def visualize(img_path, points, diag_percent, image_label):
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)
    img = cv2.merge((r, g, b))
    diag = math.sqrt(img.shape[0]**2 + img.shape[1]**2)*diag_percent
    values = np.asarray(points)
    selPoints = outlier_removal(values, diag)
    # Make heatmap and show images
    hm = heatmap(np.copy(img), selPoints)
    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img), ax[0].axis('off'), ax[0].set_title(image_label)
    ax[1].imshow(img), ax[1].axis('off'),
    ax[1].scatter(selPoints[:, 1], selPoints[:, 0]),
    ax[1].set_title('CNN Fixations')
    ax[2].imshow(img), ax[2].imshow(hm, 'jet', alpha=0.6),
    ax[2].axis('off'), ax[2].set_title('Heatmap')
    plt.show()
