import numpy as np
import cv2 as cv
from plantcv import plantcv as pcv
from skimage.feature import hog
from PIL import Image, ImageEnhance

from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from utilities.utils import bgrtorgb

color_dict_HSV = {
    'black': [[180, 255, 26], [0, 0, 0]],
    'white': [[180, 38, 255], [0, 0, 166]],
    'gray': [[180, 38, 166], [0, 0, 25]],
    'red1': [[180, 255, 255], [155, 50, 40]],
    'red2': [[9, 255, 255], [0, 50, 70]],
    'pink1': [[6, 178, 255], [0, 0, 26]],
    'pink2': [[180, 178, 255], [175, 0, 26]],
    'pink3': [[176, 255, 255], [155, 38, 25]],
    'orange': [[25, 255, 255], [5, 38, 191]],
    'brown': [[25, 255, 191], [5, 50, 25]],
    'yellow': [[40, 255, 255], [15, 15, 10]],
    'yellowgreen': [[60, 255, 250], [30, 100, 200]],
    'green': [[85, 255, 255], [41, 15, 10]],
    'bluegreen': [[90, 255, 255], [76, 38, 25]],
    'blue': [[127, 255, 255], [91, 38, 25]],
    'purple': [[155, 255, 255], [128, 38, 25]],
    'lightpurple': [[155, 128, 255], [128, 38, 25]],
}


def mask_from_lab(img, k=2, ch='S'):
    ch = 'HSV'.index(ch.upper())
    # convert from RGB to LAB color space
    lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    # convert to float
    Z = np.float32(lab[:, :, ch].flatten())

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, .01)
    _, label, center = cv.kmeans(
        Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)

    # convert the channel to a binary
    lab[:, :, ch] = np.where(center[label.flatten()].reshape(lab[:, :, ch].shape) == list(
        sorted(set(center.flatten())))[-1], 0, 255*np.ones(lab[:, :, ch].shape))

    return lab[:, :, ch]


def houghLines(img_in):
    edges = cv.Canny(img_in, 50, 50, apertureSize=7, L2gradient=True)
    minLineLength = 0
    lines = cv.HoughLinesP(image=edges, rho=0.02, theta=np.pi/256, threshold=15,
                           lines=np.array([]), minLineLength=minLineLength, maxLineGap=21)

    a, b, c = lines.shape
    for i in range(a):
        cv.line(img_in, (lines[i][0][0], lines[i][0][1]), (lines[i]
                [0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)

    return img_in


def hog_process(img):
    img = cv.resize(img, (256, 256), interpolation=cv.INTER_CUBIC)

    img1 = np.array(cv.GaussianBlur(
        np.array(img[:, :128, :]), (9, 9), cv.BORDER_DEFAULT), dtype='uint8')
    img2 = np.array(cv.GaussianBlur(
        np.array(img[:, 128:, :]), (9, 9), cv.BORDER_DEFAULT), dtype='uint8')

    _, hog_img1 = hog(img1, orientations=9, pixels_per_cell=(
        8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    _, hog_img2 = hog(img2, orientations=9, pixels_per_cell=(
        8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    return np.concatenate([hog_img1, hog_img2], axis=1)

# CLAHE (Contrast Limited Adaptive Histogram Equalization) --> https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html


def clash(img, type='HSV', cliplimit=2.0, tilegridsize=(8, 8)):
    dtype = cv.COLOR_BGR2HSV if type == 'HSV' else cv.COLOR_BGR2LAB
    dtype_in = cv.COLOR_HSV2BGR if type == 'HSV' else cv.COLOR_LAB2BGR

    Clash = cv.cvtColor(img, dtype)
    Clash_planes = cv.split(Clash)
    clahe = cv.createCLAHE(clipLimit=cliplimit, tileGridSize=tilegridsize)
    lst_Clash_planes = list(Clash_planes)
    lst_Clash_planes[0] = clahe.apply(Clash_planes[0])
    Clash_planes = tuple(lst_Clash_planes)
    Clash = cv.merge(Clash_planes)
    return cv.cvtColor(Clash, dtype_in)


def kmeans_thresh(img, K=2):
    # convert to float
    Z = np.float32(img.flatten())
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv.kmeans(
        Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)

    # convert the to a binary
    return np.where(center[label.flatten()].reshape(img.shape) == list(sorted(set(center.flatten())))[-1], 0, 255*np.ones(img.shape))


def thresh_morpho_lab(img):
    # performing thresholding and morphological operations after the input image is loaded
    l, a, b = cv.split(cv.pyrMeanShiftFiltering(
        cv.cvtColor(img, cv.COLOR_BGR2LAB), 0, 255))
    l = kmeans_thresh(l, K=2)
    a = np.array(kmeans_thresh(a, K=2), dtype='uint8')
    b = np.invert(np.array(kmeans_thresh(b, K=2), dtype='uint8'))
    mask = cv.bitwise_or(a, b)
    mask = mask if mask.ravel().mean()/255 < .9 else cv.bitwise_or(np.invert(a), b)
    masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')
    return mask, masked


def thresh_morpho(img):
    # performing thresholding and morphological operations
    op1 = cv.pyrMeanShiftFiltering(img, 0, 255)
    imggray = cv.cvtColor(op1, cv.COLOR_BGR2GRAY)
    imgthresh = cv.threshold(
        imggray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return imgthresh[1]

# This function is designed to produce a set of GaborFilters
# an even distribution of theta values equally distributed amongst pi rad / 180 degree


def create_gaborfilter():
    filters = []
    num_filters = 16
    ksize = 9  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    # Theta is the orientation for edge detection
    for theta in np.arange(0, np.pi, np.pi / num_filters):
        kern = cv.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

# This general function is designed to apply Gabor Kernel filters to our image


def apply_gaborfilter(img, filters=create_gaborfilter()):

    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)

    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1  # remain depth same as original image

    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv.filter2D(img, depth, kern)  # Apply filter to image

        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage


def remove_bg(img):
    original_img = img
    img = clash(original_img, type='LAB', cliplimit=22.0, tilegridsize=(12, 12))
    img = apply_gaborfilter(img)

    blur = np.array(cv.GaussianBlur(np.array(img), (7, 7),
                    cv.BORDER_DEFAULT), dtype='uint8')
    blur = np.array(cv.medianBlur(np.array(blur), 7), dtype='uint8')
    blur = np.array(cv.bilateralFilter(blur, 7, 75, 75), dtype='uint8')

    lab_mask = mask_from_lab(blur, ch='S', k=2)

    HSV = {}
    for k in color_dict_HSV.keys():
        HSV[k], _ = pcv.threshold.custom_range(
            img=blur, lower_thresh=color_dict_HSV[k][1], upper_thresh=color_dict_HSV[k][0], channel='HSV')

        # Get countours and fill inside contours
        cnts = cv.findContours(HSV[k], cv.RETR_TREE,
                               cv.CHAIN_APPROX_SIMPLE | cv.RETR_EXTERNAL)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        HSV[k] = cv.fillPoly(HSV[k], cnts, (255, 255, 255))

    if (HSV['brown'][:20, :].mean()/255 >= .4) | (HSV['brown'][230:, :].mean()/255 >= .4):
        HSV['brown'] = HSV['brown'] * 0
        HSV['red1'] = HSV['red1'] * 0
        HSV['red2'] = HSV['red2'] * 0

    mask = cv.bitwise_or(HSV['red1'], HSV['red2'])
    for k in color_dict_HSV.keys():
        if k in ['yellow', 'yellowgreen', 'green', 'bluegreen', 'orange', 'brown']:
            mask = cv.bitwise_or(mask, HSV[k])
    mask = cv.bitwise_or(mask, lab_mask)
    if HSV['black'].mean()/255 < .1:
        mask = cv.bitwise_or(mask, HSV['black'])
    else:
        mask = cv.bitwise_and(mask, np.invert(HSV['black']))
    masked = pcv.apply_mask(img=original_img, mask=mask, mask_color='white')

    return mask, masked


def preprocess_images(file_name):
    try:
        bgr_img = cv.resize(pcv.readimage(file_name)[
                            0], (256, 256), interpolation=cv.INTER_CUBIC)
        gray_img = cv.resize(pcv.readimage(file_name, mode='gray')[
                             0], (256, 256), interpolation=cv.INTER_CUBIC)

        mask, masked = remove_bg(bgr_img)

        # Apply mask and add sharpness on the colored
        im = Image.fromarray(bgrtorgb(masked))
        enhancer = ImageEnhance.Sharpness(im)
        enhanced = enhancer.enhance(2)
        normalized_img = cv.normalize(np.array(
            enhanced), None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        # Apply mask and add sharpness on the gray image
        im = pcv.apply_mask(img=gray_img, mask=mask, mask_color='white')
        img_gray_enh = ImageEnhance.Sharpness(Image.fromarray(im)).enhance(4)

        canny = pcv.canny_edge_detect(np.array(normalized_img))
        dilated = np.array(
            cv.dilate(canny, (5, 5), iterations=5), dtype='uint8')
        eroded = np.array(
            cv.erode(dilated, (5, 5), iterations=5), dtype='uint8')

        enh_canny = cv.bitwise_and(np.array(enhanced, dtype='uint8'), np.array(
            enhanced, dtype='uint8'), mask=np.invert(canny))
        crop_res = crop_resize_image(canny, enh_canny)
    except Exception as e:
        print("Error : ", file_name, e)
        return bgrtorgb(bgr_img), np.zeros(bgr_img.shape), np.zeros(bgr_img.shape), np.zeros(bgr_img.shape), np.zeros(bgr_img.shape), np.zeros(bgr_img.shape), np.zeros(bgr_img.shape), np.zeros(bgr_img.shape), np.zeros(bgr_img.shape), np.zeros(bgr_img.shape)

    return bgrtorgb(bgr_img), gray_img, canny, dilated, eroded, enhanced, img_gray_enh, normalized_img, enh_canny, crop_res
