'''
This file contains function to extract features from img.
Theses functions is used in cli to select the features.
'''

import sys
import numpy as np
import cv2 as cv
import pyfeats as pf
import mahotas
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from utilities.images_conversions import bgrtogray


def get_pyfeats_features(raw_bgr_img, mask):
    f = bgrtogray(raw_bgr_img)
    mask = mask // 255

    contours, _ = cv.findContours(~mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    perimeter = np.zeros_like(mask)
    for c in [c for c in contours if cv.contourArea(c) > 200]:
        approx = cv.approxPolyDP(c, 0.0001*cv.arcLength(c, True), True)
        cv.drawContours(perimeter, [approx], -1, 255, 2)
        
    result = {}
    #  1 `Textural Features`
    #

    #  1.1 `First Order Statistics/Statistical Features (FOS/SF)`
    # First Order Statistics (FOS) are calculated from the histogram of the img which is the empirical probability density function for single pixels. The FOS features are the following:
    # 1) mean,
    # 2) standard deviation,
    # 3) median,
    # 4) mode,
    # 5) skewnewss,
    # 6) kurtosis,
    # 7) energy,
    # 8) entropy,
    # 9) minimal gray level,
    # 10) maximal gray leve,
    # 11) coefficient of variation,
    # 12) percentiles (10)
    # 13) percentiles (25)
    # 14) percentiles (75)
    # 15) percentiles (90)
    # 16) histogram width.
    #
    #

    features, labels = pf.fos(f, mask)
    result = {labels[index]: features[index] for index in range(len(labels))}

    #  1.2 `Gray Level Co-occurence Matrix (GLCM/SGLDM)`
    # The Gray Level Co-occurrence Matrix (GLCM) as proposed by Haralick are based on the estimation of the second-order joint conditional probability density functions. The GLGLCM features are the following:
    # 1) angular second moment,
    # 2) contrast,
    # 3) correlation,
    # 4) sum of squares: variance,
    # 5) inverse difference moment,
    # 6) sum average,
    # 7) sum variance,
    # 8) sum entropy,
    # 9) entropy,
    # 10) difference variance,
    # 11) difference entropy,
    # 12) Information 1&2 range,
    # 13) information measures of correlation. For each feature, the mean values and the range of values are computed, and are used as two different features sets.
    #
    #

    features_mean, features_range, labels_mean, labels_range = pf.glcm_features(
        f, ignore_zeros=True)
    features, labels = np.array([features_mean, features_range]).flatten(
    ), np.array([labels_mean, labels_range]).flatten()
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  1.3 `Gray Level Difference Statistics (GLDS)`
    # The Gray Level Difference Statistics (GLDS) algorithm uses first order statistics of local property values based on absolute differences between pairs of gray levels or of average gray levels in order to extract texture measures. The GLDS features are the following: 1) homogeneity, 2) contrast, 3) energy, 4) entropy, 5) mean.
    #
    #

    features, labels = pf.glds_features(
        f, mask, Dx=[0, 1, 1, 1], Dy=[1, 1, 0, -1])
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  3.1.4 `Neighborhood Gray Tone Difference Matrix (NGTDM)`
    # Neighbourhood Gray Tone Difference Matrix (NDTDM) corresponds to visual properties of texture. The NGTDM features are the following: 1) coarseness, 2) contrast, 3) busyness, 4) complexity, 5) strength.
    #
    #

    features, labels = pf.ngtdm_features(f, mask, d=1)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  3.1.5 `Statistical Feature Matrix (SFM)`
    # The Statistical Feature Matrix measures the statistical properties of pixel pairs at several distances within an img which are used for statistical analysis. The SFM features are the following: 1) coarseness, 2) contrast, 3) periodicity, 4) roughness.
    #
    #

    features, labels = pf.sfm_features(f, mask, Lr=4, Lc=4)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  1.6 `Law's Texture Energy Measures (LTE/TEM)`
    # Law’s texture Energy Measures, are derived from three simple vectors of length 3. If these vectors are convolved with themselves, new vectors of length 5 are obtained. By further self-convolution, new vectors of length 7 are obtained. If the column vectors of length l are multiplied by row vectors of the same length, Laws l×l masks are obtained. In order to extract texture features from an img, these masks are convoluted with the img, and the statistics (e.g., energy) of the resulting img are used to describe texture:
    # 1) texture energy from LL kernel,
    # 2) texture energy from EE kernel,
    # 3) texture energy from SS kernel,
    # 4) average texture energy from LE and EL kernels,
    # 5) average texture energy from ES and SE kernels,
    # 6) average texture energy from LS and SL kernels.
    #
    #

    features, labels = pf.lte_measures(f, mask, l=7)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  1.7 `Fractal Dimension Texture Analysis (FDTA)`
    # Fractal Dimension Texture Analysis (FDTA) is based on the Fractional Brownian Motion (FBM) Model. The FBM model is used to describe the roughness of nature surfaces. It regards naturally occurring surfaces as the end result of random walks. Such random walks are basic physical processes in our universe. One of the most important parameters to represent a fractal surface is the fractal dimension. A simpler method is to estimate the H parameter (Hurst coefficient). If the img is seen under different resolutions, then the multiresolution fractal (MF) feature vector is obtained.
    #
    #

    features, labels = pf.fdta(f, mask, s=3)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  1.8 `Gray Level Run Length Matrix (GLRLM)`
    # A gray level run is a set of consecutive, collinear picture points having the same gray level value. The length of the run is the number of picture points in the run. The GLRLM features are the following:
    # 1) short run emphasis,
    # 2) long run emphasis,
    # 3) gray level non-uniformity,
    # 4) run length non-uniformity,
    # 5) run percentage,
    # 6) low gray level run emphasis,
    # 7) high gray level run emphasis,
    # 8) short low gray level emphasis,
    # 9) short run high gray level emphasis,
    # 10) long run low gray level emphasis,
    # 11) long run high level emphasis.
    #
    #

    features, labels = pf.glrlm_features(f, mask, Ng=256)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  1.9 `Fourier Power Spectrum (FPS)`
    # For digital pictures, instead of the continuous Fourier transform, one uses the discrete transform. The standard set of texture features based on a ring-shaped samples of the discrete Fourier power spectrum are of the form. Similarly, the features based on a wedge-shaped samples are of the form.
    # The FPS features are the following:
    # 1) radial sum,
    # 2) angular sum

    features, labels = pf.fps(f, mask)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  1.10 `Shape Parameters`
    # Shape parameters consists of the following features: 1) x-coordinate maximum length, 2) y-coordinate maximum length, 3) area, 4) perimeter, 5) perimeter2/area
    #
    #

    features, labels = pf.shape_parameters(
        f, mask, perimeter, pixels_per_mm2=1)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  1.11 `Gray Level Size Zone Matrix (GLSZM)`
    # Gray Level Size Zone Matrix (GLSZM) quantifies gray level zones in an img. A gray level zone is defined as the number of connected voxels that share the same gray level intensity. A voxel is considered connected if the distance is 1 according to the infinity norm (26-connected region in a 3D, 8-connected region in 2D). The GLSZM features are the following:
    # 1) small zone emphasis,
    # 2) large zone emphasis,
    # 3) gray level non-uniformity,
    # 4) zone-size non-uniformity,
    # 5) zone percentage,
    # 6) low gray level zone emphasis,
    # 7) high gray level zone emphasis,
    # 8) small zone low gray level emphasis,
    # 9) small zone high gray level emphasis,
    # 10) large zone low gray level emphasis,
    # 11) large zone high gray level emphasis,
    # 12) gray level variance,
    # 13) zone-size variance,
    # 14) zone-size entropy.
    #
    #

    features, labels = pf.glszm_features(f, mask)  # , connectivity=1)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  1.12 `Higher Order Spectra (HOS)`
    # Radon transform transforms two dimensional imgs with lines into a domain of possible line parameters, where each line in the img will give a peak positioned at the corresponding line parameters. Hence, the lines of the imgs are transformed into the points in the Radon domain. High Order Spectra (HOS) are spectral components of higher moments. The bispectrum, of a signal is the Fourier transform (FT) of the third order correlation of the signal (also known as the third order cumulant function). The bispectrum, is a complex-valued function of two frequencies. The bispectrum which is the product of three Fourier coefficients, exhibits symmetry and was computed in the non-redundant region. The extracted feature is the entropy 1.
    #
    #

    features, labels = pf.hos_features(f, th=[135, 140])
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  1.13 `Local Binary Pattern (LPB)`
    # Local Binary Pattern (LBP), a robust and efficient texture descriptor, was first presented by Ojala. The LBP feature vector, in its simplest form, is determined using the following method: A circular neighbourhood is considered around a pixel. P points are chosen on the circumference of the circle with radius R such that they are all equidistant from the centre pixel. . These P points are converted into a circular bit-stream of 0s and 1s according to whether the gray value of the pixel is less than or greater than the gray value of the centre pixel. Ojala et al. (2002) introduced the concept of uniformity in texture analysis. The uniform fundamental patterns have a uniform circular structure that contains very few spatial transitions U (number of spatial bitwise 0/1 transitions). In this work, a rotation invariant measure using uniformity measure U was calculated. Only patterns with U less than 2 were assigned the LBP code i.e., if the number of bit transitions in the circular bit-stream is less than or equal to 2, the centre pixel was labelled as uniform. Energy and entropy of the LBP img, constructed over different scales are used as feature descriptors.
    #
    #

    features, labels = pf.lbp_features(f, mask, P=[8, 16, 24], R=[1, 2, 3])
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  2 `Morphological Features`

    #  2.1 `Gray-scale Morphological Analysis`
    # In multilevel binary morphological analysis different components are extracted and investigated for their geometric properties. Three binary imgs are generated by thresholding. Here, binary img outputs are represented as sets of img coordinates where img intensity meets the threshold criteria. <br><br>Overall, this multilevel decomposition is closely related to a three-level quantization of the original img intensity. For each binary img the pattern spectrum is calculated. The Grayscale Morphological Features are the following:
    # 1) mean cumulative distribution functions (CDF) and
    # 2) mean probability density functions (PDF) of the pattern spectra using the cross $"+"$ as a structural element of the grayscale img.
    #
    #

    N = 9
    features = pf.grayscale_morphology_features(f, N)
    labels = [f'{c}{i+1}' for c in ['PDF', 'CDF'] for i in range(N)]
    features = np.array(features).flatten()
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  2.2 `Multilevel Binary Morphological Analysis`
    # Same as above but with grayscale img. The difference lies in the calculation of the pattern spectrum.
    #
    #

    features = pf.multilevel_binary_morphology_features(
        f, mask, N=N, thresholds=[25, 50])
    labels = [f'{f}{i}' for f in ['pdf_L', 'pdf_M', 'pdf_H',
                                  'cdf_L', 'cdf_M', 'cdf_H'] for i in range(N)]
    features = np.array(features).flatten()
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  3 `Histogram Based Features`

    #  3.1 `Histogram`
    # Histogram: The grey level histogram of the ROI of the img.

    bins = 8
    features, labels = pf.histogram(f, mask, bins=bins)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  3.2 `Multi-region histogram`
    # A number of equidistant ROIs are identified by eroding the img outline by a factor based on the img size. The histogram is computed for each one of the regions as described above
    #
    #

    features, labels = pf.multiregion_histogram(
        f, mask, bins, num_eros=3, square_size=3)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  3.3 `Correlogram`
    # Correlograms are histograms, which measure not only statistics about the features of the img, but also consider the spatial distribution of these features. In this work two correlograms are implemented for the ROI of the img:
    #
    # based on the distance of the distribution of the pixels’ gray level values from the centre of the img, and
    # based on their angle of distribution.
    # For each pixel the distance and the angle from the img centre is calculated and for all pixels with the same distance or angle their histograms are computed. In order to make the comparison between imgs of different sizes feasible, the distance correlograms is normalized. The angle of the correlogram is allowed to vary among possible values starting from the left middle of the img and moving clockwise. The resulting correlograms are matrices.
    #
    #

    dig, hist = 8, 8
    Hd, Ht, labels = pf.correlogram(
        f, mask, bins_digitize=dig, bins_hist=hist, flatten=True)
    labels = [f'Correlogram_{c}_{i}_{j}' for c in [
        'dist', 'angl'] for i in range(dig) for j in range(hist)]
    features = list(np.array([Hd, Ht]).flatten())
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  4 `Multi-scale Features`

    #  4.1 `Fractal Dimension Texture Analysis (FDTA)`
    # See 1.7.
    #
    #

    features, labels = pf.fdta(f, mask, s=3)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  4.2 `Amplitude Modulation – Frequency Modulation (AM-FM)`
    # We consider multi-scale Amplitude Modulation – Frequency Modulation (AM-FM) representations, under least-square approximations, for imgs. For each img an instantaneous amplitude (IA), an the instantaneous phase (IP) and an instantaneous frequency (IF) is calculated for a specific component. Given the input discrete img, we first apply the Hilbert transform to form a 2D extension of the 1D analytic signal. The result is processed through a collection of bandpass filters with the desired scale. Each processing block will produce the IA, the IP and the IF. The AM-FM features are the following: Histogram of the
    # 1) low,
    # 2) medium,
    # 3) high and
    # 4) dc reconstructed imgs.
    #
    #
    # !!!! processing takes a long time
    #
    # features, labels = pf.amfm_features(f, bins=bins)
    # labels = [l for l in labels if int(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", l)[0]) in range(bins)]
    # result.update({labels[index]: features[index] for index in range(len(labels))})

    #  4.3 `Discrete Wavelet Transform (DWT)`
    # The Discrete Wavelet Transform (DWT) of a signal is defined as its inner product with a family of functions. For imgs, i.e., 2-D signals, the 2-D DWT can be used. This consists of a DWT on the rows of the img and a DWT on the columns of the resulting img. The result of each DWT is followed by down sampling on the columns and rows, respectively. The decomposition of the img yields four sub-imgs for every level. Each approximation sub-img is decomposed into four sub imgs named approximation, detail-horizontal, detail-vertical, and detail-diagonal sub-img respectively. Each detail sub-img is the result of a convolution with two half-band filters. The DWT features are the following: 1) mean and 2) standard deviation of the absolute value of detail sub-imgs of the DWT.
    #
    #

    features, labels = pf.dwt_features(f, mask, wavelet='bior3.3', levels=3)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  4.4 `Stationary Wavelet Transform (SWT)`
    # The 2-D Stationary Wavelet Transform (SWT) is similar to the 2-D DWT, but no down sampling is performed. Instead, up sampling of the low-pass and high-pass filters is carried out. The SWT features are the following: 1) mean and 2) standard deviation of the absolute value of detail sub-imgs of the SWT.
    #
    #

    features, labels = pf.swt_features(f, mask, wavelet='bior3.3', levels=3)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  4.5 `Wavelet Packets (WP)`
    # The 2-D Wavelet Packets (WP) decomposition is a simple modification of the 2-D DWT, which offers a richer space-frequency representation. The first level of analysis is the same as that of the 2-D DWT. The second, as well as all subsequent levels of analysis consist of decomposing every sub img, rather than only the approximation sub img, into four new sub imgs. The WP features are the following: 1) mean and 2) standard deviation of the absolute value of detail sub-imgs of the Wavelet Decomposition.
    #
    #

    features, labels = pf.wp_features(f, mask, wavelet='coif1', maxlevel=2)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  4.6 `Gabor Transform (GT)`
    # The Gabor Transform (GT) of an img consists in convolving that img with the Gabor function, i.e., a sinusoidal plane wave of a certain frequency and orientation modulated by a Gaussian envelope. Frequency and orientation representations of Gabor filters are similar to those of the human visual system, rendering them appropriate for texture segmentation and classification. The GT features are the following: 1) mean and 2) standard deviation of the absolute value of detail sub-imgs of the GT of the img.
    #
    #

    features, labels = pf.gt_features(f, mask, deg=4, freq=[0.05, 0.4])
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  5 `Other Features`

    #  5.1 `Zernikes’ Moments`
    # In img processing, computer vision and related fields, an img moment is a certain particular weighted average (moment) of the img pixels' intensities, or a function of such moments, usually chosen to have some attractive property or interpretation. Zernikes’ moment is a kind of orthogonal complex moments and its kernel is a set of Zernike complete orthogonal polynomials defined over the interior of the unit disc in the polar coordinates space. Zernike's Moments are: 1-25) orthogonal moments invariants with respect to translation.
    #
    #

    features, labels = pf.zernikes_moments(f, radius=9)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  5.2 `Hu’s Moments`
    # In img processing, computer vision and related fields, an img moment is a certain particular weighted average (moment) of the img pixels' intensities, or a function of such moments, usually chosen to have some attractive property or interpretation. Hu’s Moments are: 1-7) moments invariants with respect to translation, scale, and rotation.
    #
    #

    features, labels = pf.hu_moments(f)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  5.3 `Threshold Adjacency Matrix (TAS)`
    #

    features, labels = pf.tas_features(f)
    result.update({labels[index]: features[index] for index in range(len(labels))})

    #  5.4 `Histogram of Oriented Gradients (HOG)`
    #
    # !!!! not used because it generates lot of features
    #

    # features, labels = pf.hog_features(f, ppc=8, cpb=3)

    return result


def get_lpb_histogram(img):
    '''
        Returns the LBP histogram of the given img.
    '''
    radius = 4
    n_points = 4 * radius
    img = bgrtogray(img)
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')

    (hist, _) = np.histogram(lbp.ravel(),
                             bins=10)

    # normalize the histogram
    hist = hist.astype(np.float)
    hist /= (hist.sum() + 1e-7)
    
    result = {f"lpb-histogram-{index}": hist[index] for index in range(len(hist))}
    return result

def get_hue_moment(img):
    '''
    Calculate the Hu Moments of an img.
    '''
    img = bgrtogray(img)
    hue_moment = cv.HuMoments(cv.moments(img)).flatten()
    
    result = {f"hue-moment-{index}": hue_moment[index] for index in range(len(hue_moment))}
    return result


def get_haralick(img):
    '''
    Calculate the Haralick texture features of an img.
    '''
    gray = bgrtogray(img)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    result = {f"haralick-{index}": haralick[index] for index in range(len(haralick))}
    return result

def get_hsv_histogram(img):
    '''
        Get the histogram of the hsv channels of an img.
    '''
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([img], [0, 1, 2], None, [
                       8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    hist = hist.flatten()
    result = {f"hsv-histogram-{index}": hist[index] for index in range(len(hist))}
    return result

def get_lab_histogram(img):
    '''
        Get the histogram of the lab channels of an img.
    '''

    img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    hist = cv.calcHist([img], [0, 1, 2], None, [
                       8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv.normalize(hist, hist)
    hist = hist.flatten()
    result = {f"lab-histogram-{index}": hist[index] for index in range(len(hist))}
    return result

def get_lab_img(img):
    '''
        Get the lab channels of an img.
    '''
    return cv.cvtColor(img, cv.COLOR_RGB2LAB)

def get_hsv_img(img):
    '''
        Get the hsv channels of an img.
    '''
    return cv.cvtColor(img, cv.COLOR_RGB2HSV)

def get_graycoprops(img):
    img = bgrtogray(img)

    # distance: 1, 2, 3
    # angles:
    # - pi / 4 => 45 & -45 & 135 & -135
    # - pi / 2 => 90 & -90
    # - 0 => 0 & 180

    glcm = graycomatrix(img, [1, 2, 3], angles=[
                                0, np.pi/4, np.pi/2], symmetric=True, normed=True)
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    correlation = graycoprops(glcm, 'correlation')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    ASM = graycoprops(glcm, 'ASM')
    contrast = graycoprops(glcm, 'contrast')
    features = np.array([dissimilarity, correlation,
                         homogeneity, energy, ASM, contrast])
    features = features.flatten()
    result = {f"graycomatrix-{index}": features[index] for index in range(len(features))}

    return result


def get_gabor_img(img):
    '''
      Get gabor image
      img: numpy array
    '''

    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv.getGaborKernel(
            (ksize, ksize), 3.5, theta, 10.0, 0.5, 0, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)

    accum = np.zeros_like(img)

    def f(kern):
        return cv.filter2D(img, cv.CV_8UC3, kern)
    pool = ThreadPool(processes=8)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)

    return accum