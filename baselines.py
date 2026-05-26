import numpy as np
from skimage.filters import threshold_otsu
from sklearn.svm import LinearSVC

def run_otsu_mndwi(img_s2):
    """
    Runs Otsu's thresholding on MNDWI calculated from Sentinel-2 image.
    img_s2 has shape (H, W, C) and is normalized to [0, 1]
    Channels: B11 is 0, B8 is 1, B3 is 2
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Green is Band 3, SWIR is Band 11
        band3 = img_s2[:, :, 2]
        band11 = img_s2[:, :, 0]
        denom = band3 + band11
        mndwi = np.where(denom == 0, 0.0, (band3 - band11) / denom)
    
    mndwi[np.isnan(mndwi)] = -1.0
    
    m_min, m_max = mndwi.min(), mndwi.max()
    if m_max > m_min:
        norm = (mndwi - m_min) / (m_max - m_min)
        thresh = threshold_otsu(norm)
        # Match notebook comparison logic:
        # In the original notebook, they compared mndwi (unnormalized) directly with thresh (normalized)
        pred = mndwi > thresh
    else:
        pred = mndwi > 0.0
        
    return pred.astype(np.uint8)

class BaselineSVM:
    def __init__(self, C=10.0, tol=1e-4):
        self.clf = LinearSVC(dual=False, C=C, tol=tol, random_state=42)

    def train(self, X, Y):
        """
        X: features array, shape (N_pixels, N_features)
        Y: labels array, shape (N_pixels,)
        """
        self.clf.fit(X, Y)

    def predict(self, X):
        return self.clf.predict(X)
