import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def prep_seg(input_image, model, batch_size=32):
    array = sitk.GetArrayFromImage(input_image)
    seg = array.copy()

    array = np.clip(array, -1000, 400)
    array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) / 255.0

    split = np.array_split(array, array.shape[0] // (batch_size - 2))
    mask = np.concatenate(
        [model.predict(s[..., np.newaxis], batch_size=batch_size) for s in split],
        axis=0,
    )

    mask = np.argmax(mask, axis=-1)
    mask[mask != 0] = 1

    mask = ndimage.binary_erosion(mask, iterations=2).astype("uint8")
    mask = ndimage.binary_dilation(mask, iterations=10).astype("uint8")

    seg[mask == 0] = 0
    seg_image = sitk.GetImageFromArray(seg)
    seg_image.CopyInformation(input_image)
    
    return seg_image
