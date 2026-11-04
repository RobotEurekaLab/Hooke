"""Depth-dependent optical blur of the actual inclined capillary silhouette.

This is a finite-resolution image approximation, not glass wave propagation.
Neighbouring depth planes share each pixel's opacity so partitioning cannot
increase the tool diameter or opacity. Far planes use coarser sampling.
"""

import numpy as np
from PIL import Image


def composite_capillary(image, layer, tip, back, focal_height, calibration, blur, *, geometry=None):
    pixels = np.asarray(layer, dtype=float)
    alpha = pixels[..., 3]/255.
    rows, columns = np.nonzero(alpha)
    if not len(rows):
        return image
    tip, back = np.asarray(tip), np.asarray(back)
    direction = back[:2]
    xx = (columns-calibration.pixels/2)*calibration.pixel_size_m
    yy = (calibration.pixels/2-rows)*calibration.pixel_size_m
    distance = ((xx-tip[0])*direction[0]+(yy-tip[1])*direction[1])/np.dot(direction, direction)
    depth = tip[2]+np.maximum(0., distance)*back[2]-focal_height
    if geometry is not None and geometry.bend_end_m:
        from microscopy.pipettes import NEEDLE_LENGTH_M
        direction = direction/np.linalg.norm(direction)
        path = geometry.centreline(np.linspace(0., NEEDLE_LENGTH_M, 257))
        projected = (xx-tip[0])*direction[0]+(yy-tip[1])*direction[1]
        depth = tip[2]+np.interp(projected, path[:, :2]@direction, path[:, 2])-focal_height
    positive = np.array([.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 144])*1e-6
    planes = np.r_[-positive[::-1], 0., positive]
    upper = np.clip(np.searchsorted(planes, depth), 1, len(planes)-1)
    fraction = np.clip((depth-planes[upper-1])/(planes[upper]-planes[upper-1]), 0., 1.)
    opacity = np.zeros(image.size[::-1], dtype=float)
    colour = np.zeros_like(opacity)
    # The drawn glass silhouette is grayscale. Keep floating point throughout
    # convolution so a thin glass wall is not rounded away before compositing.
    luminance = pixels[rows, columns, :3].mean(axis=1)
    source_alpha = alpha[rows, columns]
    for index in np.unique(np.r_[upper-1, upper]):
        weights = np.where(upper == index, fraction, np.where(upper-1 == index, 1-fraction, 0.))
        active = weights > 0
        if not active.any():
            continue
        sigma = np.hypot(.8, planes[index]/calibration.defocus_scale_m)
        if sigma > calibration.pixels/4:
            continue
        padding = int(np.ceil(4*sigma))
        y0, y1 = max(0, rows[active].min()-padding), min(calibration.pixels, rows[active].max()+padding+1)
        x0, x1 = max(0, columns[active].min()-padding), min(calibration.pixels, columns[active].max()+padding+1)
        patch_alpha = np.zeros((y1-y0, x1-x0))
        patch_colour = np.zeros_like(patch_alpha)
        local = (rows[active]-y0, columns[active]-x0)
        patch_alpha[local] = source_alpha[active]*weights[active]
        patch_colour[local] = luminance[active]*patch_alpha[local]
        opacity[y0:y1, x0:x1] += blur(patch_alpha, sigma=(sigma, sigma))
        colour[y0:y1, x0:x1] += blur(patch_colour, sigma=(sigma, sigma))
    result = np.asarray(image, dtype=float)*(1-opacity.clip(0, 1)[..., None])+colour[..., None]
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))
