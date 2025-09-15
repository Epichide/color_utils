#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Epichide
# @Email: no email
# @Time: 2025/9/16 0:50
# @File: file_utils.py
# @Software: PyCharm

import cv2
import numpy as np
from matplotlib import pyplot as plt


def imread(file_path, colormode="BGR"):
    # BGR
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if colormode == "RGB":
        cv_img = cv_img[:, :, ::-1]
    return cv_img


def show_multi(imgs, titles=[], nrow=1, colormode="BGR", types=[]):
    ncols = int(np.ceil(len(imgs) / nrow))
    if not types:
        types = ["img"] * len(imgs)
        f, axs = plt.subplots(nrow, ncols)
    else:
        allimg = [1 if dtype == "img" else 0 for dtype in types]
        allimg = sum(allimg)
        if allimg == len(imgs):
            f, axs = plt.subplots(nrow, ncols, sharex=1, sharey=1)
        else:
            f, axs = plt.subplots(nrow, ncols)
    if len(imgs) > 1:
        axs = axs.ravel()
        i = 0
        for ax, img, dtype in zip(axs, imgs, types):
            if dtype == "img":
                if img.ndim == 2:
                    # img to bgr img
                    # 将灰度图像扩展为RGB图像
                    # if not img.dtype == np.uint8:
                    #
                    #     img_gray = to_image255(img)
                    # else:
                    #     img_gray = img
                    # img_rgb = cv2.cvtColor(img_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

                    # rgb_img = np.stack([img, img, img], axis=-1)
                    ax.imshow(img, cmap='gray')
                elif colormode == "BGR":
                    ax.imshow(img[:, :, ::-1])
                else:
                    ax.imshow(img)
                if i < len(titles):
                    ax.set_title(titles[i])
            elif dtype == "hist":
                img = img.flatten()
                ax.hist(img[img > 0], bins=256, range=[0, 1], color='r')

            i += 1
    else:
        if colormode == "BGR":
            axs.imshow(imgs[0][:, :, ::-1])
        else:
            axs.imshow(imgs[0])


if __name__ == '__main__':
    pass