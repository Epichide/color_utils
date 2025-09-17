#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Epichide
# @Email: no email
# @Time: 2025/9/16 0:50
# @File: hist_match.py
# @Software: PyCharm

import cv2
import numpy as np
from file_utils import imread, show_multi


# 计算累积分布函数(CDF)
def cdf(array, limit_value=256, bin_num=255,method="accumulate"):
    """

    :param array:
    :param limit_value:
    :param bin_num:
    :param method: accumulate, histogram
    :return:
    """
    if array.ndim == 3:
        cdf_vals = []
        gray_levels = []
        for i in range(3):
            cdf_vals_i, gray_level = cdf(array[:, :, i],limit_value,bin_num,method)
            cdf_vals.append(cdf_vals_i)
            gray_levels.append(gray_level)
    else:
        values = array.ravel()
        pixel_num = len(values)
        # 计算源图像和参考图像的直方图
        if method== "histogram":
            densitys,gray_levels=np.histogram(values,bins=bin_num,range=[0,limit_value],density=True)
            gray_levels=gray_levels[:-1]
            cdf_vals=densitys.cumsum()
        elif method=="accumulate":
            # 排序像素（使用快速排序算法）
            # sorted_pixels = np.sort(values)
            # 提取唯一灰度值及其索引（关键优化点）
            gray_levels, counts = np.unique(values, return_counts=True)
            cdf_vals = counts.cumsum() / pixel_num
            cdf_vals = np.insert(cdf_vals, 0, 0)
            cdf_vals = np.insert(cdf_vals, len(cdf_vals), 1)
            gray_levels = np.insert(gray_levels, 0, 0)
            gray_levels = np.insert(gray_levels, len(gray_levels), limit_value - 1)
    return cdf_vals, gray_levels


def interpreter_matching(source_array, source_values, source_cdfs,
                         reference_values, reference_cdfs,maxvalue=255):
    if isinstance(source_values, list):
        source_array_match_cdf = source_array.copy()
        for i in range(len(source_values)):
            source_array_match_cdf_i = interpreter_matching(source_array[:, :, i],
                                                            source_values[i], source_cdfs[i],
                                                            reference_values[i], reference_cdfs[i])
            source_array_match_cdf[:, :, i] = source_array_match_cdf_i
    else:
        source_values_match_cdf = np.interp(source_cdfs, reference_cdfs, reference_values)
        source_array_match_cdf = np.interp(source_array, source_values, source_values_match_cdf)
    source_array_match_cdf=np.clip(source_array_match_cdf,0,maxvalue)
    return source_array_match_cdf


def get_map_curve(source_array, source_values, source_cdfs,
                  reference_values, reference_cdfs, maxvalue=255):
    maps = []
    if isinstance(source_values, list):
        for i in range(len(source_values)):
            source_values_i, source_values_match_cdf_i = interpreter_matching(source_array[:, :, i],
                                                                              source_values[i], source_cdfs[i],
                                                                              reference_values[i], reference_cdfs[i])[0]
            maps.append([source_values_i, source_values_match_cdf_i])
    else:
        source_values_match_cdf = np.interp(source_cdfs, reference_cdfs, reference_values)
        source_array_match_cdf = np.interp(source_array, source_values, source_values_match_cdf)
        maps.append([source_values, source_values_match_cdf])
    return maps


def apply_map_curve(source_array, source_values, source_values_match_cdf, maxvalue=255):
    source_array_match_cdf = np.interp(source_array, source_values, source_values_match_cdf)
    source_array_match_cdf = np.clip(source_array_match_cdf, 0, maxvalue)
    # show_multi([source_array, source_array_match_cdf],titles=["gray","match"])
    # plt.show()
    return source_array_match_cdf


def histogram_matching(source_img, reference_img, mode="gray",method="histogram"):
    """
    实现直方图匹配（规定化）
    参数:
        source_img: 源图像(PIL Image对象)
        reference_img: 参考图像(PIL Image对象)

    返回:
        matched_img: 匹配后的图像(PIL Image对象)
    """
    limit_value = 256
    bin_num = 255
    if mode == "color":
        source_array = np.array(source_img)
        reference_array = np.array(reference_img)
        source_cdfs, source_values = cdf(source_array, limit_value, bin_num,method=method)
        reference_cdfs, reference_values = cdf(reference_array, limit_value, bin_num,method=method)
        source_array_match_cdf = interpreter_matching(source_array, source_values, source_cdfs,
                                                      reference_values, reference_cdfs, maxvalue=limit_value - 1)

        show_multi([source_array, reference_array, source_array_match_cdf],
                   titles=["src", "tar", "src-matched"])
    else:
        # 将图像转换为灰度图
        source_array = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        reference_array = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        source_cdfs, source_values = cdf(source_array, limit_value, bin_num,method=method)
        reference_cdfs, reference_values = cdf(reference_array, limit_value, bin_num,method=method)
        source_array_match_cdf = interpreter_matching(source_array, source_values, source_cdfs,
                                                      reference_values, reference_cdfs, maxvalue=limit_value - 1)
        source_array_match_cdf_color=np.array(source_img*(source_array_match_cdf/(source_array+0.01))[...,None],dtype=np.float32)
        source_array_match_cdf_color=source_array_match_cdf_color/(limit_value-1)
        source_array_match_cdf_color=source_array_match_cdf_color.clip(0,1)
        show_multi([source_array, reference_array, source_array_match_cdf,source_img,reference_img,source_array_match_cdf_color],
                   titles=["src", "tar", "src-matched","src", "tar", "src-matched"],nrow=2)


    import matplotlib.pyplot as plt
    plt.show()
    return


def test_histgrom_matching():
    try:
        # 读取源图像和参考图像
        source_image = imread("img/1.jpg")
        reference_image = imread("img/2.jpg")

        # 执行直方图匹配
        result_image = histogram_matching(source_image, reference_image, mode="color")
        result_image = histogram_matching(source_image, reference_image, mode="gray")

        print("直方图匹配完成，结果已保存为 matched_result.jpg")

    except Exception as e:
        print(f"发生错误: {e}")
        raise


# 使用示例
if __name__ == "__main__":

    test_histgrom_matching()
