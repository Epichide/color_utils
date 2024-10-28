#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: leya
# @Email: no email
# @Time: 2024/10/29 2:27
# @File: visual_3d.py
# @Software: PyCharm

import  numpy as np
import cv2
import  vedo

def show_3d_points(x=[],y=[],z=[],c=[],title="",
                   xtitle="",ytitle="",ztitle=""):
    """

    :param x: n   0-255
    :param y: n   0-255
    :param z: n   0-255
    :param c: n*4 0-255
    :param xtitle:
    :param ytitle:
    :param ztitle:
    :return:
    """
    import vedo
    import numpy as np

    # 假设n是每个轴上的点的数量
    n = 10

    # 将坐标转换为n*n*n的数组，每个点的坐标是(x, y, z)
    coords = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # 为每个点生成一个随机颜色
    colors = c  # 生成随机颜色，范围是0到1

    # 创建点云对象
    pts = vedo.Points(coords)  # c是颜色参数
    pts.pointcolors=c

    # 显示点云
    vedo.show(pts, title=title,new=True)

