#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: leya
# @Email: no email
# @Time: 2024/10/29 2:44
# @File: colorspace.py
# @Software: PyCharm
import cv2
import numpy as np

SIZE=200
def generate_RGB_cube():
    size=SIZE
    spaces=np.linspace(0,1,size)
    r,g,b=np.meshgrid(spaces,spaces,spaces)

    R=(r*255).ravel()
    G=(g*255).ravel()
    B=(b*255).ravel()

    RGB=np.stack((R,G,B),axis=-1)
    RGBA=np.c_[R,G,B,np.ones_like(R)*125]
    RGB = RGB.reshape(-1, 3)

    return RGB,RGBA

def generate_Lab_cube():
    size=SIZE
    RGB,RGBA=generate_RGB_cube()
    Lab=cv2.cvtColor(RGB.reshape(-1,1,3).astype(np.uint8),cv2.COLOR_RGB2Lab)
    # Lab=Lab.reshape(size,size,size,3)
    Lab=Lab.reshape(-1,3)
    return Lab,RGBA

def visualizer_color_sapce():
    Lab,RGBA=generate_Lab_cube()
    RGB,RGBA=generate_RGB_cube()
    import visual_3d
    visual_3d.show_3d_points(RGB[:,0],RGB[:,1],RGB[:,2],RGBA,"RGB","R","G","B")
    visual_3d.show_3d_points(Lab[:,0],Lab[:,1],Lab[:,2],RGBA,"Lab","L","a","b")

if __name__ == '__main__':
    visualizer_color_sapce()
