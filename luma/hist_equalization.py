
import cv2
import numpy as np
from matplotlib import pyplot as plt

from hist_match import cdf, interpreter_matching
from file_utils import show_multi, imread


def generate_uniform_cdf(limit_value=256, bin_num=256):
    """
    生成均匀分布的cdf
    """
    cdf_vals, gray_levels = np.linspace(0, 1, bin_num), np.linspace(0, limit_value, bin_num)
    return cdf_vals, gray_levels

def clip_threshold_cdf(reference_cdfs,threshold=10):
    hist= np.diff(reference_cdfs)
    hists= np.insert(hist,0,reference_cdfs[0])
    all_sum = sum(hists)
    all_mean=all_sum/len(hists)
    threshold_value = all_mean * threshold
    total_extra = sum([h - threshold_value for h in hists if h >= threshold_value])
    print("clip total_extra",total_extra)
    mean_extra = total_extra / len(hists)

    clip_hists = [0 for _ in hists]
    for i in range(len(hists)):
        if hists[i] >= threshold_value:
            clip_hists[i] = (threshold_value + mean_extra)
        else:
            clip_hists[i] = (hists[i] + mean_extra)
    clip_hists_cdf=np.cumsum(clip_hists)
    return clip_hists_cdf
def histogram_equlizaion(source_img,
                         limit_value=256,bin_num = 255,Isshow=True):
    """
    实现直方图均衡化
    参数:
        source_img: 源图像(PIL Image对象)
    返回:
        matched_img: 均衡化后的图像(PIL Image对象)
    """
    if source_img.ndim==3:
        source_array = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    else:
        source_array=source_img
    source_cdfs, source_values = cdf(source_array, limit_value, bin_num)
    reference_cdfs, reference_values = generate_uniform_cdf(limit_value, bin_num)
    source_array_match_cdf = interpreter_matching(source_array, source_values, source_cdfs,
                                                  reference_values, reference_cdfs)
    gain=(source_array_match_cdf / (source_array + 0.01))[..., None] if source_img.ndim==3 else (source_array_match_cdf / (source_array + 0.01))
    source_array_match_cdf_color = np.array(source_img * gain,
                                            dtype=np.float32)
    source_array_match_cdf_color = source_array_match_cdf_color #/ (limit_value - 1)
    source_array_match_cdf_color = source_array_match_cdf_color.clip(0, limit_value-1)
    if Isshow:
        show_multi([source_array, source_array_match_cdf,
                    source_img,source_array_match_cdf_color/ (limit_value - 1)],
                   titles=["src","HE","src","HE"],nrow=2)

        plt.show()
    return source_array_match_cdf,source_array_match_cdf_color


def test_hist_equalization():
    try:
        # 读取源图像和参考图像
        source_image = imread("img/1.jpg")
        # 执行直方图匹配
        result_image = histogram_equlizaion(source_image, Isshow=True)

        print("直方图匹配完成，结果已保存为 matched_result.jpg")

    except Exception as e:
        print(f"发生错误: {e}")


def AHE(source_img,limit_value=256,bin_num = 255,Isshow=False):

    source_array = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    window_size = 512#32
    affect_size = 256#16

    # calculate how many blocks needed in row-axis and col-axis
    (m, n) = source_array.shape
    if (m - window_size) % affect_size == 0:
        rows = int((m - window_size) / affect_size + 1)
    else:
        rows = int((m - window_size) / affect_size + 2)
    if (n - window_size) % affect_size == 0:
        cols = int((n - window_size) / affect_size + 1)
    else:
        cols = int((n - window_size) / affect_size + 2)

    # equalize histogram of every block image
    arr = source_array.copy()
    for i in range(rows):
        for j in range(cols):
            # offset
            off = int((window_size - affect_size) / 2)

            # affect region border
            asi, aei = i * affect_size + off, (i + 1) * affect_size + off
            asj, aej = j * affect_size + off, (j + 1) * affect_size + off

            # window region border
            wsi, wei = i * affect_size, i * affect_size + window_size
            wsj, wej = j * affect_size, j * affect_size + window_size

            # equalize the window region
            window_arr = source_array[wsi: wei, wsj: wej]
            block_arr = histogram_equlizaion(window_arr,
                                             limit_value=limit_value,bin_num = bin_num,Isshow=False)[0]

            # border case
            if i == 0:
                arr[wsi: asi, wsj: wej] = block_arr[0: asi - wsi, :]
            elif i >= rows - 1:
                arr[aei: wei, wsj: wej] = block_arr[aei - wsi: wei - wsi, :]
            if j == 0:
                arr[wsi: wei, wsj: asj] = block_arr[:, 0: asj - wsj]
            elif j >= cols - 1:
                arr[wsi: wei, aej: wej] = block_arr[:, aej - wsj: wej - wsj]
            arr[asi: aei, asj: aej] = block_arr[asi - wsi: aei - wsi, asj - wsj: aej - wsj]

    source_array_match_cdf = arr
    source_array_match_cdf_color = np.array(source_img * (source_array_match_cdf / (source_array + 0.01))[..., None],
                                            dtype=np.float32)
    source_array_match_cdf_color = source_array_match_cdf_color
    source_array_match_cdf_color = source_array_match_cdf_color.clip(0, limit_value - 1)
    if Isshow:
        show_multi([source_array, source_array_match_cdf, source_img, source_array_match_cdf_color/ (limit_value - 1)],
                   titles=["src", "HE", "src", "HE"], nrow=2)
        import matplotlib.pyplot as plt
        plt.show()
    return source_array_match_cdf,source_array_match_cdf_color

def test_AHE():
    try:
        # 读取源图像和参考图像
        source_image = imread("img/1.jpg")
        # 执行直方图匹配
        result_image = AHE(source_image, mode="color")

        print("直方图匹配完成，结果已保存为 matched_result.jpg")

    except Exception as e:
        print(f"发生错误: {e}")
        raise
def CLAHE(grayimg):
    pass

def CLHE(source_img,limit_value=256,bin_num = 255,Isshow=True,threshold=5):
    """
    Contrast Limited HE
    :param source_img:
    :return:
    """
    if source_img.ndim == 3:
        source_array = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    else:
        source_array = source_img
    source_cdfs, source_values = cdf(source_array, limit_value, bin_num,method="histogram")
    clip_hists_cdf=clip_threshold_cdf(source_cdfs,threshold=threshold)

    source_array_match_cdf = interpreter_matching(source_array, source_values, source_cdfs,
                                                  source_values, clip_hists_cdf)
    gain = (source_array_match_cdf / (source_array + 0.01))[..., None] if source_img.ndim == 3 else (
                source_array_match_cdf / (source_array + 0.01))
    source_array_match_cdf_color = np.array(source_img * gain,
                                            dtype=np.float32)
    source_array_match_cdf_color = source_array_match_cdf_color  # / (limit_value - 1)
    source_array_match_cdf_color = source_array_match_cdf_color.clip(0, limit_value - 1)
    if Isshow:
        show_multi([source_array, source_array_match_cdf, source_img, source_array_match_cdf_color / (limit_value - 1)],
                   titles=["src", "CLHE", "src", "CLHE"], nrow=2)

        plt.show()
    return source_array_match_cdf,source_array_match_cdf_color

def ACLHE(source_img,limit_value=256,bin_num = 255,Isshow=True,threshold=5):
    pass
def test_CLHE():
    try:
        # 读取源图像和参考图像
        source_image = imread("img/1.jpg")
        # 执行直方图匹配
        result_image = CLHE(source_image, mode="color")

        print("直方图匹配完成，结果已保存为 matched_result.jpg")

    except Exception as e:
        print(f"发生错误: {e}")

def test_all():
    try:
        limit_value=256
        # 读取源图像和参考图像
        source_image = imread("img/1.jpg")
        source_gray  = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
        # 执行直方图匹配
        HE_match_gray,HE_match_color= histogram_equlizaion(source_image,Isshow=False)
        CLHE_match_gray,CLHE_match_color = CLHE(source_image,Isshow=False,threshold=1)
        AHE_match_gray,AHE_match_color=AHE(source_image,Isshow=False)

        show_multi([source_gray,HE_match_gray,AHE_match_gray,CLHE_match_gray,
                    source_image,HE_match_color/(limit_value - 1),AHE_match_color/(limit_value-1),CLHE_match_color/(limit_value-1)],
                   titles=["src","HE","AHE","CLHE",
                           "src","HE","AHE","CLHE"],nrow=2)
        plt.show()


        print("直方图匹配完成，结果已保存为 matched_result.jpg")

    except Exception as e:
        raise
        print(f"发生错误: {e}")

if __name__ == '__main__':
    # test_hist_equalization()
    test_all()
    # test_CLHE()
    # test_hist_equalization()
    # test_AHE()
