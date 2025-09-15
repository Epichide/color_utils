
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


def histogram_equlizaion(source_img, mode="gray",
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
        show_multi([source_array, source_array_match_cdf,source_img,source_array_match_cdf_color/ (limit_value - 1)],
                   titles=["src","HE","src","HE"],nrow=2)

        plt.show()
    return source_array_match_cdf_color


def test_hist_equalization():
    try:
        # 读取源图像和参考图像
        source_image = imread("img/1.jpg")
        # 执行直方图匹配
        result_image = histogram_equlizaion(source_image, mode="color")

        print("直方图匹配完成，结果已保存为 matched_result.jpg")

    except Exception as e:
        print(f"发生错误: {e}")


def AHE(source_img, mode="gray",limit_value=256,bin_num = 255):

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
            block_arr = histogram_equlizaion(window_arr, mode="gray",
                                             limit_value=limit_value,bin_num = bin_num,Isshow=False)

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

    show_multi([source_array, source_array_match_cdf, source_img, source_array_match_cdf_color/ (limit_value - 1)],
               titles=["src", "HE", "src", "HE"], nrow=2)
    import matplotlib.pyplot as plt
    plt.show()
    return

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


if __name__ == '__main__':
    test_hist_equalization()
    test_AHE()
