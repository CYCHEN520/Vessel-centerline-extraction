import cv2
import imageio
import nibabel as nib
import numpy as np
import p_cost
from nibabel.viewers import OrthoSlicer3D


# 获取图片的mask像素值矩阵
def read_img(img_path):
    img = nib.load(img_path)
    return img


# 读取nii图片
def read_nii(img_path):
    img = nib.load(img_path)
    return img


# 裁剪图像
def crop_img(image):
    pass


# 获取领域的坐标点
def get_neighbors(p):
    x, y, z = p
    x_left = 0 if x == 0 else x - 1
    x_right = W if x == W - 1 else x + 2
    y_front = 0 if y == 0 else y - 1
    y_back = H if y == H - 1 else y + 2
    z_top = Q if z == Q - 1 else Q + 2
    z_bottom = 0 if z == 0 else z - 1

    return [(x, y, z) for x in range(x_left, x_right) for y in range(y_front, y_back)
            for z in range(z_bottom, z_top)]


# dijkstra算法
def dijkstra():
    pass


def path_to_p(end_, steps):
    for i in range(steps):
        top_p = paths[end_]
        end_ = top_p
        if end_ == start_:
            break

    return end_


def MPP_BT(min_R, max_R,R_step,intensity_range):
    # 已经处理的集合
    processed = set()
    # 初始化成本函数
    cost_ = p_cost(image, min_R, max_R)
    start_cost = cost_.p_cost_grayscale(start_)
    # 当前路径的累积成本
    cost = {start_: start_cost}

    StopPropagation = False
    while StopPropagation == False:
        # 每次取出当前成本代价最小值
        p = min(cost, key=cost.get)
        # 获取当前成本最小值的领域节点
        neighbors = get_neighbors(p)
        # 保存已经处理过的点
        processed.add(p)
        # 新扩展点的记录累积成本和回溯跟踪点
        for next_p in [x for x in neighbors if x not in processed]:
            next_p_cost = cost_.p_cost_grayscale(next_p)
            now_cost = next_p_cost + cost[p]
            # 如果该领域点之前计算过了，则需要判断此时所用的代价小还是之前的代价小
            if next_p in cost:
                if now_cost < cost[next_p]:
                    cost.pop(next_p)
            else:
                # 如果该领域点之前没有计算过，或者需要更新
                cost[next_p] = now_cost
                processed.add(next_p)
                paths[next_p] = p

            # 回溯了Ibk步
            pbk = path_to_p(next_p, I_bk)
            if pbk == start_:
                P_pbk = cost(start_)
            else:
                pbk_front = path_to_p(next_p, 1)
                P_pbk = cost(pbk) - cost(pbk_front)
            I_bk[pbk] = I_bk[pbk] + 1 / (n_ + P_pbk)

        # cost.pop(p)


if __name__ == "__main__":
    global image, W, H, Q, intensity_range
    global start_, paths, lbk, I_bk
    img_path = 'CL_data/10361479_gray/data_0.nii'
    image_ = read_nii(img_path)
    W, H, Q = image_.dataobj.shape
    # OrthoSlicer3D(image_.dataobj).show()
    image = image_.get_fdata().transpose(1, 0, 2)
    # 初始化血管强度
    intensity_range = [0,500]
    # 初始化参数最小半径，最大半径，回溯步数
    min_R = 2
    max_R = 15
    R_step = 2
    lbk = 15
    # 初始化平衡系数k_和y_，血管中轴提取阈值yc
    k_ = 100
    y_ = 1
    n_ = 0.001

    # 初始化全局变量起始点start_，回溯路径paths，特征图I_bk
    start_ = (100,100,100)
    paths = {}
    I_bk = np.zeros([W, H, Q])
    MPP_BT(min_R, max_R,R_step )
    print(image)
