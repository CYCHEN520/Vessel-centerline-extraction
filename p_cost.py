import numpy as np


class Cost():
    def __init__(self, image_data, min_R, max_R, R_step, intensity_range, W, H, Q):
        self.image_data = image_data
        self.min_R = min_R
        self.max_R = max_R
        self.R_step = R_step
        self.intensity_range = intensity_range

        self.W = W - 1
        self.H = H - 1
        self.Q = Q - 1

        self.k_ = 1
        self.y_ = 100
        self.n_ = 0.001
        self.N_ = 6
        self.LM_SIZE = 3

        self.P_33 = np.zeros([3, 3])

    # 计算经过LU分解的系数矩阵
    def GetpLM(self):
        cross_line_d = 2 * self.max_R + 1
        for i in range(self.LM_SIZE):
            for j in range(cross_line_d):
                pass

        #  每个半径的A系数矩阵
        # for R in range(self.max_R, self.max_R + 1):
        #     for i in range(self.LM_SIZE):
        #         for j in range(self.LM_SIZE):
        #             for k in range(2 * R + 1):

    def LUDecomLeftM33(self):
        for di in range(3):
            for dj in range(di, 3):
                self.P_33[di][dj] = self.GetUkj(di, dj)
            for dj in range(di + 1, 3):
                self.P_33[dj][di] = self.GetLik(dj, di)

    def GetUkj(self, i, j):
        sum = 0
        # for di in range(j - 1 + 1):
        #     sum = sum + self.P_33[][]
        # sum = self.P_33[][] - sum
        return sum

    def GetLik(self, i, j):
        # return self.P_33[i][k] / self.P_33[][]
        pass

    def GetPnlCoef(self):
        pass

    # 2D的mask图像的成本函数
    def p_cost_2D_mask(self):
        pass

    # mask的成本函数
    def p_cost_mask(self, p):

        cross_line_value = np.zeros([2 * self.max_R])
        cost_ = float('inf')

        for q in range(-1, 1 + 1):
            for i in range(-1, 1 + 1):
                for j in range(-1, 1 + 1):

                    # 如果遍历回到了中心则停止返回成本值
                    if (abs(q) + abs(i) + abs(j)) == 0:
                        return cost_

                    for R in range(self.min_R, self.max_R + 1):

                        cross_line_value[R] = self.image_data[p[0]][p[1]][p[2]]

                        # 读取该方向上的灰度值
                        for R_l in range(1, R + 1):
                            if (p[0] + i * R_l) > self.W or (p[0] + i * R_l) < 0 or p[1] + j * R_l > self.H or p[
                                1] + j * R_l < 0 or p[2] + q * R_l > self.Q or p[2] + q * R_l < 0:
                                d_front_value = -1000
                            else:
                                d_front_value = self.image_data[p[0] + i * R_l][p[1] + j * R_l][p[2] + q * R_l]
                            cross_line_value[R + R_l] = d_front_value

                            if (p[0] - i * R_l) > self.W or (p[0] - i * R_l) < 0 or p[1] - j * R_l > self.H or p[
                                1] - j * R_l < 0 or p[2] - q * R_l > self.Q or p[2] - q * R_l < 0:
                                d_back_value = -1000
                            else:
                                d_back_value = self.image_data[p[0] - i * R_l][p[1] - j * R_l][p[2] - q * R_l]
                            cross_line_value[R - R_l] = d_back_value

    # 灰度图的成本函数
    def p_cost_grayscale(self, p):
        # 矩阵的右边和系数矩阵
        # RY = np.zeros([self.LM_SIZE])
        # RX = []

        # 记录不同半径长度对应的凸性值
        # R_convexity = {}
        # 存放横截线上的点对应的灰度值
        # cross_line_value = []
        # 为方向计数
        p_direction = 0

        p_convexity = []
        p_symmetry = []
        M_c = []

        for q in range(-1, 1 + 1):
            for i in range(-1, 1 + 1):
                for j in range(-1, 1 + 1):

                    # 如果遍历回到了中心则停止返回成本值
                    if (abs(q) + abs(i) + abs(j)) == 0:
                        # 计算p点的损失
                        M_c.sort(reverse=True)
                        M_c_p = 1
                        for i in range(self.N_):
                            M_c_p = M_c_p * M_c[i]

                        P_c = 1 / (M_c_p)
                        return P_c

                    # 初始化
                    Q_R_min = float('inf')
                    R_mc = 0
                    max_cross_line = []
                    # 初始化为半径最大值
                    # fb = self.max_R

                    for R in range(self.min_R, self.max_R + 1, self.R_step):
                        # 检测是否是背景和目标的边界
                        '''
                        博客中的代码有问题，暂时不考虑
                        '''

                        cross_line_value = []

                        # 读取横截线的灰度值
                        # 中心线的灰度值
                        cross_line_value.append(self.image_data[p[0]][p[1]][p[2]])


                        # 如果中心点的灰度值超出范围，则定义为边界值
                        if cross_line_value[0] < self.intensity_range[0]:
                            cross_line_value[0] = self.intensity_range[0]
                        elif cross_line_value[0] > self.intensity_range[1]:
                            cross_line_value[0] = self.intensity_range[1]

                        # 读取该方向上的灰度值
                        d_front_value = cross_line_value[0]
                        d_back_value = cross_line_value[0]
                        for R_l in range(1, R + 1):
                            if (p[0] + i * R_l) > self.W or (p[0] + i * R_l) < 0 or p[1] + j * R_l > self.H or p[
                                1] + j * R_l < 0 or p[2] + q * R_l > self.Q or p[2] + q * R_l < 0:
                                d_value = -1000
                            else:
                                d_value = self.image_data[p[0] + i * R_l][p[1] + j * R_l][p[2] + q * R_l]
                            if self.intensity_range[0] <= d_value <= self.intensity_range[1]:
                                d_front_value = d_value
                            cross_line_value.insert(0, d_front_value)

                            if (p[0] - i * R_l) > self.W or (p[0] - i * R_l) < 0 or p[1] - j * R_l > self.H or p[
                                1] - j * R_l < 0 or p[2] - q * R_l > self.Q or p[2] - q * R_l < 0:
                                d_value = -1000
                            else:
                                d_value = self.image_data[p[0] - i * R_l][p[1] - j * R_l][p[2] - q * R_l]
                            if self.intensity_range[0] <= d_value <= self.intensity_range[1]:
                                d_back_value = d_value
                            cross_line_value.append(d_back_value)

                        # 拟合二次曲线
                        RX = np.polyfit(range(len(cross_line_value)), cross_line_value, 2)

                        # for l in range(self.LM_SIZE):
                        #     RY[l] = 0.0

                        # 计算公式5的右矩阵
                        # for l in range(R * 2 + 1):
                        #     d_value = cross_line_value[l]
                        #     RY[0] = RY[0] + d_value
                        #     RY[1] = RY[1] + d_value * (l + 1)
                        #     RY[2] = RY[2] + d_value * (l + 1) * (l + 1)

                        # 根据右矩阵，系数矩阵(LU分解之后)，计算得出X，即二次拟合函数的系数
                        # self.GetPnlCoef(RY, )
                        # 二次项系数存在RX[0]
                        Q_coef = RX[0]
                        # R_convexity[R] = Q_coef
                        # 找到最小的Q_coef*R
                        if (Q_coef * R) < Q_R_min:
                            Q_R_min = Q_coef * R
                            R_mc = R
                            max_cross_line = cross_line_value

                    # 取出最大凸性值时的半径R和其相邻的半径的凸性值做平均，平均值作为最终的半径
                    # if R_mc + 2 > self.max_R:
                    #     d_convexity = R_convexity[R_mc] * R_mc + R_convexity[R_mc - 2] * (R_mc - 2)
                    # else:
                    #     d_convexity = R_convexity[R_mc] * R_mc + R_convexity[R_mc + 2] * (R_mc + 2)
                    #
                    # p_convexity[p_direction] = -d_convexity / 2

                    # 添加凸性的值
                    p_convexity.append(Q_R_min)
                    # 计算非对称性
                    sum_ = 0.0
                    for l in range(1, R_mc + 1):
                        tmp = max_cross_line[R_mc - l] - cross_line_value[R_mc + 1]
                        sum_ = sum_ + tmp

                    p_symmetry.append((R_mc + 1) / (abs(sum_) + self.n_))
                    M_c.append((self.k_ + p_convexity[p_direction] ** 2) / (self.y_ + 1 / p_symmetry[p_direction] ** 2))
                    p_direction = p_direction + 1
