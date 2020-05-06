import numpy as np
import math


class UAVClassification(object):

    def __init__(self, data, vertical_class_name, horizontal_class_num):
        
        self.data = data
        self.quadrant_num = horizontal_class_num - 1
        # # # vertical # # #
        # 判断输入维度,获取样本
        if len(self.data.shape) == 1:
            self.data_vertical = self.data
        else:
            self.data_vertical = self.data[:, 0]

        # 状态种类变量
        self.class_name_vertical = ['climb', 'level', 'descend']
        self.class_vertical = [[] for _ in range(vertical_class_name)]
        
        # 制作标签初始域载体，全0array，n类就是一个n*n的矩阵
        self.labels_class_vertical = np.zeros((len(self.class_name_vertical), len(self.class_name_vertical)))
        
        # 设置分类数值区间，三类就是两个分界数值
        self.cut_point_vertical = [-0.5, 0.5]
        
        # 输出样本的标签集合以及不同类的样本集合
        self.labels_vertical = self.classify_vertical()

        # # # horizontal # # #
        # 获取样本，因为水平至少两个维度
        self.data_horizontal = self.data[:, 1:3]

        # 状态种类变量
        # 将水平方向的运动分为5种，origin：hover；1-4quadrant：level fight, 4 kind of fight area
        # 这里是分了4块水平飞行的区域，可以更细化到8块，预测范围更精确，但是分类误差应该会增加，是否可以研究一下多少块区域最合适

        # 判断分几个象限
        if self.quadrant_num == 4:
            self.class_name_horizontal = ['origin', 'quadrant1', 'quadrant2', 'quadrant3', 'quadrant4']
            self.class_horizontal = [[] for _ in range(horizontal_class_num)]

        if self.quadrant_num == 8:
            self.class_name_horizontal = ['origin', 'quadrant1', 'quadrant2', 'quadrant3', 'quadrant4',
                                          'quadrant5', 'quadrant6', 'quadrant7', 'quadrant8']
            self.class_horizontal = [[] for _ in range(horizontal_class_num)]

        if self.quadrant_num == 16:
            self.class_name_horizontal = ['origin']
            for i in range(1, 17):
                self.class_name_horizontal.append('quadrant' + str(i))

            self.class_horizontal = [[] for _ in range(horizontal_class_num)]

        # 制作标签初始域载体，全0array，n类就是一个n*n的矩阵
        self.labels_class_horizontal = np.zeros((len(self.class_name_horizontal), len(self.class_name_horizontal)))

        # 设置分类数值区间，5类需要2个方向各2分界数值,
        self.cut_point_horizontal = [-1, 1]

        # 输出样本的标签集合以及不同类的样本集合
        self.labels_horizontal = self.classify_horizontal()

    def classify_vertical(self):
        
        # 确定每一类的label
        for index, _ in enumerate(self.class_name_vertical):
            self.labels_class_vertical[index, index] = 1
            
        # 将输入样本data的labels放在一个数组里，一行为一个样本的label
        labels_data = np.zeros((len(self.data_vertical), len(self.class_name_vertical)))
        
        # 遍历样本
        for index_data in range(len(self.data_vertical)):

            single_data_vertical = self.data_vertical[index_data]

            # 飞行状态改变趋势分类， 同时设置标签
            if single_data_vertical > self.cut_point_vertical[0]:
                if single_data_vertical > self.cut_point_vertical[1]:
                    self.class_vertical[0].append(single_data_vertical)
                    labels_data[index_data] = self.labels_class_vertical[0]
                else:
                    self.class_vertical[1].append(single_data_vertical)
                    labels_data[index_data] = self.labels_class_vertical[1]
            else:
                self.class_vertical[2].append(single_data_vertical)
                labels_data[index_data] = self.labels_class_vertical[2]
        
        return labels_data

    def classify_horizontal(self):

        # 确定每一类的label
        for index, _ in enumerate(self.class_name_horizontal):
            self.labels_class_horizontal[index, index] = 1

        # 将输入样本data的labels放在一个数组里，一行为一个样本的label
        labels_data = np.zeros((len(self.data), len(self.class_name_horizontal)))

        # 遍历样本
        for index_data in range(len(self.data_horizontal)):
            # 不同于vertical, 这里的单个数据是一个array，第一元素是lat，第二元素是long
            single_data_horizontal = self.data_horizontal[index_data]

            # 飞行状态改变趋势分类， 同时设置标签, 这里为不同种情况添加一个def
            if self.quadrant_num == 4:
                labels_data = self.horizontal_4_quadrant(single_data_horizontal, labels_data, index_data)
            if self.quadrant_num == 8:
                labels_data = self.horizontal_8_quadrant(single_data_horizontal, labels_data, index_data)
            if self.quadrant_num == 16:
                labels_data = self.horizontal_16_quadrant(single_data_horizontal, labels_data, index_data)

        return labels_data
    
    def horizontal_4_quadrant(self, single_data_horizontal, labels_data, index):

        if single_data_horizontal[0] ** 2 + single_data_horizontal[1] ** 2 > self.cut_point_horizontal[1]:
            # 讨论水平飞行的各个情况
            if single_data_horizontal[0] >= 0:
                if single_data_horizontal[1] >= 0:
                    self.class_horizontal[1].append(single_data_horizontal)
                    labels_data[index] = self.labels_class_horizontal[1]
                else:
                    self.class_horizontal[2].append(single_data_horizontal)
                    labels_data[index] = self.labels_class_horizontal[2]
            else:
                if single_data_horizontal[1] >= 0:
                    self.class_horizontal[4].append(single_data_horizontal)
                    labels_data[index] = self.labels_class_horizontal[4]
                else:
                    self.class_horizontal[3].append(single_data_horizontal)
                    labels_data[index] = self.labels_class_horizontal[3]
        else:
            # 其余认为悬停
            self.class_horizontal[0].append(single_data_horizontal)
            labels_data[index] = self.labels_class_horizontal[0]

        return labels_data

    def horizontal_8_quadrant(self, single_data_horizontal, labels_data, index):

        if single_data_horizontal[0] ** 2 + single_data_horizontal[1] ** 2 > self.cut_point_horizontal[1]:
            # 讨论水平飞行的各个情况
            if single_data_horizontal[0] >= 0:
                if single_data_horizontal[1] >= 0:
                    if np.abs(single_data_horizontal[1]) >= np.abs(single_data_horizontal[0]):
                        self.class_horizontal[1].append(single_data_horizontal)
                        labels_data[index] = self.labels_class_horizontal[1]
                    else:
                        self.class_horizontal[2].append(single_data_horizontal)
                        labels_data[index] = self.labels_class_horizontal[2]
                else:
                    if np.abs(single_data_horizontal[1]) <= np.abs(single_data_horizontal[0]):
                        self.class_horizontal[3].append(single_data_horizontal)
                        labels_data[index] = self.labels_class_horizontal[3]
                    else:
                        self.class_horizontal[4].append(single_data_horizontal)
                        labels_data[index] = self.labels_class_horizontal[4]
            else:
                if single_data_horizontal[1] <= 0:
                    if np.abs(single_data_horizontal[1]) >= np.abs(single_data_horizontal[0]):
                        self.class_horizontal[5].append(single_data_horizontal)
                        labels_data[index] = self.labels_class_horizontal[5]
                    else:
                        self.class_horizontal[6].append(single_data_horizontal)
                        labels_data[index] = self.labels_class_horizontal[6]
                else:
                    if np.abs(single_data_horizontal[1]) <= np.abs(single_data_horizontal[0]):
                        self.class_horizontal[7].append(single_data_horizontal)
                        labels_data[index] = self.labels_class_horizontal[7]
                    else:
                        self.class_horizontal[8].append(single_data_horizontal)
                        labels_data[index] = self.labels_class_horizontal[8]
        else:
            # 其余认为悬停
            self.class_horizontal[0].append(single_data_horizontal)
            labels_data[index] = self.labels_class_horizontal[0]

        return labels_data

    def horizontal_16_quadrant(self, single_data_horizontal, labels_data, index):

        coff = math.tan(22.5)

        if single_data_horizontal[0] ** 2 + single_data_horizontal[1] ** 2 > self.cut_point_horizontal[1]:
            # 讨论水平飞行的各个情况
            if single_data_horizontal[0] >= 0:
                # 在第一象限划分
                if single_data_horizontal[1] >= 0:
                    if np.abs(single_data_horizontal[1]) >= np.abs(single_data_horizontal[0]):
                        if np.abs(single_data_horizontal[1]) * coff >= np.abs(single_data_horizontal[0]):
                            self.class_horizontal[1].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[1]
                        else:
                            self.class_horizontal[2].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[2]

                    else:
                        if np.abs(single_data_horizontal[0]) * coff <= np.abs(single_data_horizontal[1]):
                            self.class_horizontal[3].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[3]
                        else:
                            self.class_horizontal[4].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[4]
                # 在第二象限划分
                else:
                    if np.abs(single_data_horizontal[1]) <= np.abs(single_data_horizontal[0]):
                        if np.abs(single_data_horizontal[0]) * coff >= np.abs(single_data_horizontal[1]):
                            self.class_horizontal[5].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[5]
                        else:
                            self.class_horizontal[6].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[6]
                    else:
                        if np.abs(single_data_horizontal[1]) * coff <= np.abs(single_data_horizontal[0]):
                            self.class_horizontal[7].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[7]
                        else:
                            self.class_horizontal[8].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[8]
            else:
                # 在第三象限讨论
                if single_data_horizontal[1] <= 0:
                    if np.abs(single_data_horizontal[1]) >= np.abs(single_data_horizontal[0]):
                        if np.abs(single_data_horizontal[1]) * coff >= np.abs(single_data_horizontal[0]):
                            self.class_horizontal[9].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[9]
                        else:
                            self.class_horizontal[10].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[10]
                    else:
                        if np.abs(single_data_horizontal[0]) * coff <= np.abs(single_data_horizontal[1]):
                            self.class_horizontal[11].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[11]
                        else:
                            self.class_horizontal[12].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[12]
                # 在第四象限讨论
                else:
                    if np.abs(single_data_horizontal[1]) <= np.abs(single_data_horizontal[0]):
                        if np.abs(single_data_horizontal[1]) <= np.abs(single_data_horizontal[0]) * coff:
                            self.class_horizontal[13].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[13]
                        else:
                            self.class_horizontal[14].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[14]
                    else:
                        if np.abs(single_data_horizontal[1]) * coff <= np.abs(single_data_horizontal[0]):
                            self.class_horizontal[15].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[15]
                        else:
                            self.class_horizontal[16].append(single_data_horizontal)
                            labels_data[index] = self.labels_class_horizontal[16]
        else:
            # 其余认为悬停
            self.class_horizontal[0].append(single_data_horizontal)
            labels_data[index] = self.labels_class_horizontal[0]

        return labels_data