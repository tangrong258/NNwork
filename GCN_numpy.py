import numpy as np
import tensorflow as tf

A = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float)

X = np.array([
    [i, -i]
    for i in range(A.shape[0])],
    dtype=float)

# 每个节点的第i个特征变为于其有向相连的上层节点的第i个特征之和,却不包含自己的特征
H_1 = np.matmul(A, X)
print(H_1)

# 增加自环，包含自己的上层特征
I = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]],
    dtype=float)
A = A + I
H_1 = np.matmul(A, X)
print(H_1)

# 对特征表征归一化处理，对邻接矩阵左乘度矩阵D的逆矩阵，通过节点的度对特征进行归一化
dia =  np.sum(A, axis=1)
D = np.array(np.diag(dia))
print(D)
D_inv = np.linalg.inv(D)

A_s = np.matmul(D_inv, A)
# 将A的每一行按照该节点的度进行归一化，度为n,对应的连接权重为1/n,
print('A_s:', A_s)
# 就相当于将该顶点的相连的节点第i个特征求均值（原先是求和）后得到该节点的第i个特征值
H_1 = np.matmul(A_s, X)
print(H_1)

# 增加权重矩阵W，就是改变特征是取均值中情况，为每个赋予不同权重，但是仍然是不同节点（H每一行）
# 会共用权重（W每一列）W的列就是控制输出到下一层的特征维度,行数必须等于上一层的特征维度

W = np.random.rand(2, 1)
H_1 = np.matmul(H_1, W)
print(H_1)
# 激活
H_1 = tf.nn.relu(H_1)
print(H_1)