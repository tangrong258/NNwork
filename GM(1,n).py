import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





A = np.array([560823,542386,604834,591248,583031,640636,575688,689637,570790,519574,614677])
x0 = np.array([[104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3],
             [135.6, 140.2, 140.1, 146.9, 144.0, 143.0, 133.3, 135.7, 125.8,98.5,99.8],
             [131.6, 135.5, 142.6, 143.2, 142.2, 138.4, 138.4, 135, 122.5, 87.2, 96.5],
             [54.2, 54.9, 54.8, 56.3, 54.5, 54.6, 54.9, 54.8, 49.3, 41.5, 48.9]])
[n, m] = np.shape(x0)
AGO = np.cumsum(A) #累加函数，第n个值是前面值的累加
T = 1
x1 = np.zeros((n, m+T))
Z = np.zeros((1, m-1))
for k in range(0, (m-1)):
    Z[0, k] = (AGO[k]+AGO[k+1])/2  #Z(i)为xi(1)的紧邻均值生成序列

for i in range(0, n):
    for j in range(0, m):
        for k in range(0, j):
            x1[i, j] = x1[i, j] + x0[i, k] #原始数据一次累加,得到xi(1)

#这不就是对x0的每一行使用cumsum函数嘛

x11 = x1[:, 0:m]
X = x1[:, 1:m].T #截取矩阵
#Yn = A  #Yn为常数项向量
Yn = np.delete(A, 0, 0) #从第二个数开始，即x(2),x(3)...
Yn = Yn.T  #Yn=A(:,2:m).T;
ZZ = -Z.T
B = np.hstack((ZZ, X))
C = (np.dot(np.linalg.inv(np.dot(B.T, B)), np.dot(B.T, Yn))).T #由公式建立GM(1,n)模型
a = C[0]
b = C[1:n+1]
F = np.zeros(m+1)
F[0] = A[0]
u = np.zeros(m)
for i in range(m):
    for j in range(n):
        u[i] = u[i] + b[j] * x11[j, i]

for k in range(1, m+1):
    F[k] = (A[0]-u[k-1]/a) / np.exp(a*(k-1))+u[k-1]/a

G = np.zeros(m+1)
G[0] = A[0]
for k in range(1, m+1):
    G[k] = F[k]-F[k-1] #两者做差还原原序列，得到预测数据

fig = plt.figure()
t1 = np.arange(0, m)
t2 = np.arange(0, m)
plt.title('fig')
plt.scatter(t1, A, color='r', marker='*')
plt.plot(t1, A, color='black', linestyle='-')
plt.scatter(t2, G, color='b', marker='o')
plt.plot(t2, G, color='gray', linestyle='-')
plt.legend('act', 'pred')
plt.show()