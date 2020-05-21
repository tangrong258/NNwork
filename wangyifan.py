# 使用pandas 中的read_csv读取数据，用的是所有值作为输入特征，csv里面除了时间其余都是输入，然后实际到达架次是输出
import pandas as pd
# 你的数据2018ZSNJ是放在桌面wangyifan文件夹中的
data = pd.read_csv(r'C:\Users\TR\Desktop\wangyifan\2018ZSNJ.csv')
# 读取的data是一个Dataframe格式，在其中的._values调用array数据,在._values中，是保存在csv
# 的除标题之外的数据，就是从第1行开始的，没有第0行的标题，
# 第0列是时间，不读取，其余列均读取出来,保存在新的array中
data_value = data._values[:, 1:]
# 需要将每一个特征也就是一列范围进行归一化到0-1,调用MinMaxScaler,默认按照列进行归一化
from sklearn.preprocessing import MinMaxScaler
# 定义归一化的区间
Slr = MinMaxScaler(feature_range=(0, 1))
# 将你的数据放入这个区间进行转化，得到归一化的数据
data_value_Slr = Slr.fit_transform(data_value)
# 到此，数据的基本格式就可以了，二维数据，一行一个样本，一列一个特征，归一化的数据
# 再调用一些要用的函数，其实所有的import语句都应放在最前面，为了程序工整

import numpy as np  # 数组函数
import tensorflow as tf  # 就是调用tensorflow这个模块
import os  # 路径模块


# 设定神经网络的超参数
time_step = 10  # 你的时间步长，就是你打算用前time_step组数据预测下一组数据
# hidden_size = 50  # 神经网络一层的神经元个数
layer_num = 3  # 定义隐含层的层数
# 上面三个函数就是超参数，你需要调参，在主函数中设定循环去遍历一个参数，其余两个不变，比如这里先修改神经元个数
output_size = ((1, 1))  # 定义输出的维度大小，你只有一个实际到港就是一行一列
batch_size = 100  # 一次训练的样本个数

# 定义一个生成三维样本的函数
def generate_data(sq):  # sq就是处理得到的二维数据
    x = []
    y = []
    sq = np.array(sq)  # 防止sq是一个list
    for i in range(0, len(sq)-time_step):
        x.append(sq[i:i + time_step])  # x做为输入选择前time_steps个值，并且选择所有特征作为输入
        y.append(sq[i + time_step, 1])  # y 作为输出，选择data_value_slr第2列(实际到达航班)，index是1
    xs = np.array(x, dtype=np.float32)
    ys = np.array(y, dtype=np.float32)
    return xs, ys


# 定义LSTM函数，就是从tensorflow的rnn中调用，这个模块你以后别的程序直接调用都可以，里面的东西都是死的，
def lstm_model(x, y,  is_training):
    cell_f = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_size)
                                        for _ in range(layer_num)])
    outputs, _ = tf.nn.dynamic_rnn(cell_f, x, dtype=tf.float32)
    output = outputs[:, -1, :]
    nonpredictions = tf.contrib.layers.fully_connected(output, output_size[0] * output_size[1], activation_fn = None )
    predictions = tf.nn.leaky_relu(nonpredictions,alpha = 0.1, name = None)

    if not is_training:
        return predictions, None, None

    predictions = tf.nn.dropout(predictions, 0.99)

    mse = [[] for i in range(output_size[0] * output_size[1])]


    for i in range(0, y.shape[1]):
        a = tf.reduce_mean(tf.square(y[:, i] - predictions[:, i]))
        mse[i].append(a)


    loss = tf.reduce_mean(mse)
    global_step = tf.Variable(0)
    LR = tf.train.exponential_decay(0.1, global_step, int(m / batch_size), 0.96, staircase=True)
    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),optimizer='Adagrad', learning_rate=LR)
    return predictions, loss, train_op


# 定义训练函数
def train(sess, x, y):
    ds = tf.data.Dataset.from_tensor_slices((x,y))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x,y = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        _, train_op, lossor = lstm_model(x, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(train_step):
        _, loss = sess.run([train_op,lossor])
        if i % 100 == 0:
            print('lost: ' + str(i), loss)


# 定义测试函数
def test(sess,x,y, test_step):
    pred_y = np.zeros((test_step * batch_size,  y.shape[1]))
    errorcsv = np.zeros((test_step * batch_size, 4))
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction,_,_ = lstm_model(x, [], False)   # 为了使得测试的时候lstm调用的参数值是训练好的，必须设置reuse=True
    for i in range(test_step):
        pred, ys = sess.run([prediction,y])   # 不能写y = 因为y和prediction被run以后就是ndarray,y这个变量已经存在了，类型是tensor，没法赋值，要能涵盖赋值，必须是同类型

        pred_y[i * batch_size:(i + 1) * batch_size, :] = pred

        mse = np.zeros((ys.shape[0], ys.shape[1]))
        mae = np.zeros((ys.shape[0],ys.shape[1]))
        mape = np.zeros((ys.shape[0], ys.shape[1]))
        for k in range(0, ys.shape[1]):
            for l in range(0, ys.shape[0]):
                a = np.square(ys[l, k] - pred[l, k])
                c = np.abs(ys[l, k] - pred[l, k])
                b = np.abs((ys[l, k] - pred[l, k]) / (ys[l, k] + 0.01))  # 防止分母为0
                mse[l,k] = a
                mae[l,k] = c
                mape[l,k] = b
        MSE = np.mean(mse,axis = 1)
        RMSE = np.sqrt(np.mean(mse))
        MAE = np.mean(mae,axis = 1)
        MAPE = np.mean(mape,axis = 1)
        ERROR = [MSE, RMSE, MAE, MAPE]
        for e in range(0, 4):
            errorcsv[i*batch_size:(i+1)*batch_size, e] = ERROR[e]
    return pred_y, errorcsv


# 定义主程序
if __name__=='__main__':

    #  定义存放MSE, RMSE, MAE, MAPE,一行一类，一列是一个你控制的超参数对应的, MAPE可能是inf,因为数据中要预测的值是0导致的
    MAEscalar = np.zeros((4, 70))
    for t in range(9, 10, 1):
        tf.reset_default_graph()
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        hidden_size = (t + 1) * 10  # 设置不同的神经元个数

        # 调用二维变三维函数，生成网络层的输入输出
        xs, ys = generate_data(data_value_Slr)

        # 判断ys是不是二维的，需要二维数据，即使是一个输出，也要改为 n行1列,
        if len(ys.shape) > 1:
            ys = ys
        else:
            y2d = np.zeros((ys.shape[0], 1), dtype='float32')
            for i in range(ys.shape[0]):
                y2d[i, 0] = ys[i]
            ys = y2d

        # 记录一共有多少样本
        m = len(xs)
        # 75%用作训练，其余测试
        train_end = int(len(xs) * 0.75)
        test_end = int(len(xs))
        # 设定训练次数
        train_step = 5000
        #  计算测试次数
        test_step = (test_end - train_end) // batch_size

        train_x = xs[0:train_end]
        train_y = ys[0:train_end]
        test_x = xs[train_end:test_end]
        test_y = ys[train_end:test_end]

        #  定义你的计算数据存放位置，这里是桌面wangyifan文件夹中
        root = r'C:\Users\TR\Desktop\wangyifan'
        # 判断error文件夹是否存在
        b = os.path.exists(root + "\\" + 'error')
        if b:
            print("path exist")
        else:
            os.makedirs(root + "\\" + 'error')

        #  训练
        train(sess, train_x, train_y)
        # 测试
        _, error = test(sess, test_x, test_y, test_step)

        # error 有三列，分别是MSE, RMSE, MAE, MAPE
        for i in range(MAEscalar.shape[0]):
            MAEscalar[i, t] = np.mean(error[:, i])

        #  保存error
        dt = pd.DataFrame(MAEscalar, index=['MSE', 'RMSE', 'MAE', 'MAPE'])
        dt.to_csv(root + "\\" + 'error' + "\\" + "error3.csv")