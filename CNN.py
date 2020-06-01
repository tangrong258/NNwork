
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#true表示是这类标签是1，其余为0，而false表示0123456789表示类别数

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#产生随机变量
    return tf.Variable(initial)  # 用get_Variable, 不然test的时候会报错参数没有初始化，因为tf.Variable只会创建新的变量，而不是如果已存在就调用

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1#‘VALID’抽取的图片大小比原图片小
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')# strinde 抽取的步长，【1，1，1，1】
      #中间1表示x,y方向布长为1
    #input shape为 [ batch, in_height, in_weight, in_channel ]，
    #fliter 卷积核 shape为 [ filter_height, filter_weight, in_channel, out_channels ]，

    # strides 应该是对应于input的在各个维度上的步长

def max_pool_2x2(x):#为了防止步长过大，遗漏信息，使用pooling,移动2步，就相当于把图片压缩了
    # stride [1, x_movement, y_movement, 1]
    #ksize：表示池化窗口的大小：一个长度为4的一维列表，一般为[1, height, width, 1]，因不想在batch和channels上做池化，则将其值设为1
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image= tf.reshape(xs,[-1,28,28,1])#-1表示图片的维度，1表示为黑白，彩色为3 RGB
#print(x_image.shape)#输出的应该是【n_samples,28,28,1】
## conv1 layer ##
W_conv1= weight_variable([5,5,1,32]) #隐藏层中的神经元 具有一个固定大小的感受视野去感受上一层的部分特征
#卷积层的神经元是只与前一层的部分神经元节点相连，每一条相连的线对应一个权重 w
#一个感受视野带有一个卷积核，我们将 感受视野 中的权重 w 矩阵称为 卷积核，32个卷积核就相当于32个神经元，一张图片对于这层的每一个神经元都有一个权重
# patch 5*5（范围）,in size 1（输入厚度）,out size 32(输出后厚度)等于卷积核数量
b_conv1= bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image ,W_conv1 )+b_conv1)#output size [1,28,28,32]
h_pool1=max_pool_2x2(h_conv1 ) #output size [1,14,14,32]
## conv2 layer ##
W_conv2= weight_variable([5,5,32,64]) #patch 5*5（范围）,in size 32（输入厚度）,out size 64(输出后高度)
b_conv2= bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2 )+b_conv2)#output size 14,14,64
h_pool2=max_pool_2x2(h_conv2) #output size 7,7,64
## func1 layer
W_fc1= weight_variable([7*7*64,512])#512是这层的神经元个数，不是固定的
b_fc1= bias_variable([512])
#把第二个pooling以后的形状由【n_samples,7,7,64】变为【n_samples,7*7*64】，
# 这样你的空间特征还能保持吗，或者说你默认从第0行开始，一次往右遍历，这样技术顺序
h_pool2_flat= tf.reshape(h_pool2,[-1,7*7*64] )
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat , W_fc1  )+b_fc1 )#输出格式为[?,512]
h_fc1_drop=tf.nn.dropout (h_fc1 ,keep_prob )#第一全连接层输出,dropout
## func2 layer ##
W_fc2= weight_variable([512,10])#得到10类，那个概率最大就是预测的几
b_fc2= bias_variable([10])
#两次降维度，有点像主成分分析的意思，用较少的变量去表示原始变量包含的信息
fully_connect = tf.matmul(h_fc1_drop ,W_fc2 )+b_fc2#输出格式为[?, 10]
#如果是一张图片，最后就输出10个数，代表10类
prediction= tf.nn.softmax(fully_connect)
#代表这10个类的各自概率

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))#交叉熵
#其实就是 axis = 1,就是[1,10]对10个数进行求和      # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#Adam 更快 其实是train_op,优化器这样写是怎么调用前面的神经网络的呢

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)#一次训练100张，返回值都是二维数组，行是样本数量
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys,keep_prob: 0.5
                                    })
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))#到999终止，