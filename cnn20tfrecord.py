import tensorflow as tf

def read_and_decode(filename, is_batch):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [20])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    if is_batch:
        batch_size = 50
        min_after_dequeue = 20
        num_threads = 1
        capacity = min_after_dequeue + (num_threads + 1) * batch_size
        img, label = tf.train.shuffle_batch([img, label],
                                            batch_size=batch_size,
                                            num_threads=num_threads,
                                            capacity=capacity,
                                            min_after_dequeue=min_after_dequeue)
    return img, label

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, 5*4])
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
x_image= tf.reshape(xs,[-1,5,4,1])#这里会修改维度，不需要在读取record的时候将输出维度转换为三维
W_conv1= weight_variable([5,5,1,32])
b_conv1= bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1 )+b_conv1)
h_pool1=max_pool_2x2(h_conv1 )

W_conv2= weight_variable([5,5,32,64])
b_conv2= bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2 )+b_conv2)
h_pool2=max_pool_2x2(h_conv2)


W_fc1= weight_variable([2*1*64,32])
b_fc1= bias_variable([32])
h_pool2_flat= tf.reshape(h_pool2,[-1,2*1*64] )
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat , W_fc1  )+b_fc1 )
h_fc1_drop=tf.nn.dropout (h_fc1 ,keep_prob )


W_fc2= weight_variable([32,2])
b_fc2= bias_variable([2])
fully_connect = tf.matmul(h_fc1_drop ,W_fc2 )+b_fc2

prediction= tf.nn.softmax(fully_connect)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
filename = r"D:\ZSNJAP01\flight\clf\train.tfrecords"
with tf.Session() as sess:
    sess.run(init)
    train_img, train_labels = read_and_decode(filename,is_batch=True)
    img = train_img.eval(session = sess)
    labels = train_labels.eval(session = sess)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: img, ys: labels, keep_prob: 0.5})
        if i % 50 == 0:
            print(compute_accuracy(
                img[:100], labels[:100]))