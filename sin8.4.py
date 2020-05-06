import tensorflow as tf
import numpy as np


import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt



HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIME_STEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TEST_EXAMPLES = 1000
SAMPLE_GAP = 0.01


def generate_data(seq):
    x = []
    y = []
    for i in range (len(seq)-TIME_STEPS ):
        x.append([seq[i:i+TIME_STEPS ]])
        y.append([seq[i+TIME_STEPS ]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(x,y,is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell (HIDDEN_SIZE)
                                        for _ in range(NUM_LAYERS)])
    outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    output = outputs[:,-1,:]

    predictions = tf.contrib.layers.fully_connected(output,1,activation_fn=None)

    if not is_training:
        return predictions ,None,None

    loss = tf.losses.mean_squared_error (labels= y,predictions = predictions )

    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),
                                               optimizer="Adagrad",learning_rate=0.1)
    return predictions ,loss,train_op

def train(sess,train_x,train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE )
    x,y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope ("model"):
        predictions,loss,train_op = lstm_model(x,y,True)

    sess.run(tf.global_variables_initializer() )#run的是全连接层的权重阈值参数
    for i in range(TRAINING_STEPS ):
        _,l = sess.run([train_op,loss])
        if i % 100 == 0:
            print("train step: " + str(i) +",loss:" + str(l))

def run_eval(sess,test_x,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)
    x,y = ds.make_one_shot_iterator() .get_next()

    with tf.variable_scope("model",reuse=True):
        prediction,_,_= lstm_model(x,[0.0],False)

    predictions = []
    labels = []
    for i in range (TEST_EXAMPLES ):
        p,l = sess.run([prediction,y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions ).squeeze()
    labels = np.array(labels ).squeeze()
    rmse = np.sqrt(((predictions - labels )**2).mean(axis=0))
    print("Mean Square Error is: %f" % rmse)

    plt.figure()
    plt.plot(predictions ,label = 'predictions')
    plt.plot(labels,label = 'real_sin')
    plt.legend()
    plt.show()

test_start = (TRAINING_EXAMPLES + TIME_STEPS )* SAMPLE_GAP
test_end = test_start + (TEST_EXAMPLES  + TIME_STEPS )* SAMPLE_GAP

train_x, train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES + TIME_STEPS,
                                                    dtype=np.float32)))
print("train_x:",train_x.shape,train_y.shape)
test_x,test_y = generate_data(np.sin(np.linspace(test_start ,test_end ,TEST_EXAMPLES  + TIME_STEPS,
                                                 dtype=np.float32)))


with tf.Session() as sess:
    train (sess,train_x,train_y)
    run_eval(sess,test_x,test_y)
