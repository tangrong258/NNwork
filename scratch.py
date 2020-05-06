import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

state=tf.Variable(0,name='counter')
#print(state,name)
one = tf.constant(1)

new_value=tf.add(state,one)
update=tf.assign(state,new_value)#将new_value赋给state

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(3):
    sess.run(update)
    print(sess.run(state))
