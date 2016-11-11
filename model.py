import tensorflow as tf

height = 24
width = 24
depth = 1

def model():
    # define the tensors

    #  n dimensional uniform distribution
    Z = tf.Variable(tf.random_uniform((100,)))

    # Project Z

    W_1 = tf.nn.conv2d_transpose()

def generator(x):


    W = tf.Variable(tf.zeros(()))
    b = tf.variable(tf.zeros(()))

    # Same shape as input
    W_filter_conv1 = tf.Variable(tf.random_normal(
        shape=[height, width, depth],
        mean=0.0,
        stddev=1
    ))

    # 4-D input
    # filter
    h_conv1 = tf.nn.conv2d(
        input=x,
        filter=W_filter_conv1,
        strides=[],
        padding=False,
        name=''
    )

    # apply non linearity
    h_activation1 = tf.nn.rel







def discriminator():

def train():
    x  = None # image none for now
    x = tf.reshape(x, [-1, height, width, depth])
