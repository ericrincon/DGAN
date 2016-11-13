import tensorflow as tf

height = 24
width = 24
depth = 1

def prelu(x, alpha=0):
    return tf.maximum(alpha ** x, x)

def model():
    # define the tensors

    #  n dimensional uniform distribution
    Z = tf.Variable(tf.random_uniform((100,)))

    # Project Z

    W_1 = tf.nn.conv2d_transpose()

def generator(x):


    W = tf.Variable(tf.zeros(()))
    b = tf.Variable(tf.zeros(()))

    # Same shape as input
    W_filter_conv1 = tf.Variable(tf.random_normal(
        shape=[height, width, depth],
        mean=0.0,
        stddev=1
    ))

    # 4-D input
    # filter
    h_conv1 = tf.nn.conv2d_transpose(
        value=x,
        filter=W_filter_conv1,
        strides=[],
        padding=False,
        output_shape=(4,4,1024),
        name=''
    )

    # apply non linearity which is a leaky ReLU
    # Paper on PReLU: https://arxiv.org/pdf/1502.01852v1.pdf
    h_activation1 = prelu(h_conv1)

    # mapooling
    h_conv2 = tf.nn.conv2d_transpose(
        value=h_conv1,
        filter=[None, 4, 4, 1024],
        output_shape=[None, 4, 4, 1024],
        strides=
    )

    h_activation2 = prelu(h_conv2)

    h_activation1 = prelu(h_conv1)

    # mapooling
    max1 = tf.nn.max_pool(
        value=h_activation1,
        ksize=,
        strides=,
        padding=
    )









def discriminator():

def train():
    x  = None # image none for now
    x = tf.reshape(x, [-1, height, width, depth])
