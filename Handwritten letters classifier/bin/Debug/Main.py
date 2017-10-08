import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask
import random
import os

app = Flask(__name__)

#Constants:--------------------------
input_data = np.genfromtxt("test_set.csv", delimiter=',', dtype=np.uint8)
input_label = np.genfromtxt("test_labels_set.csv", delimiter=',', dtype=np.uint8)
input_label = input_label.T[:6].T
N_EXAMPLES = input_data.shape[0]



default_saver_path = "./logs/"

Alphabet = list("ABCDEF")

sess = tf.InteractiveSession() #Main session
#-------------------------------------



def params_init(shape, name):
    w = tf.truncated_normal(shape=shape, stddev=0.1)
    #tf.summary.histogram(name, w)
    return tf.Variable(w, name= name)

def conv2D_init(x, kernel_shape,output_channels, name, activation=tf.nn.relu):
    with tf.name_scope(name):
        w = params_init(kernel_shape, "W")
        b = params_init([output_channels], "b")
    return tf.nn.relu(tf.nn.conv2d(x, filter=w, strides=[1, 1,1,1], padding='SAME', name=name))

def pool2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def fcl_init(x, in_neurons, out_neurons,name):
    with tf.name_scope(name):
        w = params_init([in_neurons, out_neurons], name="W")
        b = params_init([out_neurons], name="b")
    return(tf.matmul(x, w) + b)


s = pd.Series(Alphabet)
s = pd.get_dummies(s)

# Get one hot encoding of given letter
def get_one_hot(c):
    return s.iloc[Alphabet.index(c)].as_matrix()[np.newaxis].T

# Return letter using its index in the latin alphabet
def get_letter(i):
    return Alphabet[i]




#---Composing Latin Classifier Net---


#Input:
x = tf.placeholder(tf.float32, shape=[None, 784 * 3])  # input vectorized images
y = tf.placeholder(tf.float32, shape=[None, 6])  # Output labels (Onehot encoding)
x_image = tf.reshape(x, [-1, 28, 28, 3])

#Convolution-Pool 1:
conv1 = conv2D_init(x_image, [5, 5, 3, 32], 32, "conv1")
pool1 = pool2x2(conv1, "pool1")
#Convolution-Pool 2:
conv2 = conv2D_init(pool1, [5, 5, 32, 64], 64, "conv2")
pool2 = pool2x2(conv2, "pool2")

#Full-connected layer 1:
vectorized_pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
fcl1 = tf.nn.relu(fcl_init(vectorized_pool2, 7 * 7 * 64, 1024, "fcl1"))
#Drop out:
keep_prob = tf.placeholder(tf.float32, name="Keep_prob")
d_fcl1 = tf.nn.dropout(fcl1, keep_prob=keep_prob, name="fcl1_regul")

#Full-connected layer 2:
y_hat = fcl_init(d_fcl1, 1024, 6, "fcl2")

#Loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)

#Train Step:
train_step = tf.train.AdamOptimizer(4e-4).minimize(cross_entropy)

#Accuracy:
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#--------------------------------------------------------------------


saver = tf.train.Saver()


def Initialising_LatinClassifierNet(restore = False, path = default_saver_path, ckpt_name = ""):
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    save = tf.train.Saver()
    if (restore):
        save.restore(sess,path+ckpt_name+".ckpt")

def Learn(name,train_set = input_data, train_set_labels=input_label, batch_size=15, N_ATTEMPTIONS=20,epochs=1001,sess=sess,testing=True):
    list_accuracy = []
    for z in range(N_ATTEMPTIONS):
        k = 0
        N_EXAMPLES = train_set.shape[0]
        inx_seq = range(N_EXAMPLES)
        random.shuffle(inx_seq)
        sess.run(tf.global_variables_initializer())
        if (testing):
            test_inx = inx_seq[:N_EXAMPLES // 10]
            inx_seq = inx_seq[N_EXAMPLES // 10:]
            N_EXAMPLES = N_EXAMPLES - N_EXAMPLES // 10
        sess.run(tf.initialize_all_variables())
        #print("------Attemtion {0}------".format(z))
        for i in range(epochs):
            if ((k + 1) * batch_size >= N_EXAMPLES):
                k = 0
                np.random.shuffle(inx_seq)

            batch_data = train_set[inx_seq[k * batch_size:(k + 1) * batch_size]]
            batch_label = train_set_labels[inx_seq[k * batch_size:(k + 1) * batch_size]]
            k += 1

            if (i % 100 == 0):
                trainning_accuracy = sess.run(accuracy, feed_dict={x: batch_data, y: batch_label, keep_prob: 0.5})
               # print(i, trainning_accuracy)


            sess.run(train_step, feed_dict={x: batch_data, y: batch_label, keep_prob: 0.5})

        if (testing):
            acc = sess.run(accuracy, feed_dict={x: input_data[test_inx], y: input_label[test_inx], keep_prob: 1.})
            list_accuracy = np.hstack((list_accuracy, acc))
            #print("Attemption {0}: Accuracy = {1}".format(z, acc))
        #if (testing):
            #print("Overall: {0}".format(np.mean(list_accuracy)))
        if (not testing):
            saver.save(sess, default_saver_path+name+".ckpt",)

@app.route('/<path>')
def start(path):
    #Initialising_LatinClassifierNet()
    #Learn("test", N_ATTEMPTIONS=20)
    #Learn("complete_training", N_ATTEMPTIONS=1, testing=False,epochs=2000)
    saver = tf.train.Saver()
    saver.restore(save_path="./logs/complete_training.ckpt",sess=sess)
    #Initialising_LatinClassifierNet(restore=True,ckpt_name="complete_training")


    im = Image.open(path)
    a = np.array(im)
    a = a.flatten()
    a = a.reshape(1,2352)
    #print(sess.run(tf.nn.softmax((y_hat)), feed_dict={x:a, keep_prob:1.}))
    ans = sess.run((tf.argmax(tf.nn.softmax(y_hat),1)), feed_dict={x:a, keep_prob:1.})
    return (get_letter(ans[0]))
