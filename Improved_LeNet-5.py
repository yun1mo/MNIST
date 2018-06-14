import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 从文件读取数据
train_data = np.fromfile('train/mnist_train_data', dtype=np.uint8)
test_data = np.fromfile('test/mnist_test_data', dtype=np.uint8)
train_label = np.fromfile('train/mnist_train_label', dtype=np.uint8)
test_label = np.fromfile('test/mnist_test_label', dtype=np.uint8)


train_data = train_data.reshape(60000, 45, 45,1)
test_data = test_data.reshape(10000, 45, 45,1)

# 打乱训练和测试数据
train_num = 60000
train_index = np.arange(train_num)
np.random.shuffle(train_index)
train_data = train_data[train_index]
train_label = train_label[train_index]
test_num = 10000
test_index = np.arange(test_num)
np.random.shuffle(test_index)
test_data = test_data[test_index]
test_label = test_label[test_index]

x = tf.placeholder(tf.float32, [None, 45, 45, 1], name='x')
y_ = tf.placeholder(tf.int32, [None], name='y_')


def inference(input_tensor, train, regularizer):
    # 第一层卷积层，输入为45*45，卷积核为7*7，深度为6，步长为1，无padding，输出为39*39*6
    with tf.variable_scope('layer1-conv1', reuse=tf.AUTO_REUSE):
        conv1_weights = tf.get_variable('weight', [7, 7, 1, 6], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [6], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
        feature_map1 = tf.add(conv1, conv1_biases)
        output1 = tf.nn.relu(feature_map1)

    # 第二层池化层，采用max池化，过滤器为3*3，步长为3，padding为‘SAME’，实际上未填充，输出为13*13*6
    with tf.variable_scope('layer2-pooling1', reuse=tf.AUTO_REUSE):
        output2 = tf.nn.max_pool(output1, [1, 3, 3, 1], [1, 3, 3, 1], 'SAME')

    # 第三层卷积层，输入为13*13*6，卷积核为5*5，深度为16，步长为1，无padding，输出为9*9*16
    with tf.variable_scope('layer3-conv2', reuse=tf.AUTO_REUSE):
        conv3_weights = tf.get_variable('weight', [5, 5, 6, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias', [16], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3 = tf.nn.conv2d(output2, conv3_weights, strides=[1, 1, 1, 1], padding='VALID')
        feature_map2 = tf.add(conv3, conv3_biases)
        output3 = tf.nn.relu(feature_map2)

    # 第四层池化层，采用max池化，过滤器为3*3，步长为3，padding为‘SAME’，实际上未填充，输出为3*3*16
    with tf.variable_scope('layer4-pooling2', reuse=tf.AUTO_REUSE):
        output4 = tf.nn.max_pool(output3, [1, 3, 3, 1], [1, 3, 3, 1], 'SAME')

    # 第五层：全连接层，nodes=3×3×16=144，144->84的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×144->64×84
    pool_shape = output4.get_shape()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(output4, [-1, nodes])
    with tf.variable_scope('layer5-fc1', reuse=tf.AUTO_REUSE):
        fc1_weights = tf.get_variable('weight', [nodes, 84], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [84], initializer=tf.truncated_normal_initializer(stddev=0.1))
        output5 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            output5 = tf.nn.dropout(output5, 0.5)

    # 第六层：全连接层，84->10的全连接
    #  尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×10。最后，64×10的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
    with tf.variable_scope('layer7-fc3', reuse=tf.AUTO_REUSE):
        fc3_weights = tf.get_variable('weight', [84, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias', [10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(output5, fc3_weights) + fc3_biases
    return logit

# 定义loss fuction，采用softmax+交叉熵
regularizer = tf.contrib.layers.l2_regularizer(0.001)
y = inference(x, True, regularizer)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 每次获取batch_size个样本进行训练或测试
def get_batch(data, label, batch_size):
    for start_index in range(0, len(data)-batch_size+1, batch_size):
        slice_index = slice(start_index, start_index+batch_size)
        yield data[slice_index],  label[slice_index]


# 创建Session会话
with tf.Session() as sess:
    # 初始化所有变量(权值，偏置等)
    sess.run(tf.global_variables_initializer())

    train_num = 5000
    batch_size = 200
    lossset = []
    accset = []
    for i in range(1,train_num):
        train_loss, train_acc, batch_num = 0, 0, 0
        for train_data_batch, train_label_batch in get_batch(train_data, train_label, batch_size):
            _, err, acc = sess.run([train_op, loss, accuracy], feed_dict={x: train_data_batch, y_: train_label_batch})
            train_loss += err
            train_acc += acc
            batch_num += 1
        lossset.append(train_loss / batch_num)
        accset.append(train_acc / batch_num)
        if i % 500 == 0:
            print("train times", i)
            print("train loss:", train_loss / batch_num)
            print("train acc:", train_acc / batch_num)

    y = inference(x, False, regularizer)
    err, acc = sess.run([loss, accuracy], feed_dict={x: test_data, y_: test_label})
    print("test loss:", err)
    print("test acc:", acc)

plt.subplot(121)
plt.scatter(np.arange(1, 5000), lossset, c='r')
plt.xlabel('Train times')
plt.ylabel('Loss')
plt.subplot(122)
plt.scatter(np.arange(1, 5000), accset, c='b')
plt.xlabel('Train times')
plt.ylabel('Accuracy')
plt.show()
