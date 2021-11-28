# -*- coding: utf-8 -*-
"""
Created on: 2019/5/27 14:29
@Author: zsfeng
"""

import tensorflow as tf


def neural_tensor_layer(class_vector, query_encoder, out_size=100):
    """neural tensor layer (NTN)"""
    print('-' * 20)
    print('| NTL > class_vector: ', class_vector.shape)
    print('| NTL > query_encoder: ', query_encoder.shape)

    C, H = class_vector.shape
    # print("class_vector shape:", class_vector.shape)
    # print("query_encoder shape:", query_encoder.shape)
    M = tf.get_variable("M", [H, H, out_size], dtype=tf.float32,
                        initializer=tf.keras.initializers.glorot_normal())
    print('| NTL > M: ', M.shape)

    mid_pro = []
    for slice in range(out_size):
        class_slide = tf.matmul(class_vector, M[:, :, slice])
        print('| NTL > class_slide: ', class_slide.shape)
        slice_inter = tf.matmul(class_slide, query_encoder, transpose_b=True)  # (C,Q)
        print('| NTL > slice_inter: ', slice_inter.shape)

        mid_pro.append(slice_inter)
    tensor_bi_product = tf.concat(mid_pro, axis=0)  # (C*K,Q)
    print('| NTL > tensor_bi_product: ', tensor_bi_product.shape)

    V = tf.nn.relu(tf.transpose(tensor_bi_product))
    print('| NTL > V: ', V.shape)

    W = tf.get_variable("w", [C * out_size, C], dtype=tf.float32, initializer=tf.keras.initializers.glorot_normal())
    print('| NTL > W: ', W.shape)
    b = tf.get_variable("b", [C], dtype=tf.float32, initializer=tf.keras.initializers.glorot_normal())
    print('| NTL > b: ', b.shape)
    probs = tf.nn.sigmoid(tf.matmul(V, W) + b)  # (Q,C)
    print('| NTL > probs: ', probs.shape)

    return probs


def self_attention(inputs):
    _, sequence_length, hidden_size = inputs.shape
    with tf.variable_scope('self_attn'):
        x_proj = tf.layers.Dense(hidden_size)(inputs)
        print('| self_attention > x_proj: ', x_proj.shape)
        x_proj = tf.nn.tanh(x_proj)
        print('| self_attention > x_proj: ', x_proj.shape)
        u_w = tf.get_variable('W_a2', shape=[hidden_size, 1],
                              dtype=tf.float32, initializer=tf.keras.initializers.glorot_normal())
        print('| self_attention > u_w: ', u_w.shape)
        x = tf.tensordot(x_proj, u_w, axes=1)
        print('| self_attention > x: ', x.shape)
        alphas = tf.nn.softmax(x, axis=1)
        print('| self_attention > alphas: ', alphas.shape)
        output = tf.matmul(tf.transpose(inputs, [0, 2, 1]), alphas)
        print('| self_attention > output: ', output.shape)
        output = tf.squeeze(output, -1)
        print('| self_attention > output: ', output.shape)
        return output


def dynamic_routing(input, b_IJ, iter_routing=2):
    ''' The routing algorithm.'''

    C, K, H = input.shape
    W = tf.get_variable('W_s', shape=[H, H],
                        dtype=tf.float32, initializer=tf.keras.initializers.glorot_normal())
    print('| dynamic_routing > W: ', W.shape)
    for r_iter in range(iter_routing):
        print('-' * 20)
        with tf.variable_scope('iter_' + str(r_iter)):
            d_I = tf.nn.softmax(tf.reshape(b_IJ, [C, K, 1]), axis=1)
            print('| dynamic_routing > d_I: ', d_I.shape)
            # for all samples j = 1, ..., K in class i:
            e_IJ = tf.reshape(tf.matmul(tf.reshape(input, [-1, H]), W), [C, K, -1])  # (C,K,H)
            print('| dynamic_routing > e_IJ: ', e_IJ.shape)
            c_I = tf.reduce_sum(tf.multiply(d_I, e_IJ), axis=1, keepdims=True)  # (C,1,H)
            print('| dynamic_routing > c_I: ', c_I.shape)
            c_I = tf.reshape(c_I, [C, -1])  # (C,H)
            print('| dynamic_routing > c_I: ', c_I.shape)
            c_I = squash(c_I)  # (C,H)
            print('| dynamic_routing > c_I: ', c_I.shape)
            c_produce_e = tf.matmul(e_IJ, tf.reshape(c_I, [C, H, 1]))  # (C,K,1)
            print('| dynamic_routing > c_produce_e: ', c_produce_e.shape)
            # for all samples j = 1, ..., K in class i:
            b_IJ += tf.reshape(c_produce_e, [C, K])
            print('| dynamic_routing > b_IJ: ', b_IJ.shape)

    return c_I


def squash(vector):
    '''Squashing function corresponding to Eq. 1'''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), 1, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * vector  # element-wise
    return (vec_squashed)


def tensordot():
    import numpy as np

    x = np.array([i for i in range(24)]).reshape(2, 3, 4)
    # print(x)

    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant([[1], [2], [3],[4]], dtype=tf.float32)
    z = tf.tensordot(x, y, axes=1)
    sess = tf.Session()
    with sess.as_default():
        print(x.eval().shape)
        print(y.eval().shape)
        print(x.eval())
        print(y.eval())
        print(z.eval())


if __name__ == "__main__":
    tensordot()

    exit(0)
    import numpy as np

    inputs = np.random.random((24, 5, 10))  # (3*3+3*5,seq_len,lstm_hidden_size*2)
    print(inputs.shape)
    inputs = tf.constant(inputs, dtype=tf.float32)
    encoder = self_attention(inputs)  # (k*c,lstm_hidden_size*2)

    support_encoder = tf.slice(encoder, [0, 0], [9, 10])
    query_encoder = tf.slice(encoder, [9, 0], [15, 10])

    support_encoder = tf.reshape(support_encoder, [3, 3, -1])
    b_IJ = tf.constant(np.zeros([3, 3], dtype=np.float32))
    class_vector = dynamic_routing(support_encoder, b_IJ)
    inter = neural_tensor_layer(class_vector, query_encoder, out_size=10)

    # test accuracy
    query_label = [0, 1, 2] * 5
    print(query_label)
    predict = tf.argmax(name="predictions", input=inter, axis=1)
    correct_prediction = tf.equal(tf.cast(predict, tf.int32), query_label)
    accuracy = tf.reduce_mean(name="accuracy", input_tensor=tf.cast(correct_prediction, tf.float32))
    labels_one_hot = tf.one_hot(query_label, 3, dtype=tf.float32)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        print(encoder.eval().shape)
        print(query_encoder.eval().shape)
        print(inter.eval().shape)
        print(predict.eval().shape)
        print(correct_prediction.eval().shape)
        print(accuracy.eval().shape)
