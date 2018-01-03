#! /usr/bin/env python

import os
import sys
import time
import tensorflow as tf
import numpy as np
import utils


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_file", "../../../dataset/time-aware/real_train.txt", "Training data set")
tf.flags.DEFINE_string("validation_file", "../../../dataset/time-aware/real_validation.txt", "Validation data set")
tf.flags.DEFINE_string("test_file", "../../../dataset/time-aware/real_test.txt", "Testing data set")
tf.flags.DEFINE_string("visual_features", "../../../dataset/vggoutput.npz", "Visual feature file")
tf.flags.DEFINE_string("embedding_file", "../../../dataset/word_embedding/word2vec", "Data source")

# Model Hyperparameters
tf.flags.DEFINE_integer("wordembedding_dim", 100, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("hidden_state", 100, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("attention_hidden_state", 400, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("relu", 600, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("time_feature_dim", 50, "Dimensionality of time feature")
tf.flags.DEFINE_integer("user_feature_dim", 250, "Dimensionality of time feature")
tf.flags.DEFINE_integer("num_month", 12, "Number of times")
tf.flags.DEFINE_integer("num_week", 7, "Number of times")
tf.flags.DEFINE_integer("num_times", 24, "Number of times")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("visual_feature_dim", 4096, "Dimensionality of visual feature")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 1024, "Number of training epochs")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learing rate")
tf.flags.DEFINE_float("dropout_rate",0.5, "Dropout rate")

# Tensorflow Option Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
train, validation, test = utils.load_data(FLAGS.train_file, FLAGS.validation_file, FLAGS.test_file)

# Build Dictionary
print("Build Dictionary...")
word2id, id2word, user2id, id2user, poi2id, id2poi, post2id, id2post = utils.build_dic(train, validation, test)

# Convert Data to Index
print("Converting Data...")
train, validation, test, maximum_document_length = utils.converting(train, validation, test, word2id, user2id, poi2id, post2id)

# Load pretrained embedding
print("Load pretrained word embedding...")
_word_embedding = utils.load_embedding(FLAGS.embedding_file, word2id, FLAGS.wordembedding_dim)

# Load Visual Feature
print("Loading Visual Feature Matrix...")
with open(FLAGS.visual_features) as f:
    _visual_feature = np.load(f)["array"]

# Load visual feature
print("word dict size: "+str(len(word2id)))
print("user dict size: "+str(len(user2id)))
print("poi dict size: "+str(len(poi2id)))
print("Train/Validation/Test: {:d}/{:d}/{:d}".format(len(train), len(validation), len(test)))
print("==================================================================================")

best = [0, .0, .0, .0, .0]

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        
        # ===========================
        # input layer
        # ===========================

        dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
        input_image = tf.placeholder(tf.float32, [None, FLAGS.visual_feature_dim], name="input_image")
        input_user = tf.placeholder(tf.int32, [None], name="input_user")
        input_content = tf.placeholder(tf.int32, [None, maximum_document_length], name="input_content")
        input_poi = tf.placeholder(tf.int32, [None], name="input_poi")
        input_month = tf.placeholder(tf.int32, [None], name="input_month")
        input_week = tf.placeholder(tf.int32, [None], name="input_week")
        input_time = tf.placeholder(tf.int32, [None], name="input_time")

        # keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        with tf.device("/cpu:0"):
            # word embedding matrix
            word_embedding = tf.Variable(tf.truncated_normal([len(word2id), FLAGS.wordembedding_dim]), trainable=True, name="user_embedding")
            embedded_words = tf.nn.embedding_lookup(word_embedding, input_content)

            # user embedding matrix
            user_embedding = tf.Variable(tf.truncated_normal([len(user2id), FLAGS.user_feature_dim]), trainable=True, name="user_embedding")
            embedded_users = tf.nn.embedding_lookup(user_embedding, input_user)

            # time embedding matrix
            month_embedding = tf.Variable(tf.truncated_normal([FLAGS.num_month, FLAGS.time_feature_dim]), trainable=True, name="month_embedding")
            embedded_months = tf.nn.embedding_lookup(month_embedding, input_month)

            week_embedding = tf.Variable(tf.truncated_normal([FLAGS.num_week, FLAGS.time_feature_dim]), trainable=True, name="week_embedding")
            embedded_weeks = tf.nn.embedding_lookup(week_embedding, input_week)

            time_embedding = tf.Variable(tf.truncated_normal([FLAGS.num_times, FLAGS.time_feature_dim]), trainable=True, name="time_embedding")
            embedded_times = tf.nn.embedding_lookup(time_embedding, input_time)


        # identity matrix for POI
        identity_matrix = tf.constant(np.identity(len(poi2id)))
        corrected_poi = tf.nn.embedding_lookup(identity_matrix, input_poi)

        # ===========================
        # textual RNN layer
        # ===========================

        def length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
            length = tf.reduce_sum(used, 1)
            length = tf.cast(length, tf.int32)
            return length

        gru_f = tf.contrib.rnn.GRUCell(FLAGS.hidden_state)
        gru_b = tf.contrib.rnn.GRUCell(FLAGS.hidden_state)
        gru_f = tf.nn.rnn_cell.DropoutWrapper(gru_f, dropout_rate)
        gru_b = tf.nn.rnn_cell.DropoutWrapper(gru_b, dropout_rate)

        input_length = length(embedded_words)
        output, h_n = tf.nn.bidirectional_dynamic_rnn(gru_f, gru_b, embedded_words, dtype=tf.float32, sequence_length = input_length)
        output = tf.concat(output, 2)

        # Attention Layer
        attention1_w = tf.get_variable("attention1_w", shape=[2*FLAGS.hidden_state, FLAGS.attention_hidden_state], initializer=tf.contrib.layers.xavier_initializer())
        attention1_b = tf.Variable(tf.constant(0.1, shape=[FLAGS.attention_hidden_state]), name="attention1_b")

        attention2_w = tf.get_variable("attention2_w", shape=[FLAGS.attention_hidden_state, 1], initializer=tf.contrib.layers.xavier_initializer())
        attention2_b = tf.Variable(tf.constant(0.1, shape=[1]), name="attention2_b")

        output_weight = tf.reshape(output, [-1, 2*FLAGS.hidden_state])
        attention_hidden = tf.tanh(tf.nn.xw_plus_b(output_weight, attention1_w, attention1_b))
        attention_hidden = tf.nn.xw_plus_b(attention_hidden, attention2_w, attention2_b)
        attention_weight = tf.reshape(attention_hidden, [-1, maximum_document_length, 1])
        attention_weight = tf.nn.softmax(attention_weight, dim=1)
        textual_feature = tf.reduce_sum(tf.multiply(output, attention_weight), 1)

        # Feed Foward Layers
        textual_w = tf.get_variable("textual_w", shape=[2*FLAGS.hidden_state+FLAGS.user_feature_dim+FLAGS.visual_feature_dim+3*FLAGS.time_feature_dim, FLAGS.relu], initializer=tf.contrib.layers.xavier_initializer())
        textual_b = tf.Variable(tf.constant(0.1, shape=[FLAGS.relu]), name="textual_b")

        output = tf.concat([textual_feature, input_image, embedded_users, embedded_months, embedded_weeks, embedded_times], 1)
        output = tf.nn.dropout(output, dropout_rate)
        output = tf.nn.xw_plus_b(output,textual_w,textual_b)
        output = tf.nn.relu(output)

        textual_w2 = tf.get_variable("textual_w2", shape=[FLAGS.relu, len(poi2id)], initializer=tf.contrib.layers.xavier_initializer())
        textual_b2 = tf.Variable(tf.constant(0.1, shape=[len(poi2id)]), name="textual_b2")

        output = tf.nn.dropout(output, dropout_rate)
        output = tf.nn.xw_plus_b(output, textual_w2, textual_b2)

        _, rank = tf.nn.top_k(output, k=len(poi2id))

        # ==============================
        # Compute Loss
        # ==============================

        l2_loss += tf.nn.l2_loss(textual_w)
        l2_loss += tf.nn.l2_loss(textual_b)

        # Add dropout
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=corrected_poi)
        loss = tf.reduce_mean(losses) + FLAGS.l2_reg_lambda * l2_loss

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        def train_step(post_batch, user_batch, content_batch, poi_batch, month_batch, week_batch, time_batch):
            feed_dict = {input_image:post_batch, input_user:user_batch, input_content:content_batch, input_poi:poi_batch, input_month:month_batch, input_week:week_batch, input_time:time_batch, dropout_rate:FLAGS.dropout_rate}
            _loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            return _loss

        def validation_step(post_batch, user_batch, content_batch, poi_batch, month_batch, week_batch, time_batch, _dropout_rate):
            feed_dict = {input_image:post_batch, input_user:user_batch, input_content:content_batch, input_poi:poi_batch, input_month:month_batch, input_week:week_batch, input_time:time_batch, dropout_rate:_dropout_rate}
            _loss, _prediction = sess.run([loss,rank], feed_dict=feed_dict)
            return _prediction

        def get_score(data, _dropout_rate, current_epoch, is_test=False):
            accuracy_1 = 0.0
            accuracy_2 = 0.0
            accuracy_3 = 0.0
            mrr = 0.0
            step = 0
            batches = utils.batch_iter(data, FLAGS.batch_size)
            num_batches = int(len(data)/FLAGS.batch_size) + 1
            for batch in batches:
                post_batch, user_batch, content_batch, poi_batch, month_batch, week_batch, time_batch = batch
                post_batch = [_visual_feature[j] for j in post_batch]
                batch_prediction = validation_step(post_batch, user_batch, content_batch, poi_batch, month_batch, week_batch, time_batch, _dropout_rate)
                for i in range(len(batch_prediction)):
                    result = batch_prediction[i]
                    y = poi_batch[i]

                    # calculate accuracy
                    if y == result[0]:
                        accuracy_1 += 1
                        accuracy_2 += 1
                        accuracy_3 += 1
                    if y == result[2]:
                        accuracy_2 += 1
                        accuracy_3 += 1
                    if y == result[3]:
                        accuracy_3 += 1

                    # calculate mrr
                    rank = float(np.where(result==y)[0][0]+1)
                    mrr += (1./rank)

                step+=1
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
                print("Evaluating: [{}/{}]".format(step, num_batches))
            
            mrr /= len(data)
            accuracy_1 /= len(data)
            accuracy_2 /= len(data)
            accuracy_3 /= len(data)

            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print("Epoch #{} [MRR/Acc@1/Acc@2/Acc@3]: [{}/{}/{}/{}]".format(current_epoch,mrr, accuracy_1, accuracy_2, accuracy_3))

            if is_test:
                if mrr > best[1]:
                    best[0] = current_epoch
                    best[1] = mrr
                    best[2] = accuracy_1
                    best[3] = accuracy_2
                    best[4] = accuracy_3

                print("Best Epoch #{} [MRR/Acc@1/Acc@2/Acc@3]: [{}/{}/{}/{}]".format(best[0], best[1], best[2], best[3], best[4]))

        print("Training..\n")
        for i in range(FLAGS.num_epochs):
            # Training
            _loss = .0
            step = 0
            train_batches = utils.batch_iter(train,FLAGS.batch_size)
            num_batches = int(len(train)/FLAGS.batch_size) + 1
            for batch in train_batches:
                post_batch, user_batch, content_batch, poi_batch, month_batch, week_batch, time_batch = batch
                post_batch = [_visual_feature[j] for j in post_batch]
                _loss+=train_step(post_batch, user_batch, content_batch, poi_batch, month_batch, week_batch, time_batch)
                step+=1
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
                print("Process Context Layer Epoch: [{}/{}] Batch: [{}/{}]".format(i+1, FLAGS.num_epochs, step, num_batches))

            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print("Process Context Layer Epoch: [{}/{}] Loss: {}\n".format(i+1, FLAGS.num_epochs, _loss))

            if (i+1)%10 == 0:
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
                # Evaluation
                print("Evaluation at epoch #{:d}...".format(i+1))
                get_score(validation, FLAGS.dropout_rate, i+1)

                # Testing
                print("Testing at epoch #{:d}...".format(i+1))
                get_score(test, 1.0, i+1, is_test=True)
                print("======================================\n")
