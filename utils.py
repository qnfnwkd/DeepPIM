import numpy as np
import random
import os
from gensim.models import Word2Vec


def load_data(train_file, validation_file, test_file):
    # file load    
    train_file = [line.strip() for line in open(train_file,"r").readlines()]
    validation_file = [line.strip() for line in open(validation_file,"r").readlines()]
    test_file = [line.strip() for line in open(test_file,"r").readlines()]

    # ==============================
    # data format
    # post_id \t user_id \t content \t poi \t time
    # ==============================

    train, validation, test = [], [], []

    for line in train_file:
        post, user, content, poi, month, week, time = line.split("\t")
        train.append((post, user, content, poi, int(month)-1, int(week), int(time)))

    for line in validation_file:
        post, user, content, poi, month, week, time = line.split("\t")
        validation.append((post, user, content, poi, int(month)-1, int(week), int(time)))

    for line in test_file:
        post, user, content, poi, month, week, time = line.split("\t")
        test.append((post, user, content, poi, int(month)-1, int(week), int(time)))

    return train, validation, test

def build_dic(train, validation, test):

    # for zero padding
    word2id = {"<ZERO>":0}
    id2word = ["<ZERO>"]

    user2id = {}
    id2user = []

    post2id = {}
    id2post = []

    poi2id = {}
    id2poi = []

    for line in train+validation+test:
        if line[0] not in post2id:
            post2id[line[0]] = len(post2id)

        if line[1] not in user2id:
            user2id[line[1]] = len(user2id)
            id2user.append(line[1])

        if line[3] not in poi2id:
            poi2id[line[3]] = len(poi2id)
            id2poi.append(line[3])

        for word in line[2].split():
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word.append(word)
    post_list = post2id.keys()
    post_list.sort()
    post2id = {post_list[i]:i for i in range(len(post_list))}
    id2post = post_list


    return word2id, id2word, user2id, id2user, poi2id, id2poi, post2id, id2post

def converting(_train, _validation, _test, word2id, user2id, poi2id, post2id):
    train, validation, test = [], [], []

    # Calucate Maximum Lenth of Text Content for Zero Padding
    maximum_document_length = max([len(line[2].split()) for line in _train+_validation+_test])

    for line in _train:
        words = [word2id[word] for word in line[2].split()]
        words.extend([0]*(maximum_document_length - len(words)))
        train.append((post2id[line[0]], user2id[line[1]], words, poi2id[line[3]], line[4], line[5], line[6]))

    for line in _validation:
        words = [word2id[word] for word in line[2].split()]
        words.extend([0]*(maximum_document_length - len(words)))
        validation.append((post2id[line[0]], user2id[line[1]], words, poi2id[line[3]], line[4], line[5], line[6]))

    for line in _test:
        words = [word2id[word] for word in line[2].split()]
        words.extend([0]*(maximum_document_length - len(words)))
        test.append((post2id[line[0]], user2id[line[1]], words, poi2id[line[3]], line[4], line[5], line[6]))

    return train, validation, test, maximum_document_length

def load_embedding(embedding_file, word2id, embedding_dim):
    model = Word2Vec.load(embedding_file)
    word_embedding = np.ndarray((len(word2id), embedding_dim), dtype=np.float32)
    for word, idx in word2id.items():
        if word in model:
            word_embedding[idx] = np.asarray(model[word])
        else:
            word_embedding[idx] = np.random.uniform(-1, 1, embedding_dim)

    word_embedding[0] = np.zeros(embedding_dim, dtype=np.float32)
    return word_embedding


def batch_iter(data, batch_size):
    num_batches = int(len(data)/batch_size) + 1
    random.shuffle(data)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(data))
        post, user, content, poi, month, week, time = [], [], [] ,[] ,[], [], []
        for line in data[start_index:end_index]:
            post.append(line[0])
            user.append(line[1])
            content.append(line[2])
            poi.append(line[3])
            month.append(line[4])
            week.append(line[5])
            time.append(line[6])

        yield post, user, content, poi, month, week, time

def validation_batch_iter(data, batch_size):
    data = np.array(data)
    data_size = len(data)
    num_batches = int(len(data)/batch_size) + 1
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
