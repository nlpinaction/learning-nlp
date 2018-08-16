# -*- coding:utf-8 -*-
import numpy as np
import time
import sys
import os
import re
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from dynamic_seq2seq_model import dynamicSeq2seq
import jieba


class Seq2seq():
    '''
        params:
        encoder_vec_file    encoder向量文件
        decoder_vec_file    decoder向量文件
        encoder_vocabulary  encoder词典
        decoder_vocabulary  decoder词典
        model_path          模型目录
        batch_size          批处理数
        sample_num          总样本数
        max_batches         最大迭代次数
        show_epoch          保存模型步长
    '''

    def __init__(self):
        tf.reset_default_graph()

        self.encoder_vec_file = "./tfdata/enc.vec"
        self.decoder_vec_file = "./tfdata/dec.vec"
        self.encoder_vocabulary = "./tfdata/enc.vocab"
        self.decoder_vocabulary = "./tfdata/dec.vocab"
        self.batch_size = 1
        self.max_batches = 100000
        self.show_epoch = 100
        self.model_path = './model/'

        self.model = dynamicSeq2seq(encoder_cell=LSTMCell(40),
                                    decoder_cell=LSTMCell(40),
                                    encoder_vocab_size=600,
                                    decoder_vocab_size=1600,
                                    embedding_size=20,
                                    attention=False,
                                    bidirectional=False,
                                    debug=False,
                                    time_major=True)
        self.location = ["杭州", "重庆", "上海", "北京"]
        self.dec_vocab = {}
        self.enc_vocab = {}
        self.dec_vecToSeg = {}
        tag_location = ''
        with open(self.encoder_vocabulary, "r") as enc_vocab_file:
            for index, word in enumerate(enc_vocab_file.readlines()):
                self.enc_vocab[word.strip()] = index
        with open(self.decoder_vocabulary, "r") as dec_vocab_file:
            for index, word in enumerate(dec_vocab_file.readlines()):
                self.dec_vecToSeg[index] = word.strip()
                self.dec_vocab[word.strip()] = index

    def data_set(self, file):
        _ids = []
        with open(file, "r") as fw:
            line = fw.readline()
            while line:
                sequence = [int(i) for i in line.split()]
                _ids.append(sequence)
                line = fw.readline()
        return _ids

    def data_iter(self, train_src, train_targets, batches, sample_num):
        ''' 获取batch
            最大长度为每个batch中句子的最大长度
            并将数据作转换:
            [batch_size, time_steps] -> [time_steps, batch_size]

        '''
        batch_inputs = []
        batch_targets = []
        batch_inputs_length = []
        batch_targets_length = []

        # 随机样本
        shuffle = np.random.randint(0, sample_num, batches)
        en_max_seq_length = max([len(train_src[i]) for i in shuffle])
        de_max_seq_length = max([len(train_targets[i]) for i in shuffle])

        for index in shuffle:
            _en = train_src[index]
            inputs_batch_major = np.zeros(
                shape=[en_max_seq_length], dtype=np.int32)  # == PAD
            for seq in range(len(_en)):
                inputs_batch_major[seq] = _en[seq]
            batch_inputs.append(inputs_batch_major)
            batch_inputs_length.append(len(_en))

            _de = train_targets[index]
            inputs_batch_major = np.zeros(
                shape=[de_max_seq_length], dtype=np.int32)  # == PAD
            for seq in range(len(_de)):
                inputs_batch_major[seq] = _de[seq]
            batch_targets.append(inputs_batch_major)
            batch_targets_length.append(len(_de))

        batch_inputs = np.array(batch_inputs).swapaxes(0, 1)
        batch_targets = np.array(batch_targets).swapaxes(0, 1)

        return {self.model.encoder_inputs: batch_inputs,
                self.model.encoder_inputs_length: batch_inputs_length,
                self.model.decoder_targets: batch_targets,
                self.model.decoder_targets_length: batch_targets_length, }

    def train(self):
        # 获取输入输出
        train_src = self.data_set(self.encoder_vec_file)
        train_targets = self.data_set(self.decoder_vec_file)

        f = open(self.encoder_vec_file)
        self.sample_num = len(f.readlines())
        f.close()
        print("样本数量%s" % self.sample_num)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # 初始化变量
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            loss_track = []
            total_time = 0
            for batch in range(self.max_batches + 1):
                # 获取fd [time_steps, batch_size]
                start = time.time()
                fd = self.data_iter(train_src,
                                    train_targets,
                                    self.batch_size,
                                    self.sample_num)
                _, loss, _, _ = sess.run([self.model.train_op,
                                          self.model.loss,
                                          self.model.gradient_norms,
                                          self.model.updates], fd)

                stop = time.time()
                total_time += (stop - start)

                loss_track.append(loss)
                if batch == 0 or batch % self.show_epoch == 0:

                    print("-" * 50)
                    print("n_epoch {}".format(sess.run(self.model.global_step)))
                    print('  minibatch loss: {}'.format(
                        sess.run(self.model.loss, fd)))
                    print('  per-time: %s' % (total_time / self.show_epoch))
                    checkpoint_path = self.model_path + "nlp_chat.ckpt"
                    # 保存模型
                    self.model.saver.save(
                        sess, checkpoint_path, global_step=self.model.global_step)

                    # 清理模型
                    self.clearModel()
                    total_time = 0
                    for i, (e_in, dt_pred) in enumerate(zip(
                            fd[self.model.decoder_targets].T,
                            sess.run(self.model.decoder_prediction_train, fd).T
                    )):
                        print('  sample {}:'.format(i + 1))
                        print('    dec targets > {}'.format(e_in))
                        print('    dec predict > {}'.format(dt_pred))
                        if i >= 0:
                            break

    def add_to_file(self, strs, file):
        with open(file, "a") as f:
            f.write(strs + "\n")

    def add_voc(self, word, kind):
        if kind == 'enc':
            self.add_to_file(word, self.encoder_vocabulary)
            index = max(self.enc_vocab.values()) + 1
            self.enc_vocab[word] = index
        else:
            self.add_to_file(word, self.decoder_vocabulary)
            index = max(self.dec_vocab.values()) + 1
            self.dec_vocab[word] = index
            self.dec_vecToSeg[index] = word
        return index

    def onlinelearning(self, input_strs, target_strs):
        input_seg = jieba.lcut(input_strs)
        target_seg = jieba.lcut(target_strs)

        input_vec = []
        for word in input_seg:
            if word not in self.enc_vocab.keys():
                vec = self.add_voc(word, "enc")
            else:
                vec = self.enc_vocab.get(word)
            input_vec.append(vec)

        target_vec = []
        for word in target_seg:
            if word not in self.dec_vocab.keys():
                vec = self.add_voc(word, "dec")
            else:
                vec = self.dec_vocab.get(word)
            target_vec.append(vec)

        with tf.Session() as sess:
            # 初始化变量
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            fd = self.data_iter([input_vec], [target_vec], 1, 1)
            for i in range(100):
                _, loss, _, _ = sess.run([self.model.train_op,
                                          self.model.loss,
                                          self.model.gradient_norms,
                                          self.model.updates], fd)
                checkpoint_path = self.model_path + "nlp_chat.ckpt"
                # 保存模型
                self.model.saver.save(
                    sess, checkpoint_path, global_step=self.model.global_step)

                for i, (e_in, dt_pred) in enumerate(zip(
                        fd[self.model.decoder_targets].T,
                        sess.run(self.model.decoder_prediction_train, fd).T
                )):
                    print('    sample {}:'.format(i + 1))
                    print('    dec targets > {}'.format(e_in))
                    print('    dec predict > {}'.format(dt_pred))
                    if i >= 0:
                        break

    def segement(self, strs):
        return jieba.lcut(strs)

    def make_inference_fd(self, inputs_seq):
        sequence_lengths = [len(seq) for seq in inputs_seq]
        max_seq_length = max(sequence_lengths)

        inputs_time_major = []
        for sents in inputs_seq:
            inputs_batch_major = np.zeros(
                shape=[max_seq_length], dtype=np.int32)  # == PAD
            for index in range(len(sents)):
                inputs_batch_major[index] = sents[index]
            inputs_time_major.append(inputs_batch_major)

        inputs_time_major = np.array(inputs_time_major).swapaxes(0, 1)
        return {self.model.encoder_inputs: inputs_time_major,
                self.model.encoder_inputs_length: sequence_lengths}

    def predict(self):
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("没找到模型")

            action = False
            while True:
                if not action:
                    inputs_strs = input("me > ")
                if not inputs_strs:
                    continue

                inputs_strs = re.sub(
                    "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", inputs_strs)

                action = False
                segements = self.segement(inputs_strs)
                # inputs_vec = [enc_vocab.get(i) for i in segements]
                inputs_vec = []
                for i in segements:
                    inputs_vec.append(self.enc_vocab.get(i, self.model.UNK))
                fd = self.make_inference_fd([inputs_vec])
                inf_out = sess.run(self.model.decoder_prediction_inference, fd)
                inf_out = [i[0] for i in inf_out]

                outstrs = ''
                for vec in inf_out:
                    if vec == self.model.EOS:
                        break
                    outstrs += self.dec_vecToSeg.get(vec, self.model.UNK)
                print(outstrs)

    def clearModel(self, remain=3):
        try:
            filelists = os.listdir(self.model_path)
            re_batch = re.compile(r"nlp_chat.ckpt-(\d+).")
            batch = re.findall(re_batch, ",".join(filelists))
            batch = [int(i) for i in set(batch)]
            if remain == 0:
                for file in filelists:
                    if "nlp_chat" in file:
                        os.remove(self.model_path + file)
                os.remove(self.model_path + "checkpoint")
                return
            if len(batch) > remain:
                for bat in sorted(batch)[:-(remain)]:
                    for file in filelists:
                        if str(bat) in file and "nlp_chat" in file:
                            os.remove(self.model_path + file)
        except Exception as e:
            return

    def test(self):
        with tf.Session() as sess:

            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 获取输入输出
            train_src = [[2, 3, 5], [7, 8, 2, 4, 7], [9, 2, 1, 2]]
            train_targets = [[2, 3], [6, 4, 7], [7, 1, 2]]

            loss_track = []

            for batch in range(self.max_batches + 1):
                # 获取fd [time_steps, batch_size]
                fd = self.data_iter(train_src,
                                    train_targets,
                                    2,
                                    3)

                _, loss, _, _ = sess.run([self.model.train_op,
                                          self.model.loss,
                                          self.model.gradient_norms,
                                          self.model.updates], fd)
                loss_track.append(loss)

                if batch == 0 or batch % self.show_epoch == 0:
                    print("-" * 50)
                    print("epoch {}".format(sess.run(self.model.global_step)))
                    print('  minibatch loss: {}'.format(
                        sess.run(self.model.loss, fd)))

                    for i, (e_in, dt_pred) in enumerate(zip(
                            fd[self.model.decoder_targets].T,
                            sess.run(self.model.decoder_prediction_train, fd).T
                    )):
                        print('  sample {}:'.format(i + 1))
                        print('    dec targets > {}'.format(e_in))
                        print('    dec predict > {}'.format(dt_pred))
                        if i >= 3:
                            break


if __name__ == '__main__':
    seq_obj = Seq2seq()
    if sys.argv[1]:
        if sys.argv[1] == 'train':
            seq_obj.train()
        elif sys.argv[1] == 'infer':
            seq_obj.predict()
