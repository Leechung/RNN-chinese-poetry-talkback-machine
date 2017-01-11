import os
import sys
import tensorflow as tf

from config import FLAGS
import data_util

from train import create_model, get_predicted_sentence

'''
author: lee chung
description: this is a project for computer linguitics course.
           i used cells from https://github.com/NickShahML/tensorflow_with_latest_papers, many thanks to those hardcore coders
'''
def chat():
    with tf.Session() as sess:
        model = create_model(sess, forward_only = True)
        model.batch_size = 1
        vocab, rev_vocab = data_util.init_vocab(FLAGS.vocab)
        #print vocab
        #print rev_vocab
        sys.stdout.write('>')
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            predicted_sentence = get_predicted_sentence(sentence, vocab, rev_vocab, model, sess)
            print(predicted_sentence)
            print('>')
            sentence = sys.stdin.readline()



if __name__ == '__main__':
    chat()
