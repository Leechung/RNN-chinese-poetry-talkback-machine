import tensorflow as tf

'''
author: lee chung
description: this is a project for computer linguitics course.
           i used cells from https://github.com/NickShahML/tensorflow_with_latest_papers, many thanks to those hardcore coders
'''

tf.app.flags.DEFINE_string('data_set', '../data/dat2' ,'data used to train chat bot.')
tf.app.flags.DEFINE_string('vocab','../data/vocab','save vocab to this file.')
tf.app.flags.DEFINE_string('model','../data/nn_model','save model to this directory.')
tf.app.flags.DEFINE_string('results','../data/results','save results to this directory.')


tf.app.flags.DEFINE_float('learning_rate',0.5,'learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay',0.99,'learning rate decay factor.')
tf.app.flags.DEFINE_float('max_gradient_norm',5.0,'clip gradient by this norm.')
tf.app.flags.DEFINE_integer('batch_size',128,'batch size for training.')

tf.app.flags.DEFINE_integer('vocab_size',10000,'vocabulary size.')
tf.app.flags.DEFINE_integer('layer_size',128,'size of each model layer.')
tf.app.flags.DEFINE_integer('num_layers',2,'number of layers in the model.')

tf.app.flags.DEFINE_integer('max_train_data_size',0,'limit on the size of training data (0, no limit).')
tf.app.flags.DEFINE_integer('steps_per_checkpoint',100,'training steps per checkpoint.')

tf.app.flags.DEFINE_string('cell_unit','lstm','rnn cell type, options: lstm, lstm with attention; hwrnn, highway rnn network with numliplicative intergration; bgc, basic gated cell; mgu, minimal gated unit; lstm_memarr, recurrent memory array structure; JZS1, JZS2, JZS3, http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf ')

FLAGS = tf.app.flags.FLAGS

BUCKETS = [(2, 3), (5,7), (7, 9), (8, 10)]
