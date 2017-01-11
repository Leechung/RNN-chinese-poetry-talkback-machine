import sys
sys.path.append('cells')
import os
import math
import time
import random

from six.moves import xrange

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile

from config import FLAGS, BUCKETS

import data_util

import rnn_cell_modern as mcell

'''
author: lee chung
description: this is a project for computer linguitics course.
           i used cells from https://github.com/NickShahML/tensorflow_with_latest_papers, many thanks to those hardcore coders
'''



class Seq2SeqModel(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
        num_layers, max_gradient_norm, batch_size, learning_rate,
        learning_rate_decay_factor, cell = 'lstm_att',
        num_samples=512, forward_only=False):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)


        output_projection = None
        softmax_loss_function = None

        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable("proj_w", [size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,self.target_vocab_size)
            softmax_loss_function = sampled_loss



        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if cell is 'lstm':
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        elif cell is 'hwrnn':
            single_cell = mcell.HighwayRNNCell(size)
        elif cell is 'bgc':
            single_cell = mcell.BasicGatedCell(size)
        elif cell is 'mgu':
            single_cell = mcell.MGUCell(size)
        elif cell is 'lstm_memarr':
            single_cell = mcell.LSTMCell_MemoryArray(size)
        elif cell is 'JZS1':
            single_cell = mcell.JZS1Cell(size)
        elif cell is 'JZS2':
            single_cell = mcell.JZS2Cell(size)
        elif cell is  'JZS3Cell':
            single_cell = mcell.JZS3Cell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)


        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode)


        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, targets,self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs[b]]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):


        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))


        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updates[bucket_id],
                     self.gradient_norms[bucket_id],
                     self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
        for l in xrange(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None
        else:
            return None, outputs[0], outputs[1:]

    def get_batch(self, data, bucket_id):


        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            encoder_pad = [data_util.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_util.GO_ID] + decoder_input + [data_util.PAD_ID] * decoder_pad_size)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):

                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_util.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights






def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    model = Seq2SeqModel(
        source_vocab_size=FLAGS.vocab_size,
        target_vocab_size=FLAGS.vocab_size,
        buckets=BUCKETS,
        size=FLAGS.layer_size,
        num_layers=FLAGS.num_layers,
        max_gradient_norm=FLAGS.max_gradient_norm,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay_factor=FLAGS.learning_rate_decay,
        cell = FLAGS.cell_unit,
        forward_only=forward_only)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
    return model


def get_predicted_sentence(input_sentence, vocab, rev_vocab, model, sess):
    input_token_ids = data_util.sentence2tokenids(input_sentence, vocab)

    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    outputs = []

    feed_data = {bucket_id: [(input_token_ids, outputs)]}

    encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)


    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

    outputs = []

    for logit in output_logits:
        selected_token_id = int(np.argmax(logit, axis=1))

        if selected_token_id == data_util.EOS_ID:
            break
        else:
            outputs.append(selected_token_id)

    output_sentence = ' '.join([rev_vocab[output] for output in outputs])

    return output_sentence


def main(_):
    print('creating vocabulary from %s and storing in %s' % (FLAGS.data_set, FLAGS.vocab))
    data_util.create_vocab(FLAGS.data_set, FLAGS.vocab)
    vocab, vocab_re = data_util.init_vocab(FLAGS.vocab)


    with tf.Session() as sess:
        # print 'run sess'
        model = create_model(sess, forward_only=False)
        train_data = data_util.get_train_data(FLAGS.data_set, vocab)
        train_buckets_sizes = [len(train_data[b]) for b in xrange(len(BUCKETS))]
        train_total_size = float(sum(train_buckets_sizes))

        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size for i in
                               xrange(len(train_buckets_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            start_time = time.time()

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_data, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                         forward_only=False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')

                print('step')
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)

                previous_losses.append(loss)

                checkpoint_path = os.path.join(FLAGS.model, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                sys.stdout.flush()


if __name__ == '__main__':
   tf.app.run()
