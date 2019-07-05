import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from preprocess import *
from config import *
from model import *

def main(_):
    # GPU type setting
    gpu_config = tf.ConfigProto(device_count={'GPU' : 1})
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 1

    # -------------------- Import data ---------- #
    X, Y, en_vocab, de_vocab, zip_file = preprocess()

    # -------------------- Building Training Graph -------------------- #
    with tf.Graph().as_default(), tf.Session(config=gpu_config) as sess:
        tf.set_random_seed(1170)
        np.random.seed(seed=1170)

        initializer = tf.random_normal_initializer()

        # -------------------- Make training graph -------------------- #
        with tf.variable_scope("Model", initializer=initializer):
            tr_model = model_graph(source=de_vocab, target=en_vocab)
            _, tr_loss, tr_mean_loss, tr_accuracy = tr_model.loss_fn()
            tr_global_step, _, tr_train_op = tr_model.train_fn()

        # -------------------- Save Model -------------------- #
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./train_dir', graph=sess.graph)
        print ('Training start with initialized variable')

        # -------------------- Training Start -------------------- #
        for epoch_time in range(FLAGS.epoch):
            train_loss = 0.0
            start = time.time()

            for idx, (x, y) in enumerate(zip_file):
                # index 0 ~ 4109
                input_ = {tr_model.enc_inputs : x, tr_model.dec_inputs : y}
                loss, mean_loss, global_step, train_op, accuracy = sess.run([tr_loss, tr_mean_loss, \
                                                                            tr_global_step, tr_train_op, \
                                                                            tr_accuracy], input_)

                idx += 1
                train_loss += mean_loss
                if (idx+1) % FLAGS.verbose == 0:
                    print ('epoch : %d, step : %d, mean_loss : %.3f, time : %.2f, accuracy : %.3f' % (epoch_time, global_step, \
                            mean_loss, time.time() - start, accuracy))

            print ('one epoch done, spend time :', time.time() - start)
            saver.save(sess, '%s/epoch%d_%.3f.model' % ('./train_dir', epoch_time, train_loss / X.shape[0]))
            print ('Successfully saved model')
            summary = tf.Summary(value=[tf.Summary.Value(tag="Training_loss", simple_value=train_loss / X.shape[0])])
            summary_writer.add_summary(summary, global_step)


if __name__ == "__main__":
    tf.app.run()
