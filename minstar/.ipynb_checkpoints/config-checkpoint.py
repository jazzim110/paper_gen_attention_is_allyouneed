import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('data_path', './dataset/de-en/', 'IWSLT16 TED training data with preprocessed')

flags.DEFINE_integer('sentence_maxlen', 20, 'Max length of the sentence')
flags.DEFINE_integer('batch_size', 32, 'batch size of the data')
flags.DEFINE_integer('stack_layer', 6, 'a number of encoder and decoder identical layers, denoted as N')
flags.DEFINE_integer('multi_head', 8, 'a number of parallel attention layers, denoted as h')
flags.DEFINE_integer('key_dim', 64, 'reduced dimension of each head, denoted as d_k')
flags.DEFINE_integer('value_dim', 64, 'reduced dimension of each head, denoted as d_v')
flags.DEFINE_integer('inner_layer', 2048, 'Position-Wise Feed-Forward Networks inner-layer dimensionality, denoted as d_ff')
flags.DEFINE_integer('model_dim', 512, 'To faciliate residual connections, output of dimension is 512, denoted as d_model')
flags.DEFINE_integer('warmup_steps', 4000, 'First Warmup_step training steps')
flags.DEFINE_integer('epoch', 20, 'a number of training epoch')
flags.DEFINE_integer('verbose', 100, 'a number of how many times to print loss')

flags.DEFINE_float('dropout', 0.1, 'After softmax layer, use dropout when needed')
flags.DEFINE_float('beta1', 0.9, 'Adam optimizer beta 1 number')
flags.DEFINE_float('beta2', 0.98, 'Adam optimizer beta 2 number')
flags.DEFINE_float('adam_epsilon', 1e-8, 'Adam optimizer epsilon number')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate of adam optimizer')
