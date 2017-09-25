# -*- coding: utf-8 -*-
##################
# Document Maker #
################################################
# Desc   : A Nice RNN for Generating Documents #
# Author : Abe Hoffman                         #
# Date   : Aug 21 2017                         #
################################################

from __future__ import print_function
from collections import namedtuple
import string, textract
import tensorflow as tf
import numpy as np
import glob, time

###################
# I. Architecture #
###################
# A. Load Document (from the Corpus)
# B. Pre-process Document
# C. Encode the Document
# D. Define Batch Generator
# E. Define Model Inputs
# F. Define LSTM Cell
# G. Define Model Output
# H. Define Model Loss
# I. Define Model Optimizer
# J. The LSTM Network
###################

######################
# II. Implementation #
######################
# A. Define Hyperparameters
# B. Model Training and Checkpoints
# C. Sampling
######################

# I. Architecture
# A. Load the Document
# A1. Sanitized Patient Document
#printable = set(string.printable)
#adoc = textract.process('12345678.rtf')
#text = filter(lambda x: x in printable, adoc)
with open('anna.txt', 'r') as f:
    text=f.read()

# B. Pre-process Document
# B1. Set the Vocabulary : Unique set of characters that are found in the document
vocab = set(text)
# B2. Assign each unique character an integer (starting from 0) : {'<character>': 0, ... }
vocab_to_int = {c: i for i, c in enumerate(vocab)}
# B3. Flip assignment so integer indicates character : {<integer>: '<character>', ...}
int_to_vocab = dict(enumerate(vocab))

# C. Encode the Document
# C1. For every occurence of a character in the document, assign the integer
# (Use syntactic sugar and convert to a Numpy array)
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
# C2. This results in our entire document encoded as character : integers, based on the defined vocabulary
print('Encoded document shape: {}'.format(encoded.shape))
encoded[0:10]

# D. Define Batch Generator
# D1. A Generator Function that returns an Iterator for Batches
def get_batches(arr, n_seqs, n_steps):
    '''
    Desc : Generate batches of size n_seqs * n_steps
    Variables :
        - arr     : Input array
        - n_seqs  : Sequences per batch (batch size)
        - n_steps : Sequence steps per batch
    '''
    # Get the batch size and number of batches we can make
    batch_size = n_seqs * n_steps
    n_batches  = len(arr) // batch_size

    # Keep only enough characters to make full batches
    arr =  arr[:n_batches * batch_size]

    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs,-1))

    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:,n:n+n_steps]
        # The targets, shifted by one
        y = np.zeros(x.shape)
        y[:,:-1],y[:,-1] = x[:,1:] ,x[:,0]
        yield x, y

# E. Define Model Inputs
# E1. Shaping input placeholders and preparing for optimization
def build_inputs(batch_size, num_steps):
    '''
    Desc : Tensorflow Placeholders
    Variables :
        - batch_size : Number of sequences per batch
        - num_steps  : Sequence steps per batch
    '''
    # Graph placeholders
    inputs = tf.placeholder(tf.int32,[batch_size,num_steps],name="inputs")
    targets = tf.placeholder(tf.int32,[batch_size,num_steps],name="targets")

    # Retain probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32,name="keep_prob") # Scalar

    return inputs, targets, keep_prob

# F. Define LSTM Cell
# F1. Basic LSTM Cells
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    '''
    Desc : LSTM Cell for Hidden Layers
    Variables :
        - keep_prob  : Dropout optimization (scalar placeholder)
        - lstm_size  : Size of the hidden layers in the LSTM cells
        - num_layers : Number of LSTM layers
        - batch_size : Batch size
    '''

    # LSTM cell and dropout to the cell outputs
    # Stack LSTM layers, for vector ops (syntactic sugar)
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper\
                                       (tf.contrib.rnn.BasicLSTMCell(lstm_size)) \
                                        for _ in range(num_layers)])
    # Fill the initial state with Zeros
    initial_state = cell.zero_state(batch_size,tf.float32)

    return cell, initial_state

# G. Define Model Output
# G1. RNN Softmax Output Layer
def build_output(lstm_output, in_size, out_size):
    '''
    Desc : Softmax layer that returns softmax output and logits (logarithm of the odds p/(1 − p))
    Variables:
        - lstm_output : Output tensor list from LSTM layer
        - in_size     : Size of the input tensor, for example, size of the LSTM cells
        - out_size    : Size of this softmax layer
    '''
    # Reshape output: one row for each step for each sequence.
    # Concatenate lstm_output over axis 1 (the columns)
    seq_output = tf.concat(lstm_output,axis=1)
    # Reshape seq_output to 2D tensor with lstm_size columns
    x = tf.reshape(seq_output,[-1,in_size])

    # Connect RNN outputs to Softmax Layer
    with tf.variable_scope('softmax'):
        # Weight and Bias
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size),stddev=0.1))
        softmax_b = tf.Variable(tf.zeros([out_size]))

    # Outputs are RNN cell output rows, therefore logits are output rows (one for each step and sequence)
    logits =  tf.add(tf.matmul(x,softmax_w),softmax_b)

    # Softmax for predicted character probabilities
    out = tf.nn.softmax(logits,name ="out")
    print(out)
    return out, logits

# H. Define Model Loss
# H1. Discover mean to calculate loss
def build_loss(logits, targets, lstm_size, num_classes):
    '''
    Desc : Calculate loss from logits and targets
    Variables :
        - logits      : Logits from final fully connected layer
        - targets     : Targets for supervised learning
        - lstm_size   : Number of LSTM hidden units
        - num_classes : Number of classes in targets
    '''
    # One-hot encode targets and reshape to match logits, one row per sequence per step
    y_one_hot = tf.one_hot(targets,num_classes)
    y_reshaped =  tf.reshape(y_one_hot,logits.get_shape())

    # Softmax cross entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_reshaped))

    return loss

# I. Define Model Optimizer
# I1. Clip Gradients to Ensure non-exponential growth
def build_optimizer(loss, learning_rate, grad_clip):
    '''
    Desc : Build optmizer for training, using gradient clipping
    Variables :
        - loss: Network loss
        - learning_rate: Learning rate for optimizer
    '''

    # Optimizer for training (clipping the exploding gradients)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

# J. The LSTM Network
# J1. A class to define our model
class CharRNN:

    def __init__(self,
                 num_classes,
                 batch_size=64,
                 num_steps=50,
                 lstm_size=128,
                 num_layers=2,
                 learning_rate=0.001,
                 grad_clip=5,
                 sampling=False):

        # Sampling: Pass one character at a time
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size,num_steps)
        # Build the LSTM cell
        cell, self.initial_state = build_lstm(lstm_size,num_layers,batch_size,self.keep_prob)
        # (Run the data through the RNN layers)
        # One-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs,num_classes)

        # Run each sequence step through the RNN
        outputs, state = tf.nn.dynamic_rnn(cell,x_one_hot,initial_state=self.initial_state)
        self.final_state = state

        # Softmax predictions and logits
        self.prediction, self.logits = build_output(outputs,lstm_size,num_classes)

        # Loss and optimizer (with gradient clipping)
        self.loss =  build_loss(self.logits,self.targets,lstm_size,num_classes)
        self.optimizer = build_optimizer(self.loss,learning_rate,grad_clip)

# II. Implementation
# A. Define Hyperparameters
# A1. Larger Networks + Dropout Values (0.0 - 1.0)
batch_size = 10         # Sequences per batch
num_steps  = 150        # Sequence steps per batch
lstm_size  = 550        # LSTMs' Hidden layer sizes
num_layers = 5          # LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob  = 0.5        # Dropout 'keep_prob' probability

# B. Model Training and Checkpoints
# B1. Declare Epochs
epochs = 100
# B2. Savepoints every n iterations
save_every_n = 200

# B3. Try first with one GPU
#with tf.device("/gpu:0"):
#    model = CharRNN(len(vocab),
#                    batch_size=batch_size,
#                    num_steps=num_steps,
#                    lstm_size=lstm_size,
#                    num_layers=num_layers,
#                    learning_rate=learning_rate)
#
#saver = tf.train.Saver(max_to_keep=100)
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#
#    # Load a checkpoint and resume training: saver.restore(sess, 'checkpoints/______.ckpt')
#    counter = 0
#    for e in range(epochs):
#        # Train network
#        new_state = sess.run(model.initial_state)
#        loss = 0
#        for x, y in get_batches(encoded, batch_size, num_steps):
#            counter += 1
#            start = time.time()
#            feed = {model.inputs: x,
#                    model.targets: y,
#                    model.keep_prob: keep_prob,
#                    model.initial_state: new_state}
#            batch_loss, new_state, _ = sess.run([model.loss,
#                                                 model.final_state,
#                                                 model.optimizer],
#                                                 feed_dict=feed)
#
#            end = time.time()
#            print('Epoch: {}/{}... '.format(e+1, epochs),
#                  'Training Step: {}... '.format(counter),
#                  'Training loss: {:.4f}... '.format(batch_loss),
#                  '{:.4f} sec/batch'.format((end-start)))
#
#            if (counter % save_every_n == 0):
#                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
#
#    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

# C. Sampling
# C1. Next Char
def pick_top_n(preds, vocab_size, top_n=5):
    '''
    Desc : Feed in char, receive next char.
    Variables:
        - preds      : Char to predict
        - vocab_size : Length of the vocab
        - top_n      : Return the top selection 'n' of chars
    '''
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

# C2. Sample based on Checkpoint
def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    '''
    Desc : Resturn a sample from a checkpoint restore
    Variables:
        - checkpoint : ckpt file to restore
        - n_samples  : Number of samples to return
        - lstm_size  : LSTM size
        - vocab_size : Vocabulary size
        - prime      : Prime the results with a string
    '''
    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)

checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab))
print(samp)

