import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
import math
import numpy as np
import os, pathlib, random
from contextlib import redirect_stdout

class MAMLNets(Model):

    def __init__(self, shot, xdim, ydim, zdim,
            filter1, filter2, update, training=True):
        super(MAMLNets, self).__init__()

        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.support_samples = shot
        self.filter1, self.filter2 = filter1, filter2
        self.update = update
        self.training = training

        self.initializer = tf.initializers.GlorotUniform()
        self.inner_optimizer = tf.keras.optimizers.Adam()
        self.loss_func = keras.losses.MeanSquaredError()

        # input layer - 2 Convolutional layers [Conv-batchnormalization-activation] - output layer
        inputs = keras.Input(shape=(self.xdim,self.ydim,self.zdim))
       
        conv1 = Conv2D(strides=(2,2),filters=self.filter1, kernel_size=3, padding='same',
                kernel_initializer=self.initializer)(inputs)
        batch1 = BatchNormalization()(conv1)
        act1 = Activation('tanh')(batch1)

        conv2 = Conv2D(filters=self.filter2, strides=(2,2), kernel_size=3, padding='same',
                kernel_initializer=self.initializer)(act1)
        batch2 = BatchNormalization()(conv2)
        act2 = Activation('tanh')(batch2)
        dropout = Dropout(0.2)(act2, training=self.training)
        flatten = Flatten()(dropout)
        outputs = Dense(1)(flatten)

        self.forward = tf.keras.Model(inputs=inputs, outputs=outputs, trainable=self.training)

    def call(self, inp_support, lab_support, inp_query, lab_query):

        inp_support = tf.reshape(inp_support,
                [self.support_samples, self.xdim, self.ydim, self.zdim])
        lab_support = tf.reshape(lab_support,[self.support_samples])
        inp_query = tf.reshape(inp_query,[-1, self.xdim, self.ydim, self.zdim])
        lab_query = tf.reshape(lab_query,[-1])

        em_loss, em_pred = [], []

        # Support data prediction
        with tf.GradientTape() as train_tape:
            pred = self.forward(inp_support)
            inner_loss = self.loss_func(lab_support, pred)

        # Parameters update for support data
        inner_gradients = train_tape.gradient(inner_loss, self.forward.trainable_variables)
        self.inner_optimizer.apply_gradients(zip(inner_gradients, self.forward.trainable_variables))

        # Query data prediction          
        query_pred = self.forward(inp_query)
        outer_loss = self.loss_func(query_pred, lab_query)

        # Query loss record
        em_loss.append(outer_loss)
        losses = tf.stack(em_loss, name='query loss')
        em_pred.append(query_pred)
        preds = tf.stack(em_pred, name='query predictions')

        for _ in range(self.update-1):

            # Support data prediction
            with tf.GradientTape() as train_tape:
                pred = self.forward(inp_support)
                inner_loss = self.loss_func(lab_support, pred)

            # Parameters update for support data
            inner_gradients = train_tape.gradient(inner_loss, self.forward.trainable_variables)
            self.inner_optimizer.apply_gradients(zip(inner_gradients, self.forward.trainable_variables))

            # Query data prediction
            query_pred = self.forward(inp_query)
            outer_loss = self.loss_func(query_pred, lab_query)

            # Query loss record
            em_loss.append(outer_loss)
            losses = tf.stack(em_loss, name='query loss')
            em_pred.append(query_pred)
            preds = tf.stack(em_pred, name='query predictions')

        # loss for outer loop
        loss = tf.reduce_mean(losses[-1])
        predictions = tf.squeeze(preds[-1])

        return predictions, loss
                
    def load(self, dir_path):           
        encoder_path = os.path.join(dir_path, 'cnn_encoder.h5')
        self.forward(tf.zeros([1, self.xdim, self.ydim, self.zdim]))
        self.forward.load_weights(encoder_path)

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.forward.save_weights(os.path.join(dir_path, 'cnn_encoder.h5'))
        with open(dir_path+'model_summary.md','w') as f:
            with redirect_stdout(f):
                self.forward.summary()

