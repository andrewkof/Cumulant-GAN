"""
@author: andrewkof

Cumulant GAN class.
"""
import time
import tensorflow as tf         # 2.0 or higher version
from utils import *

class Cumulant_GAN_dense(tf.keras.Model):

    def __init__(self, name, data_name, beta, gamma, epochs, BATCH_SIZE):
        super(Cumulant_GAN_dense, self).__init__()
        self.divergence_name = name
        self.data_name = data_name
        self.beta = beta
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = BATCH_SIZE

        self.X_dim = 2
        self.Z_dim = 8
        self.disc_iters = 5

        self.gen_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.disc_optimizer = tf.keras.optimizers.Adam(2e-3)

        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units = 32, activation='relu'),
                tf.keras.layers.Dense(units = 32, activation='relu'),
                tf.keras.layers.Dense(units = self.X_dim, activation='linear')
            ])
        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units = 32, activation='relu'),
                tf.keras.layers.Dense(units = 32, activation='relu'),
                tf.keras.layers.Dense(units = self.X_dim, activation='linear')
            ])

    def __repr__(self):
        return 'generator: {}, discriminator: {}'.format(self.generator, self.discriminator)

    def sample(self, Z_dim, eps=None):
        if eps is None:
            z = tf.random.normal(shape=(self.batch_size, Z_dim), mean=0.0, stddev=1.0)
            return z

    def generate(self):
        z = self.sample(self.Z_dim)
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def discriminator_loss(self,x):
        x_hat = self.generate()
        D_real = self.discriminate(x)
        D_fake = self.discriminate(x_hat)

        if self.beta == 0:
            D_loss_real = tf.reduce_mean(D_real)
        else:
            max_val = tf.reduce_max((-self.beta) * D_real)
            D_loss_real = -(1.0 / self.beta) * (tf.math.log(tf.reduce_mean(tf.math.exp((-self.beta) * D_real - max_val))) + max_val)
        if self.gamma == 0:
            D_loss_fake = tf.reduce_mean(D_fake)
        else:
            max_val = tf.reduce_max((self.gamma) * D_fake)
            D_loss_fake = (1.0 / self.gamma) * (tf.math.log(tf.reduce_mean(tf.math.exp(self.gamma * D_fake - max_val))) + max_val)

        D_loss = D_loss_real - D_loss_fake
        return D_loss

    def generator_loss(self):
        x_hat = self.generate()
        D_fake = self.discriminate(x_hat)
        if self.gamma == 0:
            G_loss = -tf.reduce_mean(D_fake)
        else:
            max_val = tf.reduce_max((self.gamma) * D_fake)
            G_loss = - (1.0 / self.gamma) * (tf.math.log(tf.reduce_mean(tf.math.exp(self.gamma * D_fake - max_val))) + max_val)

        return G_loss

    def train_step(self, x):
        # discriminator's parameters update
        for i in range(self.disc_iters):
            with tf.GradientTape() as disc_tape:
                disc_loss = -self.discriminator_loss(x) # we maximize the discrimination loss

            gradients_of_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_disc, self.discriminator.trainable_variables))

        # generator's parameters update
        with tf.GradientTape() as gen_tape:
            gen_loss = self.generator_loss()

        gradients_of_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_gen, self.generator.trainable_variables))


    def train(self, dataset):
        partitionings = partition(self.data_name)
        for epoch in range(self.epochs+1):
            start = time.time()

            for batch in dataset:
                self.train_step(batch)

            if epoch in partitionings:
                generate_and_save_plots(self.divergence_name, self.data_name, self.beta, self.gamma, self.generate(), epoch)

            # print time cost per epoch
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        print("""========================================== """)
        print('Finished training the model!!')
        print("""========================================== """)