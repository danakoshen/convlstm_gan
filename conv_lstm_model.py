from __future__ import print_function

import matplotlib

matplotlib.use('Qt5Agg')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
t=4
GENERATE_RES = 4
GENERATE_SQUARE = 32 * GENERATE_RES

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=9)
parser.add_argument('--mode', type=str, default='test', help='train, test')
parser.add_argument('--n_epoch', type=int, default='10')
parser.add_argument('--train_continue', type=bool, default = False)
args = parser.parse_args()

### 0. prepare data

from keras.models import Model
from keras.layers import Input, Reshape, Dense, ConvLSTM2D, Flatten, Dropout, GaussianNoise, Conv2D
from keras.layers import Conv3DTranspose, LeakyReLU, ReLU, TimeDistributed, UpSampling2D, UpSampling3D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from tqdm import tqdm
import math

from keras.utils.generic_utils import Progbar

global_disc = None
is_disc_loaded = False


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

### combine images for visualization
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img[:, :, :]
    return image


### generator model define
def generator_model():

    inputs  = Input((100,))
    fc1 = Dense(input_dim = 100, units = 32 * t * int(GENERATE_SQUARE/4) * int(GENERATE_SQUARE/4))(inputs)
    fc2 = Reshape((t, int(GENERATE_SQUARE/4), int(GENERATE_SQUARE/4), 32), input_shape=(32 * t * int(GENERATE_SQUARE/4) * int(GENERATE_SQUARE/4),))(fc1)
    up1 = TimeDistributed(Conv2D(128, (5,5), strides = (2,2), padding='same'))(fc2)
    conv1 = BatchNormalization(momentum=0.8)(up1)
    conv1 = ReLU()(conv1)
    conv1 = ConvLSTM2D(128, (3, 3), strides=(1, 1), padding='same', return_sequences=True)(conv1)
    conv2 = ConvLSTM2D(64, (3,3), strides=(1,1), padding='same', return_sequences=True)(conv1) #3_kernel
    conv3 = ConvLSTM2D(48, (3,3), strides=(1,1), padding='same', return_sequences=True)(conv2) #5_kernel
    conv3 = TimeDistributed(Conv2DTranspose(128, (5,5), strides = (2,2), padding='same'))(conv3)

    conv3 = BatchNormalization(momentum=0.8)(conv3)
    conv3 = ReLU()(conv3)
    conv3 = TimeDistributed(Conv2DTranspose(1, (11, 11), strides=(4, 4), padding='same'))(conv3)
    outputs = Activation('tanh')(conv3)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model



### discriminator model define
def discriminator_model():

    inputs = Input((t, GENERATE_SQUARE, GENERATE_SQUARE, 1))
    conv1 = ConvLSTM2D(128, (3,3), strides=(2,2), padding='same', return_sequences=True)(inputs) #16
    conv1 = Dropout(0.25)(conv1)
    conv2 = Dropout(0.25)(conv2)
    conv3 = ConvLSTM2D(48, (3,3), strides=(1,1), padding='same', return_sequences=True)(conv2) #128
    conv3 = BatchNormalization(momentum=0.8)(conv3)
    conv3 = ReLU()(conv3)
    up1 = TimeDistributed(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))(conv3)
    fc1 = Flatten()(up1)
    outputs = Dense(1)(fc1)

    model = Model(inputs = [inputs], outputs = [outputs])
    return model


### d_on_g model for training generator
def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(100,))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    return gan


def load_model(class_n):
    d = discriminator_model()
    g = generator_model()
    d_optim = RMSprop()
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    d.load_weights('./weights/' + class_n + '/discriminator.h5')
    g.load_weights('./weights/' + class_n + '/generator.h5')
    return g, d


d = None
global_int_model_feat_extract = None
is_feature_extracted = False



z = None
intermidiate_model = None
d_x = None
loss = None
similar_data = None

### discriminator intermediate layer feautre extraction
def feature_extractor():
    global global_disc, is_disc_loaded, d, global_int_model_feat_extract, is_feature_extracted
    d = discriminator_model()
    if is_disc_loaded:
        d = global_disc
    else:
        d.load_weights('weights/discriminator.h5')
        is_disc_loaded = True
        global_disc = d
        print("Discriminator loaded from " + 'weights/discriminator.h5')

    if is_feature_extracted:
        return global_int_model_feat_extract
    else:
        intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-9].output)
        intermidiate_model.compile(loss=wasserstein_loss, optimizer='rmsprop')
        is_feature_extracted = True
        global_int_model_feat_extract = intermidiate_model
    return intermidiate_model


def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.normal(0, 1, size=(n_samples, latent_dim))
    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    return X


def summarize_performance(step, g_model, latent_dim, n_samples=4):
    X = generate_fake_samples(g_model, latent_dim, n_samples)
    X = (X + 1) / 2.0
    # plot images
    for i in range(2 * 2):
        plt.subplot(2, 2, 1 + i)
        plt.axis('off')
        plt.imshow(X[1, i, :, :, 0], cmap='gray_r')
    plt.savefig('results_baseline/generated_plot_%03d.png' % (step + 1))
    plt.close()


def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist, a):
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='d-real')
    plt.plot(d2_hist, label='d-fake')
    plt.plot(g_hist, label='gen')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(a1_hist, label='acc-real')
    plt.plot(a2_hist, label='acc-fake')
    plt.legend()
    plt.savefig('/results_baseline/plot_line_plot_loss_%03d.png' % a)
    plt.close()




### train generator and discriminator
def train(BATCH_SIZE, X_train, n_epoch):
    ### model define
    d = discriminator_model()
    print("#### discriminator ######")
    d.summary()
    g = generator_model()
    print("#### generator ######")
    g.summary()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = RMSprop(lr=0.00005) #0.0003
    g_optim = RMSprop(lr=0.00005) #0.0002
    d_on_g.compile(loss=wasserstein_loss, optimizer=g_optim)
    d.trainable = True
    d.compile(loss=wasserstein_loss, optimizer=d_optim)
    clip_value = 0.01
    n_critic = 5
    valid = -np.ones((BATCH_SIZE, 1))
    fake = np.ones((BATCH_SIZE, 1))
    if args.train_continue:
        d.load_weights('weights/discriminator.h5')
        g.load_weights('weights/generator.h5')
    for epoch in tqdm(range(int(n_epoch))):
        print("Epoch is", epoch)
        n_iter = int(X_train.shape[0] / BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)

        for index in range(n_iter):
            for _ in range(n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
                imgs = X_train[idx]



                # Sample noise as generator input
                noise = np.random.normal(0, 1, (BATCH_SIZE, 100))

                # Generate a batch of new images
                gen_imgs = g.predict(noise, verbose=0)

                # Train the critic
                d_loss_real = d.train_on_batch(imgs, valid)
                d_loss_fake = d.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in d.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                    l.set_weights(weights)

            # training generator
            d.trainable = False
            #g_loss = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            g_loss = d_on_g.train_on_batch(noise, valid)
            d.trainable = True

            progress_bar.update(index, values=[('g', g_loss), ('d', d_loss)])
        print('')

        # save weights for each epoch
        g.save_weights('weights/generator.h5', True)
        d.save_weights('weights/discriminator.h5', True)
        summarize_performance(epoch, g, 100, n_samples=4)
        a = g.predict(np.random.normal(0, 1, size=(1, 100)))
        plt.imshow(a[0, 0, :, :, 0], cmap='gray_r')
        plt.savefig('noise_gen%3d.png' % epoch)
        plt.close()
        b = g.predict(np.random.normal(0, 1, size=(1, 100)))
        plt.imshow(b[0, 0, :, :, 0], cmap='gray_r')
        plt.savefig('noise_gen_1%3d.png' % epoch)
        plt.close()
    return d, g


### generate images
def generate(BATCH_SIZE):
    g = generator_model()
    g.load_weights('weights/generator.h5')
    noise = np.random.normal(0, 1, (BATCH_SIZE, 100))
    generated_images = g.predict(noise)
    return generated_images


### anomaly loss function
def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))


### anomaly detection model define
def anomaly_detector(g=None, d=None):
    if g is None:
        g = generator_model()
        g.load_weights('weights/generator.h5')
    intermidiate_model = feature_extractor()
    intermidiate_model.trainable = False
    g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False
    # Input layer cann't be trained. Add new layer as same size & same distribution
    aInput = Input(shape=(100,))
    gInput = Dense((100), trainable=True)(aInput)
    gInput = Activation('tanh')(gInput)

    # G & D feature
    G_out = g(gInput)
    D_out = intermidiate_model(G_out)
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights=[0.90, 0.10], optimizer='rmsprop')

    # batchnorm learning phase fixed (test) : make non trainable
    K.set_learning_phase(0)

    return model





### anomaly detection
def compute_anomaly_score(model, x, iterations=500, d=None):
    global z, intermidiate_model, d_x, loss, similar_data, global_int_model_feat_extract, is_feature_extracted
    z = np.random.normal(0, 1, size=(1, 100))
    if is_feature_extracted:
        intermidiate_model = global_int_model_feat_extract
    else:
        intermidiate_model = feature_extractor()

    d_x = intermidiate_model.predict(x)

    # learning for changing latent
    loss = model.fit(z, [x, d_x], batch_size=1, epochs=iterations, verbose=0)
    similar_data, _ = model.predict(z)

    loss = loss.history['loss'][-1]

    return loss, similar_data

import hickle

if args.mode == 'train':
    X_train = hickle.load('X_train.hkl')
    Model_d, Model_g = train(16, X_train, n_epoch = args.n_epoch)

X_test = hickle.load('X_test.hkl')
y_test = np.load('Test_ped1.npy', allow_pickle=True)




ano_score = None
similar_img = None
original_x = None
similar_x = None
np.residual = None


def anomaly_detection(model, test_img, g=None, d=None):
    global ano_score, similar_img, similar_x, original_x, np_residual
    ano_score, similar_img = compute_anomaly_score(model, test_img.reshape(1, t, GENERATE_SQUARE, GENERATE_SQUARE, 1), iterations=500, d=d)

    # anomaly area, 255 normalization
    np_residual = test_img.reshape(t, GENERATE_SQUARE, GENERATE_SQUARE, 1) - similar_img.reshape(t, GENERATE_SQUARE, GENERATE_SQUARE, 1)
    np_residual = (np_residual + 2) / 4

    np_residual = (255 * np_residual.reshape(t, GENERATE_SQUARE, GENERATE_SQUARE)).astype(np.uint8)
    original_x = (test_img.reshape(t, GENERATE_SQUARE, GENERATE_SQUARE)*127.5 + 127.5).astype(np.uint8)
    similar_x = (similar_img.reshape(t, GENERATE_SQUARE, GENERATE_SQUARE) * 127.5 + 127.5).astype(np.uint8)


    return ano_score, original_x, similar_x, np_residual



model = anomaly_detector(g=None, d=None)

scores = np.ndarray((len(X_test)), np.float32)
model = anomaly_detector()
for i in range(len(X_test)):
    score, orig, sim, res = anomaly_detection(model, X_test[i], t)
    scores[i] = score
    if(i+1)%2==0:
        plt.imshow(orig[0], cmap='gray')
        plt.show()
        plt.imshow(sim[0], cmap='gray')
        plt.show()
        plt.imshow(res[0], cmap='gray')
        plt.show()
    if(i+1)%2==0:
        print((i+1)*t," tested")
        print(score)

from PIL import Image

scores_1 = np.squeeze(np.array(Image.fromarray(np.expand_dims(scores, 1).astype(np.float32)).resize((1,len(y_test)))))

np.save(OUTPUT_PATH + '/scores.npy',scores_1)
