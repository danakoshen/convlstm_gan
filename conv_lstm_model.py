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
 #   fc1 = BatchNormalization(momentum=0.8)(fc1)
#    fc1 = LeakyReLU(0.2)(fc1) kernel_size = (4,4)
    fc2 = Reshape((t, int(GENERATE_SQUARE/4), int(GENERATE_SQUARE/4), 32), input_shape=(32 * t * int(GENERATE_SQUARE/4) * int(GENERATE_SQUARE/4),))(fc1)

 #   up1 = Conv3DTranspose(1, (1,3,3), strides = (1,2,2), padding='same')(fc2)
#    up1 = TimeDistributed(UpSampling2D())(fc2)
    up1 = TimeDistributed(Conv2D(128, (5,5), strides = (2,2), padding='same'))(fc2)  #reconsider for the next test
#    conv1 = ConvLSTM2D(64, (3,3), strides=(1,1), padding='same', return_sequences=True)(up1) #2_kernel
    conv1 = BatchNormalization(momentum=0.8)(up1)
    conv1 = ReLU()(conv1)
    conv1 = ConvLSTM2D(128, (3, 3), strides=(1, 1), padding='same', return_sequences=True)(conv1)
#    conv1 = Dropout(0.5)(conv1)

#    up2 = Conv3DTranspose(1, (1,4,4), strides = (1,2,2), padding='same')(conv1)
#    up2 = TimeDistributed(UpSampling2D())(conv1)
#    up2 = UpSampling3D(size = (1,2,2))(conv1)

    conv2 = ConvLSTM2D(64, (3,3), strides=(1,1), padding='same', return_sequences=True)(conv1) #3_kernel
#    conv2 = BatchNormalization(momentum=0.8)(conv2)
#    conv2 = ReLU()(conv2)
#    conv2 = Dropout(0.5)(conv2)
    conv3 = ConvLSTM2D(48, (3,3), strides=(1,1), padding='same', return_sequences=True)(conv2) #5_kernel
#    conv3 = ReLU()(conv3)
#    conv3 = Dropout(0.5)(conv3)
#    up1 = Conv3DTranspose(1, (1,4,4), strides = (1,2,2), padding='same')(conv3)
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
    #discriminator.add(GaussianNoise(0.1, input_shape=(t, GENERATE_SQUARE, GENERATE_SQUARE, 1)))
    #noise1 = GaussianNoise(0.1)(inputs)
    conv1 = ConvLSTM2D(128, (3,3), strides=(2,2), padding='same', return_sequences=True)(inputs) #16
#    conv1 = LeakyReLU(0.2)(conv1)
    conv1 = Dropout(0.25)(conv1)
    conv2 = ConvLSTM2D(64, (3,3), strides=(2,2), padding='same', return_sequences=True)(conv1) #32
#    conv2 = BatchNormalization(momentum=0.8)(conv2)
#    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = Dropout(0.25)(conv2)
#    conv4 = ConvLSTM2D(48, (3,3), strides=(2,2), padding='same', return_sequences=True)(conv2) #64
#    conv4 = BatchNormalization(momentum=0.8)(conv4)
#    conv4 = LeakyReLU(0.2)(conv4)
#    conv4 = Dropout(0.25)(conv4)
    conv3 = ConvLSTM2D(48, (3,3), strides=(1,1), padding='same', return_sequences=True)(conv2) #128
    conv3 = BatchNormalization(momentum=0.8)(conv3)
    conv3 = ReLU()(conv3)
#    conv3 = LeakyReLU(0.2)(conv3)
#    conv3 = Dropout(0.5)(conv3)
    up1 = TimeDistributed(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))(conv3)
    fc1 = Flatten()(up1)
#    fc1 = Dense(1)(fc1)
    outputs = Dense(1)(fc1)
#    outputs = Activation('tanh')(fc1)

    model = Model(inputs = [inputs], outputs = [outputs])
    return model


### d_on_g model for training generator
def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(100,))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    # gan.compile(loss='binary_crossentropy', optimizer='adam')
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
#    g.compile(loss='mse', optimizer=g_optim)
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
            # create random noise -> U(0,1) 10 latent vectors
            # noise = np.random.normal(0, 1, size=(BATCH_SIZE, 100))

            # load real data & generate fake data
            # image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            # generated_images = g.predict(noise, verbose=0)

            # visualize training results
#            if index % 20 == 0:
#                image = combine_images(generated_images)
#                image = image * 127.5 + 127.5
#                cv2.imwrite('./result/' + str(epoch) + "_" + str(index) + ".png", image)

            # attach label for training discriminator
            #X = np.concatenate((image_batch, generated_images))
            #y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)

            # training discriminator
            #d_loss = d.train_on_batch(X, y)
            # d_loss_real = d.train_on_batch(image_batch, valid)
            # d_loss_fake = d.train_on_batch(generated_images, fake)
            # d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

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

            # for l in d.layers:
            #     weights = l.get_weights()
            #     weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            #     l.set_weights(weights)

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

### 1. train generator & discriminator
# if args.mode == 'train':
#     X_train = np.load('training_data_28_28.npy', allow_pickle=True)
#     print('Original train shape: ' + str(X_train.shape))
#     X_train = X_train[:, :, :, None]
#     if args.class_n == '1':
#         X_train_1 = np.load('training_data_1_28_28.npy', allow_pickle=True)
#         print('Class 1 train shape: ' + str(X_train_1.shape))
#         X_train_1 = X_train_1[:, :, :, None]
#         Model_d, Model_g = train(64, X_train_1, class_n='1', n_epoch=args.n_epoch)
#     elif args.class_n == '2':
#         X_train_2 = np.load('training_data_2_28_28.npy', allow_pickle=True)
#         print('Class 2 train shape: ' + str(X_train_2.shape))
#         X_train_2 = X_train_2[:, :, :, None]
#         Model_d, Model_g = train(64, X_train_2, class_n='2', n_epoch=args.n_epoch)
#     else:
#         Model_d, Model_g = train(64, X_train, class_n='0', n_epoch=args.n_epoch)
X_test = hickle.load('X_test.hkl')
y_test = np.load('Test_ped1.npy', allow_pickle=True)

### 2. test generator
# generated_img = generate(25)
# img = combine_images(generated_img)
# img = (img * 127.5) + 127.5
# img = img.astype(np.uint8)
# img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

### opencv view
# cv2.namedWindow('generated', 0)
# cv2.resizeWindow('generated', 256, 256)
# cv2.imshow('generated', img)
# cv2.imwrite('result_latent_10/generator.png', img)
# cv2.waitKey()

### plt view
# plt.figure(num=0, figsize=(4, 4))
# plt.title('trained generator')
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# exit()

### 3. other class anomaly detection


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



### compute anomaly score - sample from test set
# test_img = X_test_original[y_test==1][30]

### compute anomaly score - sample from strange image
# test_img = X_test_original[y_test==0][30]

### compute anomaly score - sample from strange image
# test_img = globals()['X_test_'+class_n][img_idx]
model = anomaly_detector(g=None, d=None)

scores = np.ndarray((len(X_test)), np.float32)
model = anomaly_detector()
for i in range(len(X_test)):
#for i in range(500,511):
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
# test_img = np.random.normal(-1,1, (28,28,1))
#
# start = cv2.getTickCount()
# score, qurey, pred, diff = anomaly_detection(test_img, class_n)
# time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
# print ('%d : done'%(img_idx), '%.2f'%score, '%.2fms'%time)
## cv2.imwrite('./qurey.png', qurey)
## cv2.imwrite('./pred.png', pred)
## cv2.imwrite('./diff.png', diff)
#
### matplot view
# plt.figure(1, figsize=(3, 3))
# plt.title('query image')
# plt.imshow(qurey.reshape(28,28), cmap=plt.cm.gray)
#
# print("anomaly score : ", score)
# plt.figure(2, figsize=(3, 3))
# plt.title('generated similar image')
# plt.imshow(pred.reshape(28,28), cmap=plt.cm.gray)
#
# plt.figure(3, figsize=(3, 3))
# plt.title('anomaly detection')
# plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
# plt.show()


### 4. tsne feature view

### t-SNE embedding
### generating anomaly image for test (radom noise image)

# from sklearn.manifold import TSNE
#
# random_image = np.random.normal(0, 1, (100, 28, 28, 1))
# print("random noise image")
# plt.figure(4, figsize=(2, 2))
# plt.title('random noise image')
# plt.imshow(random_image[0].reshape(28,28), cmap=plt.cm.gray)
#
## intermidieate output of discriminator
# model = anogan.feature_extractor(class_n='1')
# feature_map_of_random = model.predict(random_image, verbose=1)
# feature_map_of_minist = model.predict(X_test[:len(X_test)], verbose=1)
# feature_map_of_minist_1 = model.predict(X_test[:len(X_test)], verbose=1)
#
## t-SNE for visulization
# output = np.concatenate((feature_map_of_random, feature_map_of_minist, feature_map_of_minist_1))
# output = output.reshape(output.shape[0], -1)
# anomaly_flag = np.array([1]*100+ [0]*300)
#
# X_embedded = TSNE(n_components=2).fit_transform(output)
# plt.figure(5)
# plt.title("t-SNE embedding on the feature representation")
# plt.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
# plt.scatter(X_embedded[100:400,0], X_embedded[100:400,1], label='mnist(anomaly)')
# plt.scatter(X_embedded[400:,0], X_embedded[400:,1], label='mnist(normal)')
# plt.legend()
# plt.show()
