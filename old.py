# https://zhangruochi.com/Create-a-Siamese-Network-with-Triplet-Loss-in-Keras/2020/08/11/

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

import os, glob, io
from PIL import Image 

print('TensorFlow version:', tf.__version__)

image_size = (250, 150)
image_shape = (image_size[1], image_size[0], 3)


embbeding_size = 128
train_history_dir = 'train_history'
tfrecord_dir = 'tfrecords'

if not os.path.exists(train_history_dir):
    os.mkdir(train_history_dir)

from sklearn.decomposition import PCA


class PCAPlotter(tf.keras.callbacks.Callback):
    
    def __init__(self, plt, embedding_model, x_test, y_test):
        super(PCAPlotter, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = batch_list_to_array(x_test)
        self.y_test = y_test

        self.fig = plt.figure(figsize=(9, 4))
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax2 = plt.subplot(1, 2, 2)
        plt.ion()
        
        self.losses = []
    
    def plot(self, epoch=None, plot_loss=False):
        x_test_embeddings = self.embedding_model.predict(self.x_test)
        pca_out = PCA(n_components=2).fit_transform(x_test_embeddings)
        self.ax1.clear()
        self.ax1.scatter(pca_out[:, 0], pca_out[:, 1], c=self.y_test, cmap='seismic')
        if plot_loss:
            self.ax2.clear()
            self.ax2.plot(range(epoch), self.losses)
            self.ax2.set_xlabel('Epochs')
            self.ax2.set_ylabel('Loss')
        self.fig.canvas.draw()
    
    def on_train_begin(self, logs=None):
        self.losses = []
        #self.fig.show()
        self.fig.canvas.draw()
        self.plot()
        self.fig.savefig(os.path.join(train_history_dir, 'Train_begin.jpg'))
        
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.plot(epoch+1, plot_loss=True)
        self.fig.savefig(os.path.join(train_history_dir,'Epoch{}.jpg'.format(epoch+1)))


class DataLoader:

    def __init__(self, tfrecord_dir):

        self.tfrecord_files = glob.glob(os.path.join(tfrecord_dir, '*.record'))
        self.index = 0

        '''
            Index will point to which one of the tfrecords files the data is taken from.
        '''

    def update_index(self):
        self.index += 1
        self.index = self.index % len(self.tfrecord_files)


    def load_tfrecord(self):

        '''
            Take all data in tfrecord_file
        '''

        print('Loading tfrecord file {}.'.format(self.index))
        dataset_partition = tf.data.TFRecordDataset(tfrecord_files[self.index]).take(-1)
        dataset_partition.shuffle(-1)
        self.update_index()
        
        x = []
        y = []

        for raw_record in dataset_partition:

            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            info = example.features.feature
        
            im_width = info['image/width'].int64_list.value[0]
            im_height = info['image/height'].int64_list.value[0]

            xmins = info['image/object/bbox/xmin'].float_list.value
            xmaxs = info['image/object/bbox/xmax'].float_list.value
            ymins = info['image/object/bbox/ymin'].float_list.value
            ymaxs = info['image/object/bbox/ymax'].float_list.value

            category = info['image/object/class/text'].bytes_list.value[0].decode("utf-8")
            label = info['image/object/class/label'].int64_list.value[0]
            
            label_map[label] = category

            enc_image = info['image/encoded'].bytes_list.value[0]
            image = Image.open(io.BytesIO(enc_image))


            for i in range(len(xmins)):
                cropped_image = image.crop((
                    xmins[i] * im_width,  
                    ymins[i] * im_height, 
                    xmaxs[i] * im_width,
                    ymaxs[i] * im_height)).resize(image_size)
            
                np_image = np.asarray(cropped_image)
                if np_image.shape == image_shape:
                    np_image = np_image[np.newaxis,...]
                    x.append(np_image)
                    y.append([label])

        x = np.concatenate(x)
        y = np.concatenate(y)

        return x, y


    def load_data(self, samples):
        
        x_partitions = []
        y_partitions = []
    
        while len(x) < samples:

            x_partition, y_partition = self.load_tfrecord()
            x_partitions.append(x_partition)
            y_partitions.append(y_partition)

        x = np.concatenate(x_partitions[:samples])
        y = np.concatenate(y_partitions[:samples])

        return x, y


    def get_batch(self, batch_size):

        x, y = self.load_data(batch_Size ** 2)

        batch_array_size = (batch_size, image_shape[0], image_shape[1], image_shape[2])
        
        x_anchors = np.zeros(batch_array_size)
        x_positives = np.zeros(batch_array_size)
        x_negatives = np.zeros(batch_array_size)
        
        for i in range(0, batch_size):
            # We need to find an anchor, a positive example and a negative example
            random_index = random.randint(0, x.shape[0] -1)
            x_anchor = x[random_index]
            y_anchor = y[random_index]

            
            indices_for_pos = np.squeeze(np.where(y == y_anchor))
            indices_for_neg = np.squeeze(np.where(y != y_anchor))

            try:
                len(indices_for_pos)
            except:
                indices_for_pos = [random_index]

            x_positive = x[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
            x_negative = x[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]
            
            x_anchors[i] = x_anchor
            x_positives[i] = x_positive
            x_negatives[i] = x_negative
            
        return [x_anchors, x_positives, x_negatives]




def load_data(samples = 1000, split = 0.9, tfrecord_file = None):

    label_map = {}
    x = []
    y = []
 

    if tfrecord_file is None:
        tfrecord_files = glob.glob(os.path.join(tfrecord_dir, '*.record'))
    else:
        tfrecord_files = [tfrecord_file]

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files).take(samples)

    for raw_record in raw_dataset.take(samples):

        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        info = example.features.feature
    
        im_width = info['image/width'].int64_list.value[0]
        im_height = info['image/height'].int64_list.value[0]

        xmins = info['image/object/bbox/xmin'].float_list.value
        xmaxs = info['image/object/bbox/xmax'].float_list.value
        ymins = info['image/object/bbox/ymin'].float_list.value
        ymaxs = info['image/object/bbox/ymax'].float_list.value

        category = info['image/object/class/text'].bytes_list.value[0].decode("utf-8")
        label = info['image/object/class/label'].int64_list.value[0]
        
        label_map[label] = category

        enc_image = info['image/encoded'].bytes_list.value[0]
        image = Image.open(io.BytesIO(enc_image))


        for i in range(len(xmins)):
            cropped_image = image.crop((
                xmins[i] * im_width,  
                ymins[i] * im_height, 
                xmaxs[i] * im_width,
                ymaxs[i] * im_height)).resize(image_size)
        
            np_image = np.asarray(cropped_image)
            if np_image.shape == image_shape:
                np_image = np_image[np.newaxis,...]
                x.append(np_image)
                y.append([label])

    x = np.concatenate(x)
    y = np.concatenate(y)

    train_samples = int(x.shape[0] * split)
    test_samples = x.shape[0] - train_samples

    x_train = x[:train_samples,:,:,:]
    x_test = x[-test_samples:,:,:,:]

    y_train = y[:x_train.shape[0]]
    y_test = y[-x_test.shape[0]:]

    print('Loaded {} samples ({} training, {} testing).'.format(x.shape[0], x_train.shape[0], x_test.shape[0]))

    return (x_train, y_train), (x_test, y_test)
    
    

(x_train, y_train), (x_test, y_test) = load_data()


def plot_triplets(examples):
    plt.figure(figsize=(6, 2))
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.reshape(examples[i] / 255.0, image_shape), cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    #plt.savefig('Result.jpg')



def create_batch(x_train, y_train, batch_size=32):

    batch_array_size = (batch_size, image_shape[0], image_shape[1], image_shape[2])
    x_anchors = np.zeros(batch_array_size)
    x_positives = np.zeros(batch_array_size)
    x_negatives = np.zeros(batch_array_size)
    
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, x_train.shape[0] -1)
        x_anchor = x_train[random_index]
        y_anchor = y_train[random_index]

        
        indices_for_pos = np.squeeze(np.where(y_train == y_anchor))
        indices_for_neg = np.squeeze(np.where(y_train != y_anchor))

        try:
            len(indices_for_pos)
        except:
            indices_for_pos = [random_index]

        x_positive = x_train[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = x_train[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]
           
        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        
    return [x_anchors, x_positives, x_negatives]


#examples = create_batch(x_train, y_train, 1)
#plot_triplets(examples)


embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = image_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    #tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(embbeding_size)
])

embedding_model.summary()


input_anchor = tf.keras.layers.Input(shape=image_shape)
input_positive = tf.keras.layers.Input(shape=image_shape)
input_negative = tf.keras.layers.Input(shape=image_shape)

embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)
embedding_negative = embedding_model(input_negative)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
net.summary()

alpha = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:embbeding_size], y_pred[:,embbeding_size:2*embbeding_size], y_pred[:,2*embbeding_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


def data_generator(x_train, y_train, batch_size=16):
    
    while True:
        x = create_batch(x_train, y_train, batch_size)
        y = np.zeros((len(x), 3 * embbeding_size))
        
        yield x, y



batch_size = 16
epochs = 10
steps_per_epoch = int(len(x_train)/batch_size)

net.compile(loss=triplet_loss, optimizer='adam')

_ = net.fit(
    data_generator(x_train, y_train, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, verbose=False,
    callbacks=[
        PCAPlotter(
            plt, embedding_model,
            x_train[:300], y_train[:300]
        )]
)

net.save('net_model')
embedding_model.save('embbeding_model')