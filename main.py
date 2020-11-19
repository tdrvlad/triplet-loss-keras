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
dataset_dir = 'dataset'

if os.path.exists(train_history_dir):
    fs = glob.glob(os.path.join(train_history_dir, "*"))
    for f in fs:
        os.remove(f)
else:
    os.mkdir(train_history_dir)

from sklearn.decomposition import PCA


class PCAPlotter(tf.keras.callbacks.Callback):
    
    def __init__(self, plt, embedding_model, x_test, y_test):
        super(PCAPlotter, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = x_test
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

    def __init__(self, tfrecord_dir, batch_size, test = False):

        if test == False:
            self.tfrecord_files = glob.glob(os.path.join(tfrecord_dir, 'train*.record'))
        else:
            self.tfrecord_files = glob.glob(os.path.join(tfrecord_dir, 'test*.record'))
        self.index = 0
        '''
            Index will point to which one of the tfrecords files the data is taken from.
        '''

        self.loaded_batches = []
        self.batch_size = batch_size
    

    def update_index(self):
        self.index += 1
        self.index = self.index % len(self.tfrecord_files)


    def load_tfrecord(self):

        '''
            Take all data in tfrecord_file
        '''

        print('Loading tfrecord file {}.'.format(self.index))
        dataset_partition = tf.data.TFRecordDataset(self.tfrecord_files[self.index]).take(-1)
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
            
            #label_map[label] = category

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

        return x, y


    def load_data(self):
        
        def divide_chunks(l, n): 
            '''
                Function to split a list in even chunks.
            '''
            for i in range(0, len(l), n):  
                yield l[i:i + n] 
        
        x, y = self.load_tfrecord()
        x_batches = list(divide_chunks(x, self.batch_size)) 
        y_batches = list(divide_chunks(y, self.batch_size)) 
        
        for _ in range(x_batches):
            if len(x_batches[i]) == len(y_batches) == batch_size:
                batch = (np.concatenate(x_batches[i]), np.concatenate(y_batches[i]))
                self.batches.append(batch)
        

    def get_batch(self):

        if len(self.batches) == 0:
            self.load_batches()
        if batch_size < 8:
            x, y = self.load_data(8 ** 2)
        else:
            x, y = self.load_data((batch_size + 1 ) ** 2)

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

    
    def model_data_generator(self, batch_size):

        while True:
            x = self.get_batch(batch_size)

            print(x[0].shape[0])
            y = np.zeros((x[0].shape[0], 3 * embbeding_size))
            
            print('Generated data: x ({}), y({}).'.format(len(x), y.shape), flush= True)
            yield x, y



def plot_triplets(examples):
    plt.figure(figsize=(6, 2))
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.reshape(examples[i] / 255.0, image_shape), cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    #plt.savefig('Result.jpg')


data_loader = DataLoader(tfrecord_dir)
test_data_loader = DataLoader(tfrecord_dir, test = True)

x, y = data_loader.load_tfrecord()


[x_anch, x_pos, x_neg] = data_loader.get_batch(8)

ex = data_loader.get_batch(1)
plot_triplets(ex)
'''
embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = image_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    #tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(embbeding_size)
])

#embedding_model.summary()


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


batch_size = 16
epochs = 10
steps_per_epoch = 10

net.compile(loss=triplet_loss, optimizer='adam')


x_test, y_test = test_data_loader.load_tfrecord()

_ = net.fit(
    data_loader.model_data_generator(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, verbose=False,
    callbacks=[
        PCAPlotter(
            plt, embedding_model,
            x_test, y_test
        )]
)

net.save('net_model')
embedding_model.save('embbeding_model')

'''