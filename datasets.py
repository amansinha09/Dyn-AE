# import numpy as np
# import os
from PIL import Image
from sklearn.utils import shuffle
import os, numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import cv2
# from skimage import color
# from skimage.feature import hog
def extract_vgg16_features(x):
    from keras.preprocessing.image import img_to_array, array_to_img
    from keras.applications.vgg16 import preprocess_input, VGG16
    from keras.models import Model

    # im_h = x.shape[1]
    im_h = 224
    model = VGG16(include_top=True, weights='imagenet', input_shape=(im_h, im_h, 3))
    # if flatten:
    #     add_layer = Flatten()
    # else:
    #     add_layer = GlobalMaxPool2D()
    # feature_model = Model(model.input, add_layer(model.output))
    feature_model = Model(model.input, model.get_layer('fc1').output)
    print('extracting features...')
    x1 = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x[0:1000]])
    x1 = preprocess_input(x1)
    features = feature_model.predict(x1)
    for i in range(int(x.shape[0] / 1000)-1):
      x1 = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x[1000*(i+1):1000*(i+2)]])
      x1 = preprocess_input(x1)  # data - 127. #data/255.#
      features1 = feature_model.predict(x1)
      features = np.concatenate((features, features1)).astype(float)
    print('Features shape = ', features.shape)
    return features

def load_flicker(folder = 'gdrive/My Drive/flickr-sarcasm-dataset/train/transformed'):
    x = []
    y = []
    i = 0
    for i, filename in enumerate(os.listdir(folder + "/0_transformed")):
        #if(i > 20):
         #   break
        img = np.array(Image.open(os.path.join(folder + "/0_transformed",filename)))
        if img is not None:
            x.append(img)
#             x.append(img.flatten())
            y.append(0)
        i += 1
        
    for i, filename in enumerate(os.listdir(folder + "/1_transformed")):
        #if(i > 10):
         #   break
        img = np.array(Image.open(os.path.join(folder + "/1_transformed",filename)))
        if img is not None:
            x.append(img)
#             x.append(img.flatten())
            y.append(1)
        i += 1
  
    x = np.array(x)/255.0
    y = np.array(y)
    x, y = shuffle(x, y, random_state=0)
    return x, y   


def load_data_conv(dataset, datapath):
 #    if dataset == 'mnist':
 #        return load_mnist()
 #    elif dataset == 'mnist-test':
 #        return load_mnist_test()
 #    elif dataset == 'fmnist':
 #        return load_fashion_mnist()
 #    elif dataset == 'usps':
 #        return load_usps(datapath)
 #    elif dataset == 'pendigits':
 #        return load_pendigits(datapath)
 #    elif dataset == 'reuters10k' or dataset== 'reuters':
 #        return load_reuters(datapath)
 #    elif dataset == 'stl':
 #        return load_stl(datapath)
 #    elif dataset == 'cifar10':
 #        return load_cifar10(datapath)
	# el
    #if dataset == 'flicker':
	return load_flicker(datapath)
    #else:
    #    raise ValueError('Not defined for loading %s' % dataset)

def load_data(dataset, datapath):
    x, y = load_data_conv(dataset, datapath)
    return x.reshape([x.shape[0], -1]), y

def generate_data_batch(x, y=None, batch_size=256):
    index_array = np.arange(x.shape[0])
    index = 0
    while True:
        idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
        if y is None:
            yield x[idx]
        else: 
            yield x[idx], y[idx]

def generate_transformed_batch(x, datagen, batch_size=256):
    if len(x.shape) > 2:  # image
        gen0 = datagen.flow(x, shuffle=False, batch_size=batch_size)
        while True:
            batch_x = next(gen0)
            yield batch_x
    else:
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen0 = datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=batch_size)
        while True:
            batch_x = next(gen0)
            batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
            yield batch_x      

def generate(x, datagen, batch_size=256):
    gen1 = generate_data_batch(x, batch_size=batch_size)
    if len(x.shape) > 2:  # image
        gen0 = datagen.flow(x, shuffle=False, batch_size=batch_size)
        while True:
            batch_x1 = next(gen0)
            batch_x2 = next(gen1)
            yield [batch_x1, batch_x2]
    else:
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen0 = datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=batch_size)
        while True:
            batch_x1 = next(gen0)
            batch_x1 = np.reshape(batch_x1, [batch_x1.shape[0], x.shape[-1]])
            batch_x2 = next(gen1)
            yield [batch_x1, batch_x2]


