import os

# to suppress the excessive tf logging on cmd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
import gc

from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, GlobalMaxPooling3D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D
from tensorflow.python.util import deprecation

# To suppress the annoying 1.15 deprecation warnings
deprecation._PRINT_DEPRECATION_WARNINGS = False

TARGET_SIZE_LRCN = (224,224)
IMG_COL_LRCN = [f'img_col_{i}' for i in range(1,31) if i%6 == 0]
SEQ_LRCN = len(IMG_COL_LRCN)

TARGET_SIZE_3DCNN = (112,112)
IMG_COL_3DCNN = [f'img_col_{i}' for i in range(1,31,2)] + ['img_col_30']
SEQ_3DCNN = len(IMG_COL_3DCNN)

TARGET_COLS = ['is_fight','is_falldown']

SEED = 123
DIR = 'assault-falldown-data\\'
BATCH_SIZE=16

IMG_TYPES = ['raw','hm','kp','hhb','hkb','rhb']

def populate_image_cols(df, img_type, img_dir, img_cols):
    
    df_copy = df.copy()
    
    assert img_type in ['raw','hm','kp','hhb','hkb','rhb'], f'Image type "{img_type}" not available'
    
    if img_type in ['hhb','hkb','rhb']:
        folder_suffix = f'_3ch_{img_type}\\'
    else:
        folder_suffix = f'_{img_type}\\'
    
    for col in img_cols:
        
        i = int(col.split('_')[-1])
        
        df_copy[f'img_col_{i}'] = \
            f'{img_dir}' + \
            df_copy['data_source'] + \
            folder_suffix + \
            df_copy['subclip_name'] + \
            f'_frame{i:03d}' + \
            f'_{img_type}.png'
    
    return df_copy

def extract_features_lrcn(model, df):
    
    num_model_features = model.output_shape[-1]
    
    features = np.empty((df.shape[0], len(IMG_COL_LRCN), num_model_features))

    print(f'\n\nShape of empty array {features.shape}')
    steps = np.ceil(df.shape[0] / BATCH_SIZE)

    print(f'Extracting Features for {df.shape[0]} rows')
    print('===========================================')

    generator = ImageDataGenerator(rescale=1./255)

    for index, col in enumerate(IMG_COL_LRCN):

        img_gen = generator.flow_from_dataframe(df, 
                                                directory=DIR, 
                                                x_col=col, 
                                                class_mode=None, 
                                                seed=SEED, 
                                                batch_size=BATCH_SIZE,
                                                target_size=TARGET_SIZE_LRCN,
                                                shuffle=False)


        model_output = model.predict_generator(img_gen, steps=steps, verbose=1)
        features[:,index,:] = model_output
            
    return features

def create_3dcnn_model():
    """ Creates model object with the functional API:
     https://keras.io/models/model/
     """
    inputs = Input(shape=(16, 112, 112, 3,))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu',
                   padding='same', name='conv1')(inputs)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                         padding='valid', name='pool1')(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu',
                   padding='same', name='conv2')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool2')(conv2)

    conv3a = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3a')(pool2)
    conv3b = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3b')(conv3a)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool3')(conv3b)

    conv4a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4a')(pool3)
    conv4b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4b')(conv4a)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool4')(conv4b)

    conv5a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5a')(pool4)
    conv5b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5b')(conv5a)
    zeropad5 = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)),
                             name='zeropad5')(conv5b)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool5')(zeropad5)

    flattened = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flattened)
    dropout1 = Dropout(rate=0.5)(fc6)

    fc7 = Dense(4096, activation='relu', name='fc7')(dropout1)
    dropout2 = Dropout(rate=0.5)(fc7)

    predictions = Dense(487, activation='softmax', name='fc8')(dropout2)

    return Model(inputs=inputs, outputs=predictions)

def C3D():
    
    model = create_3dcnn_model()
    model.load_weights(os.path.join('pretrained_model','C3D_Sport1M_weights_keras_2.2.4.h5'))
    
    globalmax = GlobalMaxPooling3D(name='globalmax')(model.get_layer('pool5').output)
    
    feature_extractor = Model(inputs=model.input, outputs=globalmax, name='c3d')
    
    return feature_extractor

def multi_images_gen(df):
    generator = ImageDataGenerator(rescale=1./255)
    
    img_gen = dict()

    for index, col in enumerate(IMG_COL_3DCNN):

        img_gen[index] = generator.flow_from_dataframe(df, 
                                                directory=DIR, 
                                                x_col=col, 
                                                class_mode=None, 
                                                seed=SEED, 
                                                batch_size=BATCH_SIZE,
                                                target_size=TARGET_SIZE_3DCNN,
                                                shuffle=False)
    
    frames_gen = zip(*img_gen.values())
    
    while True:
        imgs = next(frames_gen)
        current_batch_size = imgs[0].shape[0]
        img_arr = np.zeros((current_batch_size, len(IMG_COL_3DCNN), *TARGET_SIZE_3DCNN, 3))

        for x in range(current_batch_size):
            img_arr[x] = np.stack([imgs[num][x] for num in range(len(IMG_COL_3DCNN))])

        yield img_arr

def extract_features_3dcnn(model, df):
    
    num_model_features = model.output_shape[-1]
    
    features = np.empty((df.shape[0], num_model_features))

    print(f'Shape of empty array {features.shape}')
    steps = np.ceil(df.shape[0] / BATCH_SIZE)

    print(f'Extracting Features for {df.shape[0]} rows')
    print('===========================================')

    generator = multi_images_gen(df)
    
    features = model.predict_generator(generator, steps=steps, verbose=1)
            
    return features

def save_train_test_bottle_neck_features_by_model(df_train, df_test, model, img_types, img_cols, extract_func, **model_params):
    
    feature_extractor = model(**model_params)
    
    model_name = feature_extractor.name.split('_')[0].lower()
    
    if not os.path.isdir(model_name):
        os.mkdir(model_name)

    for img_type in tqdm(img_types):
        
        # populate image links
        
        df_train_w_links = populate_image_cols(df_train, img_type, '', img_cols)
        df_test_w_links = populate_image_cols(df_test, img_type, '', img_cols)
        
        train_features = extract_func(feature_extractor, df_train_w_links)
        test_features = extract_func(feature_extractor, df_test_w_links)
        
        train_feature_name = f'{model_name}_{img_type}_train_features.npy'
        test_feature_name = f'{model_name}_{img_type}_test_features.npy'
        
        np.save(os.path.join(model_name, train_feature_name), train_features)
        np.save(os.path.join(model_name, test_feature_name), test_features)
        
        print('Saved train and test features.')

    del feature_extractor

    tf.keras.backend.clear_session()

    gc.collect()

    return


def main():
    
    df_train = pd.read_csv('annotation_train.csv')
    df_test = pd.read_csv('annotation_test.csv')

    print(' Extracting features from Resnet, Mobilenet, C3D.')
    for model in tqdm([ResNet50V2, MobileNet, C3D]):

        if model == C3D:
            img_cols = IMG_COL_3DCNN
            extract_func = extract_features_3dcnn
            model_params = {}
        else:
            img_cols = IMG_COL_LRCN
            extract_func = extract_features_lrcn
            model_params = {
                'include_top':False, 
                'weights':'imagenet', 
                'pooling':'max',
                'input_shape':(224,224,3)}

        save_train_test_bottle_neck_features_by_model(
            df_train, 
            df_test, 
            model, 
            IMG_TYPES, 
            img_cols, 
            extract_func,
            **model_params)

    return

if __name__ == '__main__':
    main()