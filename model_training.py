import os

# to suppress the excessive tf logging on cmd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import gc

from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Input, Bidirectional, CuDNNLSTM, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.util import deprecation

# To suppress the annoying 1.15 deprecation warnings
deprecation._PRINT_DEPRECATION_WARNINGS = False

SEED = 123
DIR = 'assault-falldown-data\\'
TARGET_COLS = ['is_fight','is_falldown']

def gss_get_fold_data(df, total_folds, fold_no):
    
    gss = GroupShuffleSplit(n_splits=total_folds, random_state=SEED)
    
    for index, (train_id, test_id) in enumerate(gss.split(df, df[TARGET_COLS], df['video_name'])):
        
        if index != fold_no:
            continue
        else:
            df_train = df.iloc[train_id].copy()
            df_test = df.iloc[test_id].copy()

            # Check

            df_check = df_train['video_name'].value_counts().to_frame()\
                .merge(df_test['video_name'].value_counts(),
                       left_index=True, right_index=True, how='outer')\
                .notna()\
                .sum(axis=1)
            
            assert (df_check == 1).all(), 'Subclips are not split according to videos'

            break
    
    return df_train, df_test

def apply_split_according_to_source(df, func, as_index=True, **func_params):
    
    datasources = df['data_source'].unique()
    
    df_train_source = dict()
    df_test_source = dict()
    
    for source in datasources:
        df_subset = df.loc[df['data_source'] == source].copy()
        df_subset_train, df_subset_test = func(df_subset, **func_params)
        df_train_source[source] = df_subset_train
        df_test_source[source] = df_subset_test
        
    df_train = pd.concat([*df_train_source.values()], axis=0, ignore_index=False, sort=False)
    df_test = pd.concat([*df_test_source.values()], axis=0, ignore_index=False, sort=False)
    
    # Check 1 : duplicated videos in train and test
    
    train_videos = set(df_train['video_name'].unique())
    test_videos = set(df_test['video_name'].unique())
    
    assert len(train_videos.intersection(test_videos)) == 0, 'Duplicated videos in train & test set'
    
    # Check 2 : df_train and df_test same size as original df
    
    assert (df_train.shape[0] + df_test.shape[0]) == df.shape[0], 'output df different size from original df'
    
    if as_index:
        return df_train.index, df_test.index
    else:
        return df_train, df_test

def get_lrcn_model(feature_shape):
    
    # using parameters from best bayesian trial
    nn_input = Input((5,feature_shape))
    x = Bidirectional(CuDNNLSTM(128, 
                           return_sequences=True))(nn_input)

    x = GlobalMaxPooling1D()(x)

    x = Dropout(0.5)(x)
    x = Dense(32, 
              activation='relu', 
              kernel_regularizer=l2(0.15))(x)

    x = Dropout(0.5)(x)
    x = Dense(32, 
              activation='relu', 
              kernel_regularizer=l2(0.15))(x)

    x = Dropout(0.5)(x)

    # Multi Label
    nn_output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=nn_input, outputs=nn_output)
    
    return model

def get_3dcnn_model(feature_shape):
    
    nn_input = Input((feature_shape))
    
    x = Dense(1028, activation='relu', name='dense')(nn_input)
    x = Dropout(rate=0.5, name='dropout2')(x)
    
    nn_output = Dense(2, activation='sigmoid', name='output')(x)

    model = Model(inputs=nn_input, outputs=nn_output)

    return model


def get_callbacks(monitor='val_loss', mode='min', patience=10, verbose=1):
    
    earlystop = EarlyStopping(monitor=monitor, mode=mode, verbose=verbose, patience=patience, restore_best_weights=True)
    
    lr_reducer = ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                   cooldown=0, patience=round(patience // 2 - 1), 
                                   mode=mode, verbose=verbose)
    call_backs = [lr_reducer, earlystop]
    
    return call_backs

def load_features(extractor, img_type):
    
    assert extractor in ['mobilenet','c3d','resnet50v2'], 'Extractor not available'

    assert img_type in ['raw','hm','kp','hhb','hkb','rhb'], 'Image type not available'

    extractor = extractor

    train_features_file = f'_{img_type}_train_features.npy'
    test_features_file = f'_{img_type}_test_features.npy'

    train_features = np.load(os.path.join(extractor, extractor + train_features_file))
    test_features = np.load(os.path.join(extractor, extractor + test_features_file))

    return train_features, test_features

def load_df_and_labels(curated=True):
    df_train = pd.read_csv('annotation_train.csv')
    df_test = pd.read_csv('annotation_test.csv')

    train_labels = df_train[TARGET_COLS].to_numpy()
    test_labels = df_test[TARGET_COLS].to_numpy()
    
    if curated:
        df_train = df_train[(df_train['curated'] == True) | (df_train['curated'].isna())]

    return df_train, train_labels, test_labels

def run_cross_validation(total_folds, extractor, img_types, curated=True, total_rounds=1):

    start_time = time.time()

    df, train_labels, test_labels = load_df_and_labels(curated)

    folder_dir = extractor + '_training_results'
    preds_csv_path = folder_dir + '\\' + extractor + '_test_preds.csv'

    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)

    if os.path.isfile(preds_csv_path):
        df_preds_master = pd.read_csv(preds_csv_path)
    else:
        df_preds_master = pd.DataFrame(test_labels, columns=['is_fight','is_falldown'])

    for img_type in tqdm(img_types):

        img_type_folder_dir = os.path.join(folder_dir, img_type)
        if not os.path.isdir(img_type_folder_dir):
            os.mkdir(img_type_folder_dir)

        train_features, test_features = load_features(extractor, img_type)

        for training_round in range(total_rounds):
            print(f'\n {extractor.capitalize()} : Training Round {training_round + 1} of {total_rounds} for {img_type}')
            for fold_no in range(total_folds):
                
                train_idx, val_idx = apply_split_according_to_source(df,
                            gss_get_fold_data,
                            as_index=True,
                            total_folds=total_folds, 
                            fold_no=fold_no)
                print(f' Fold No {fold_no + 1} of {total_folds} : {len(train_idx)} train samples, {len(val_idx)} val samples.')

                file_name = f'{img_type_folder_dir}\\{extractor}_{img_type}_run{training_round + 1}_fold{fold_no + 1}'

                test_preds = training_loop(
                    extractor,
                    train_features[train_idx], train_labels[train_idx], 
                    train_features[val_idx], train_labels[val_idx],
                    test_features, test_labels, file_name)

                df_preds = pd.DataFrame(test_preds, columns=[
                    f'{img_type}_run{training_round + 1}_fold{fold_no + 1}_fight',
                    f'{img_type}_run{training_round + 1}_fold{fold_no + 1}_fall'])

                df_preds_master = df_preds_master.merge(df_preds, left_index=True, right_index=True)

    df_preds_master.to_csv(preds_csv_path, index=False)
    
    output_msg = \
        f' {extractor.capitalize()} : {len(img_types) * total_folds * total_rounds} models' + \
        f' completed in {(time.time() - start_time) / 60 :.2f} mins'

    print(output_msg)


def training_loop(model, train_features, train_labels, 
        val_features, val_labels, 
        test_features, test_labels, file_name):

    
    num_features = train_features.shape[-1]

    if model == 'c3d':
        model = get_3dcnn_model(num_features)

    else:
        model = get_lrcn_model(num_features)

    callbacks = get_callbacks(monitor='val_loss', mode='min', patience=50, verbose=0)

    model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy', 
              metrics=['acc', AUC(name='auc')])

    history = model.fit(train_features,train_labels, 
                  batch_size=128, 
                  validation_data=(val_features, val_labels), 
                  callbacks=callbacks, 
                  verbose=0,
                  epochs=1000)

    test_preds = model.predict_on_batch(test_features)

    # Plot train & val AUC, Loss
    plot_and_save_train_val_metrics(history, file_name)

    print(f'   Fight AUC : {roc_auc_score(test_labels[:,0], test_preds[:,0]):.3f}')
    print(f'   Fall AUC  : {roc_auc_score(test_labels[:,1], test_preds[:,1]):.3f}\n')

    del model

    tf.keras.backend.clear_session()

    gc.collect()

    return test_preds

def plot_and_save_train_val_metrics(history, file_name):
    fig, ax = plt.subplots(1,2, figsize=(8,4))

    for index, metric in enumerate(['auc','loss']):
        ax[index].plot(history.history[f'val_{metric}'], label=f'Val {metric.capitalize()}')
        ax[index].plot(history.history[f'{metric}'], label=f'Train {metric.capitalize()}')
        ax[index].legend(loc='center right')
        ax[index].set_title(f'{metric.upper()}', fontsize=16)
        if metric == 'auc':
            ax[index].set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(f'{file_name}_metrics.png', dpi=150)
    
    plt.close(fig)
    gc.collect()

    return 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", 
        type=str,   
        help="Select model you want to build.", 
        required=True,
        choices=['mobilenet','resnet50v2','c3d','all']
        )

    parser.add_argument("--folds", 
        type=int, 
        help="Specify N-fold validations.",
        required=True, 
        choices=range(1,11))

    parser.add_argument("--runs", 
        type=int, 
        help="Specify N runs. During each run, N-fold CV is done.",
        required=True, 
        choices=range(1,11))

    parser.add_argument("--imgtype", 
        type=str, 
        help="Specify feature representation.",
        required=True, 
        choices=['raw','hm','kp','hhb','rhb','hkb','all'])

    args = parser.parse_args()

    return args

def main():

    args =get_args()

    if args.model == 'all':
        models = ['mobilenet','resnet50v2','c3d']
    else:
        models = [args.model]

    if args.imgtype == 'all':
        img_types = ['raw','hm','kp','hhb','rhb','hkb']
    else:
        img_types = [args.imgtype]

    for model in models:
        run_cross_validation(
            total_folds=args.folds, 
            extractor=model, 
            img_types=img_types,
            total_rounds=args.runs)

    return

if __name__ == '__main__':    
    main()

