import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import glob
import re
import scipy
import time
import collections
import itertools
import librosa
import pickle
from librosaFeatureExtract import feature_extract


extractTestingFeatures = False
extractValidationFeatures = False

def instrument_code(filename):
    """
    Function that takes in a filename and returns instrument based on naming convention
    """
    class_names = ['bass', 'brass', 'flute', 'guitar',
                   'keyboard', 'mallet', 'organ', 'reed',
                   'string', 'synth_lead', 'vocal']

    for name in class_names:
        if name in filename:
            return class_names.index(name)
    else:
        return None


def prepare_files_with_extracted_features():
    # directory to training data and json file
    train_dir = 'C:/Users/dmarcu/Desktop/nsynth/nsynth-train/audio/'
    # directory to training data and json file
    valid_dir = 'C:/Users/dmarcu/Desktop/nsynth/nsynth-valid/audio/'
    # directory to training data and json file
    test_dir = 'C:/Users/dmarcu/Desktop/nsynth/nsynth-test/audio/'

    # read the raw json files as given in the training set
    df_train_raw = pd.read_json(path_or_buf='C:/Users/dmarcu/Desktop/nsynth/nsynth-train/examples.json', orient='index')
    # Get a count of instruments in ascending order
    n_class_train = df_train_raw['instrument_family'].value_counts(ascending=True)

    # Sample n files
    df_train_sample = df_train_raw.groupby('instrument_family', as_index=False,  # group by instrument family
                                           group_keys=False).apply(lambda df: df.sample(200))  # number of samples
    # drop the synth_lead from the training dataset
    df_train_sample = df_train_sample[df_train_sample['instrument_family'] != 9]

    # Display instrument distrution
    # df_train_sample['instrument_family'].value_counts().reindex(np.arange(0, len(n_class_train), 1)).plot(kind='bar')
    # plt.title("Instrument Family Distribution of Sampled Data: Training")
    # plt.xlabel('Instrument Family')
    # plt.ylabel('Number of Samples in Dataset')
    # plt.show()

    # save the train file index as list
    filenames_train = df_train_sample.index.tolist()
    # save the list to a pickle file
    with open('C:/Users/dmarcu/Desktop/nsynth/DataWrangling/filenames_train.pickle', 'wb+') as f:
        pickle.dump(filenames_train, f)
    # extract the filenames from the validation dataset
    df_valid = pd.read_json(path_or_buf='C:/Users/dmarcu/Desktop/nsynth/nsynth-valid/examples.json', orient='index')

    # save the train file index as list
    filenames_valid = df_valid.index.tolist()

    # save the list to a pickle file
    with open('C:/Users/dmarcu/Desktop/nsynth/DataWrangling/filenames_valid.pickle', 'wb+') as f:
        pickle.dump(filenames_valid, f)

    # extract the filenames from the testing dataset
    df_test = pd.read_json(path_or_buf='C:/Users/dmarcu/Desktop/nsynth/nsynth-test/examples.json', orient='index')

    # save the train file index as list
    filenames_test = df_test.index.tolist()

    # save the list to a pickle file
    with open('C:/Users/dmarcu/Desktop/nsynth/DataWrangling/filenames_test.pickle', 'wb+') as f:
        pickle.dump(filenames_test, f)

    # extract features testing set @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
    if (extractTestingFeatures):
        start_valid = time.time()

        # create dictionary to store all test features
        dict_test = {}
        # loop over every file in the list
        i = 0
        for file in filenames_test:
            i += 1
            print("Exctracting from testing file number # " + str(i))
            # extract the features
            features = feature_extract(test_dir + file + '.wav')  # specify directory and .wav
            # add dictionary entry
            dict_test[file] = features

        end_valid = time.time()
        print('Time to extract {} files is {} seconds'.format(len(filenames_valid), end_valid - start_valid))

        # convert dict to dataframe
        features_test = pd.DataFrame.from_dict(dict_test, orient='index',
                                               columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast'])

        # extract mfccs
        mfcc_test = pd.DataFrame(features_test.mfcc.values.tolist(), index=features_test.index)
        mfcc_test = mfcc_test.add_prefix('mfcc_')

        # extract spectro
        spectro_test = pd.DataFrame(features_test.spectro.values.tolist(), index=features_test.index)
        spectro_test = spectro_test.add_prefix('spectro_')

        # extract chroma
        chroma_test = pd.DataFrame(features_test.chroma.values.tolist(), index=features_test.index)
        chroma_test = chroma_test.add_prefix('chroma_')

        # extract contrast
        contrast_test = pd.DataFrame(features_test.contrast.values.tolist(), index=features_test.index)
        contrast_test = chroma_test.add_prefix('contrast_')

        # drop the old columns
        features_test = features_test.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast'], axis=1)

        # concatenate
        df_features_test = pd.concat([features_test, mfcc_test, spectro_test, chroma_test, contrast_test],
                                     axis=1, join='inner')

        targets_test = []
        for name in df_features_test.index.tolist():
            targets_test.append(instrument_code(name))

        df_features_test['targets'] = targets_test

        # save the dataframe to a pickle file
        with open('C:/Users/dmarcu/Desktop/nsynth/DataWrangling/df_features_test.pickle', 'wb+') as f:
            pickle.dump(df_features_test, f)



    # extract features training set @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
    start_train = time.time()

    # create dictionary to store all test features
    dict_train = {}
    # loop over every file in the list
    i = 0
    for file in filenames_train:
        i += 1
        print("Exctracting from training file number # " + str(i))
        # extract the features
        features = feature_extract(train_dir + file + '.wav')  # specify directory and .wav
        # add dictionary entry
        dict_train[file] = features

    end_train = time.time()
    print('Time to extract {} files is {} seconds'.format(len(filenames_train), end_train - start_train))

    # convert dict to dataframe
    features_train = pd.DataFrame.from_dict(dict_train, orient='index',
                                            columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast'])

    # extract mfccs
    mfcc_train = pd.DataFrame(features_train.mfcc.values.tolist(),
                              index=features_train.index)
    mfcc_train = mfcc_train.add_prefix('mfcc_')

    # extract spectro
    spectro_train = pd.DataFrame(features_train.spectro.values.tolist(),
                                 index=features_train.index)
    spectro_train = spectro_train.add_prefix('spectro_')

    # extract chroma
    chroma_train = pd.DataFrame(features_train.chroma.values.tolist(),
                                index=features_train.index)
    chroma_train = chroma_train.add_prefix('chroma_')

    # extract contrast
    contrast_train = pd.DataFrame(features_train.contrast.values.tolist(),
                                  index=features_train.index)
    contrast_train = chroma_train.add_prefix('contrast_')

    # drop the old columns
    features_train = features_train.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast'], axis=1)

    # concatenate
    df_features_train = pd.concat([features_train, mfcc_train, spectro_train, chroma_train, contrast_train],
                                  axis=1, join='inner')

    targets_train = []
    for name in df_features_train.index.tolist():
        targets_train.append(instrument_code(name))

    df_features_train['targets'] = targets_train

    # save the dataframe to a pickle file
    with open('C:/Users/dmarcu/Desktop/nsynth/DataWrangling/df_features_train.pickle', 'wb+') as f:
        pickle.dump(df_features_train, f)


    # extract features validation set @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
    if (extractValidationFeatures):
        start_valid = time.time()

        # create dictionary to store all test features
        dict_valid = {}
        # loop over every file in the list
        i = 0
        for file in filenames_valid:
            i += 1
            print("Exctracting from validation file number # " + str(i))
            # extract the features
            features = feature_extract(valid_dir + file + '.wav')  # specify directory and .wav
            # add dictionary entry
            dict_valid[file] = features

        end_valid = time.time()
        print('Time to extract {} files is {} seconds'.format(len(filenames_valid), end_valid - start_valid))

        # convert dict to dataframe
        features_valid = pd.DataFrame.from_dict(dict_valid, orient='index',
                                                columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast'])

        # extract mfccs
        mfcc_valid = pd.DataFrame(features_valid.mfcc.values.tolist(),
                                  index=features_valid.index)
        mfcc_valid = mfcc_valid.add_prefix('mfcc_')

        # extract spectro
        spectro_valid = pd.DataFrame(features_valid.spectro.values.tolist(),
                                     index=features_valid.index)
        spectro_valid = spectro_valid.add_prefix('spectro_')

        # extract chroma
        chroma_valid = pd.DataFrame(features_valid.chroma.values.tolist(),
                                    index=features_valid.index)
        chroma_valid = chroma_valid.add_prefix('chroma_')

        # # extract contrast
        contrast_valid = pd.DataFrame(features_valid.contrast.values.tolist(),
                                      index=features_valid.index)
        contrast_valid = chroma_valid.add_prefix('contrast_')

        # drop the old columns
        features_valid = features_valid.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast'], axis=1)


        # concatenate
        df_features_valid = pd.concat([features_valid, mfcc_valid, spectro_valid, chroma_valid, contrast_valid],
                                      axis=1, join='inner')

        targets_valid = []
        for name in df_features_valid.index.tolist():
            targets_valid.append(instrument_code(name))

        df_features_valid['targets'] = targets_valid

        # save the dataframe to a pickle file
        with open('C:/Users/dmarcu/Desktop/nsynth/DataWrangling/df_features_valid.pickle', 'wb+') as f:
            pickle.dump(df_features_valid, f)
