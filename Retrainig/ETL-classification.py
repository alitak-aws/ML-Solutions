import os, io, glob, csv, json, sys
import boto3
from awsglue.utils import getResolvedOptions
import argparse

import numpy as np
import pandas as pd

import tarfile
import scipy.sparse as sp
from scipy.sparse import hstack
from time import gmtime, strftime
import collections
from collections import Counter
import re
import unicodedata
import six

import doc2table
import utils
import tokenization


import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# nlp = spacy.load("en_core_web_sm", disable=['ner','parser'])


s3 = boto3.client('s3')

args = getResolvedOptions(sys.argv, ['Bucket_source', 'Bucket_target', 'isNew'])

bucket_source = args['Bucket_source'] 
bucket_target = args['Bucket_target'] 
isNew = args['isNew'] 


jsonl_key = 'test.jsonl'   

NGRAM = 2
TABLE_TYPE = 'tfidf'
TOKENIZER = 'wordpiece'
BOW_FEATURE_PREFIX = f'data'

# Download full text corpus
s3.download_file(bucket_source, f'Raw-data/{jsonl_key}', os.path.basename('test.jsonl'))

# import vocab.txt
s3.download_file(bucket_source, 'Raw-data/vocab.txt', os.path.basename('vocab.txt'))
# WordPiece_tokenizer = tokenization.FullTokenizer(vocab_file='vocab.txt', do_lower_case=True)

#     transform to count table and tfidf table
d2t = doc2table.doc2tables(ngram=NGRAM, 
                          files=open(os.path.basename(jsonl_key),'r'), 
                          tokenizer=TOKENIZER
                          )
d2t.get_table();
print('finished processing Jsonlfile')

# save features
local_feature_file='BOW_feature.npz' #####
np.savez(
    local_feature_file,
    count=d2t.count_matrix,
    tfidf=d2t.tfidf_matrix,
    binary=d2t.binary_matrix,
    meta={
        'Term2idx':d2t.Term2idx,
        'Terms': d2t.Terms,
        'Doc2idx':d2t.Doc2idx,
        'Docs': d2t.Docs,
        'ngram':d2t.ngram,
        'Ns': d2t.Ns,
    }
)

s3.upload_file(
    local_feature_file,
    bucket_target,
    f'Raw-data/features.npz'
)


# Create train/test data
#     master_split_prefix = 'Mastersplit-{}'.format(DATASET)

CLF_COLS = [
        'label_1'# , 'label_2', 'label_3', 'label_4', 'label_5'
]

# metric_definitions = [
#     {"Name": "precision_macro", "Regex": "'precision_macro': ([0-9\\.]+)"},
#     {"Name": "precision_micro", "Regex": "'precision_micro': ([0-9\\.]+)"},
#     {"Name": "recall_macro", "Regex": "'recall_macro': ([0-9\\.]+)"},
#     {"Name": "recall_micro", "Regex": "'recall_micro': ([0-9\\.]+)"},
#     {"Name": "f1_macro", "Regex": "'f1_macro': ([0-9\\.]+)"},
#     {"Name": "f1_micro", "Regex": "'f1_micro': ([0-9\\.]+)"},
#     {"Name": "accuracy", "Regex": "'accuracy': ([0-9\\.]+)"},
# ]


s3.download_file(bucket_source, f'Raw-data/{isNew}_data_train.csv', os.path.basename(f'meta_data_train.csv'))
s3.download_file(bucket_source, f'Raw-data/{isNew}_data_test.csv', os.path.basename(f'meta_data_test.csv'))
s3.download_file(bucket_source, f'Raw-data/dic.json', os.path.basename(f'dic.json'))
s3.download_file(bucket_source, f'Raw-data/vocab.txt', os.path.basename(f'vocab.txt'))

# Upload dic.json, vocab.txt
print('copying dic.json')
s3.upload_file('dic.json', bucket_target,'Raw-data/dic.json')
s3.upload_file('vocab.txt',bucket_target,'Raw-data/vocab.txt')

for col in CLF_COLS:

    print(f'Preparing data for {col} classifier...')

    train_y = pd.read_csv(f'meta_data_train.csv', dtype={col: int}, usecols=['file_name',col] )
    test_y = pd.read_csv(f'meta_data_test.csv', dtype={col: int}, usecols=['file_name',col] )

    n_classes = len(set(train_y[col].unique()).union(set(test_y[col].unique())))

    print(f'{col} has {n_classes} classes.')

    # x table
    x_file = 'BOW_feature.npz'

    xdata = np.load(x_file, allow_pickle=True)
    x_table = xdata[TABLE_TYPE].item()
    Doc2idx = xdata['meta'].item()['Doc2idx']

    train_x = x_table[[Doc2idx[docname] for docname in train_y['file_name'].to_numpy()], :]
    train_data = hstack((np.array(train_y[col])[:,None], train_x)).tocsr()

    test_x  = x_table[[Doc2idx[docname] for docname in  test_y['file_name'].to_numpy()], :]
    test_data = hstack((np.array(test_y[col])[:,None], test_x)).tocsr()

    # Save npz
    sp.save_npz(f'{col}_train.npz', train_data)
    sp.save_npz(f'{col}_test.npz', test_data)

    # Upload npz
    s3.upload_file(
        f'{col}_train.npz',
        bucket_target,
        f'features_{isNew}/train_eval/{col}/data/npz/train/train.npz'
    )
    s3.upload_file(
        f'{col}_test.npz',
        bucket_target,
        f'features_{isNew}/train_eval/{col}/data/npz/validation/validation.npz'
    )
    


    print(f'{col} data uploaded to s3.')