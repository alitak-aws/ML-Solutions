# coding=utf-8
# Copyright 2020 AWS Data Scientist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" This script contains helper tokenizer functions for getting count matrix. """

import os
import spacy
import unicodedata
import regex
import tokenization
import boto3

import en_core_web_sm
nlp = en_core_web_sm.load()

s3 = boto3.client('s3')
bucket = 'iqvia-blog'
s3.download_file(bucket, 'Raw-data/vocab.txt', os.path.basename('vocab.txt'))

WordPiece_tokenizer = tokenization.FullTokenizer(vocab_file='vocab.txt',do_lower_case=True)


# nlp = spacy.load("en_core_web_sm", disable=['ner','parser'])
nlp.max_length = 15000000

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}

for stopword in STOPWORDS:
    nlp.vocab[u"{}".format(stopword)].is_stop = True
    #nlp.vocab[stopword].is_stop = True

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def filter_token(token):
    """Take out english stopwords, punctuation, and compound endings."""
    text = normalize(token.text)
    if regex.match(r'^\p{P}+$', text):
        return True
    if regex.match(r'[\p{Z}\p{C}]', text):
        return True
    if token.is_stop:
        return True
    return False


def rawtext_2_ngrams(text, ngram):
    """ convert a raw text into a list of grams using
    simple tokenizer
     """
    #normalize text
    text = normalize(text)
    #lower case
    text = text.lower()
    # tokenize
    doc = nlp(text)
    grams = []

    # remove stop words, punctuation and get lemma
    for s in range(len(doc)):
        for e in range(s, min(s + ngram, len(doc))):
            tokens = doc[s:e + 1]
            if True not in [filter_token(token) for token in tokens]:
                grams.append( ' '.join( [ token.lemma_ for token in tokens ] ) )

    return grams

def wordpiece_2_ngrams(text, ngram):
    """ convert a piece of raw text into a list of grams
     using wordpiece tokenizer
     """
    doc = WordPiece_tokenizer.tokenize(text)
    grams = []

    for s in range(len(doc)):
        for e in range(s, min(s + ngram, len(doc))):
            tokens = doc[s:e + 1]
            grams.append( ' '.join( [ token for token in tokens ] ) )

    return grams






