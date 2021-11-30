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
""" Preprocess a set of documents into Bag of words table """

from utils import rawtext_2_ngrams,wordpiece_2_ngrams
import scipy.sparse as sp
import numpy as np
from collections import Counter
import pandas as pd
import json

class doc2tables(object):
    """
    this class transform a set of documents into count_table, binary table and tfidf_table
    """

    def __init__(self,ngram,files, tokenizer):
        """
        Constructor

        Args:
            ngram (int): number of grams to construct
            files (object): a file-like object to read jsonl file
            tokenizer (function): a function to convert raw text to grams
        """

        self.Term2idx = {}
        self.Terms=[]
        self.Doc2idx={}
        self.Docs = []
        self.ngram = ngram
        self.files = files
        if tokenizer=='simple':
            self.tokenizer = rawtext_2_ngrams
        elif tokenizer =='wordpiece':
            self.tokenizer = wordpiece_2_ngrams
        else:
            raise TypeError('unknown tokenizer')

    def count(self,text, docidx):
        """ get the gram counts of text of a document """

        print('processing... {}'.format( self.Docs[docidx] ))
        grams = self.tokenizer(text,ngram=self.ngram)
        gram_idx = []
        #update Terms and Term2idx
        for gram in grams:
            if gram in self.Term2idx:
                gram_idx.append( self.Term2idx[gram] )
            else:
                self.Terms.append(gram)
                self.Term2idx[gram]=len(self.Terms)-1
                gram_idx.append( self.Term2idx[gram] )

        #count
        counts = Counter(gram_idx)

        # Return in sparse matrix data format.
        row = counts.keys()
        col = [docidx] * len(counts)
        data =counts.values()
        return row, col, data

    def get_count_matrix(self, files):
        """ get the gram count matrix of whole document set """

        row, col, data = [], [], []

        for line in files:
            if len( line.strip() ) == 0:
                continue
            docid, text = list(json.loads(line.strip()).items())[0]
            docid = str(docid)
            text = str(text)
            text = text + '. ' + docid.replace('_',' ').replace('-',' ')
            # update Docs and Docs2idx
            if docid in self.Doc2idx:
                docidx = self.Doc2idx[docid]
            else:
                self.Docs.append(docid)
                self.Doc2idx[docid] = len(self.Docs)-1
                docidx = self.Doc2idx[docid]

            p_row, p_col, p_data = self.count(text,docidx)
            row.extend(p_row)
            col.extend(p_col)
            data.extend(p_data)

        count_matrix = sp.csr_matrix(
            (data, (row, col)), shape=(len(self.Terms), len(self.Docs))
        )

        assert len(self.Terms)==len(self.Term2idx.keys())
        assert len(self.Docs) == len(self.Doc2idx.keys())

        return count_matrix

    def get_tfidf_matrix(self,cnts):
        """Convert the word count matrix into tfidf one.

            tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
            * tf = term frequency in document
            * N = number of documents
            * Nt = number of occurences of term in all documents
            """
        # document frequency
        binary = (cnts > 0).astype(int)
        Ns = np.array(binary.sum(1)).squeeze()

        idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0
        idfs = sp.diags(idfs, 0)
        tfs = cnts.log1p()
        tfidfs = idfs.dot(tfs)
        return tfidfs, binary, Ns

    def get_table(self):
        """ compute count matrix, tfidf matrix and binary matrix of whole document set """

        count_matrix = self.get_count_matrix(self.files)
        tfidf_matrix, binary_matrix, self.Ns = self.get_tfidf_matrix(count_matrix)
        
        self.count_matrix = count_matrix.transpose()
        self.tfidf_matrix = tfidf_matrix.transpose()
        self.binary_matrix = binary_matrix.transpose()
        
        return

        #count_table = pd.DataFrame.sparse.from_spmatrix(count_matrix)
        #tfidf_table = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)

        #for table in [count_table,tfidf_table]:
        #    table.columns = self.Docs
        #    table.index = self.Terms

        #return count_table, tfidf_table