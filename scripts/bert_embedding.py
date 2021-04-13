# Import Libraries
import os
from pathlib import Path
import argparse
import datetime, dateutil
import json
import pickle

import numpy as np
import pandas as pd
import csv

import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()

import torch
import torchtext

from transformers import AutoModel, AutoTokenizer, AutoConfig

device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tweet_df = '/h/ypark/tweet_covid/hatespeech/hatebase/output'

class BERT:
    def __init__(self):
        self.config = AutoConfig.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', output_hidden_states=True)
        self.model = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', config=self.config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        
        self.unknown_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)

    def fine_tune(self):
        return
    
    def get_model(self):
        return self.model, self.tokenizer

    def check_unknown(self, word):
        first_token = self.tokenizer.encode(word)[1]
        # check = self.tokenizer.convert_tokens_to_ids(word) == self.unknown_id
        check = first_token == self.unknown_id
        if check:
            print("!!! Warning !!! Word: {} cannot be processed.".format(word))
        return check

    def get_word_rep(self, word, word_df):
        sentences = list(word_df['cleaned_text'])
        bsz = len(sentences)
        
        # tokenized_texts = [self.tokenizer.tokenize(s) for s in sentences]

        # processed_texts = []
        # word_idx = []

        word_idx = []
        ids = []
        word_id = self.tokenizer.encode(word)[1]
        sentences_ids = [self.tokenizer.encode(s) for s in sentences]

        # for idx, tokens in enumerate(tokenized_texts):
        #   first_token = self.tokenizer.convert_ids_to_tokens((self.tokenizer.encode(word)))
        #   if first_token in tokens:
        #       word_idx.append(tokens.index(first_token))
        #       processed_texts.append(tokenized_texts[idx])
        #   else:
        #       print("Not in tokenizer")

        for idx, sen_ids in enumerate(sentences_ids):
            if word_id in sen_ids:
                word_idx.append(sen_ids.index(word_id))
                ids.append(sentences_ids[idx])
            else:
                print("Not in tokenizer")
        
        # ids = [self.tokenizer.convert_tokens_to_ids(s) for s in processed_texts]

        all_lens = np.clip([len(s) for s in ids], 0, 510)
        if len(all_lens) == 0:
            # raise ValueError
            print("No text to process")
            return None

        maxlen = max(all_lens)
        padded_ids = np.zeros((bsz, maxlen))
        for i in range(bsz):
            padded_ids[i][:all_lens[i]] = ids[i]
        padded_ids = torch.tensor(padded_ids).int().to(device)

        with torch.no_grad():
            raw_outputs = self.model(padded_ids)[0]

        embedded = [raw_outputs[i][word_idx[i]] for i in range(bsz)] if word_idx[i] >= 0 else None
        embedded_mean = torch.stack(embedded, dim=0).mean(dim = 0)

        return embedded_mean

def get_mapping(args):
    fn = Path(args.unigram_dir, 'mapping_chunk_{}.pkl'.format(args.chunk_id))
    with open(fn, 'rb') as f:
        mapping = pickle.load(f)
    
    return mapping

def main(args):

    # Load dictionary (key: hatespeech, value: dataframe)
    mapping = get_mapping(args)
    model = BERT()  

    result = {}

    hateterms = mapping.keys()
    hateterms = list(hateterms)
    for word in hateterms:
        mapping_word = mapping[word]
        if not mapping[word].empty and not model.check_unknown(word.lower()):
            print('****** Word: {} in progress ******'.format(word))
            gb = mapping_word.groupby('day')
            mapping_by_days = [gb.get_group(x) for x in gb.groups]
            word_by_days = {}
            for group in mapping_by_days:
                day = group['day'].iloc[0]
                print("\t Day {} ... ".format(day))
                embedding = model.get_word_rep(word, group)
                if embedding is not None:
                    word_by_days[day] = embedding
            result[word] = word_by_days
    
            # embedding = model.get_word_rep(word, mapping_word)
            # embedding_s = pd.Series(embedding, dtype='object')
            # mapping_word.insert(0, 'embedding', embedding_s)
            # result[word] = mapping_word

    Path(args.export, "embedding").mkdir(exist_ok=True)
    export_fn = Path(args.export, "embedding/chunk_{}_embedding.pkl".format(args.chunk_id))

    with open(export_fn, 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", typel=str, defaut="../../data/panacealab_covid_tweets")
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--export", type=str, default="/h/ypark/tweet_covid/hatespeech/output")
    parser.add_argument("--unigram_dir", type=str, default = '/h/ypark/tweet_covid/output/unigrams/mapping') 

    args = parser.parse_args()

    main(args)

    






