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
# import gensim

import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()

import torch
import torchtext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction

# Collect hate speech
nfiles = 444
hb_dir = Path("/h/ypark/tweet_covid/hatespeech/hatebase")
tweet_dir = Path("/h/ypark/tweet_covid/output")
# and create a data frame

def load_speech(hb_dir, i):
	
	fn = Path(hb_dir, "hatebase_eng{}".format(i))
	
	words = {}
	try:
		with open(fn, 'rb') as f:
			words = json.loads(f.read())
	except:
		print("No file exists")
	
	return words['result']

def load_tweet(args):
	fn = Path(args.tweet_dir, "tweet_{}day.csv".format(args.days))
	# fn = Path(args.tweet_dir, "chunk_39.csv")
	with open(fn, 'rb') as f:
		tweet_df = pd.read_csv(fn)

	return tweet_df

def extract_features(corpus, export, days):
	'''Extract TF-IDF features from corpus'''

	stop_words = nltk.corpus.stopwords.words("english")

	model_fn = Path(export, "model/count_vec_{}.pkl".format(days))
	if os.path.exists(model_fn):
		with open(model_fn, 'rb') as f:
			data = pickle.load(f)
			cv, corpus = data[0], data[1]
		return cv, corpus

	print("Setting up count vectorizer")

	# vectorize means we turn non-numerical data into an array of numbers
	count_vectorizer = CountVectorizer(
		lowercase=False,  # for demonstration, True by default
		tokenizer=nltk.word_tokenize,  # use the NLTK tokenizer
		min_df=5,  # minimum document frequency, i.e. the word must appear more than once.
		ngram_range=(1, 2),
		stop_words=stop_words
	)

	print("Processing corpus")
	processed_corpus = count_vectorizer.fit_transform(corpus)
	print("Processing corpus2")
	processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(
		processed_corpus)

	return count_vectorizer, processed_corpus 

def save_model(data, export, days):

	
	model_pkl_fn = Path(export, "model/count_vec_{}.pkl".format(days))
	if os.path.exists(model_pkl_fn):
		return

	with open(model_pkl_fn, 'wb') as f:
		pickle.dump(data, f)

	print("Done saving a model")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument("--data_dir", type=str, default="../../data/panacealab_covid_tweets")
	parser.add_argument("--days", type=int, default=30)
	parser.add_argument("--export", type=str, default="/h/ypark/tweet_covid/hatespeech")
	parser.add_argument("--tweet_dir", type=str, default = "/h/ypark/tweet_covid/output") 
	parser.add_argument("--hb_dir", type=str, default = "/h/ypark/tweet_covid/hatespeech/hatebase") 

	args = parser.parse_args()

	Path(args.export, "model").mkdir(exist_ok=True)

	hate_df = pd.DataFrame()
	for i in range(1,16):
		df = load_speech(args.hb_dir, i)    
		df = pd.DataFrame(df) 
		hate_df = hate_df.append(df)

	tweet_df = load_tweet(args)

	tweet_text = tweet_df['cleaned_text'].tolist()
	cv, X = extract_features(tweet_text, args.export, args.days)

	save_model([cv, X], args.export, args.days)







