import pandas as pd
from nltk import word_tokenize
from pathlib import Path
from nltk.util import ngrams
from IPython.display import display, HTML
import pickle
import argparse
import nltk
import os
import json

# Get topics from LDA, Combine 9 chunkcs. 
def get_terms(hatespeech_dir):

	terms_chunks = []

	for i in range(1,16):
		fn = Path(hatespeech_dir, "hatebase_eng{}".format(i))
		with open(fn, 'rb') as f:
			data = json.loads(f.read())
			terms_chunks.append(data['result'])
	
	all_terms = pd.concat(pd.DataFrame(item) for item in terms_chunks)
			
	
	return all_terms
		
# Filter tweets which contains topics
# Output Dictionary (key:topic, value:dataframe that contains topic)
def filter_tweet(hate_df, tweet_df_tokens):
	assert 'tokens' in tweet_df_tokens.columns
	terms = list(hate_df['term'])

	hate_dict = {}
	for term in terms:
		hate_dict[term] = pd.DataFrame()

	for i, row in tweet_df_tokens.iterrows():
		tokens = row['tokens']
		for term in terms:
			if term in tokens:
				hate_dict[term] = hate_dict[term].append(row, ignore_index=True)
	
	return hate_dict

# Tokenize cleaned_text into bigram as topics only include unigram tokens. 
def tokenize_text(tweet_df):
	s = pd.Series([], dtype='object')
	for idx, row in tweet_df.iterrows():
		text = row['cleaned_text']
		try: 
			tokens = nltk.word_tokenize(text)
		except:
			tokens = []
		# bigrams_tuple = list(ngrams(token, 2)) 
		# bigrams = ["{}_{}".format(bigram[0], bigram[1]).lower() for bigram in bigrams_tuple]
		s[idx] = tokens
	
	tweet_df.insert(2, "tokens", s)
	return tweet_df

# Filter df that includes topics and save dictionary as pickel files. 
def filter_tweet_topic(args):

	Path(args.export).mkdir(exist_ok=True)
	Path(args.export, 'mapping').mkdir(exist_ok=True)

	export_fn = Path(args.export, 'days_{}_unigram.csv'.format(args.days))

	if os.path.exists(export_fn):
		print("days_{}_unigram.csv file already exists".format(args.days))
		tokenized_tweet_df = pd.read_csv(export_fn)
	else:
		fn = Path(args.tweet_dir, "tweet_{}day.csv".format(args.days))
		with open(fn, "r") as f:
			tweet_df = pd.read_csv(fn)

		tokenized_tweet_df = tokenize_text(tweet_df)

		export_fn = Path(args.export, 'days_{}_unigram.csv'.format(args.days))
		tokenized_tweet_df.to_csv(export_fn, index=False)

	mapping_fn = Path(args.export, 'mapping/mapping_days_{}.pkl'.format(args.days))
	if not os.path.exists(mapping_fn):
		terms = get_terms(args.hatebase_dir)
		terms_dict = filter_tweet(terms, tokenized_tweet_df)
	
		with open(mapping_fn, 'wb') as f:
			pickle.dump(terms_dict, f)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--tweet_dir", type=str, default="../../data/tweets_with_mfd_scores")
	parser.add_argument("--days", type=int, default=0, choices=range(0,444))
	parser.add_argument("--hatebase_dir", type=str, default="../../data/topics")
	parser.add_argument("--export", type=str, default="../../data/mapping")
	args = parser.parse_args()

	print(args)
	filter_tweet_topic(args)

