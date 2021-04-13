# hatespeech-semantic-shift

This project aims to trace semantic shifts of hate speech during the COVID-19 pandemic. It uses contextualized embeddings for time-specific word representation, and trace the semantic difference using cosine similarity. 

 There are four main folders in this repository:

**Data**: This folder contains hate terms extracted from [Hatebase](https://hatebase.org/), which contains 1,500 English hate terms which divides terms based on the intrinsic characteristics of a group of people. There are seven different hate classes. 

Note that we did not include the Twitter corpus that we used due to the memory limitation. Data is available in [Panacea lab](http://www.panacealab.org/covid19/). 

**Notebook**: This folder contains analysis of the dataset we used for our project. There are three files, 

tweet_analysis.ipynb :  It plots the number of tweets in each predefined time slice. 
analyze_freq.ipynb: It plots the frequency of hate terms in each hate class. It also shows other detailed analysis. 
final_analysis.ipynb: It contains final analysis that traces the semantic distance from the start time period. 

**output**: It contains plots for cosine distance of each word used in our analysis, average cosine distance per group, frequency plots that are used for our analysis.

**scripts**: It contains python scripts that are used as our main analysis. 
