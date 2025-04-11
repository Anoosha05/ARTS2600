"""
can't run on the whole dataset because we are 
running a very heavy transformer model (BART-large) locally on MPS
which is way slower than a proper GPU (like those on Colab or in a datacenter).
"""

from transformers import pipeline
import pandas as pd
import os
import re
from tqdm import tqdm

os.chdir("/Users/anooshamallakanti/Desktop/twitter_data")
df = pd.read_csv("Longest_Conversation(Justin).csv")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

fallacy_labels = [ 'strawman', 
    'ad hominem', 
    'False Equivalence', 
    'slippery slope', 
    'appeal to emotion', 
    'fallacy cause/post hoc', 
    'red herring', 
    'hasty generalization', 
    'false dilemma' ,
    'circular reasoning', 
    'bandwagon', 
    'tu quoque', 
    'anecdotal' ,
    'moral equivalence', 
    'appeal to ignorance', 
    'no fallacy'
]

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"…", "", text)        # remove unicode ellipsis
    return text.strip()

batch_size = 4
results = []
# Process only the first 10 tweets
tweets = df["Tweet contents"].head(10).astype(str).tolist()

for tweet in tqdm(tweets):
    try:
        result = classifier(clean_text(tweet), candidate_labels=fallacy_labels, multi_label=False)
        results.append(result['labels'][0])
    except Exception as e:
        print(f"Error on: {tweet[:60]}... | {e}")
        results.append("error")

# After processing, make sure the lengths match
df_subset = df.head(10).copy()  # Only keep the first 10 rows in the subset
df_subset["fallacy_type"] = results
df_subset.to_csv("fallacy_labeled_subset.csv", index=False)