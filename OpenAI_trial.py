"""
Splitting into batches and using openai API 
"""


import openai 
from transformers import pipeline
import pandas as pd
import re
from tqdm import tqdm


openai.api_key = "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"

df = pd.read_csv("Longest_Conversation(Justin).csv")

# Labels for fallacies
fallacy_labels = [
    'strawman', 'ad hominem', 'False Equivalence', 'slippery slope',
    'appeal to emotion', 'fallacy cause/post hoc', 'red herring', 'hasty generalization',
    'false dilemma', 'circular reasoning', 'bandwagon', 'tu quoque',
    'anecdotal', 'moral equivalence', 'appeal to ignorance', 'no fallacy'
]

# Clean function
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))  # remove URLs
    text = re.sub(r"…", "", text)            # remove unicode ellipsis
    return text.strip()


def classify_tweet(tweet):
    try:
        # OpenAI API call to GPT-3 (using old v0.28 syntax)
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can use any available model like text-davinci-003
            prompt=f"Identify the logical fallacy in this statement: {tweet}",
            max_tokens=60,
            temperature=0.0
        )
        # Extract the response text (the identified fallacy)
        fallacy = response['choices'][0]['text'].strip()
        return fallacy
    except Exception as e:
        print(f"Error processing tweet: {e}")
        return "error"

batch_size = 16  # Adjust batch size as needed
results = []

# Iterate through the tweets in batches
for i in tqdm(range(0, len(df), batch_size)):
    batch = df[i:i+batch_size]
    for tweet in batch["Tweet contents"].astype(str).tolist():
        cleaned_tweet = clean_text(tweet)
        fallacy = classify_tweet(cleaned_tweet)
        results.append(fallacy)
    
    print(f"Processed batch {i // batch_size + 1}")

# Ensure the results length matches the dataframe
df = df.iloc[:len(results)]
df["fallacy_type"] = results
df.to_csv("fallacies_batches_gpt.csv", index=False)
