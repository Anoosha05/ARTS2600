"""
trying to run it using the HugginFace's model 
"""

from transformers import pipeline
import pandas as pd
import re
from tqdm import tqdm


df = pd.read_csv("Longest_Conversation(Justin).csv")
classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")

fallacy_labels = [
    'strawman', 'ad hominem', 'False Equivalence', 'slippery slope',
    'appeal to emotion', 'fallacy cause/post hoc', 'red herring', 'hasty generalization',
    'false dilemma', 'circular reasoning', 'bandwagon', 'tu quoque',
    'anecdotal', 'moral equivalence', 'appeal to ignorance', 'no fallacy'
]

def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))  # Remove URLs
    text = re.sub(r"…", "", text)  # Remove ellipsis
    return text.strip()

# Run the classification in batches to prevent overloading the system
batch_size = 16
results = []

for i in tqdm(range(0, len(df), batch_size)):
    batch = df["Tweet contents"][i:i+batch_size].astype(str).tolist()  # Process tweets in batches
    try:
        outputs = classifier(batch, candidate_labels=fallacy_labels, multi_label=False)
        for output in outputs:
            results.append(output["labels"][0])  # Store the most likely label
    except Exception as e:
        print(f"Error in batch {i}: {e}")
        results.extend(["error"] * len(batch))

# Ensure the results have the same length as the number of tweets
df = df.iloc[:len(results)]
df["fallacy_type"] = results

df.to_csv("fallacies_huggingface.csv", index=False)
