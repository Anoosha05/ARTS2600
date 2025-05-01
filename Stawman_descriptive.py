"""
Testing out the presence of Strawman in the first 100 tweets using the descriptive prompt
- save the progress as we go 
"""

import pandas as pd
import time
from hugchat import hugchat
from hugchat.login import Login 
from hugchat.exceptions import ChatError
import os 
from tqdm import tqdm 

EMAIL = "m.anoosha.lk@gmail.com"
PASSWD = "huggingFace@2005"
cookie_path_dir = "./cookies/" 
sign = Login(EMAIL, PASSWD)
cookies= sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)

# Initialize the HugChat bot with the obtained cookies
try:
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    print("Chatbot initialized successfully!")
except Exception as e:
    print(f"Error initializing chatbot: {e}")
    exit()


df = pd.read_csv("Longest_Conversation(Justin).csv")
tweets = df["Tweet contents"][:100]


output_file = "Strawman_descriptive_results.csv"
if os.path.exists(output_file):
    saved_df = pd.read_csv(output_file)
    processed_indices = set(saved_df.index)
    fallacy_results = saved_df.values.tolist()
    print(f"Resuming from existing file. {len(processed_indices)} tweets already processed.")
else:
    processed_indices = set()
    fallacy_results = []
    print("Starting fresh.")

def process_tweet(tweet, idx, retries=3):
    for attempt in range(retries):
        try:
            prompt = f"""You're an expert in logical reasoning. Does the tweet misrepresent someone's argument to make it easier to attack?
Tweet: "{tweet}"
Fallacy:"""
            response = chatbot.chat(prompt)
            result_text = str(response)
            return [tweet, result_text]
        except ChatError as e:
            if "429" in str(e):
                print(f"Rate limit hit at idx {idx}. Waiting 60 seconds...")
                time.sleep(60)
            else:
                print(f"ChatError at idx {idx}: {e}")
                break
        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            break
    return [tweet, "Error"]

# Loop through tweets
for idx in tqdm(range(len(tweets)), desc="Processing Tweets"):
    if idx in processed_indices:
        continue
    tweet = tweets.iloc[idx]
    result = process_tweet(tweet, idx)
    fallacy_results.append(result)

    # Save after each tweet
    pd.DataFrame(fallacy_results, columns=["Tweet", "Fallacy"]).to_csv(output_file, index=False)

    # Wait 3 seconds between tweets
    time.sleep(3)


