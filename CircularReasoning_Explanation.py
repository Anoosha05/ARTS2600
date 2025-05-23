"""
Testing out the presence of Circular Reasoning in the first 100 tweets using an explanation prompt
from HugChat API. No context is provided to the model. 
"""

import pandas as pd
import time
from hugchat import hugchat
from hugchat.login import Login 
from hugchat.exceptions import ChatError
import os 
from tqdm import tqdm 

EMAIL = " "
PASSWD = " "
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

# Group directly by Conversation ID
grouped = df.groupby("Conversation ID")

output_file = "CircularReasoning_Explanation_results.csv"
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
            prompt = f"""You're an expert in logical reasoning. Does this tweet use a Circular Reasoning fallacy? Answer Yes or No, and explain.
Tweet: "{tweet}"
Results:"""
            response = chatbot.chat(prompt)
            result_text = str(response)
            return [tweet, result_text]
        except ChatError as e:
            if "429" in str(e):
                print(f"Rate limit hit at idx {idx}. Waiting 60 seconds...")
                time.sleep(60)
            elif "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
                print(f"Network error at idx {idx}. Waiting 30 seconds before retrying...")
                time.sleep(30)
            else:
                print(f"ChatError at idx {idx}: {e}")
                break
        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            break
    return [tweet, "Error"]


# Added a try/except block to save the progress of the script in case of an error.
try:
    # Loop through tweets
    for idx in tqdm(range(len(tweets)), desc="Processing Tweets"):
        if idx in processed_indices:
            continue
        
        tweet = tweets.iloc[idx]
        result = process_tweet(tweet, idx)
        fallacy_results.append(result)

        # Save after each tweet
        pd.DataFrame(fallacy_results, columns=["Tweet", "Explanation Results"]).to_csv(output_file, index=False)

        # Wait 3 seconds between tweets
        time.sleep(3)

except KeyboardInterrupt:
    print("\nManual interrupt received (Ctrl+C). Saving progress and exiting...")

except Exception as e:
    print(f"\nUnexpected error: {e}. Saving progress and exiting...")

finally:
    print("\nFinal save of results before exit.")
    pd.DataFrame(fallacy_results, columns=["Tweet", "Explanation Results"]).to_csv(output_file, index=False)
