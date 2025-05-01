"""
Testing the presence of 'Appeal to Ignorance' fallacy 
- no context is provided
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

# Group directly by Conversation ID
grouped = df.groupby("Conversation ID")

output_file = "AppealToIgnorance_results.csv"
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
            prompt = f"""You are a logical reasoning expert trained to identify fallacies in short social media posts.

    Your task is to determine whether the following tweet commits the **Appeal to Ignorance Fallacy** —
    a fallacy that occurs when someone claims something is true simply because it hasn't been proven false,
    or claims it's false simply because it hasn't been proven true.

    This fallacy typically misuses the absence of evidence as if it were concrete evidence.

    --- Common signs of this fallacy ---
    - "There's no proof it didn't happen, so it did."
    - "You can't disprove my claim, so it must be valid."
    - "Since no one has shown the opposite, this must be true."

    --- Example of an Appeal to Ignorance Fallacy ---
    Tweet: "Or maybe the exit polls were manipulated like many of the polls which predicted an overwhelming victory for HRC"
    Explanation: The tweet suggests that exit polls may have been manipulated not based on evidence,
    but on the fact that previous polls were wrong — treating a lack of disproof or a past failure as sufficient 
    reason for current suspicion.

    --- Example of a tweet that is NOT an Appeal to Ignorance Fallacy ---
    Tweet: "Polls can be inaccurate due to flawed methodologies, so it's important to consider margins of error."
    Explanation: This tweet critiques polling based on method and evidence, not due to absence of disproof.

    Now analyze the following tweet:
    Tweet: "{tweet}"

    Does this tweet commit the Appeal to Ignorance Fallacy?
    If yes, explain clearly why it fits this fallacy.
    If no, explain why not and what reasoning is being used instead.
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
        pd.DataFrame(fallacy_results, columns=["Tweet", " Results"]).to_csv(output_file, index=False)

        # Wait 3 seconds between tweets
        time.sleep(3)

except KeyboardInterrupt:
    print("\nManual interrupt received (Ctrl+C). Saving progress and exiting...")

except Exception as e:
    print(f"\nUnexpected error: {e}. Saving progress and exiting...")

finally:
    print("\nFinal save of results before exit.")
    pd.DataFrame(fallacy_results, columns=["Tweet", " Results"]).to_csv(output_file, index=False)