"""
Testing out the presence of Tu Quoque fallacy in the first 100 tweets using an descriptive prompt
from HugChat API. 
Changes made 
- Context provided using the conversation ID
- Few-shot learning Prompt 
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
df = df.head(100) 


output_file = "TuQuoque_descriptive_results.csv"
if os.path.exists(output_file):
    saved_df = pd.read_csv(output_file)
    processed_indices = set(saved_df.index)
    fallacy_results = saved_df.values.tolist()
    print(f"Resuming from existing file. {len(processed_indices)} tweets already processed.")
else:
    processed_indices = set()
    fallacy_results = []
    print("Starting fresh.")


# Prompt Template with Examples
def make_prompt(context_tweet, reply_tweet):
    return f"""
You're an expert in logical reasoning.

Your task is to identify whether the *Tu Quoque* fallacy is present in a Twitter exchange.
The Tu Quoque fallacy (also called "you too") is committed when someone deflects criticism 
by accusing the critic of the same or similar wrongdoing instead of addressing the argument.


Examples:

Example 1:
Person A: "You shouldn't be smoking; it's bad for your health."
Person B: "Well, you used to smoke in college!"
→ Tu Quoque: Yes. (Person B is deflecting the argument by attacking Person A.)

Example 2:
Person A: "I don't think cheating on taxes is right."
Person B: "Coming from someone who bragged about doing that last year?"
→ Tu Quoque: Yes.

Example 3:
Person A: "We need to reduce our carbon footprint."
Person B: "You drive an SUV every day. Practice what you preach."
→ Tu Quoque: Yes.

Example 4:
Person A: "You were late to the meeting."
Person B: "Yes, I was. Sorry about that."
→ Tu Quoque: No.

---

Now analyze the following exchange:
Person A: "{context_tweet}"
Person B replies: "{reply_tweet}"

Does Person B commit a Tu Quoque fallacy? Respond with "Yes" or "No", and briefly explain your reasoning.
Result:
""".strip()


# Get context within a conversation
def get_context(convo_df, current_index):
    current_row = convo_df.iloc[current_index]
    current_user = current_row['User']
    current_text = current_row['Tweet contents']

    tagged_users = [u.strip('@').lower() for u in current_text.split() if u.startswith('@')]

    context_candidates = convo_df.iloc[:current_index]
    context_tweet = None
    for i in reversed(range(len(context_candidates))):
        candidate = context_candidates.iloc[i]
        if candidate['User'].lower() in tagged_users:
            context_tweet = candidate['Tweet contents']
            break
    return context_tweet, current_text


# Process each tweet
def process_tweet_pair(context_tweet, reply_tweet, idx, retries=3):
    for attempt in range(retries):
        try:
            if context_tweet is None:
                return [reply_tweet, "No tagged context found. Skipping."]
            prompt = make_prompt(context_tweet, reply_tweet)
            response = chatbot.chat(prompt)
            return [reply_tweet, response]
        except ChatError as e:
            if "429" in str(e):
                print(f"Rate limit hit at idx {idx}. Waiting 60 seconds...")
                time.sleep(60)
            elif "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
                print(f"Network error at idx {idx}. Waiting 30 seconds...")
                time.sleep(30)
            else:
                print(f"ChatError at idx {idx}: {e}")
                break
        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            break
    return [reply_tweet, "Error"]



# Loop through grouped tweets
try:
    for convo_id, convo_df in tqdm(df.groupby("Conversation ID"), desc="Processing Conversations"):
        for i in range(1, len(convo_df)):
            idx = convo_df.index[i]
            if idx in processed_indices:
                continue

            context_tweet, reply_tweet = get_context(convo_df.reset_index(drop=True), i)
            result = process_tweet_pair(context_tweet, reply_tweet, idx)
            fallacy_results.append(result)

            pd.DataFrame(fallacy_results, columns=["Tweet", "TuQuoque Result"]).to_csv(output_file, index=False)
            time.sleep(3)

except KeyboardInterrupt:
    print("Manual interrupt received. Saving progress...")

except Exception as e:
    print(f"Unexpected error: {e}. Saving progress...")

finally:
    print("Final save before exit.")
    pd.DataFrame(fallacy_results, columns=["Tweet", "TuQuoque Result"]).to_csv(output_file, index=False)