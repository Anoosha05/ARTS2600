"""
This script uses Hugging Face's transformers library and a text-generation pipeline with the Phi-3.5-mini-instruct 
model to detect Ad hominem fallacies in tweets. 
Sampled run of only the first 10 tweets with a few shot technique. 

authors: Anoosha Mallakanti, Prof. Tristram Alexander 
"""

import pandas as pd
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch 

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map = "cuda", 
    torch_dtype="auto",
    attn_implementation="flash_attention_2", 
    trust_remote_code=True
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

pipe = pipeline(
    "text-generation", 
    model= model, 
    tokenizer = tokenizer,
    )

df = pd.read_csv("Longest_Conversation(Justin).csv")
tweets = df["Tweet contents"][:10].values 
results = []    

generation_args = {
    "max_new_tokens": 500,
    "temperature": None,
    "do_sample": False,
    "return_full_text": False,
}


idx = 0
for tweet in tweets:
    print(f"Processing tweet {idx +1}/{len(tweets)}")

    messages = [
        {"role": "system", "content": "You are an expert in logical reasoning and fallacy detection"},
        {"role": "user", "content": "An Ad Hominem fallacy occurs when someone attacks the person making an argument "
        "rather than addressing the argument itself. It attempts to undermine the argument by attacking the character, "
        "motive, or other attributes of the person instead of addressing their reasoning.Answer with 'Yes' if the tweet "
        "contains an Ad Hominem fallacy or 'No' if it doesn't. "
        "Example of Ad Hominem fallacies:" 
        "Don't listen to Dr. Smith's climate research - he drives an SUV, so he's a hypocrite."

        "Examples that are NOT Ad Hominem fallacies: "
        "I disagree with the senator's proposal because it would increase the deficit by 20%."}, 

        {"role": "assistant", "content": "I'll analyze the tweet you provide and determine if it contains an Ad Hominem fallacy. "
        "I'll respond with **Yes** if it does or **No** if it doesn't, based on whether the tweet attacks the person making an "
        "argument rather than addressing the substance of their argument."},
        {"role": "user", "content": f"Message: {tweet}"},
    ]

    output = pipe(messages, **generation_args, use_cache=False)
    idx+=1
    reply = output[0]['generated_text'].strip()

    results.append(
        {"Tweet": tweet,
        "Fallacy Evaluation": reply
        })
    time.sleep(2)  
    print()

output_df = pd.DataFrame(results)
output_df.to_csv("Ad_hominem_results_sample.csv", index=False)