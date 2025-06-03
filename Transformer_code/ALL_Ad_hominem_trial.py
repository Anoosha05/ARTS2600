"""
This script uses Hugging Face's transformers library and a text-generation pipeline with the Phi-3.5-mini-instruct 
model to detect Ad hominem fallacies in all 17000+ tweets. 
Authors: Anoosha Mallakanti, Prof. Tristram Alexander 
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
tweets = df["Tweet contents"].values 

start_index =  10111
tweets = tweets[start_index :]   

results = []

generation_args = {
    "max_new_tokens": 500,
    "temperature": None,
    "do_sample": False,
    "return_full_text": False,
}


idx = start_index
for tweet in tweets:
    print(f"Processing tweet {idx +1}/{len(tweets)}") # should have changed the len(tweets) to total count 

    messages = [
        {"role": "system", "content": "You are an expert in logical reasoning and fallacy detection"},
        {"role": "user", "content": f"An Ad Hominem fallacy occurs when someone attacks the person making an argument "
        f"rather than addressing the argument itself. Answer with **Yes** if the tweet contains an Ad Hominem fallacy "
        f"or **No** if it doesn't. Message: {tweet}"}, 
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
    
    # Save intermediate result
    pd.DataFrame(results).to_csv("ALL_Ad_hominem_results_5.csv", index=False)

