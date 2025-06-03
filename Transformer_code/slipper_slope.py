"""
This script uses Hugging Face's transformers library and a text-generation pipeline with the Phi-3.5-mini-instruct 
model to detect Slippery Slope fallacies in tweets. (few shot)
Authors: Anoosha Mallakanti and Prof. Tristram Alexander 
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
tweets = df["Tweet contents"][:100].values 
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
        {"role": "user", "content": f"A Slippery Slope fallacy occurs when a person argues that a relatively small action or event will lead to a series"
        "of increasingly severe consequences without sufficient evidence for that progression."
        "Examples:"

        "1. Message: If we allow students to redo their assignments, next they'll want to retake entire courses, and soon nobody will study seriously."
            "Answer: **Yes**"

        "2. Message: If we don't stop using plastic bags now, it could have long-term environmental effects."
            "Answer: **No**"

        "Analyze the following message and if it contains a Slippery Slope fallacy, respond with:"
        "**Yes** - and give a brief explanation why" 
        "If it does not, respond with:"
        "**No** - and give a brief explanation why"
        f"Message: {tweet}"}, 
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
output_df.to_csv("Slippery_slope_results.csv", index=False)