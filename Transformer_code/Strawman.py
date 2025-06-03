"""
Strawman_descriptive.py uses HuggingChat's unofficial API to classify tweets using a prompt. 
In this file, we replace HuggingChat with a local model using HuggingFace's transformers 
pipeline, using a chat_template format. 
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
        {"role": "system", "content": "You are an expert in logical reasoning"},
        {"role": "user", "content": f"Does the following message exhibit a Strawman fallacy? Answer Yes or No with a brief explanation.\n\nMessage: {tweet}"}
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
output_df.to_csv("Strawman_transformer_results2.csv", index=False)

