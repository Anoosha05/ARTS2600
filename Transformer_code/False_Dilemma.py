"""
This script uses Hugging Face's transformers library and a text-generation pipeline with the Phi-3.5-mini-instruct 
model to detect False Dilemma fallacies in tweets. (few shot)
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
        {"role": "user", "content": f"A False Dilemma fallacy (also known as False Dichotomy or"
        "Black-and-White Thinking) occurs when a speaker presents two choices as the only possibilities, when more alternatives may exist."
         
        "Examples:"
        "Message: We either raise taxes or the economy will collapse."
        "Answer: **Yes**"

        "Message: Raising taxes is one of many ways we could fund education reform."
        "Answer: **No**"
    
        "Analyze the following message and if it contains a False Dilemma fallacy, respond with:"
        "**Yes** - and briefly explain why" 
        "If it does not, respond with:"
        "**No**  - and briefly explain why"
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
output_df.to_csv("False_dilemma_results.csv", index=False)