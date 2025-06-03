"""
This script uses Hugging Face's transformers library and a text-generation pipeline with the Phi-3.5-mini-instruct 
model to detect Anecdotal fallacies in tweets. (few shot)
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
        {"role": "user", "content": " An **Anecdotal Fallacy** occurs when someone uses a personal story or isolated case to argue against broader evidence or general trends."
         
        "Examples:"
        "Message: This is sad. I have many friends and family with whom I disagree politically. Refusing to engage outside of your bubble is neither healthy nor unifying."
        "Answer: **Yes**"

        "Message: I saw a documentary that used peer-reviewed data to explain global warming."
        "Answer: **No**"
    
        "Analyze the following message and if it contains a Anecdotal fallacy, respond with:"
        "**Yes**" 
        "If it does not, respond with:"
        "**No**"
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
output_df.to_csv("Anecdotal_results.csv", index=False)