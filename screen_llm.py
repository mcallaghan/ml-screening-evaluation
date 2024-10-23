import typer                                                                                                            
import time                                                                                                             
import torch                                                                                                            
import pandas as pd                                                                                                     
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from synergy_dataset import Dataset, iter_datasets
from string import Template

def binary_probs(tokenizer, model, prompt, no_words=['no'], yes_words=['yes'], return_all=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoded_text = tokenizer(prompt, return_tensors="pt").to(device)
    #1. step to get the logits of the next token
    with torch.inference_mode():
        outputs = model(**encoded_text)
    
    next_token_logits = outputs.logits[0, -1, :]
    
    # 2. step to convert the logits to probabilities
    next_token_probs = torch.softmax(next_token_logits, -1)
    
    topk_next_tokens= torch.topk(next_token_probs, 50)
    tokens = [tokenizer.decode(x).strip().lower() for x in topk_next_tokens.indices]
    p = topk_next_tokens.values
    
    df = pd.DataFrame.from_dict({'t': tokens,'p': p.cpu()})
    y = df[df['t'].isin(yes_words)]['p'].sum()
    n = df[df['t'].isin(no_words)]['p'].sum()
    
    if return_all:
        return df
    return y, n
                                                                                                                        
def main():                                                                                                             
    """                                                                                                                 
    Run evaluations testing ML prioritisation with LLMs                                                                 
    """                                                                                                                 
                                                                                                                        
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")                                                                                                                      
    model = LlamaForCausalLM.from_pretrained(                                                                           
        "meta-llama/Llama-2-7b-chat-hf",                                                               
        load_in_8bit=True,                                                                                             
        #load_in_4bit=True                                                                                              
    )

    prompt = Template('''<s>[INST] <<SYS>>
    You are a systematic review helper tasked with finding out whether a study is relevant to the review $t
    
    Answer 'yes' if the study is relevant, or 'no' if not
    <</SYS>>
    
    Study: $s 
    
    Should the study be included? Answer yes or no. [/INST] ''')
    
    for d in iter_datasets():
        name = d.name
        # try:
        #     df = pd.read_csv(f'output_data/{name}_LLM.csv').dropna()
        #     if df.shape[0] > 10:
        #         continue
        # except:
        #     pass
        df = d.to_frame().rename(columns={'label_included':'y'})
        df['text'] = df['title'].astype(str) + '\n ' + df['abstract'].astype(str)
        df['text'] = df['text'].str[:10000]
        title = d.metadata['publication']['title']
        
        print(name)

        for i, row in df.iterrows():
            p = prompt.substitute({'s':row['text'], 't':title})
            df.loc[i, ['py','pn']] = binary_probs(tokenizer, model, p)
        df[['y','py','pn']].to_csv(f'output_data/{name}_LLM.csv')
                                                                                                                        
if __name__ == '__main__':                                                                                              
    start_time = time.monotonic()                                                                                       
    typer.run(main)                                                                                                     
    end_time = time.monotonic()                                                                                         
    print(timedelta(seconds=end_time - start_time))  