import typer
import time
from datetime import datetime
import torch
import pandas as pd
from transformers import LlamaForCausalLM, AutoConfig, pipeline, BitsAndBytesConfig, AutoTokenizer
from string import Template
from synergy_dataset import iter_datasets
import pyarrow.parquet as pq


def binary_probs(tokenizer, model, prompt, no_words=['no'], yes_words=['yes'], return_all=False, print_answer=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoded_text = tokenizer(prompt, return_tensors="pt").to(device)
    #1. step to get the logits of the next token
    with torch.inference_mode():
        outputs = model(**encoded_text)

    if print_answer:
        out = model.generate(**encoded_text, max_new_tokens=1024)
        print(tokenizer.decode(out[0]))   
    
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
        return df.groupby('t').sum().reset_index().sort_values('p', ascending=False).reset_index(drop=True)
    return y, n

def generate(model, tokenizer, text):
    t = tokenizer(text, return_tensors="pt").to('cuda')
    out = model.generate(**t)
    print(tokenizer.decode(out[0]))

def main(model_name: str):
    """
    Run evaluations testing ML prioritisation with LLMs
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4_bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #config = AutoConfig.from_pretrained(
    #    model,
    #    max_new_tokens=1024
    #)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config, 
    )
    print('loaded model')

    l2_prompt = Template('''<s>[INST] <<SYS>>
    You are a systematic review helper tasked with finding out whether a study is relevant to the review $t
    
    Answer 'yes' if the study is relevant, or 'no' if not
    <</SYS>>
    
    Study: $s 
    
    Should the study be included? Answer yes or no. [/INST] ''')

    l3_prompt = Template('''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a systematic review helper tasked with finding out whether a study is relevant to the review $t

    Answer 'yes' if the study is relevant, or 'no' if not
    <|eot_id|><|start_header_id|>user<|end_header_id|>

    Study: $s 

    Should the study be included? Answer yes or no. <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    ''')

    if "Llama-3" in model_name:
        prompt = l3_prompt
    else:
        prompt = l2_prompt

    
    for d in iter_datasets():
        print(f'processing dataset {d.name} at {datetime.now()}')
        print(time.time())
        df = pq.read_table(
            f'/p/tmp/maxcall/ml-screening/llm_preds',
            filters=[
                ('review', '=', d.name),
                ('model', '=', model_name)
            ]
        ).to_pandas()
        if df.shape[0] > 0:
            print('skipping this review, as I already have data for it!')
            continue
        df = d.to_frame().rename(columns={'label_included':'y'})
        df['text'] = df['title'].astype(str) + '\n ' + df['abstract'].astype(str)
        df['text'] = df['text'].str[:10000]
        df['review'] = d.name
        df['model'] = model_name
        title = d.metadata['publication']['title']
        j = 0
        for i, row in df.iterrows():
            p = prompt.substitute({'s':row['text'], 't': title})
            df.loc[i, ['py','pn']] = binary_probs(tokenizer, model, p, print_answer=False)
            j += 1
        df[['y','py','pn','review','model']].to_parquet(
            '/p/tmp/maxcall/ml-screening/llm_preds',
            partition_cols=['review','model'],
            existing_data_behavior='delete_matching'
        )
        
    
if __name__ == '__main__':
    start_time = time.monotonic()
    typer.run(main)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
