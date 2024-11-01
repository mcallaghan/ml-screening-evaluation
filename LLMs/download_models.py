from huggingface_hub import login, snapshot_download
import json
with open('llms/llm_list.json','r') as f:
    l = json.load(f)

login()

for m in l:  
    snapshot_download(m, cache_dir='/p/tmp/maxcall/hf/hub')