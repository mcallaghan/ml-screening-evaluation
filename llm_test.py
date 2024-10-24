# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

print(torch.cuda.mem_get_info())

pipe = pipeline("text-generation", model="meta-llama/Llama-2-70b-hf", device=0)

out = pipe('Hello, is this thing on?')
print(out)
#with open('test.txt','w') as f:
#    f.write(out)
