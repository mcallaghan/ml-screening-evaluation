import json
from importlib import import_module
from sklearn.pipeline import Pipeline






def deserialize_pipes(fp):
    pipelines = []
    names = []
    with open(fp,'r') as f:
        for p in json.load(f):
            steps = []
            for s in p['steps']:
                cls = getattr(import_module(s['module']),s['package'])
                steps.append(
                    (s['name'], cls(**s['args']))
                )
            pipelines.append(Pipeline(steps=steps))
            names.append(p['name'])
    return pipelines, names

pipelines, names = deserialize_pipes('pipelines.json')
for p, n in zip(pipelines, names):
    print(p)
    print(n)
    
    