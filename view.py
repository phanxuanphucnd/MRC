import pickle

# file = open('squad/squad2_albert-xxlarge-v2/20000_res.pkl', 'rb')
# rep = pickle.load(file)

# print(rep)

import json

with open("data/visquad-v1/test_ViQuAD.json", 'r+', encoding='utf-8') as f:
    data = json.loads(f.read())

from pprint import pprint
pprint(data)