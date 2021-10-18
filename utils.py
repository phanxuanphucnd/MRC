import torch
import numpy as np

from tqdm import tqdm

tqdm.pandas()

def convert_lines(df, tokenizer, max_sequence_length):
    outputs = np.zeros((len(df), max_sequence_length))
    
    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        inputs = tokenizer.encode_plus(
            row.text,
            add_special_tokens=True,
            max_length=max_sequence_length,
            return_token_type_ids=True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[: max_sequence_length] 
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
        outputs[idx,:] = np.array(input_ids)
    return outputs

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))