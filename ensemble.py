# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import json

def ensemble_from_nbest(folder_dir: str='ensembles/dep'):
    files_path = []
    datas = []
    
    for file in os.listdir(folder_dir):
        file_path = f"{folder_dir}/{file}"
        if 'NOT' not in file_path and 'results' not in file_path:
            files_path.append(file_path)
    
    for file_path in files_path:
        print(file_path)
        with open(file_path, 'r+', encoding='utf-8') as f:
            datas.append(json.load(f))

    list_id_qas = list(datas[0].keys())

    print("The numbers of predicted files: ", len(files_path), files_path)
    print("The numbers of questions      : ", len(list_id_qas))

    prediction = {}

    for id_qas in list_id_qas:
        list_candidate = {}
        for data in datas:
            al_ans = data[id_qas]
            for can in al_ans:
                if can['text'] not in list_candidate:
                    list_candidate[can['text']] = can['probability']
                else:
                    list_candidate[can['text']] += can['probability']
            
        MAX_SCORE = 0
        prediction[id_qas] = ""
        for key, value in list_candidate.items():
            if value >= MAX_SCORE:
                prediction[id_qas] = key
                MAX_SCORE = value

    
    with open(f"{folder_dir}/results.json", 'w', encoding='utf-8') as wf:
        wf.write(json.dumps(prediction, indent=4, ensure_ascii=False) + "\n")


ensemble_from_nbest(folder_dir='ensembles/all')
