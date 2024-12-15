import os
import glob
import json
import pandas as pd
import numpy as np

base_dir = os.path.join('./data')
#base_dir = os.path.join('./temp')
train_dir = base_dir

train_files = glob.glob(train_dir+'/*.json')

utterance_text = []
utterance_eval = []
conversation_summary = []
conversation_evaluation = []

for train_file in train_files:
    #print(train_file) 
    with open(train_file, 'r') as f:
        json_data = json.load(f)
    
    conver_size = len(json_data['dataset']['conversations'])
    utter_size = len(json_data['dataset']['conversations'][0]['utterances'])

    for i in range(utter_size):
        utterance_text.append(json_data['dataset']['conversations'][0]['utterances'][i]['utterance_text'])
        df = pd.DataFrame(json_data['dataset']['conversations'][0]['utterances'][i]['utterance_evaluation'])
        df = df.replace('yes', 1)
        df = df.replace('no', 0)

        if (len(df) == 0) :
            u_eval = ['none']
        else : 
            u_eval = ['eval']
            dfsum = df.sum()
            dfsum.replace(to_replace=1, value=0, inplace=True)
            dfsum.replace(to_replace=2, value=1, inplace=True)
            dfsum.replace(to_replace=3, value=1, inplace=True)

            dfsum.replace(to_replace=0, value='no', inplace=True)
            dfsum.replace(to_replace=1, value='yes', inplace=True)

            u_eval.append(list(dfsum.values))

        utterance_eval.append(u_eval)

    for i in range(conver_size):
        conversation_summary.append(json_data['dataset']['conversations'][0]['conversation_summary'])
        c_eval = 0;
        if (json_data['dataset']['conversations'][0]['conversation_evaluation']['likeability'][0]=='yes'):
            c_eval += 1
        if (json_data['dataset']['conversations'][0]['conversation_evaluation']['likeability'][1]=='yes'):
            c_eval += 1
        if (json_data['dataset']['conversations'][0]['conversation_evaluation']['likeability'][2]=='yes'):
            c_eval += 1
        if (c_eval > 1): 
            conversation_evaluation.append('yes')
        else: 
            conversation_evaluation.append('no')

#print(len(utterance_text))
#print(len(utterance_eval))
#print(len(conversation_summary))
#print(len(conversation_evaluation))

df1 = pd.DataFrame(zip(utterance_text, utterance_eval))
df1.columns = ['text', 'eval']

print(df1)

df2 = pd.DataFrame(zip(conversation_summary, conversation_evaluation))
df2.columns = ['summary', 'c_eval']

print(df2)

fn1 = base_dir+'/trip_utter0.csv'
df1.to_csv(fn1, sep='\t')
fn2 = base_dir+'/trip_conv0.csv'
df2.to_csv(fn2, sep='\t')
