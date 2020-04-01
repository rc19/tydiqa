import glob, json
from statistics import median
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# mode = 'train'
mode = 'dev'
path = '/mnt/c/Users/rasto/Documents/'
if mode == 'train':
    path = 'tydi*keep*.json'
    idx = 29
if mode == 'dev':
    path += 'tydiqa-goldp-v1.1-dev/tydi*.json'
    idx = 39
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

files = glob.glob(path)
print("Found files", files)
avg_context_len = {}
avg_question_len = {}
total_avg_question_len = 0
total_avg_context_len = 0
overall_medain_context = []
for fi in files:    
    with open(fi,'r') as f:
        data = json.load(f)['data']
    
    median_context = []
    avg_context_len[fi[idx:-5]] = 0
    avg_question_len[fi[idx:-5]] = 0
    ignore_len = 0
    
    for entry in data:
        paragraph = entry['paragraphs']
        
        if len(paragraph) == 0:
            ignore_len += 1
            continue
        
        paragraph = paragraph[0]
        context = paragraph['context']
        question = paragraph['qas'][0]['question']
        median_context.append(len(tokenizer.tokenize(context)))
        avg_context_len[fi[idx:-5]] += len(tokenizer.tokenize(context))
        avg_question_len[fi[idx:-5]] += len(tokenizer.tokenize(question))
    
    print(fi[idx:-5], ignore_len, len(data), median(median_context))
    overall_medain_context.extend(median_context)
    total_avg_context_len += avg_context_len[fi[idx:-5]]
    total_avg_question_len += avg_question_len[fi[idx:-5]]
    avg_context_len[fi[idx:-5]] /= (len(data)-ignore_len)
    avg_question_len[fi[idx:-5]] /= (len(data)-ignore_len)

# print(avg_context_len, avg_question_len, total_avg_context_len/49881, total_avg_question_len/49881, median(overall_medain_context))
print(avg_context_len, avg_question_len, total_avg_context_len/5077, total_avg_question_len/5077, median(overall_medain_context))