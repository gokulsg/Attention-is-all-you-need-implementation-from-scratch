import pandas as pd
from sklearn.model_selection import train_test_split

data = open('./data/news-commentary-v15.de-en.tsv', encoding='utf8').read().split('\n')[:110000]

de_data = []
en_data = []

for i in data:
    d = i.split('\t')
    if len(d) == 2 and d[0] != '':
        de_data.append(d[0])
        en_data.append(d[1])
        
df = pd.DataFrame(list(zip(en_data, de_data)),columns =['English', 'German'])
dataset_40k = df.iloc[:40000, :]

train, test_valid = train_test_split(dataset_40k, test_size=0.1)
test, valid = train_test_split(test_valid, test_size=0.5)

train.to_json('./data/news_commentry/train.json', orient='records', lines=True)
test.to_json('./data/news_commentry/test.json', orient='records', lines=True)
valid.to_json('./data/news_commentry/valid.json', orient='records', lines=True)