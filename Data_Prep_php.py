import pandas as pd

de_data = open('./data/PHP.de-en.de', encoding='utf8').read().split('\n')
en_data = open('./data/PHP.de-en.en', encoding='utf8').read().split('\n')

raw_data = {'English':[line for line in en_data], 'German' : [line for line in de_data]}
df = pd.DataFrame(raw_data, columns=['English','German'])

# Drop Duplicates #
df.drop_duplicates(inplace=True)

train = df.iloc[:10281, :]
valid = df.iloc[10281:11281, :]
test = df.iloc[11281:12281, :]

train.to_json('./data/php/train_duplicate_removed_php.json', orient='records', lines=True)
test.to_json('./data/php/test_duplicate_removed_php.json', orient='records', lines=True)
valid.to_json('./data/php/valid_duplicate_removed_php.json', orient='records', lines=True)