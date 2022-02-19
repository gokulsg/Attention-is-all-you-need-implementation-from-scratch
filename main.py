import torch
import torch.nn as nn
import torchtext
import spacy
import time
from TranslateSentence import translate_sentence
from TrainEvaluate import train, evaluate, epoch_time
from Transformer import Transformer
from Encoder import Encoder
from Decoder import Decoder
from utils import count_parameters, initialize_weights
# from bleu import calculate_bleu

# Spacy tokenizers #

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    """
    Tokenize German sentence to list of words
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenize English sentence to list of words
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


# Source language is English, target language is German #
src_field = torchtext.legacy.data.Field(tokenize = tokenize_en, init_token = '<sos>', eos_token = '<eos>', lower = True, batch_first = True, fix_length = 30, use_vocab=True)
trg_field = torchtext.legacy.data.Field(tokenize = tokenize_de, init_token = '<sos>', eos_token = '<eos>', lower = True, batch_first = True, fix_length = 30, use_vocab=True)

fields = {'English':('eng',src_field), 'German':('ger',trg_field)}

# Loading Data #
# News Commentry Dataset #
train_data, valid_data, test_data = torchtext.legacy.data.TabularDataset.splits(path='./data/news_commentry',train='train.json',validation='valid.json',test='test.json',format='json', fields=fields)

# PHP Dataset - After removing the duplicates  #
#train_data, valid_data, test_data = torchtext.legacy.data.TabularDataset.splits(path='./data/php',train='train_duplicate_removed_php.json',validation='valid_duplicate_removed_php.json',test='test_duplicate_removed_php.json',format='json', fields=fields)

# Vocabulary #
src_field.build_vocab(train_data, min_freq = 2)
trg_field.build_vocab(train_data, min_freq = 2)

# Selecting decice #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128

# Train Test and Validation Iterators #
train_iterator = torchtext.legacy.data.BucketIterator(train_data, batch_size = batch_size, device = device)
valid_iterator = torchtext.legacy.data.BucketIterator(valid_data,  batch_size = batch_size, device = device)
test_iterator = torchtext.legacy.data.BucketIterator(test_data, batch_size = batch_size, device = device)

## Hyperparameters ##

input_dim = len(src_field.vocab)
output_dim = len(trg_field.vocab)
embed_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8 
dec_heads = 8 
enc_expansion = 512
dec_expansion = 512
enc_dropout = 0.1
dec_dropout = 0.1
max_len = 30

# Encoder and Decoder #
enc = Encoder(input_dim, embed_dim, enc_layers, enc_heads, enc_expansion, enc_dropout, device)
dec = Decoder(output_dim, embed_dim, dec_layers, dec_heads, dec_expansion, dec_dropout, device)

src_pad_idx = src_field.vocab.stoi[src_field.pad_token]
trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]

# Transformer model #
model = Transformer(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)

print(f'The Transformer model has {count_parameters(model):,} trainable parameters')      
model.apply(initialize_weights)

learning_rate = 0.0005

# Optimizer and Loss Function #
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)

# Training #

print('Training...')
num_epoch = 100
clip = 1

for epoch in range(num_epoch):
    start_time = time.time()
    train_loss = train(model, train_iterator, optimizer, criterion, clip)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1} | Time: {epoch_mins}mins {epoch_secs}seconds')
    print(f'\t Train Loss: {train_loss:.3f}')
    print(f'\t Validation Loss: {valid_loss:.3f}')
    print(f'________________________________________________________________________________')

print('Training Complete')
print('Testing...')

test_loss = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f}')

# Saving the model #
#torch.save(model.state_dict(), './model/model_state.pt')

## Translating one English to German sentence #

print("Translating one English sentence to German sentence")

translate_idx = 22
src = vars(test_data.examples[translate_idx])['eng'] [:max_len-3]
trg = vars(test_data.examples[translate_idx])['ger'] [:max_len-3]

print(f'Source sentence = {" ".join(src)}\n')
print(f'Target sentence = {" ".join(trg)}\n')

translation = translate_sentence(src, src_field, trg_field, model, device)
print(f'Translated sentence = {" ".join(translation[:-1])}')


# BLEU #
# bleu_score = calculate_bleu(test_data, src_field, trg_field, model, device)
# print(f'BLEU score = {bleu_score*100:.2f}')