from torchtext.data.metrics import bleu_score
from TranslateSentence import translate_sentence

def calculate_bleu(data_set, src_field, trg_field, model, device, max_len = 30):
    trgs = []
    pred_trgs = []
    for data in data_set: 
        src = vars(data)['eng'] [:max_len-3]
        trg = vars(data)['ger'] [:max_len-3]
        pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len)

        #cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)