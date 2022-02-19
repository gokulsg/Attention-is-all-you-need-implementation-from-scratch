import torch

def train(model, iterator, optimizer, criterion, clip):
    model.train() 
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.eng 
        trg = batch.ger 
        
        optimizer.zero_grad()
        
        output = model(src, trg[:,:-1])
        output_dim = output.shape[-1]

        output = output.reshape(-1, output_dim)
        trg = trg[:,1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.eng 
            trg = batch.ger 
            output = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            
            output = output.reshape(-1, output_dim)
            trg = trg[:,1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs