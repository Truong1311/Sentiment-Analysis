import torch
from models.LSTM_model import LSTMTweet
from utils.preprocessing import preprocessing_text, load_file

path = 'data/'
batch_size = 32
max_document_len = 100
embeding_size = 300
hidden_size = 100
max_size = 5000
device = 'cpu'

## load dataset
tokenize = lambda x: x.split()
vocab_size, train_iter, valid_iter, test_iter, Text, Label = load_file(path, tokenize,\
                                                                       batch_size, max_document_len, max_size, device)
model = LSTMTweet(vocab_size, embeding_size, hidden_size,device)

def accuracy(prediction, actual):
    return 100* torch.sum([prediction.argmax(dim = 1)== actual])
def train(model, iterator, optimizer, criterion):
    total_loss = 0
    total_acc = 0
    for batch in iterator:
        optimizer.zero_grab()
        text = batch.text[0]
        preds = model(text)
        acc = accuracy(preds, batch.target)
        loss = criterion(preds, batch.target.squeeze())

        # back propagation
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_acc +=acc
    return total_loss/len(iterator), total_acc/len(iterator)
def evaluate(model, iterator, criterion):
    total_loss = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text = batch.text[0]
            preds = model(text)
            acc = accuracy(preds, batch.target)
            loss = criterion(preds, batch.target.squeeze())
            total_loss += loss
            total_acc +=acc
    return total_loss/len(iterator), total_acc/len(iterator)


