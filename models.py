#!/usr/bin/python3 -OO

"""

This file contains several text classification models.

Owner : paul-tqh-nguyen

Created : 03/30/2020

File Name : classification_models.py

File Organization:
* Imports
* Misc. Globals & Global State Initializations
* Models
* Classifiers
* Driver

"""

###########
# Imports #
###########

import random
import spacy
from collections import OrderedDict
from misc_utilities import timer, debug_on_error, tqdm_with_message

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets

################################################
# Misc. Globals & Global State Initializations #
################################################

SEED = 1234
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

NLP = spacy.load('en')

##########
# Models #
##########

MAX_VOCAB_SIZE = 25_000

NUMBER_OF_EPOCHS = 5
BATCH_SIZE = 32

DROPOUT_PROBABILITY = 0.5
NUMBER_OF_ENCODING_LAYERS = 1
EMBEDDING_SIZE = 100
ENCODING_HIDDEN_SIZE = 128
ATTENTION_INTERMEDIATE_SIZE = 8
NUMBER_OF_ATTENTION_HEADS = 2
OUTPUT_SIZE = 2

class AttentionLayers(nn.Module):
    def __init__(self, encoding_hidden_size, attention_intermediate_size, number_of_attention_heads, dropout_probability):
        super().__init__()
        self.encoding_hidden_size = encoding_hidden_size
        self.number_of_attention_heads = number_of_attention_heads
        self.attention_layers = nn.Sequential(OrderedDict([
            ("intermediate_attention_layer", nn.Linear(encoding_hidden_size*2, attention_intermediate_size)),
            ("intermediate_attention_dropout_layer", nn.Dropout(dropout_probability)),
            ("attention_activation", nn.ReLU(True)),
            ("final_attention_layer", nn.Linear(attention_intermediate_size, number_of_attention_heads)),
            ("final_attention_dropout_layer", nn.Dropout(dropout_probability)),
            ("softmax_layer", nn.Softmax(dim=0)),
        ]))

    def forward(self, encoded_batch, text_lengths):
        max_sentence_length = encoded_batch.shape[1]
        batch_size = text_lengths.shape[0]
        assert tuple(encoded_batch.shape) == (batch_size, max_sentence_length, self.encoding_hidden_size*2)

        attended_batch = Variable(torch.empty(batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads).to(encoded_batch.device))

        for batch_index in range(batch_size):
            sentence_length = text_lengths[batch_index]
            sentence_matrix = encoded_batch[batch_index, :sentence_length, :]
            assert encoded_batch[batch_index ,sentence_length:, :].data.sum() == 0
            assert tuple(sentence_matrix.shape) == (sentence_length, self.encoding_hidden_size*2)

            # Intended Implementation
            sentence_weights = self.attention_layers(sentence_matrix)
            assert tuple(sentence_weights.shape) == (sentence_length, self.number_of_attention_heads)
            assert (sentence_weights.data.sum(dim=0)-1).abs().mean() < 1e-4
            
            # Mean Implementation
            # sentence_weights = torch.ones(sentence_length, self.number_of_attention_heads).to(encoded_batch.device) / sentence_length
            # assert tuple(sentence_weights.shape) == (sentence_length, self.number_of_attention_heads)

            weight_adjusted_sentence_matrix = torch.mm(sentence_matrix.t(), sentence_weights)
            assert tuple(weight_adjusted_sentence_matrix.shape) == (self.encoding_hidden_size*2, self.number_of_attention_heads,)

            concatenated_attention_vectors = weight_adjusted_sentence_matrix.view(-1)
            assert tuple(concatenated_attention_vectors.shape) == (self.encoding_hidden_size*2*self.number_of_attention_heads,)

            attended_batch[batch_index, :] = concatenated_attention_vectors

        assert tuple(attended_batch.shape) == (batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads)
        return attended_batch

class EEAPNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoding_hidden_size, number_of_encoding_layers, attention_intermediate_size, number_of_attention_heads, output_size, dropout_probability, pad_idx):
        super().__init__()
        if __debug__:
            self.embedding_size = embedding_size
            self.encoding_hidden_size = encoding_hidden_size
            self.number_of_encoding_layers = number_of_encoding_layers
            self.number_of_attention_heads = number_of_attention_heads
        self.embedding_layers = nn.Sequential(OrderedDict([
            ("embedding_layer", nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx, max_norm=1.0)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
        ]))
        self.encoding_layers = nn.LSTM(embedding_size,
                                       encoding_hidden_size,
                                       num_layers=number_of_encoding_layers,
                                       bidirectional=True,
                                       dropout=dropout_probability)
        self.attention_layers = AttentionLayers(encoding_hidden_size, attention_intermediate_size, number_of_attention_heads, dropout_probability)
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("fully_connected_layer", nn.Linear(encoding_hidden_size*2*number_of_attention_heads, output_size)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("softmax_layer", nn.Softmax(dim=1)),
        ]))

    def forward(self, text_batch, text_lengths):
        if __debug__:
            max_sentence_length = max(text_lengths)
            batch_size = text_batch.shape[0]
        assert batch_size <= BATCH_SIZE
        assert tuple(text_batch.shape) == (batch_size, max_sentence_length)
        assert tuple(text_lengths.shape) == (batch_size,)

        embedded_batch = self.embedding_layers(text_batch)
        assert tuple(embedded_batch.shape) == (batch_size, max_sentence_length, self.embedding_size)

        embedded_batch_packed = nn.utils.rnn.pack_padded_sequence(embedded_batch, text_lengths, batch_first=True)
        if __debug__:
            encoded_batch_packed, (encoding_hidden_state, encoding_cell_state) = self.encoding_layers(embedded_batch_packed)
            encoded_batch, encoded_batch_lengths = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        else:
            encoded_batch_packed, _ = self.encoding_layers(embedded_batch_packed)
            encoded_batch, _ = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        assert tuple(encoded_batch.shape) == (batch_size, max_sentence_length, self.encoding_hidden_size*2)
        assert tuple(encoding_hidden_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoding_cell_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoded_batch_lengths.shape) == (batch_size,)
        assert (encoded_batch_lengths.to(DEVICE) == text_lengths).all()

        # original implementation
        # hidden = torch.cat((encoding_hidden_state[-2,:,:], encoding_hidden_state[-1,:,:]), dim = 1) # one-line implementation
        #
        # hidden = Variable(torch.zeros(batch_size, self.encoding_hidden_size*2).to(encoded_batch.device))
        # hidden[:, self.encoding_hidden_size:] = encoding_hidden_state[-2,:,:]
        # hidden[:, :self.encoding_hidden_size] = encoding_hidden_state[-1,:,:]
        # assert tuple(hidden.shape) == (batch_size, self.encoding_hidden_size*2)
        # prediction = self.prediction_layers(hidden)
        
        # Sum Implementation (didn't work)
        # hidden = encoded_batch.sum(dim=1) 
        # assert tuple(hidden.shape) == (batch_size, self.encoding_hidden_size*2)
        # prediction = self.prediction_layers(hidden)

        # Last Output Value Implementation (works)
        # hidden = Variable(torch.zeros(batch_size, self.encoding_hidden_size*2).to(encoded_batch.device))
        # for batch_index in range(batch_size):
        #     last_word_index = text_lengths[batch_index]-1
        #     hidden[batch_index, :] = encoded_batch[batch_index,last_word_index,:]
        # assert tuple(hidden.shape) == (batch_size, self.encoding_hidden_size*2)
        # prediction = self.prediction_layers(hidden)

        # First Output Value Implementation (works (since bidirectional and last term works), but takes a few more epochs)
        # hidden = Variable(torch.zeros(batch_size, self.encoding_hidden_size*2).to(encoded_batch.device))
        # for batch_index in range(batch_size):
        #     hidden[batch_index, :] = encoded_batch[batch_index,0,:]
        # assert tuple(hidden.shape) == (batch_size, self.encoding_hidden_size*2)
        # prediction = self.prediction_layers(hidden)

        # Mean Output Value Implementation
        # hidden = Variable(torch.zeros(batch_size, self.encoding_hidden_size*2).to(encoded_batch.device))
        # for batch_index in range(batch_size):
        #     batch_sequence_length = text_lengths[batch_index]
        #     last_word_index = batch_sequence_length-1
        #     hidden[batch_index, :] = encoded_batch[batch_index,:batch_sequence_length,:].mean(dim=0)
        #     assert encoded_batch[batch_index,batch_sequence_length:,:].sum() == 0
        # assert tuple(hidden.shape) == (batch_size, self.encoding_hidden_size*2)
        # prediction = self.prediction_layers(hidden)
        
        # Attention Implementation
        attended_batch = self.attention_layers(encoded_batch, text_lengths)
        assert tuple(attended_batch.shape) == (batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads)
        prediction = self.prediction_layers(attended_batch)
        assert tuple(prediction.shape) == (batch_size, OUTPUT_SIZE)
        
        return prediction

###########################
# Domain Specific Helpers #
###########################

def discrete_accuracy(y_hat, y):
    y_hat_indices_of_max = y_hat.argmax(dim=1)
    number_of_correct_answers = (y_hat_indices_of_max == y).float().sum(dim=0)
    mean_accuracy = number_of_correct_answers / y.shape[0]
    return mean_accuracy

###############
# Classifiers #
###############

class EEAPClassifier():
    def __init__(self):
        self.load_data()
        self.initialize_model_and_optimizer()
        self.best_valid_loss = float('inf')
    
    def load_data(self):
        self.text_field = data.Field(tokenize = 'spacy', include_lengths = True, batch_first = True)
        self.label_field = data.LabelField(dtype = torch.long)
        
        train_data, test_data = datasets.IMDB.splits(self.text_field, self.label_field)
        train_data, valid_data = train_data.split(random_state = random.seed(SEED))
        
        self.text_field.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
        self.label_field.build_vocab(train_data)
        
        assert self.text_field.vocab.vectors.shape[0] <= MAX_VOCAB_SIZE+2
        assert self.text_field.vocab.vectors.shape[1] == EMBEDDING_SIZE
        
        self.vocab_size = len(self.text_field.vocab)
        
        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size = BATCH_SIZE,
            sort_within_batch = True,
            device = DEVICE)
        
        self.pad_idx = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.unk_idx = self.text_field.vocab.stoi[self.text_field.unk_token]        
            
    def initialize_model_and_optimizer(self):
        self.model = EEAPNetwork(self.vocab_size,
                                 EMBEDDING_SIZE,
                                 ENCODING_HIDDEN_SIZE,
                                 NUMBER_OF_ENCODING_LAYERS,
                                 ATTENTION_INTERMEDIATE_SIZE,
                                 NUMBER_OF_ATTENTION_HEADS,
                                 OUTPUT_SIZE,
                                 DROPOUT_PROBABILITY,
                                 self.pad_idx)
        self.model.embedding_layers.embedding_layer.weight.data.copy_(self.text_field.vocab.vectors)
        self.model.embedding_layers.embedding_layer.weight.data[self.unk_idx] = torch.zeros(EMBEDDING_SIZE)
        self.model.embedding_layers.embedding_layer.weight.data[self.pad_idx] = torch.zeros(EMBEDDING_SIZE)
        self.model = self.model.to(DEVICE)
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = self.loss_function.to(DEVICE)
        
    def count_parameters(self): # @todo use this
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def predict_sentiment(self, sentence): # @todo use this
        self.model.eval()
        tokenized = [token.text for token in NLP.tokenizer(sentence)]
        indexed = [self.text_field.vocab.stoi[t] for t in tokenized]
        lengths = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(DEVICE)
        tensor = tensor.view(1,-1)
        length_tensor = torch.LongTensor(lengths).to(DEVICE)
        prediction_as_index = self.model(tensor, length_tensor).argmax(dim=1).item()
        prediction = self.label_field.vocab.itos[prediction_as_index]
        return prediction
    
    def train_one_epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        number_of_batches = len(self.train_iterator)
        for batch in tqdm_with_message(self.train_iterator, post_yield_message_func = lambda index: f'Training Accuracy {epoch_acc/(index+1)*100:.8f}%', total=number_of_batches, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = self.model(text, text_lengths)
            loss = self.loss_function(predictions, batch.label)
            acc = discrete_accuracy(predictions, batch.label)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / number_of_batches, epoch_acc / number_of_batches
    
    def validate(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        number_of_batches = len(self.valid_iterator)
        with torch.no_grad():
            for batch in tqdm_with_message(self.valid_iterator, post_yield_message_func = lambda index: f'Validation Accuracy {epoch_acc/(index+1)*100:.8f}%', total=number_of_batches, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
                text, text_lengths = batch.text
                predictions = self.model(text, text_lengths).squeeze(1)
                loss = self.loss_function(predictions, batch.label)
                acc = discrete_accuracy(predictions, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / number_of_batches, epoch_acc / number_of_batches
        
    def train(self):
        print(f'Starting training')
        for epoch_index in range(NUMBER_OF_EPOCHS):
            with timer(section_name=f"Epoch {epoch_index}"):
                train_loss, train_acc = self.train_one_epoch()
                valid_loss, valid_acc = self.validate()
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'tut2-model.pt')
            print(f'\tTrain Loss: {train_loss:.8f} | Train Acc: {train_acc*100:.8f}%')
            print(f'\t Val. Loss: {valid_loss:.8f} |  Val. Acc: {valid_acc*100:.8f}%')
        self.model.load_state_dict(torch.load('tut2-model.pt'))
        # test_loss, test_acc = validate(model, self.test_iterator, loss_function) # @todo make this work

##########
# Driver #
##########

@debug_on_error
def main():
    classifier = EEAPClassifier()
    classifier.train()

if __name__ == '__main__':
    main()
# @todo update the main to just print what this does
