#!/usr/bin/python3
"#!/usr/bin/python3 -OO" # @todo make this the default

"""

This file contains several text classification models.

Owner : paul-tqh-nguyen

Created : 03/30/2020

File Name : models.py

File Organization:
* Imports
* Misc. Globals & Global State Initializations
* Domain Specific Helpers
* Models
* Classifiers
* Driver

"""

# @todo add type declarations

###########
# Imports #
###########

import os
import random
import spacy
from collections import OrderedDict
from misc_utilities import implies, timer, debug_on_error, tqdm_with_message

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets

################################################
# Misc. Globals & Global State Initializations #
################################################

SEED = 1234
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_SIZE = 2

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

###########################
# Domain Specific Helpers #
###########################

def dimensionality_from_pre_trained_embedding_specification(pre_trained_embedding_specification: str) -> int:
    if 'dim' in torchtext.vocab.pretrained_aliases[pre_trained_embedding_specification].keywords:
        return int(torchtext.vocab.pretrained_aliases[pre_trained_embedding_specification].keywords['dim'])
    else:
        return int(pre_trained_embedding_specification.split('.')[-1][:-1])

def discrete_accuracy(y_hat, y):
    y_hat_indices_of_max = y_hat.argmax(dim=1)
    number_of_correct_answers = (y_hat_indices_of_max == y).float().sum(dim=0)
    mean_accuracy = number_of_correct_answers / y.shape[0]
    return mean_accuracy

##########
# Models #
##########

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

            sentence_weights = self.attention_layers(sentence_matrix)
            assert tuple(sentence_weights.shape) == (sentence_length, self.number_of_attention_heads)
            assert (sentence_weights.data.sum(dim=0)-1).abs().mean() < 1e-4

            weight_adjusted_sentence_matrix = torch.mm(sentence_matrix.t(), sentence_weights)
            assert tuple(weight_adjusted_sentence_matrix.shape) == (self.encoding_hidden_size*2, self.number_of_attention_heads,)

            concatenated_attention_vectors = weight_adjusted_sentence_matrix.view(-1)
            assert tuple(concatenated_attention_vectors.shape) == (self.encoding_hidden_size*2*self.number_of_attention_heads,)

            attended_batch[batch_index, :] = concatenated_attention_vectors

        assert tuple(attended_batch.shape) == (batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads)
        return attended_batch

class EEAPNetwork(nn.Module):
    def __init__(self, pad_idx, vocab_size, embedding_size, encoding_hidden_size, number_of_encoding_layers, attention_intermediate_size, number_of_attention_heads, output_size, dropout_probability, final_representation):
        super().__init__()
        assert final_representation in ['mean', 'final_output', 'initial_output', 'hidden', 'attention']
        self.final_representation = final_representation
        assert implies(final_representation != 'attention', number_of_attention_heads == 1)
        self.encoding_hidden_size = encoding_hidden_size
        if __debug__:
            self.embedding_size = embedding_size
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
        if final_representation == 'attention':
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
        assert tuple(text_batch.shape) == (batch_size, max_sentence_length)
        assert tuple(text_lengths.shape) == (batch_size,)

        embedded_batch = self.embedding_layers(text_batch)
        assert tuple(embedded_batch.shape) == (batch_size, max_sentence_length, self.embedding_size)

        embedded_batch_packed = nn.utils.rnn.pack_padded_sequence(embedded_batch, text_lengths, batch_first=True)
        if __debug__:
            encoded_batch_packed, (encoding_hidden_state, encoding_cell_state) = self.encoding_layers(embedded_batch_packed)
            encoded_batch, encoded_batch_lengths = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        else:
            if self.final_representation == 'hidden':
                encoded_batch_packed, (encoding_hidden_state, _) = self.encoding_layers(embedded_batch_packed)
            else:
                encoded_batch_packed, _ = self.encoding_layers(embedded_batch_packed)
            encoded_batch, _ = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        assert tuple(encoded_batch.shape) == (batch_size, max_sentence_length, self.encoding_hidden_size*2)
        assert tuple(encoding_hidden_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoding_cell_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoded_batch_lengths.shape) == (batch_size,)
        assert (encoded_batch_lengths.to(DEVICE) == text_lengths).all()
        
        if self.final_representation == 'mean':
            final_representation = Variable(torch.empty(batch_size, self.encoding_hidden_size*2).to(encoded_batch.device))
            for batch_index in range(batch_size):
                batch_sequence_length = text_lengths[batch_index]
                final_representation[batch_index, :] = encoded_batch[batch_index,:batch_sequence_length,:].mean(dim=0)
                assert encoded_batch[batch_index,batch_sequence_length:,:].sum() == 0
            assert tuple(final_representation.shape) == (batch_size, self.encoding_hidden_size*2)
        
        if self.final_representation == 'final_output':
            final_representation = Variable(torch.empty(batch_size, self.encoding_hidden_size*2).to(encoded_batch.device))
            for batch_index in range(batch_size):
                last_word_index = text_lengths[batch_index]-1
                final_representation[batch_index, :] = encoded_batch[batch_index,last_word_index,:]
            assert tuple(final_representation.shape) == (batch_size, self.encoding_hidden_size*2)

        if self.final_representation == 'initial_output':
            final_representation = Variable(torch.empty(batch_size, self.encoding_hidden_size*2).to(encoded_batch.device))
            for batch_index in range(batch_size):
                final_representation[batch_index, :] = encoded_batch[batch_index,0,:]
            assert tuple(final_representation.shape) == (batch_size, self.encoding_hidden_size*2)
            
        if self.final_representation == 'hidden':
            final_representation = Variable(torch.empty(batch_size, self.encoding_hidden_size*2).to(encoded_batch.device))
            final_representation[:, self.encoding_hidden_size:] = encoding_hidden_state[-2,:,:]
            final_representation[:, :self.encoding_hidden_size] = encoding_hidden_state[-1,:,:]
            assert tuple(final_representation.shape) == (batch_size, self.encoding_hidden_size*2)
        
        if self.final_representation == 'attention':
            final_representation = self.attention_layers(encoded_batch, text_lengths)
            assert tuple(final_representation.shape) == (batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads)

        prediction = self.prediction_layers(final_representation)
        assert tuple(prediction.shape) == (batch_size, OUTPUT_SIZE)
        
        return prediction

###############
# Classifiers #
###############

class EEAPClassifier():
    def __init__(self, number_of_epochs, batch_size, max_vocab_size, pre_trained_embedding_specification, encoding_hidden_size, number_of_encoding_layers, attention_intermediate_size, number_of_attention_heads, output_size, dropout_probability, final_representation, output_directory):
        self.nlp = spacy.load('en')
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.number_of_epochs = number_of_epochs
        self.load_data(batch_size, pre_trained_embedding_specification, max_vocab_size)
        self.initialize_model_and_optimizer(pre_trained_embedding_specification, encoding_hidden_size, number_of_encoding_layers, attention_intermediate_size, number_of_attention_heads, output_size, dropout_probability, final_representation)
        self.best_valid_loss = float('inf')
        self.current_epoch = 0
        self.training_epoch_accuracy_loss_triples = []
        self.validation_epoch_accuracy_loss_triples = []
    
    def load_data(self, batch_size, pre_trained_embedding_specification, max_vocab_size):
        self.text_field = data.Field(tokenize = 'spacy', include_lengths = True, batch_first = True)
        self.label_field = data.LabelField(dtype = torch.long)
        
        train_data, test_data = datasets.IMDB.splits(self.text_field, self.label_field)
        train_data, valid_data = train_data.split(random_state = random.seed(SEED))
        
        self.text_field.build_vocab(train_data, max_size = max_vocab_size, vectors = pre_trained_embedding_specification, unk_init = torch.Tensor.normal_)
        self.label_field.build_vocab(train_data)
        
        assert self.text_field.vocab.vectors.shape[0] <= max_vocab_size+2
        assert self.text_field.vocab.vectors.shape[1] == dimensionality_from_pre_trained_embedding_specification(pre_trained_embedding_specification)
        
        self.vocab_size = len(self.text_field.vocab)
        
        self.train_iterator, self.validation_iterator, self.test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size = batch_size,
            sort_within_batch = True,
            device = DEVICE)
        
        self.pad_idx = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.unk_idx = self.text_field.vocab.stoi[self.text_field.unk_token]        
            
    def initialize_model_and_optimizer(self, pre_trained_embedding_specification, encoding_hidden_size, number_of_encoding_layers, attention_intermediate_size, number_of_attention_heads, output_size, dropout_probability, final_representation):
        embedding_size = dimensionality_from_pre_trained_embedding_specification(pre_trained_embedding_specification)
        self.model = EEAPNetwork(self.pad_idx,
                                 self.vocab_size,
                                 embedding_size,
                                 encoding_hidden_size,
                                 number_of_encoding_layers,
                                 attention_intermediate_size,
                                 number_of_attention_heads,
                                 output_size,
                                 dropout_probability,
                                 final_representation)
        self.model.embedding_layers.embedding_layer.weight.data.copy_(self.text_field.vocab.vectors)
        self.model.embedding_layers.embedding_layer.weight.data[self.unk_idx] = torch.zeros(embedding_size)
        self.model.embedding_layers.embedding_layer.weight.data[self.pad_idx] = torch.zeros(embedding_size)
        self.model = self.model.to(DEVICE)
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = self.loss_function.to(DEVICE)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def predict_sentiment(self, sentence): # @todo use this
        self.model.eval()
        tokenized = [token.text for token in self.nlp.tokenizer(sentence)]
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
        for batch_index, batch in enumerate(tqdm_with_message(self.train_iterator, post_yield_message_func = lambda index: f'Training Accuracy {epoch_acc/(index+1)*100:.8f}%', total=number_of_batches, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')):
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = self.model(text, text_lengths)
            loss = self.loss_function(predictions, batch.label)
            acc = discrete_accuracy(predictions, batch.label)
            loss.backward()
            self.optimizer.step()
            self.training_epoch_accuracy_loss_triples.append((self.current_epoch+batch_index/number_of_batches, loss.item(), acc.item()))
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / number_of_batches, epoch_acc / number_of_batches
        
    def evaluate(self, iterator, is_validation):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        number_of_batches = len(iterator)
        with torch.no_grad():
            for batch in tqdm_with_message(iterator, post_yield_message_func = lambda index: f'Validation Accuracy {epoch_acc/(index+1)*100:.8f}%', total=number_of_batches, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
                text, text_lengths = batch.text
                predictions = self.model(text, text_lengths).squeeze(1)
                loss = self.loss_function(predictions, batch.label)
                acc = discrete_accuracy(predictions, batch.label)
                if is_validation:
                    self.validation_epoch_accuracy_loss_triples.append((self.current_epoch+batch_index/number_of_batches, loss.item(), acc.item()))
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / number_of_batches, epoch_acc / number_of_batches

    def validate(self):
        return self.evaluate(self.validation_iterator, True)
    
    def test(self):
        return self.evaluate(self.test_iterator, False)
    
    def train(self):
        print(f'This model has {self.count_parameters()} parameters.')
        saved_model_location = os.path.join(self.output_directory, 'best-performing-model.pt')
        print(f'Best performing models will be saved at {saved_model_location}')
        print(f'Starting training')
        for epoch_index in range(self.number_of_epochs):
            self.current_epoch = epoch_index
            with timer(section_name=f"Epoch {epoch_index}"):
                train_loss, train_acc = self.train_one_epoch()
                valid_loss, valid_acc = self.validate()
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), saved_model_location)
            print(f'\tTrain Loss: {train_loss:.8f} | Train Acc: {train_acc*100:.8f}%')
            print(f'\t Val. Loss: {valid_loss:.8f} |  Val. Acc: {valid_acc*100:.8f}%')
        training_df = pd.DataFrame(self.training_epoch_accuracy_loss_triples, columns=['epoch','accuracy','loss'])
        training_df.to_csv(os.path.join(self.output_directory, 'training_results.csv'))
        self.validation_epoch_accuracy_loss_triples
        validation_df.to_csv(os.path.join(self.output_directory, 'validation_results.csv'))
        self.model.load_state_dict(torch.load(saved_model_location))
        test_loss, test_acc = self.testate()
        print(f'\t Test Loss: {test_loss:.8f} |  Test Acc: {test_acc*100:.8f}%')
        os.remove(saved_model_location)
        # @todo make charts visualizing the loss during this training process
        # @todo make a JSON file summarizing the overall performance. Track best validation score, best training score, # epochs to within 3% of best validation score, # epochs to within 3% of best training score, testing scores, number of parameters

##########
# Driver #
##########

if __name__ == '__main__':
    print("This file contains several text classification models.")
