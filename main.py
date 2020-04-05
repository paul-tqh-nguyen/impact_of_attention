#!/usr/bin/python3
"#!/usr/bin/python3 -OO" # @todo make this the default

"""

This file contains the main driver for our functionality to:
* Train models.
* Generate documents comparing the performance of our models.
* Deploy the documents comparing the performance of our models to GitHub Pages.

Owner : paul-tqh-nguyen

Created : 03/30/2020

File Name : main.py

File Organization:
* Imports
* Functionality
* Driver

"""

###########
# Imports #
###########

import argparse
import random
import json
import os
from typing import Generator
from misc_utilities import debug_on_error

#################
# Functionality #
#################

def random_model_hyperparameter_specification() -> dict:
    
    choices_for_final_representation = ['hidden', 'attention']
    final_representation = random.choice(choices_for_final_representation)
    
    choices_for_batch_size = [32]
    batch_size = random.choice(choices_for_batch_size)
    
    choices_for_max_vocab_size = [25_000, 50_000]
    max_vocab_size = random.choice(choices_for_max_vocab_size)
    
    choices_for_pre_trained_embedding_specification = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']
    pre_trained_embedding_specification = random.choice(choices_for_pre_trained_embedding_specification)
    
    choices_for_encoding_hidden_size = [128, 256, 512]
    encoding_hidden_size = random.choice(choices_for_encoding_hidden_size)
    
    choices_for_number_of_encoding_layers = [1, 2]
    number_of_encoding_layers = random.choice(choices_for_number_of_encoding_layers)
    
    choices_for_dropout_probability = [0.0, 0.25, 0.5]
    dropout_probability = random.choice(choices_for_dropout_probability)
    
    choices_for_number_of_attention_heads = [1, 2, 32] if final_representation == 'attention' else [1]
    number_of_attention_heads = random.choice(choices_for_number_of_attention_heads)
    
    choices_for_attention_intermediate_size = [8, 32] if final_representation == 'attention' else [0]
    attention_intermediate_size = random.choice(choices_for_attention_intermediate_size)
    
    output_directory = f"./results/final_repr_{final_representation}_batch_{batch_size}_max_vocab_{max_vocab_size}_embedding_spec_{pre_trained_embedding_specification}_encoding_size_{encoding_hidden_size}_num_encoding_{number_of_encoding_layers}_attn_inter_size_{attention_intermediate_size}_num_attn_heads_{number_of_attention_heads}_dropout_{dropout_probability}/"
    
    return {
        'final_representation': final_representation,
        'batch_size': batch_size,
        'max_vocab_size': max_vocab_size,
        'pre_trained_embedding_specification': pre_trained_embedding_specification,
        'encoding_hidden_size': encoding_hidden_size,
        'number_of_encoding_layers': number_of_encoding_layers,
        'attention_intermediate_size': attention_intermediate_size,
        'number_of_attention_heads': number_of_attention_heads,
        'dropout_probability': dropout_probability,
        'output_directory': output_directory,
    }

def model_hyperparameter_specification_has_been_evaluated(model_hyperparameter_specification: dict) -> bool:
    import models
    output_directory = model_hyperparameter_specification['output_directory']
    if os.path.isdir(output_directory):
        result_summary_json_file_location = os.path.join(output_directory,'result_summary.json')
        if os.path.isfile(result_summary_json_file_location):
            with open(result_summary_json_file_location, 'r') as result_summary_json_string:
                result_summary = json.load(result_summary_json_string)
                if models.EXPECTED_RESULT_SUMMARY_KEY_WORDS == set(result_summary.keys()):
                    return True
    return False

def unevaluated_model_hyperparameter_specification() -> dict:
    model_hyperparameter_specification = random_model_hyperparameter_specification()
    has_already_been_evaluated = model_hyperparameter_specification_has_been_evaluated(model_hyperparameter_specification)
    while has_already_been_evaluated:
        model_hyperparameter_specification = random_model_hyperparameter_specification()
        has_already_been_evaluated = model_hyperparameter_specification_has_been_evaluated(model_hyperparameter_specification)
    return model_hyperparameter_specification

def train_models() -> None:
    import models
    number_of_epochs = 5
    output_size = 2
    while True:
        model_hyperparameter_specification = unevaluated_model_hyperparameter_specification()
        batch_size = model_hyperparameter_specification['batch_size']
        max_vocab_size = model_hyperparameter_specification['max_vocab_size']
        pre_trained_embedding_specification = model_hyperparameter_specification['pre_trained_embedding_specification']
        encoding_hidden_size = model_hyperparameter_specification['encoding_hidden_size']
        number_of_encoding_layers = model_hyperparameter_specification['number_of_encoding_layers']
        attention_intermediate_size = model_hyperparameter_specification['attention_intermediate_size']
        number_of_attention_heads = model_hyperparameter_specification['number_of_attention_heads']
        dropout_probability = model_hyperparameter_specification['dropout_probability']
        final_representation = model_hyperparameter_specification['final_representation']
        output_directory = model_hyperparameter_specification['output_directory']
        print()
        print(f"Model hyperparameters are:")
        print(f'        number_of_epochs: {number_of_epochs}')
        print(f'        batch_size: {batch_size}')
        print(f'        max_vocab_size: {max_vocab_size}')
        print(f'        pre_trained_embedding_specification: {pre_trained_embedding_specification}')
        print(f'        encoding_hidden_size: {encoding_hidden_size}')
        print(f'        number_of_encoding_layers: {number_of_encoding_layers}')
        if final_representation == 'attention':
            print(f'        attention_intermediate_size: {attention_intermediate_size}')
            print(f'        number_of_attention_heads: {number_of_attention_heads}')
        print(f'        output_size: {output_size}')
        print(f'        dropout_probability: {dropout_probability}')
        print(f'        final_representation: {final_representation}')
        print(f'        output_directory: {output_directory}')
        print()
        print(f"Saving in {output_directory}")
        print()
        classifier = models.EEAPClassifier(number_of_epochs, batch_size, max_vocab_size, pre_trained_embedding_specification, encoding_hidden_size, number_of_encoding_layers, attention_intermediate_size, number_of_attention_heads, output_size, dropout_probability, final_representation, output_directory)
        classifier.train()
        print("\n\n")
    return

def generate_comparison_documents() -> None:
    # @todo implement this
    raise NotImplementedError

def deploy_comparison_documents() -> None:
    # @todo implement this
    raise NotImplementedError

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-train-models', action='store_true', help=" Instantiate the process of training models we intend to compare. This is intended to run indefinitely as the number of models is explosive.")
    parser.add_argument('-generate-comparison-documents', action='store_true', help=" Instantiate the process of generating documents and visualizations comparing all of our models.")
    parser.add_argument('-deploy-comparison-documents', action='store_true', help=" Deploy the documents comparing the performance of our models.") # @todo add the actual URL here
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,vars(args).values()))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print("Please specify exactly one action.")
    elif args.train_models:
        train_models()
    elif args.generate_comparison_documents:
        generate_comparison_documents()
    elif args.deploy_comparison_documents:
        deploy_comparison_documents()
    else:
        raise Exception("Unexpected args received.")
        
if __name__ == '__main__':
    main()
