#!/usr/bin/python3
"#!/usr/bin/python3 -OO" # @todo make this the default

"""

This file contains the main driver for our functionality to:
* Train all of our models.
* Evaluate all of our models on test data.
* Generate documents comparing the performance of our models.
* Deploy the documents comparing the performance of our models to GitHub Pages.
* End-to-end performance of all of the above.

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
from misc_utilities import debug_on_error

#################
# Functionality #
#################

def model_hyperparameter_specification_generator():
    choices_for_final_representation = ['hidden', 'attention']
    choices_for_batch_size = [32]
    choices_for_max_vocab_size = [25_000]
    choices_for_pre_trained_embedding_specification = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']
    choices_for_encoding_hidden_size = [128, 256]
    choices_for_number_of_encoding_layers = [1, 2]
    choices_for_dropout_probability = [0.5]
    for final_representation in choices_for_final_representation:
        choices_for_number_of_attention_heads = [1, 2, 32] if final_representation == 'attention' else [1]
        choices_for_attention_intermediate_size = [8, 32] if final_representation == 'attention' else [0]
        for batch_size in choices_for_batch_size:
            for max_vocab_size in choices_for_max_vocab_size:
                for pre_trained_embedding_specification in choices_for_pre_trained_embedding_specification:
                    for encoding_hidden_size in choices_for_encoding_hidden_size:
                        for number_of_encoding_layers in choices_for_number_of_encoding_layers:
                            for attention_intermediate_size in choices_for_attention_intermediate_size:
                                for number_of_attention_heads in choices_for_number_of_attention_heads:
                                    for dropout_probability in choices_for_dropout_probability:
                                        output_directory = f"./final_representation_{final_representation}_batch_size_{batch_size}_max_vocab_size_{max_vocab_size}_pre_trained_embedding_specification_{pre_trained_embedding_specification}_encoding_hidden_size_{encoding_hidden_size}_number_of_encoding_layers_{number_of_encoding_layers}_attention_intermediate_size_{attention_intermediate_size}_number_of_attention_heads_{number_of_attention_heads}_dropout_probability_{dropout_probability}/"
                                        yield {
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

def train_models():
    import models
    number_of_epochs = 5
    output_size = 2
    model_hyperparameter_specifications = model_hyperparameter_specification_generator()
    for model_index, model_hyperparameter_specification in enumerate(model_hyperparameter_specifications):
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
        print(f"Working on model {model_index}")
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

def generate_comparison_documents():
    # @todo implement this
    raise NotImplementedError

def deploy_comparison_documents():
    # @todo implement this
    raise NotImplementedError

def end_to_end():
    train_models()
    generate_comparison_documents()
    deploy_comparison_documents()
    return

##########
# Driver #
##########

@debug_on_error
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train-models', action='store_true', help="Instantiate the process of training all of the models we intend to compare.")
    parser.add_argument('-generate-comparison-documents', action='store_true', help="Instantiate the process of generating documents and visualizations comparing all of our models.")
    parser.add_argument('-deploy-comparison-documents', action='store_true', help="Deploy the documents comparing the performance of our models.") # @todo add the actual URL here
    parser.add_argument('-end-to-end', action='store_true', help="Train our models, evaluate our models, generate documents comparing them, and deploy those documents.") # @todo add the actual URL here
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,vars(args).values()))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print("Please only specify one argument.")
    elif args.train_models:
        train_models()
    elif args.evaluate_models:
        evaluate_models()
    elif args.generate_comparison_documents:
        generate_comparison_documents()
    elif args.deploy_comparison_documents:
        deploy_comparison_documents()
    elif args.end_to_end:
        end_to_end()
    else:
        raise Exception("Unexpected args received.")
        
if __name__ == '__main__':
    main()
