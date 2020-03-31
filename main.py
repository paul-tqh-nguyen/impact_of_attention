#!/usr/bin/python3 -OO

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
import models
from misc_utilities import debug_on_error

#################
# Functionality #
#################

def train_models():
    output_size = 2
    number_of_epochs = 5
    
    max_vocab_size = 25_000
    batch_size = 32
    dropout_probability = 0.5
    pre_trained_embedding_specification = 'glove.6B.100d'
    encoding_hidden_size = 128
    number_of_encoding_layers = 1
    attention_intermediate_size = 8
    number_of_attention_heads = 2
    classifier = models.EEAPClassifier(number_of_epochs,
                                       batch_size,
                                       dropout_probability,
                                       max_vocab_size,
                                       pre_trained_embedding_specification,
                                       encoding_hidden_size,
                                       number_of_encoding_layers,
                                       attention_intermediate_size,
                                       number_of_attention_heads,
                                       output_size)
    classifier.train()
    # @todo finish this implementation

def evaluate_models():
    # @todo implement this
    raise NotImplementedError

def generate_comparison_documents():
    # @todo implement this
    raise NotImplementedError

def deploy_comparison_documents():
    # @todo implement this
    raise NotImplementedError

def end_to_end():
    # @todo implement this
    raise NotImplementedError

##########
# Driver #
##########

@debug_on_error
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train-models', action='store_true', help="Instantiate the process of training all of the models we intend to compare.")
    parser.add_argument('-evaluate-models', action='store_true', help="Instantiate the process of evaluating all of the models we intend to compare and generating visualizations specific to each model.")
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
