#!/usr/bin/python3
"#!/usr/bin/python3 -OO" # @todo make this the default

"""
"""

# @todo update the doc string

###########
# Imports #
###########

import os
import json
import pandas as pd
from typing import List, Tuple

import sys ; sys.path.append('..')
from misc_utilities import *

###########
# Globals #
###########

RESULTS_DIR = os.path.join(os.path.dirname(__file__), './results')

NUMBER_OF_BEST_RESULTS_PER_ARCHITECTURE_TO_GATHER = 10
BEST_RESULTS_PER_ARCHITECTURE_OUTPUT_JSON_FILE_LOCATION = os.path.join(os.path.dirname(__file__), './../docs/visualization_data/best_results.json')

ACCURACY_VS_NUMBER_OF_PARAMETERS_JSON_FILE_LOCATION = os.path.join(os.path.dirname(__file__), './../docs/visualization_data/accuracy_vs_number_of_parameters.json')

################
# Process Data #
################

def get_sorted_result_dicts() -> Tuple[List[dict],List[dict]]:
    attention_result_dicts: List[dict] = []
    plain_rnn_result_dicts: List[dict] = []
    result_dirs = os.listdir(RESULTS_DIR)
    for result_base_dir in tqdm_with_message(result_dirs, post_yield_message_func = lambda index: f'Processing model results...'):
        result_dir = os.path.join(RESULTS_DIR, result_base_dir)
        result_summary_json_file_location = os.path.join(result_dir, 'result_summary.json')
        if {'result_summary.json', 'training_results.csv', 'validation_results.csv'}.issubset(os.listdir(result_dir)):
            with open(result_summary_json_file_location, 'r') as file_handle:
                training_progress_csv_location = os.path.join(result_dir, 'training_results.csv')
                training_progress_df = pd.read_csv(training_progress_csv_location)
                validation_progress_csv_location = os.path.join(result_dir, 'validation_results.csv')
                validation_progress_df = pd.read_csv(validation_progress_csv_location)
                result_dict = json.loads(file_handle.read())
                result_dict['result_dir'] = result_dir
                result_dict['training_progress'] = json.loads(training_progress_df.to_json(orient='records'))
                result_dict['validation_progress'] = json.loads(validation_progress_df.to_json(orient='records'))
                for key in {
                        'best_training_accuracy',
                        'best_training_accuracy_epoch',
                        'best_training_loss',
                        'best_training_loss_epoch',
                        'best_validation_accuracy',
                        'best_validation_accuracy_epoch',
                        'best_validation_loss',
                        'best_validation_loss_epoch',
                        'training_number_epochs_to_within_three_percent_of_max_accuracy',
                        'training_number_epochs_to_within_five_percent_of_max_accuracy',
                        'training_number_epochs_to_within_ten_percent_of_max_accuracy',
                        'validation_number_epochs_to_within_three_percent_of_max_accuracy',
                        'validation_number_epochs_to_within_five_percent_of_max_accuracy',
                        'validation_number_epochs_to_within_ten_percent_of_max_accuracy',
                }:
                    result_dict.pop(key, None)
                is_attention_result_dict = result_dict['attention_intermediate_size'] > 0
                if is_attention_result_dict:
                    attention_result_dicts.append(result_dict)
                else:
                    plain_rnn_result_dicts.append(result_dict)
    attention_result_dicts = sorted(attention_result_dicts, key=lambda result_dict: result_dict['test_accuracy'], reverse=True)
    plain_rnn_result_dicts = sorted(plain_rnn_result_dicts, key=lambda result_dict: result_dict['test_accuracy'], reverse=True)
    return attention_result_dicts, plain_rnn_result_dicts

def generate_best_results_per_architecture_json_file(attention_result_dicts_sorted: List[dict], plain_rnn_result_dicts_sorted: List[dict]) -> None:
    best_attention_results = attention_result_dicts_sorted[:NUMBER_OF_BEST_RESULTS_PER_ARCHITECTURE_TO_GATHER]
    best_plain_rnn_results = plain_rnn_result_dicts_sorted[:NUMBER_OF_BEST_RESULTS_PER_ARCHITECTURE_TO_GATHER]
    with open(BEST_RESULTS_PER_ARCHITECTURE_OUTPUT_JSON_FILE_LOCATION, 'w') as file_handle:
        best_results_dict = {'attention': best_attention_results, 'plain_rnn': best_plain_rnn_results}
        json.dump(best_results_dict, file_handle)
    return

def generate_accuracy_vs_number_of_parameters_json_file(attention_result_dicts_sorted: List[dict], plain_rnn_result_dicts_sorted: List[dict]) -> None:
    with open(ACCURACY_VS_NUMBER_OF_PARAMETERS_JSON_FILE_LOCATION, 'w') as file_handle:
        accuracy_vs_number_of_parameters_dict = {
            'attention': [
                {
                    'test_loss': result_dict['test_loss'],
                    'test_accuracy': result_dict['test_accuracy'],
                    'number_of_parameters': result_dict['number_of_parameters'],
                }
                          for result_dict in attention_result_dicts_sorted],
            'plain_rnn': [
                {
                    'test_loss': result_dict['test_loss'],
                    'test_accuracy': result_dict['test_accuracy'],
                    'number_of_parameters': result_dict['number_of_parameters'],
                }
                          for result_dict in plain_rnn_result_dicts_sorted],
        }
        json.dump(accuracy_vs_number_of_parameters_dict, file_handle)
    return

@debug_on_error
def process_data() -> None:
    attention_result_dicts_sorted, plain_rnn_result_dicts_sorted = get_sorted_result_dicts()
    generate_best_results_per_architecture_json_file(attention_result_dicts_sorted, plain_rnn_result_dicts_sorted)
    generate_accuracy_vs_number_of_parameters_json_file(attention_result_dicts_sorted, plain_rnn_result_dicts_sorted)
    return

if __name__ == '__main__':
    process_data()
