import argparse
import json
import os
import platform
import sys
import time
import warnings

import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
)
from transformers import (
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoConfig,
)

import Common
from Common import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
script_dir = ""
# Variable holds the dictionary from json file
modelDefinitions = {}
# Variable holds all the models as single list
modelDefinitionsList = []
# Variable holds indexes of all available models
available_model_numbers = []

__version__ = 0.2


def printHeader():
    Common.printHeader(
        "Abominable Intelligence: Tech-Priest Chronologist: (Module 5 - AI Timescale Contrast)",
        os.path.basename(__file__),
        "A script that enables posing single questions to multiple AI models simultaneously, facilitating comparative development analysis.",
        __version__, )


def run_model(model_data, device, query_list):
    (model_id, model_name, model_url, model_comment, model_type) = model_data
    model_timings = []
    try:
        start = time.time()
        model_path = os.path.join(MODEL_DIR, model_id)
        print(GREEN("------------------------------------------------------------------------------------"))
        print(GREEN(f"Answer according to model ") + GREEN(BOLD(f"{model_name}")) + GREEN(f" located in {model_path}"))

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = None
        if model_type == 'gpt2':
            model = GPT2LMHeadModel.from_pretrained(model_path)
        elif model_type == 'mt0':
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        elif model_type == 'bloomz':
            model = AutoModelForCausalLM.from_pretrained(model_path)
        elif model_type == 'gptj':
            model = GPTJForCausalLM.from_pretrained(model_path)
        elif model_type == 'gptneo':
            config = GPTNeoConfig.from_pretrained(model_path)
            model = GPTNeoForCausalLM.from_pretrained(model_path, config=config)
        elif model_type == 'HF':
            model = AutoModelForCausalLM.from_pretrained(model_path)
        elif model_type == 'IXL':
            print(GREEN(BOLD("The following model is used as a helper model and cannot be queried.")))
            return []
        # Add other model types here

        if model is not None:
            model.to(device)
            model.eval()

            for query in query_list:
                input_ids = torch.tensor(tokenizer.encode(query)).unsqueeze(0)
                input_ids = input_ids.to(device)  # move input tensor to device
                print(YELLOW("> Query:"))
                print(query)

                with torch.no_grad():
                    # Modify this part according to the model type and desired settings
                    attention_mask = torch.ones_like(input_ids)
                    repetition_penalty = 2.0

                    output_ids = model.generate(
                        input_ids, attention_mask=attention_mask, min_length=100,
                        max_length=160, pad_token_id=50256, repetition_penalty=repetition_penalty
                    )

                output_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print(YELLOW("> Answer:"))
                print(output_sentence)

            end = time.time()
            elapsed_time = end - start
            print(GREEN("Time elapsed (sec): ") + WHITE(str(int(elapsed_time))))
            model_timings.append((model_name, elapsed_time))
        else:
            print(GREEN(f"Model type {model_type} not recognized for model {model_name}"))

    except Exception as e:
        print(GREEN(f"Error processing model {model_name}: {str(e)}"))

    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return model_timings


def print_time_summary(model_timings):
    print(GREEN("===================================================================================="))
    print(GREEN("Time Summary:"))
    sorted_timings = sorted(model_timings, key=lambda x: x[1])
    for modelName, elapsed_time in sorted_timings:
        print(GREEN(f"- {modelName}: ") + WHITE(f"{elapsed_time:.2f} sec"))
    total_time = sum(timing[1] for timing in model_timings)
    print(GREEN("Total time: ") + WHITE(f"{total_time:.2f} sec\n"))


def build_models_list_with_type():
    for modelType in modelDefinitions:
        for modelData in modelDefinitions[modelType]:
            modelData.append(modelType)
            modelDefinitionsList.append(modelData)


def print_models_sectioned():
    legacy_cutoff = 12
    print(GREEN("Legacy models:"))
    i = 1
    for (model_id, model_name, model_url, model_comment, model_type) in modelDefinitionsList[:legacy_cutoff]:
        isDownloaded = os.path.exists(os.path.join(MODEL_DIR, model_id))
        if isDownloaded:
            print(GREEN(BOLD(f"{i}. (+){model_name} - {model_comment}")))
        else:
            print(WHITE(f"{i}. {model_name} - {model_comment}"))
        i += 1
    print()

    print(GREEN("Modern models:"))
    for (model_id, model_name, model_url, model_comment, model_type) in modelDefinitionsList[legacy_cutoff:]:
        isDownloaded = os.path.exists(os.path.join(MODEL_DIR, model_id))
        if isDownloaded:
            print(GREEN(BOLD(f"{i}. (+){model_name} - {model_comment}")))
        else:
            print(WHITE(f"{i}. {model_name} - {model_comment}"))
        i += 1
    print()


def print_device(device):
    print(GREEN("\nUsing device: ") + WHITE(device))

    # Get the number of CPUs
    num_cpus = os.cpu_count()

    # Get the name of the installed CPU
    cpu_name = platform.processor()

    # Get the amount of RAM installed on the system in bytes
    total_mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

    # Convert bytes to GB
    total_mem_gb = total_mem / (1024 ** 3)

    # Round off the result to 2 decimal places
    total_mem_gb = round(total_mem_gb, 2)

    # Print the results
    print(GREEN("Number of threads: ") + WHITE(str(num_cpus)))
    print(GREEN("CPU name: ") + WHITE(str(cpu_name)))
    print(GREEN("Total RAM installed: ") + WHITE(str(total_mem_gb) + GREEN(" GB")))


def get_available_models():
    available_model_numbers = []
    i = 1
    for (model_id, model_name, model_url, model_comment, model_type) in modelDefinitionsList:
        isDownloaded = os.path.exists(os.path.join(MODEL_DIR, model_id))
        if isDownloaded:
            available_model_numbers.append(i)  # Add to the list if the model is available
        i = i + 1
    return available_model_numbers


def main():
    printHeader()
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, choices=['cpu', 'cuda'], help="Device used by a model: 'cpu' or 'cuda'.")
    parser.add_argument("query", nargs="+", type=str, help="Queries to be passed to the model separated by space.")
    args = parser.parse_args()

    # Reading model definitions from file
    with open('model_defs.json') as data:
        global modelDefinitions
        modelDefinitions = json.load(data)
    # Function creates a list from dictionary
    build_models_list_with_type()

    # Print all models as a list
    print_models_sectioned()

    while True:
        print(GREEN(
            "Enter the comma-separated model numbers to run or '0' to run all available models ('end' to exit): "))
        user_input = input().strip()
        if user_input.lower() == 'end':
            sys.exit(0)

        model_numbers_str = user_input.split(",")

        available_model_numbers = get_available_models()

        if user_input == '0':  # Run all available models if the user input is '0'
            model_numbers = available_model_numbers
            break

        # Validate the input, it should be only integers within the available model range
        if all(num_str.isdigit() for num_str in model_numbers_str):
            model_numbers = list(map(int, model_numbers_str))

            # Remove duplicates by converting list to set and then back to list
            model_numbers = list(set(model_numbers))

            if all(num in available_model_numbers for num in
                   model_numbers):  # Check if the numbers are in the available models list
                break
            else:
                print(GREEN(BOLD(
                    "One or more of your models are not available (use DownloadModels.py to download required models). Please enter numbers within the range of available models.\n")))
        else:
            print(GREEN("Invalid input. Please enter numbers separated by comma."))

    # Print the details about chosen device (GPU or CPU)
    print_device(args.device)

    device = torch.device(args.device)
    all_model_timings = []
    for num in model_numbers:
        model_data = modelDefinitionsList[num - 1]
        model_timings = run_model(model_data, device, args.query)
        all_model_timings += model_timings

    print_time_summary(all_model_timings)

    printFooter()


if __name__ == "__main__":
    warnings.filterwarnings("ignore",
                            message="You have modified the pretrained model configuration to control generation.")
    main()
