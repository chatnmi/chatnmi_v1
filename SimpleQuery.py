import argparse
import json
import os
import platform
import sys
import time

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

from Common import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
script_dir = ""
# Variable holds the dictionary from json file
modelDefinitions = {}
# Variable holds all the models as single list
modelDefinitionsList = []
# Variable holds indexes of all available models
available_model_numbers = []

__version__ = 0.1


def run_model(model_data, device, query_list):
    (model_id, model_name, model_url, model_comment, model_type) = model_data
    model_timings = []
    try:
        start = time.time()
        model_path = os.path.join(MODEL_DIR, model_id)
        logGreen("------------------------------------------------------------------------------------")
        logGreen(f"Answer according to model {model_name} located in {model_path}")

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
        # Add other model types here

        if model is not None:
            model.to(device)
            model.eval()

            for query in query_list:
                input_ids = torch.tensor(tokenizer.encode(query)).unsqueeze(0)
                input_ids = input_ids.to(device)  # move input tensor to device

                with torch.no_grad():
                    # Modify this part according to the model type and desired settings
                    attention_mask = torch.ones_like(input_ids)
                    repetition_penalty = 2.0
                    output_ids = model.generate(
                        input_ids, attention_mask=attention_mask, min_length=100,
                        max_length=160, pad_token_id=50256, repetition_penalty=repetition_penalty
                    )

                output_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                logGreen("", output_sentence)

            end = time.time()
            elapsed_time = end - start
            logGreen("Time elapsed (sec): ", str(int(elapsed_time)))
            model_timings.append((model_name, elapsed_time))
        else:
            logGreen(f"Model type {model_type} not recognized for model {model_name}")

    except Exception as e:
        logGreen(f"Error processing model {model_name}: {str(e)}")

    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return model_timings



def print_time_summary(model_timings):
    logGreen("=================================================================================")
    logGreen("Time Summary:")
    sorted_timings = sorted(model_timings, key=lambda x: x[1])
    for modelName, elapsed_time in sorted_timings:
        logGreen(f"- {modelName}: ", end="")
        logWhite(f"{elapsed_time:.2f} sec")
    total_time = sum(timing[1] for timing in model_timings)
    logGreen("Total time: ", end="")
    logWhite(f"{total_time:.2f} sec\n")


def build_models_list_with_type():
    for modelType in modelDefinitions:
        for modelData in modelDefinitions[modelType]:
            modelData.append(modelType)
            modelDefinitionsList.append(modelData)


def print_models():
    global available_model_numbers  # New variable to keep track of available model numbers
    available_model_numbers = []
    i = 1
    for (model_id, model_name, model_url, model_comment, model_type) in modelDefinitionsList:
        isDownloaded = os.path.exists(os.path.join(MODEL_DIR, model_id))
        if isDownloaded:
            logGreen(f"{i}. (+){model_name} - {model_comment}")
            available_model_numbers.append(i)  # Add to the list if the model is available
        else:
            logWhite(f"{i}. {model_name} - {model_comment}")
        i = i + 1


def print_device(device):
    logGreen("\nUsing device: ", device)

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
    logGreen("Number of threads: ", str(num_cpus))
    logGreen("CPU name: ", str(cpu_name))
    logGreen("Total RAM installed: ", str(total_mem_gb) + " GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, choices=['cpu', 'cuda'], help="specify 'cpu' or 'cuda'")
    parser.add_argument("query", nargs="+", type=str, help="query to be passed to the model")
    args = parser.parse_args()
    printHeader("SimpleQuery script - used to compare different models.", __version__)

    # Reading model definitions from file
    with open('model_defs.json') as data:
        global modelDefinitions
        modelDefinitions = json.load(data)
    # Function creates a list from dictionary
    build_models_list_with_type()

    # Temporary information
    logWhite("!!! Below models are only legacy models. Modern models will be added in further releases !!!\n")

    # Print all models as a list
    print_models()

    while True:
        logGreen("Enter the comma-separated model numbers to run, '0' to run all available models, or 'exit' to quit: ")
        user_input = input().strip()
        if user_input.lower() == 'exit':
            sys.exit(0)

        model_numbers_str = user_input.split(",")

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
                logGreen(
                    "One or more of your numbers are not available models. Please enter numbers within the range of available models.")
        else:
            logGreen("Invalid input. Please enter numbers separated by comma.")

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
    main()
