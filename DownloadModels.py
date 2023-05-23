import json
import os
import shutil
import subprocess
import sys

from Common import *

__version__ = 0.1
# Variable holds the dictionary from json file
modelDefinitions = {}
# Variable holds all the models as single list
modelDefinitionsList = []


def get_available_space(path="."):
    _, _, free = shutil.disk_usage(path)
    return free


def build_models_list():
    for modelType in modelDefinitions:
        for modelData in modelDefinitions[modelType]:
            modelDefinitionsList.append(modelData)


def print_models():
    i = 1
    for (model_id, model_name, model_url, model_comment) in modelDefinitionsList:
        isDownloaded = os.path.exists(os.path.join(MODEL_DIR, model_id))
        if isDownloaded:
            logGreen(f"{i}. (+){model_name} - {model_comment}")
        else:
            logWhite(f"{i}. {model_name} - {model_comment}")
        i = i + 1


def clone_model(url, name, folder_name):
    model_path = os.path.join(MODEL_DIR, folder_name)
    if os.path.exists(model_path):
        logGreen(f"Model {name} already exists. Do you want to redownload it? (y/N): ", end="")
        if input().lower() != "y":
            return

    # Check for available disk space
    total, used, free = shutil.disk_usage(os.path.dirname(model_path))
    required_space = 10 * (2 ** 30)  # 10 GB in bytes
    if free < required_space:
        logGreen(
            f"Not enough disk space available to download {name}. "
            f"Please free up at least {required_space / (2 ** 30):.2f} GB and try again."
        )
        return
    logGreen(f"Downloading {name}...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, model_path], check=True)
        logGreen(f"Download of {name} completed.")
        subprocess.run(["git", "lfs", "pull"], cwd=model_path, check=True)
    except subprocess.CalledProcessError:
        logGreen(f"Failed to download {name}. The model may not be available online.")


def remove_existing_model(folder_name):
    model_path = os.path.join(MODEL_DIR, folder_name)
    shutil.rmtree(model_path)


def validate_user_input(user_input, max_value):
    try:
        model_numbers = list(map(int, user_input.split(",")))
        for num in model_numbers:
            if num < 1 or num > max_value:
                logGreen("Invalid model number, please enter a valid number.")
                return None
    except ValueError:
        logGreen("Invalid input, please enter a comma separated list of numbers.")
        return None
    return model_numbers


def main():
    printHeader("DownloadModel - used to download particular models.", __version__)

    # Check if the models directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open('model_defs.json') as data:
        global modelDefinitions
        modelDefinitions = json.load(data)
    build_models_list()

    # Temporary information
    print("!!! Below models are only legacy models. Modern models will be added in further releases !!!\n")

    # Print all models as a list
    print_models()

    model_numbers = None
    while model_numbers is None:
        logGreen("Enter the comma-separated model numbers to run: ")
        user_input = input().strip()
        if not user_input:
            sys.exit(0)
        model_numbers = validate_user_input(user_input, len(modelDefinitionsList))

    for num in model_numbers:
        model_id, model_name, model_url, _ = modelDefinitionsList[num - 1]

        if os.path.exists(os.path.join(MODEL_DIR, model_id)):
            logGreen(f"Model {model_name} already exists. Do you want to redownload it? (y/N): ")
            confirm = input().lower().strip()
            if confirm == "y":
                remove_existing_model(model_id)
                clone_model(model_url, model_name, model_id)
        else:
            clone_model(model_url, model_name, model_id)

    logGreen("\nDownload summary:")
    print_models()


if __name__ == "__main__":
    main()
