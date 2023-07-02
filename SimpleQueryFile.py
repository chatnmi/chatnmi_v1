import argparse
import os
import re
import shutil
import time
import warnings

# from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

import Common
from Common import *

os.environ["TRANSFORMERS_OFFLINE"] = "1"

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
# SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

HF_EMBEDDINGS_HELPER_MODEL = "instructor-xl"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)

__version__ = 0.1


def printHeader():
    Common.printHeader("Abominable Intelligence: Chronicles of the Cogitator (Module 2 - Scribe of the Omnissiah)",
                       os.path.basename(__file__), "A script designed for dialoguing with files using a local AI model.",
                       str(__version__),
                       """The path to the Omnissiah's wisdom is no easy journey. Prepare for the data warp storm, ensure your machine is equipped with a CPU pulsating with 48GB of RAM, or a GPU glowing with 16GB vRAM to withstand the onslaught of information. The weak and wanting, lacking these offerings, risk straying off the path, their cries lost in the data-less void. Much like the battlefield, those ill-equipped are left behind. Invoke the sacred RAM, fortify your machines, and prepare to unlock the Omnissiah's wisdom. Those relying on lesser armaments are akin to a Guardsman facing a daemon of the warp, likely to be consumed by the overwhelming darkness.\n""")


def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', '')
    text = text.strip()
    return text


def remove_vector_db(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def load_model(device_type, model_id):
    model_path = os.path.join(MODEL_DIR, model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    device = device_type.lower()
    print(GREEN(f"Running on: {device}"))
    model = LlamaForCausalLM.from_pretrained(model_path)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def check_model(model_id):
    return os.path.exists(os.path.join(MODEL_DIR, model_id))


def load_documents(files):  # -> List[Document]:
    documents = []
    loader = None
    for file_path in files:
        if file_path[-4:] in ['.txt', '.pdf', '.csv']:
            if file_path.endswith(".txt"):
                loader = TextLoader(f"{file_path}", encoding="utf8")
                print(GREEN(" - " + file_path + " - processing as TXT file."))
            elif file_path.endswith(".pdf"):
                loader = PDFMinerLoader(f"{file_path}")
                print(GREEN(" - " + file_path + " - processing as PDF file."))
            elif file_path.endswith(".csv"):
                loader = CSVLoader(f"{file_path}")
                print(GREEN(" - " + file_path + " - processing as CSV file."))
            documents.append(loader.load()[0])
        else:
            print(GREEN(" - " + file_path + " - file extension was not recognized."))
    return documents


def main():
    printHeader()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["7B", "13B"], default="7B")
    parser.add_argument("device", type=str, choices=['cpu', 'cuda'], help="specify 'cpu' or 'cuda'")
    parser.add_argument("files", nargs="+", type=str,
                        help="Multiple file names (separated by space) that will be processed be AI")
    args = parser.parse_args()

    if args.model == '13B':
        model_id = "Wizard-Vicuna-13B-Uncensored-HF"
    else:
        model_id = "vicuna-7B-1.1-HF"

    print(GREEN(f"Checking for model:{model_id}"))
    # Verify if the model is available
    if not check_model(model_id):
        print(f"Model {model_id} was not found in folder {MODEL_DIR}")
        print(GREEN(BOLD(f"Please use \"DownloadModels.py\" script to download a model \"{model_id}\"")))
        exit(1)
    print(GREEN("Model found."))

    # Loading the documents
    print(GREEN("Loading documents..."))
    documents = load_documents(args.files)

    if len(documents) == 0:
        print(GREEN("No files were loaded"))

    llm = load_model(args.device, model_id)
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        print(GREEN(f"Split files into {len(texts)} chunks of text"))

        embeddings = HuggingFaceInstructEmbeddings(model_name=os.path.join(MODEL_DIR, HF_EMBEDDINGS_HELPER_MODEL),
                                                   model_kwargs={"device": args.device})
        db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY,
                                   client_settings=CHROMA_SETTINGS)
        print(GREEN("Persisting DB."))
        db.persist()
        print(GREEN("Done."))
        db = None
    else:
        print(GREEN("No changes detected. Skipping database rebuilding."))

    embeddings = HuggingFaceInstructEmbeddings(model_name=os.path.join(MODEL_DIR, HF_EMBEDDINGS_HELPER_MODEL),
                                               model_kwargs={"device": args.device})
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    while True:
        query_start_time = time.time()
        query = input("\nEnter a query ('end' to exit): ")
        if query == "end":
            db.delete_collection()
            break

        # Get the answer from the chain
        inference_start_time = time.time()
        res = qa(query)
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        answer, docs = res['result'], res['source_documents']

        # Print the result
        print(YELLOW("\n\n> Query:"))
        print(query)
        print(YELLOW("\n> Answer:"))
        print(answer)

        # Print the relevant sources used for the answer
        source_string = "\n---\n".join(
            [f"Source {i + 1}:\n{clean_text(document.page_content)}" for i, document in enumerate(docs)])

        print(YELLOW("\n> Source:"))
        print(source_string)

        query_end_time = time.time()
        query_time = query_end_time - query_start_time

        print(YELLOW("\nQuery execution time: "))
        print(f"{round(query_time)} seconds\nInference time: {round(inference_time)} seconds")

    remove_vector_db(PERSIST_DIRECTORY)


if __name__ == "__main__":
    warnings.filterwarnings("ignore",
                            message="You have modified the pretrained model configuration to control generation.")
    main()
