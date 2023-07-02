import argparse
import os
import re
import sys
import time

import openai
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import GPT2TokenizerFast

import Common
from Common import *

MAX_TOKENS_FOR_MODEL = 4096
CHUNK_SIZE = 1000
__version__ = 0.1


def printHeader():
    Common.printHeader("Abominable Intelligence: Adeptus Telepathica (Module 7 - Noospheric Echo)",
                       os.path.basename(__file__), "A script for seamless interaction with files using the OpenAI API.",
                       __version__)


def preprocess_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def load_and_split_file(filename):
    _, file_extension = os.path.splitext(filename)
    file_size = os.path.getsize(filename)

    if file_extension == ".pdf":
        with open(filename, "rb") as f:
            pdf_reader = PdfReader(f)
            num_pages = len(pdf_reader.pages)
            content = [page.extract_text() for page in pdf_reader.pages]
    elif file_extension == ".txt":
        with open(filename, "r") as f:
            content = f.read()
        content = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
        num_pages = len(content)
    else:
        print(f"Unsupported file type: {file_extension}")
        sys.exit(1)

    return content, num_pages, file_size


def format_file_size(size_in_bytes):
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024 ** 2:
        return f"{size_in_bytes / 1024:.2f} kB"
    elif size_in_bytes < 1024 ** 3:
        return f"{size_in_bytes / 1024 ** 2:.2f} MB"
    else:
        return f"{size_in_bytes / 1024 ** 3:.2f} GB"


def sliding_window(text, window_size, stride):
    start = 0
    end = window_size
    windows = []
    while end <= len(text):
        if start > 0:
            while text[start] != " " and start > 0:
                start -= 1
        windows.append(text[start:end].strip())
        start += stride
        end += stride
    return windows


def clean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _generate(prompt, max_tokens=800):
    openai.api_key = os.environ["OPENAI_API_KEY"]

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.2,
    )

    return clean_text(response.choices[0].text), 1


def main():
    printHeader()
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='The filename to process.')
    parser.add_argument('--key', required=True, type=str, help='OpenAI API key.')
    parser.add_argument('query', type=str, help='Query to be passed to the model.')
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.key

    filename = args.filename
    if not os.path.exists(filename):
        print(GREEN(BOLD(f"The files {filename} was not found!")))
        exit(1)

    print(GREEN("Loading and splitting the file..."))
    pages, num_pages, file_size = load_and_split_file(filename)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.encode = lambda text: tokenizer(text)['input_ids']

    print(GREEN("Tokenizing the text and generating embeddings...\n"))

    filtered_page_texts = []

    count_tokens = lambda input_text: len(tokenizer.encode(input_text[:1024]))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200,
        length_function=count_tokens
    )

    for text in pages:
        cleaned_text = ' '.join(filter(None, text.split()))
        if cleaned_text:
            filtered_page_texts.append(preprocess_text(cleaned_text))

    chunks = text_splitter.create_documents(filtered_page_texts)
    token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
    total_chunks = len(chunks)
    total_tokens = sum(token_counts)
    avg_tokens_per_chunk = total_tokens / total_chunks
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    formatted_file_size = format_file_size(file_size)
    summary = (
        "--- Summary Information ---\n\n"
        f"Total chunks: {total_chunks}\n"
        f"Total tokens: {total_tokens}\n"
        f"Average tokens per chunk: {avg_tokens_per_chunk:.2f}\n"
        f"Filename: {filename}\n"
        f"Number of pages: {num_pages}\n"
        f"File size: {formatted_file_size}\n"
        f"Maximum tokens for model: {MAX_TOKENS_FOR_MODEL}\n"
    )
    print(GREEN(summary))

    while True:
        if args.query is None:
            print(GREEN("Please provide the query to process ('end' to exit):"))
            args.query = input()

        query = args.query
        if query.lower() == "end":
            break

        query_start_time = time.time()
        print(GREEN("Performing similarity search..."))
        docs = db.similarity_search(query)
        top_docs = docs[:3]
        top_docs_text = " ".join([doc.page_content for doc in top_docs])
        new_query = f"Based on the following text, {top_docs_text}, {query}"
        print(GREEN("Sending request to generate a response...\n"))
        response, num_generate_requests = _generate(new_query, max_tokens=800)
        query_end_time = time.time()
        query_time = query_end_time - query_start_time
        source_string = "\n---\n".join(
            [f"Source {i + 1}:\n{clean_text(doc.page_content)}" for i, doc in enumerate(top_docs)])

        print(
            GREEN("--- Query and Response ---\n") +
            YELLOW("> Query: \n") +
            f"{clean_text(query)}\n" +
            YELLOW("> Response: \n") +
            f"{clean_text(response)}\n" +
            YELLOW("Source text: \n") +
            f"{source_string}\n" +
            YELLOW("Query execution time: \n") +
            f"{query_time:.2f} seconds\n" +
            YELLOW(BOLD("Total number of generated requests: \n")) +
            f"{num_generate_requests}\n"
        )

        args.query = None

    printFooter()


if __name__ == "__main__":
    main()
