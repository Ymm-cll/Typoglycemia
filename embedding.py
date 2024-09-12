import tiktoken
from openai import OpenAI
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import run_typoglycemia_api
import methods
import os
import argparse  # Importing argparse for handling command-line arguments

# Set your OpenAI API key for the client
openai_api_key = "your api key"
client = OpenAI(
    api_key=openai_api_key,
)


# Function to calculate cosine similarity between two vectors
def cosine_similarity(list1, list2):
    return dot(list1, list2) / (norm(list1) * norm(list2))


# Function to embed input data based on mode and dataset
def embed_input(mode, dataset):
    # Create directory for saving embeddings
    methods.create_directory(f"./embedding/input/{dataset}")
    # Open the dataset file for reading
    with open(f"./dataset/{dataset}.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Process each line and generate embeddings
        for line in tqdm(lines, desc=f"Embed {mode}"):
            line = eval(line)
            with open(f"./embedding/{dataset}/{mode}.csv", "a", encoding="utf-8") as f_emb:
                # Generate prompt based on dataset type
                if "gsm8k" in dataset:
                    prompt = run_typoglycemia_api.prompt_gsm8k(line, mode)
                if "mbpp" in dataset:
                    prompt = run_typoglycemia_api.prompt_mbpp(line, mode)
                if "bool_q" in dataset:
                    prompt = run_typoglycemia_api.prompt_bool_q(line, mode)
                if "csqa" in dataset:
                    prompt = run_typoglycemia_api.prompt_csqa(line, mode)
                if "squad" in dataset:
                    prompt = run_typoglycemia_api.prompt_squad(line, mode)
                # Generate embeddings using OpenAI API and save
                embedding = (
                    client.embeddings.create(
                        input=[prompt], model="text-embedding-3-large"
                    )
                    .data[0]
                    .embedding
                )
                f_emb.write(str(embedding) + "\n")


# Function to embed output data based on mode, model, dataset, and output directory
def embed_output(mode, model, dataset, output_dir):
    # Create directory for saving embeddings
    methods.create_directory(f"./embedding/{output_dir}/{dataset}/{model}")
    fields = {"gsm8k_400": "process", "bool_q_600": "reason", "csqa_1000": "reason", "squad_500": "reason",
              "mbpp_700": "code"}
    type = ""
    # Handle special output types like summarized or translated
    if "summarize" in output_dir:
        for key in fields.keys():
            fields[key] = "summarized"
        type = "_summarize"
    if "translate" in output_dir:
        for key in fields.keys():
            fields[key] = "translated"
        type = "_translate"

    # Determine the correct path based on mode and model
    if "base" in mode:
        mode_path = f"{model}/base{type}.csv"
    else:
        mode_path = f"{model}/{mode.split('_')[0]}/{mode}{type}.csv"

    # Check if the file exists before processing
    if not os.path.exists(f"./{output_dir}/{dataset}/{mode_path}"):
        print(f"Cannot find ./{output_dir}/{dataset}/{mode_path}")

    # Open the output file and read lines
    with open(f"./{output_dir}/{dataset}/{mode_path}", "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Process each line and generate embeddings
        for line in tqdm(lines, desc=f"Embedding {mode}"):
            line = eval(line)
            text = "DO NOT FIND THE KEY."
            # Find the appropriate text field to embed
            if fields[dataset] in line.keys():
                text = line[fields[dataset]]
            if len(text) == 0:
                text = "DO NOT FIND THE KEY."

            # Create directory for the embeddings
            methods.create_directory(os.path.dirname(f"./embedding/{output_dir}/{dataset}/{mode_path}"))
            with open(f"./embedding/{output_dir}/{dataset}/{mode_path}", "a", encoding="utf-8") as f_emb:
                # Tokenize the text and handle maximum token length
                encoding = tiktoken.get_encoding("cl100k_base")  # Use cl100k_base encoding for smaller embeddings
                tokens = encoding.encode(text.strip())
                max_tokens = 8192  # Set maximum token limit for the model
                if len(tokens) > max_tokens:
                    truncated_tokens = tokens[:max_tokens]
                    text = encoding.decode(truncated_tokens)

                # Generate embeddings using OpenAI API and save
                embedding = (
                    client.embeddings.create(
                        input=[text.strip()], model="text-embedding-3-small"
                    )
                    .data[0]
                    .embedding
                )
                f_emb.write(str(embedding) + "\n")


# Main function to run embedding functions based on command-line arguments
if __name__ == "__main__":
    # Argument parser for running the script with options
    parser = argparse.ArgumentParser(description="Run embedding process for input or output")
    parser.add_argument("--task", type=str, required=True, choices=["embed_input", "embed_output"],
                        help="Task to run: embed_input or embed_output")
    parser.add_argument("--mode", type=str, required=True, help="Mode for embedding")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for embedding")
    parser.add_argument("--model", type=str, help="Model to use for embed_output")
    parser.add_argument("--output_dir", type=str, help="Output directory for embed_output")

    args = parser.parse_args()

    # Handle different tasks based on command-line arguments
    if args.task == "embed_input":
        embed_input(args.mode, args.dataset)
    elif args.task == "embed_output":
        if not args.model or not args.output_dir:
            print("For embed_output, both model and output_dir must be provided.")
        else:
            embed_output(args.mode, args.model, args.dataset, args.output_dir)
