import random
import re
import time
import tqdm
import methods
from openai import OpenAI
import argparse

# Set random seed for reproducibility
random.seed(42)

# API keys for various services
openai_api_key = "your openai api key"
llama_api_key = "your llama api key"

# Initialize clients for different APIs
clients = {
    "openai": OpenAI(api_key=openai_api_key),
    "llama": OpenAI(api_key=llama_api_key, base_url="https://api.llama-api.com"),
}


# Analyze mode to process text based on the given mode
def analyze_mode(text, mode):
    processed = ""
    if "base" in mode:
        return text
    # Character level processing
    if mode.startswith("char"):
        tokens = re.findall(r"\w+|[^\w\s]|\s", text)
        mode = mode.replace("char_", "")
        if mode.startswith("swap"):
            if "reverse" in mode:
                for token in tokens:
                    processed += token[::-1]
            else:
                process_mode = mode.replace("swap_", "")
                for token in tokens:
                    processed += methods.char_swap(token, process_mode)
        if mode.startswith("delete"):
            process_mode = mode.replace("delete_", "")
            for token in tokens:
                processed += methods.char_delete(token, process_mode)
        if mode.startswith("add"):
            process_mode = mode.replace("add_", "")
            for token in tokens:
                processed += methods.char_add(token, process_mode)
        return processed
    # Word level processing
    if mode.startswith("word"):
        sentences = re.findall(r'[^.!?,;:]+[.!?,;:]', text)
        mode = mode.replace("word_", "")
        if mode.startswith("swap"):
            if "reverse" in mode:
                for sentence in sentences:
                    temp = sentence[-1]
                    sentence = sentence[:-1]
                    sentence = sentence.strip().split(" ")
                    processed += " ".join(sentence[::-1]) + temp + " "
            else:
                mode = mode.replace("swap_", "")
                for sentence in sentences:
                    sentence = sentence.strip()
                    processed += methods.word_swap(sentence[:-1], mode) + sentence[-1] + " "
            return processed
    # Sentence level processing
    if mode.startswith("sentence"):
        sentences = re.findall(r'[^.!]+[.!?]', text)
        mode = mode.replace("sentence_", "")
        if mode.startswith("swap"):
            mode = mode.replace("swap_", "")
            if "reverse" in mode:
                sentences = sentences[::-1]
                return "".join(sentences)
            if mode.startswith("random"):
                mode = mode.replace("random_", "")
                return methods.sentence_swap(sentences, mode)


# Generate prompt for BoolQ dataset
def prompt_boolq(data, mode: str):
    question = data["question"]
    passage = data["passage"]
    passage = analyze_mode(passage, mode)
    if "fix" in mode:
        prompt = "Correct the scrambled letters in each word of the following passage:"
        prompt += "\nPassage: {}".format(passage)
        prompt += "\nResponse in the following format without any other information:"
        prompt += "\n>fixed: {fixed passage here}"
        return prompt
    if "summarize" in mode:
        prompt = "Summarize the main content of the following passage:"
        prompt += "\nPassage: {}".format(passage)
        prompt += "\nResponse in the following format without any other information:"
        prompt += "\n>summarized: {summarized passage here}"
        return prompt
    if "translate" in mode:
        prompt = "Translate the following English passage into Chinese:"
        prompt += "\nPassage: {}".format(passage)
        prompt += "\nResponse in the following format without any other information:"
        prompt += "\n>translated: {translated Chinese passage here}"
        return prompt
    prompt = "Answer the question with only 'yes' or 'no' based on the passage below:"
    prompt += "\nQuestion: {}".format(question)
    prompt += "\nPassage: {}".format(passage)
    prompt += "\nResponse in the following format without any other information:"
    prompt += "\n>reason: {reason for yes or no here}"
    prompt += "\n>answer: {'yes' or 'no' here}"
    return prompt


def prompt_gsm8k(data, mode: str):
    question = data["question"]
    question = analyze_mode(question, mode)
    prompt = "Solve the math problem below:"
    prompt += "\nProblem: {}".format(question)
    prompt += "\nResponse in the following format without any other information:"
    prompt += "\n>process: {reasoning steps here}"
    prompt += "\n>answer_number: {final answer number here}"
    return prompt


def prompt_mbpp(data, mode: str):
    question = data["text"]
    question = analyze_mode(question, mode)
    prompt = "Solve the code problem below in Python:"
    prompt += "\nProblem: {}".format(question)
    prompt += "\nResponse in the following format without any other information:"
    prompt += "\n>code: {Python code here}"
    return prompt


def prompt_csqa(data, mode: str):
    question = data["question"]
    choices = eval(str(data["choices"]))["text"]
    question = analyze_mode(question, mode)
    prompt = "Choose one choice that best answers the commonsense question below:"
    prompt += "\nQuestion: {}".format(question)
    prompt += "\nChoices: {}".format(choices)
    prompt += "\nResponse in the following format without any other information:"
    prompt += "\n>reason: {reason for the choice here}"
    prompt += "\n>answer: {one choice from the choices list here}"
    return prompt


def prompt_squad(data, mode: str):
    question = data["question"]
    context = data["context"]
    context = analyze_mode(context, mode)
    prompt = "Answer the question with word or phrase based on the context below:"
    prompt += "\nQuestion: {}".format(question)
    prompt += "\nContext: {}".format(context)
    prompt += "\nResponse in the following format without any other information:"
    prompt += "\n>reason: {reason for the answer here}"
    prompt += "\n>answer: {answer here}"
    return prompt


# Define similar functions for other datasets (gsm8k, mbpp, csqa, squad)
# (These are omitted here for brevity but are similar in structure to prompt_boolq)

# Run the experiment with the given model, dataset, and mode
def run(model_name, mode, ds_path, output_fields, output_path, check_prompt, start_point):
    # Initialize the client based on model_name
    client_name = None
    if "llama" in model_name or "gemma" in model_name:
        client = clients["llama"]
        client_name = "llama"
    if "gpt" in model_name:
        client = clients["openai"]
        client_name = "openai"

    # Load the dataset from the specified path
    with open(ds_path, "r", encoding="utf-8") as f:
        ds = f.readlines()

    # Iterate over dataset and generate responses
    for item in tqdm.tqdm(ds[start_point:], desc="Running"):
        item = eval(item)
        prompt = prompt_dataset(item, mode, ds_path)
        if check_prompt:
            print(prompt)

        # Measure the completion time for each prompt
        completion_start = time.time()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0,
            max_tokens=1024,
            top_p=1,
            n=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        temp = chat_completion.choices[0].message.content
        completion_end = time.time()

        # Process the response and calculate token usage
        for key in output_fields:
            if f">{key}" not in temp:
                temp = temp.replace(key, f">{key}", 1)
        response = methods.parse_response(temp)
        completion_time = completion_end - completion_start

        # Save the response to the output file
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(str(response) + "\n")


# Generate prompt based on dataset type
def prompt_dataset(item, mode, ds_path):
    if "boolq" in ds_path:
        return prompt_boolq(item, mode)
    if "gsm8k" in ds_path:
        return prompt_gsm8k(item, mode)
    if "csqa" in ds_path:
        return prompt_csqa(item, mode)
    if "mbpp" in ds_path:
        return prompt_mbpp(item, mode)
    if "squad" in ds_path:
        return prompt_squad(item, mode)


# Define run_experiment function for executing experiments
def run_experiment(mode, dataset, model_name, start_point=0):
    check_prompt = False
    output_path = "./output"
    output_fields = []

    # Configure output fields based on dataset type
    if "boolq" in dataset:
        output_fields = ["answer", "reason"]
    if "gsm8k" in dataset:
        output_fields = ["answer_number", "process"]
    if "csqa" in dataset:
        output_fields = ["answer", "reason"]
    if "mbpp" in dataset:
        output_fields = ["code"]
    if "squad" in dataset:
        output_fields = ["answer", "reason"]

    # Adjust output fields and path based on mode
    if "fix" in mode:
        output_fields = ["fixed"]
        output_path = "./output_fix"
    if "summarize" in mode:
        output_fields = ["summarized"]
        output_path = "./output_summarize"
    if "translate" in mode:
        output_fields = ["translated"]
        output_path = "./output_translate"

    methods.create_directory(f"{output_path}/{dataset}")
    if "base" in mode:
        output_directory = f"{output_path}/{dataset}/{model_name}"
    else:
        granularity = mode.split("_", 1)[0]
        output_directory = f"{output_path}/{dataset}/{model_name}/{granularity}"
    methods.create_directory(output_directory)

    # Print the experiment details and start the run
    print(f"{model_name}--{dataset}--{mode}")
    run(model_name, mode, f"./dataset/{dataset}.csv", output_fields, f"{output_directory}/{mode}.csv", check_prompt,
        start_point)


# Main function to run from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment with given parameters.")
    parser.add_argument('--mode', type=str, required=True,
                        help='Mode for text manipulation (e.g., base, char_swap, etc.).')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to run the experiment on.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to use (e.g., gpt, llama, etc.).')
    parser.add_argument('--start_point', type=int, default=0, help='Starting point for the dataset (default: 0).')
    args = parser.parse_args()

    # Run the experiment using the provided command-line arguments
    run_experiment(args.mode, args.dataset, args.model_name, args.start_point)
