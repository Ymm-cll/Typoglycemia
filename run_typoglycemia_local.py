import argparse
import json
import random
import re
import time
import tqdm
import methods
import transformers
import torch
from modelscope import snapshot_download
import time
from datasets import Dataset
from torch.utils.data import DataLoader
import logging
import numpy as np

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

torch.cuda.empty_cache()
random.seed(42)


def analyze_mode(text, mode):
    processed = ""
    if "base" in mode:
        return text
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
    prompt = "Answer the question with a word or phrase based on the context below:"
    prompt += "\nQuestion: {}".format(question)
    prompt += "\nContext: {}".format(context)
    prompt += "\nResponse in the following format without any other information:"
    prompt += "\n>reason: {reason for the answer here}"
    prompt += "\n>answer: {answer here}"
    return prompt


def chat(pipeline, tokenizer, prompt):
    messages = [
        {"role": "user", "content": f"{prompt}"},
    ]

    prompt_start_time = time.time()
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    input_token_count = input_tokens.shape[1]

    generation_start_time = time.time()
    outputs = pipeline(
        messages,
        max_length=8192,
        temperature=1e-6,
        top_p=1,
        num_return_sequences=1,
    )
    output = outputs[0]["generated_text"][-1]["content"]

    generation_end_time = time.time()
    prompt_time = generation_start_time - prompt_start_time
    completion_time = generation_end_time - generation_start_time
    output_tokens = tokenizer.encode(output, return_tensors="pt")
    output_token_count = output_tokens.shape[1]

    usage = {
        "completion_tokens": output_token_count,
        "prompt_tokens": input_token_count,
        "total_tokens": input_token_count + output_token_count,
        "completion_time": completion_time,
        "prompt_time": prompt_time,
        "total_time": completion_time + prompt_time
    }
    return output, usage


def run(model_name, mode, ds_path, output_fields, output_path, check_prompt, save_emb=False):
    def tokenize_function(item):
        # return tokenizer(item["prompt"], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        tokenized_output = tokenizer.apply_chat_template(item["prompt"], max_length=512, truncation=True,
                                                         add_generation_prompt=True, return_tensors="pt")
        print(tokenized_output)
        return tokenized_output

    def batch_iterable(iterable, batch_size):
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]

    def parse(s):
        if "llama" in model_name:
            index = s.rfind("assistant")
            if index != -1:
                return s[index + len("assistant"):]
            else:
                return ""
        if "gemma" in model_name:
            index = s.rfind("model")
            if index != -1:
                return s[index + len("model"):]
            else:
                return ""

    dataset = ds_path.split("/")[-1].replace(".csv", "")
    mode_path = output_path.split("/")[-1].replace(".csv", "")

    if model_name == "llama-3.1-8b":
        model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct",
                                     local_dir="../models/llama-3.1-8b-instruct")
    if model_name == "llama-3.1-70b":
        model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-70B-Instruct",
                                     local_dir="../models/llama-3.1-70b-instruct")
    if model_name == "gemma-2-2b":
        model_id = snapshot_download("LLM-Research/gemma-2-2B-it", local_dir="../models/gemma-2-2b-it")
    if model_name == "gemma-2-9b":
        model_id = snapshot_download("LLM-Research/gemma-2-9B-it", local_dir="../models/gemma-2-9b-it")
    if model_name == "gemma-2-27b":
        model_id = snapshot_download("LLM-Research/gemma-2-27B-it", local_dir="../models/gemma-2-27b-it")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 1
    emb_path = f"./embedding/hidden_states/{dataset}/{model_name}"
    methods.create_directory(emb_path)
    with open(ds_path, "r", encoding="utf-8") as f:
        ds = f.readlines()
        ds_dataset = []
        for item in ds:
            item = eval(item.strip())
            prompt = prompt_dataset(item, mode, ds_path)
            if check_prompt:
                print(prompt)
            ds_dataset.append([{"role": "user", "content": str(prompt)}])

    model.eval()
    batched_ds = list(batch_iterable(ds_dataset, batch_size))
    num = 0
    hidden_records = []
    for batch in tqdm.tqdm(batched_ds, desc="Running"):
        input_ids = []
        prompt_start = time.time()
        for item in batch:
            input_id = tokenizer.apply_chat_template(item, tokenize=True, return_tensors="pt",
                                                     add_generation_prompt=True)
            input_ids.append(input_id[0])
        prompt_end = time.time()
        prompt_time = (prompt_end - prompt_start) / batch_size

        max_length = max(len(lst) for lst in input_ids)
        pad_value = tokenizer.pad_token_id
        padded_lists = [list(lst) + [pad_value] * (max_length - len(lst)) for lst in input_ids]
        input_ids = torch.tensor(padded_lists)

        with torch.no_grad():
            completion_start = time.time()
            output_ids = model.generate(input_ids.to("cuda"), max_new_tokens=512, output_hidden_states=True,
                                        return_dict_in_generate=True, temperature=1e-6, top_p=1,
                                        repetition_penalty=1.0)
            completion_end = time.time()
            completion_time = (completion_end - completion_start) / batch_size
            for i in range(len(input_ids)):
                num += 1
                response_text = tokenizer.decode(output_ids.sequences[i], skip_special_tokens=True)
                response = parse(response_text).strip()

                prompt_tokens = len(input_ids[i])
                total_tokens = len(output_ids.sequences[i])

                temp = response
                for key in output_fields:
                    if f">{key}" not in temp:
                        temp = temp.replace(key, f">{key}", 1)
                response = methods.parse_response(temp)
                response[
                    "usage"] = f"CompletionUsage(completion_tokens={total_tokens - prompt_tokens}, prompt_tokens={prompt_tokens}, total_tokens={total_tokens}, completion_time={completion_time}, prompt_time={prompt_time}, total_time={completion_time + prompt_time})"
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(str(response) + "\n")
                if save_emb:
                    hidden_states_tensor = torch.cat(output_ids.hidden_states[0], dim=0)
                    hidden_states_numpy = hidden_states_tensor.cpu().numpy()
                    ave_hidden = np.mean(hidden_states_numpy, axis=1)
                    hidden_records.append(ave_hidden)
    methods.create_directory(f"{emb_path}")
    np.save(f"{emb_path}/{mode_path}.npy", hidden_records)


def prompt_dataset(item, mode, ds_path):
    if "bool" in ds_path:
        return prompt_boolq(item, mode)
    if "gsm8k" in ds_path:
        return prompt_gsm8k(item, mode)
    if "csqa" in ds_path:
        return prompt_csqa(item, mode)
    if "mbpp" in ds_path:
        return prompt_mbpp(item, mode)
    if "squad" in ds_path:
        return prompt_squad(item, mode)


def run_experiment(mode, dataset, model_name):
    check_prompt = False
    save_emb = True
    output_path = "./output"
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
    print(f"{model_name}--{dataset}--{mode}")
    run(model_name, mode, f"./dataset/{dataset}.csv", output_fields,
        f"{output_directory}/{mode}.csv", check_prompt, save_emb)


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
    run_experiment(args.mode, args.dataset, args.model_name)
