import random
import methods
from datasets import load_dataset
import argparse

# Set random seed for reproducibility
random.seed(42)


# Function to generate and save a dataset
def generate_dataset(dataset_name, sample_num=None, save_dir="./dataset"):
    if dataset_name == "boolq":
        dataset_path = ["goole/boolq"]
    if dataset_name == "gsm8k":
        dataset_path = ["openai/gsm8k", "main"]
    if dataset_name == "mbpp":
        dataset_path = ["goole-research-datasets/mbpp"]
    if dataset_name == "squad":
        dataset_path = ["rajpurkar/squad"]

    # Load dataset using Hugging Face's datasets library
    ds = load_dataset(*dataset_path)
    ds_total = []

    # Create the output directory if it doesn't exist
    methods.create_directory(save_dir)

    # Determine the save path
    save_path = f"{save_dir}/{dataset_name}.csv"

    # Collect all data points across all dataset splits (e.g., train, test, validation)
    for key in ds.keys():
        for item in ds[key]:
            ds_total.append(item)

    # If a sample number is specified, shuffle and select a subset of the dataset
    if sample_num:
        random.shuffle(ds_total)
        sampled_ds = random.sample(ds_total, sample_num)
    else:
        sampled_ds = ds_total

    # Write the selected data samples to a CSV file
    with open(save_path, "w", encoding="utf-8") as f:
        for item in sampled_ds:
            f.write(f"{item}\n")


# Main function to parse command-line arguments
if __name__ == '__main__':
    # Create an argument parser for command-line inputs
    # generate_dataset(dataset_name="gsm8k", sample_num=1000, save_dir="./dataset")
    parser = argparse.ArgumentParser(description="Generate and save a dataset sample.")

    # Add arguments for dataset name, dataset path, sample number, and save directory
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--sample_num", type=int, default=None, help="Number of samples to generate (optional)")
    parser.add_argument("--save_dir", type=str, default="./dataset", help="Directory to save the dataset")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the dataset generation function with the provided arguments
    generate_dataset(args.dataset, args.sample_num, args.save_dir)
