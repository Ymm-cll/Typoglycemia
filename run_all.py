from run_experiment import run_experiment
import time
import methods

model_names = [
        # "llama-3.1-8b",
        # "llama3.1-70b",
        "gemma-2-2b",
        # "gemma-2-9b",
        # "gemma-2-27b",
        # "gpt-3.5-turbo",
        # "gpt-4o-mini",
        # "gpt-4o",
    ]

def run_char():
    datasets = ["gsm8k", "boolq", "mbpp", "csqa", "squad_1000"][:1]
    modes = []
    modes += ["base", "char_swap_random_int", "char_swap_random_all"]
    # modes += ["char_swap_random_int_2", "char_swap_random_int_3", "char_swap_random_int_4"]
    # modes += ["char_swap_beg", "char_swap_end"]
    modes += ["char_swap_reverse"]
    modes += ["char_delete_random_int_1", "char_delete_random_int_2", "char_delete_random_int_3","char_delete_random_int_4"][2:3]
    modes += ["char_delete_beg", "char_delete_end"]

    modes += ["char_add_random_int_1", "char_add_random_int_2", "char_add_random_int_3", "char_add_random_int_4"][2:3]
    # modes += ["char_add_beg", "char_add_end"]
    modes += ["word_swap_random_near",  "word_swap_random_all", "word_swap_reverse"]
    modes += ["sentence_swap_random_near", "sentence_swap_random_all", "sentence_swap_reverse"]
    methods.create_directory(f"./log")
    for dataset in datasets:
        for model in model_names:
            with open(f"./log/{model}.txt", "a", encoding="utf-8") as f:
                for mode in modes:
                    start_time = time.time()
                    run_experiment(mode, dataset, model)
                    end_time = time.time()
                    f.write(f"{model}--{dataset}--{mode}--{end_time - start_time}\n")

if __name__ == '__main__':
    run_char()
