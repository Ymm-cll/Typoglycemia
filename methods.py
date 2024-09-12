import os
import random
import re
import string

# Set a random seed for reproducibility
random.seed(42)

# Create a directory if it doesn't already exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Parse text to extract key-value pairs separated by ">"
def parse_response(text):
    # Split text by regex pattern ">key:"
    split_text = re.split(r">([a-z_]*):", text)
    data = {}
    # Loop through split parts and extract key-value pairs
    for i in range(1, len(split_text), 2):
        key = split_text[i].strip().lower()  # Extract key
        value = split_text[i + 1].strip()    # Extract value
        data[key] = value
    return data

# Parse usage data from a string and convert it into a dictionary
def parse_usage(usage_text):
    # Regex pattern to capture key=value pairs
    pattern = r"(\w+)=([\d.]+|None)"
    matches = re.findall(pattern, usage_text)
    # Convert matched values to float, int, or None based on their format
    usage_dict = {
        key: (
            float(value) if "." in value else (int(value) if value.isdigit() else None)
        )
        for key, value in matches
    }
    return usage_dict

# Swap two characters in a word at specified positions
def swap_char(word, a, b):
    word = list(word)  # Convert string to list to allow mutation
    temp = word[a]
    word[a] = word[b]
    word[b] = temp
    return "".join(word)  # Convert list back to string

# Delete a character from a word at a specified position
def delete_char(word, a):
    return word[:a] + "_" + word[a + 1:]  # Replace character with an underscore

# Add a random character to the word at a specified position
def add_char(word, a):
    local_random = random.Random(hash(word) + len(word) + a)  # Generate local random seed
    related_chars = word + string.ascii_letters  # Pool of related characters
    random_char = local_random.choice(related_chars)  # Choose random character
    new_word = word[:a] + random_char + word[a:]  # Insert random character
    return new_word

# Randomly swap two characters in the word
def random_swap_char(word, mode):
    if mode == "int":  # Swap internal characters (excluding first and last)
        a, b = random.sample(range(1, len(word) - 1), 2)
        return swap_char(word, a, b)
    if mode == "all":  # Swap any two characters in the word
        a, b = random.sample(range(0, len(word)), 2)
        return swap_char(word, a, b)

# Randomly delete k characters from the word
def random_delete_char(word, mode, k):
    if mode == "int":  # Delete internal characters
        k = min(len(word) // 2, k)
        positions = random.sample(range(1, len(word) - 1), k)
        for pos in positions:
            word = delete_char(word, pos)
        return word

# Randomly add k characters to the word
def random_add_char(word, mode, k):
    if mode == "int":  # Add random characters to internal positions
        k = min(len(word) // 2, k)
        positions = random.sample(range(1, len(word) - 1), k)
        for pos in positions:
            word = add_char(word, pos)
        return word

# Shuffle characters within a specified range of a word
def char_shuffle(word, begin=None, end=None):
    if begin is None:
        begin = 0
    if end is None:
        end = len(word)
    beg_part, end_part = word[:begin], word[end:]
    int_part = list(word[begin:end])  # Internal part to shuffle
    random.shuffle(int_part)
    return "".join(beg_part) + "".join(int_part) + "".join(end_part)

# Swap characters in a word based on the mode
def char_swap(word: str, mode: str, min=3):
    mode = mode.replace("_fix", "").replace("_summarize", "").replace("_translate", "")
    if any(char.isdigit() for char in word):  # Skip words with digits
        return word
    if len(word) <= min:  # Skip short words
        return word
    if mode == "beg":  # Swap first two characters
        return swap_char(word, 0, 1)
    if mode == "int":  # Swap middle characters
        m_1 = len(word) // 2 - 1
        m_2 = len(word) // 2
        return swap_char(word, m_1, m_2)
    if mode == "end":  # Swap last two characters
        return swap_char(word, len(word) - 2, len(word) - 1)
    if mode.startswith("random"):  # Perform random swaps
        mode = mode.replace("random_", "")
        if "int" in mode:
            if mode == "int":
                return char_shuffle(word, 1, len(word) - 1)
            else:
                if mode.startswith("int_"):
                    k = int(mode.split("_")[1])
                    for _ in range(k):
                        word = random_swap_char(word, "int")
                    return word
        if mode.startswith("all"):
            if mode == "all":
                return char_shuffle(word)
        print("Undefined Mode: char_swap")
        return None

# Delete characters from a word based on the mode
def char_delete(word: str, mode: str, min=3):
    mode = mode.replace("_fix", "").replace("_summarize", "").replace("_translate", "")
    if any(char.isdigit() for char in word):  # Skip words with digits
        return word
    if len(word) <= min:  # Skip short words
        return word
    if mode == "beg":  # Delete first character
        return delete_char(word, 0)
    if mode == "end":  # Delete last character
        return delete_char(word, len(word) - 1)
    if mode.startswith("random"):  # Perform random deletions
        mode = mode.replace("random_", "")
        k = int(mode.split("_")[1])
        word = random_delete_char(word, "int", k)
        return word

# Add characters to a word based on the mode
def char_add(word: str, mode: str, min=3):
    mode = mode.replace("_fix", "").replace("_summarize", "").replace("_translate", "")
    if any(char.isdigit() for char in word):  # Skip words with digits
        return word
    if len(word) <= min:  # Skip short words
        return word
    if mode == "beg":  # Add character at the beginning
        return add_char(word, 0)
    if mode == "end":  # Add character at the end
        return add_char(word, len(word))
    if mode.startswith("random"):  # Perform random additions
        mode = mode.replace("random_", "")
        k = int(mode.split("_")[1])
        word = random_add_char(word, "int", k)
        return word

# Swap words in a sentence based on the mode
def word_swap(sentence: str, mode: str, swap_p=0.5):
    mode = mode.replace("_fix", "").replace("_summarize", "").replace("_translate", "")
    if " " not in sentence:  # Skip if no spaces
        return sentence
    if mode.startswith("random"):  # Perform random word swaps
        if "all" in mode:
            words = sentence.split(" ")
            random.shuffle(words)  # Shuffle all words
            return " ".join(words)
        if "near" in mode:  # Swap adjacent words
            words = sentence.split(" ")
            i = 0
            while i < len(words) - 1:
                if random.random() < swap_p:
                    words[i], words[i + 1] = words[i + 1], words[i]
                    i += 2
                else:
                    i += 1
            return " ".join(words)

# Swap sentences in a passage based on the mode
def sentence_swap(passage, mode: str, swap_p=0.5):
    mode = mode.replace("_fix", "").replace("_summarize", "").replace("_translate", "")
    if "all" in mode:  # Shuffle all sentences
        random.shuffle(passage)
        return "".join(passage)
    if "near" in mode:  # Swap adjacent sentences
        i = 0
        while i < len(passage) - 1:
            if random.random() < swap_p:
                temp = passage[i]
                passage[i] = passage[i + 1]
                passage[i + 1] = temp
                i += 2
            else:
                i += 1
        return "".join(passage)

# Example usage of functions
if __name__ == "__main__":
    print(char_swap("abcd", "end"))
    # Uncomment the lines below to test additional functions
    # print(char_delete("abcd", "random_int_3"))
    # print(char_delete("Have", "random_int_2"))
    # print(char_add("Have", "beg"))
