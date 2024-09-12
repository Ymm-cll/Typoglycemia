import os
import re
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine


def evaluate_gsm8k(model_name, output_path):
    def equal(y, pred):
        return y == pred

    def clean(s, item):
        s_clean = ''.join(re.findall(r'\d+', s))
        try:
            s_clean = float(s_clean)
            return s_clean
        except ValueError:
            return s_clean

    with open('./dataset/gsm8k.csv', 'r', encoding="utf-8") as f:
        correct = f.readlines()
    all_files = [str(file) for file in Path(f"{output_path}/gsm8k/{model_name}").rglob('*') if file.is_file()]
    for file in all_files:
        count = 0
        mode = file.split("/")[-1].replace(".csv", "")
        with open(file, 'r', encoding="utf-8") as f:
            test = f.readlines()
        for i in range(len(test)):
            y_item = eval(correct[i].strip())
            pred_item = eval(test[i].strip())
            if "answer_number" in pred_item.keys():
                y = y_item["answer"].split("\n#### ")[-1]
                pred = pred_item["answer_number"]
                if equal(clean(y, y_item), clean(pred, pred_item)):
                    count += 1
                else:
                    if i < 10 and "special" in mode:
                        print(mode, i + 1)
        print(f"{mode}: {count / len(test)}")


def evaluate_bool_q(model_name, output_path):
    def equal(y, pred):
        return (y == "True" and pred.replace("'", "") == "yes") or (y == "False" and pred.replace("'", "") == "no")

    with open('./dataset/boolq.csv', 'r', encoding="utf-8") as f:
        correct = f.readlines()
    all_files = [str(file) for file in Path(f"{output_path}/boolq/{model_name}").rglob('*') if file.is_file()]
    for file in all_files:
        count = 0
        mode = file.split("/")[-1].replace(".csv", "")
        with open(file, 'r', encoding="utf-8") as f:
            test = f.readlines()
        for i in range(len(test)):
            y_item = eval(correct[i].strip())
            pred_item = eval(test[i].strip())
            if "answer" in pred_item.keys():
                y = str(y_item["answer"])
                pred = pred_item["answer"]
                if equal(y, pred):
                    count += 1
        print(f"{mode}: {count / len(test)}")


def evaluate_squad(model_name, output_path):
    def equal(y, pred):
        return (y in pred) or (pred in y)

    with open('./dataset/squad.csv', 'r', encoding="utf-8") as f:
        correct = f.readlines()
    all_files = [str(file) for file in Path(f"{output_path}/squad/{model_name}").rglob('*') if file.is_file()]
    for file in all_files:
        count = 0
        mode = file.split("/")[-1].replace(".csv", "")
        with open(file, 'r', encoding="utf-8") as f:
            test = f.readlines()
        for i in range(len(test)):
            y_item = eval(correct[i].strip())
            pred_item = eval(test[i].strip())
            if "answer" in pred_item.keys():
                y = str(y_item["answers"]["text"][0])
                pred = pred_item["answer"]
                if equal(y.lower(), pred.lower()):
                    count += 1
        print(f"{mode}: {count / len(test)}")


def evaluate_mbpp(model, output_path):
    path = f"./embedding/output/mbpp/{model}"
    base_path = f"./embedding/output/mbpp/gpt-4o-songshu/base.csv"
    file_names = []
    file_names.append(f"base.csv")
    for type in ["char", "word", "sentence"]:
        for filename in os.listdir(path + f"/{type}"):
            file_names.append(f"/{type}/{filename}")
    for file in file_names:
        # 读取文件并将每一行转换为list
        def read_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                # 假设每行是一个list的字符串格式，使用eval将其转换为实际list
                data = [eval(line.strip()) for line in lines]
            return data

        # 计算两个列表之间的余弦相似度
        def cosine_similarity(list1, list2):
            # scipy的cosine计算的是余弦距离，余弦相似度是 1 - 余弦距离
            return 1 - cosine(list1, list2)

        # 计算两个文件对应行的余弦相似度的平均值
        def average_cosine_similarity(file1, file2):
            data1 = read_file(file1)
            data2 = read_file(file2)
            # 确保两个文件的行数相同
            assert len(data1) == len(data2), "两个文件的行数不同！"
            similarities = []
            for list1, list2 in zip(data1, data2):
                sim = cosine_similarity(list1, list2)
                similarities.append(sim)
            # 计算余弦相似度的平均值
            avg_similarity = np.mean(similarities)
            return avg_similarity

        # 示例：文件路径
        file1 = base_path
        file2 = f"{path}/{file}"

        # 计算平均余弦相似度
        avg_cos_sim = average_cosine_similarity(file1, file2)
        print(f"{file2}: {avg_cos_sim}")


def evaluate_csqa(model_name, output_path):
    def equal(y, pred):
        return (y in pred) or (pred in y)

    with open('./dataset/csqa.csv', 'r', encoding="utf-8") as f:
        correct = f.readlines()
    all_files = [str(file) for file in Path(f"{output_path}/csqa/{model_name}").rglob('*') if file.is_file()]
    for file in all_files:
        count = 0
        mode = file.split("/")[-1].replace(".csv", "")
        with open(file, 'r', encoding="utf-8") as f:
            test = f.readlines()
        for i in range(len(test)):
            y_item = eval(correct[i].strip())
            pred_item = eval(test[i].strip())
            if "answer" in pred_item.keys():
                y_choices_text = y_item["choices"]["text"]
                y_answer_key = y_item["answerKey"]
                if y_answer_key == "":
                    continue
                index = ord(y_answer_key.upper()) - ord('A')
                y = y_choices_text[index]
                pred = pred_item["answer"]
                if equal(y.lower(), pred.lower()):
                    count += 1
        print(f"{mode}: {count / len(test)}")


def evaluate_token(model_name, dataset, output_path, fields, target_mode):
    def extract_usage(usage):
        pattern = r'(\w+)=([\d.]+)'
        matches = re.findall(pattern, usage)
        extracted_data = {}
        for key, value in matches:
            extracted_data[key] = float(value)
        return extracted_data

    with open(f'./output/{dataset}/{model_name}/base.csv', 'r', encoding="utf-8") as f:
        correct = f.readlines()
    all_files = [str(file) for file in Path(f"{output_path}/{dataset}/{model_name}").rglob('*') if file.is_file()]
    data = {dataset: {}}
    for file in all_files:
        if "base" in file:
            continue
        mode = file.split("\\")[-1].replace(".csv", "")
        # count = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0,
        #          "completion_time": 0, "prompt_time": 0, "total_time": 0}
        count = {}
        for key in fields:
            count[key] = 0
        with open(file, 'r', encoding="utf-8") as f:
            test = f.readlines()
            for i in range(min(len(test), len(correct))):
                y_item = eval(correct[i].strip())
                pred_item = eval(test[i].strip())
                y_usage = y_item["usage"]
                pred_usage = pred_item["usage"]
                y_usage_extracted = extract_usage(y_usage)
                pred_usage_extracted = extract_usage(pred_usage)
                for key in y_usage_extracted.keys():
                    if key in fields:
                        if y_usage_extracted[key] != 0:
                            count[key] += pred_usage_extracted[key] / y_usage_extracted[key]
            for key in count.keys():
                count[key] = count[key] / min(len(test), len(correct))
            if mode.endswith(target_mode):
                if mode not in data[dataset].keys():
                    data[dataset][mode] = {}
                data[dataset][mode][model_name] = count
    print(data)
    print("---------------")
    return {'prompt_tokens': data[dataset][target_mode][model_name]['prompt_tokens'],
            'completion_time': data[dataset][target_mode][model_name]['completion_time']}


def evaluate_fix(model_name, dataset, output_path):
    def equal(y, pred):
        count_fix = 0
        clean_y = re.sub(r'[^\w\s]', '', y)
        clean_pred = re.sub(r'[^\w\s]', '', pred)
        words_y = clean_y.split()
        words_pred = clean_pred.split()
        num = min(len(words_y), len(words_pred))
        if num == 0:
            return 0
        for i in range(num):
            if words_y[i].lower() == words_pred[i].lower():
                count_fix += 1
        return count_fix / num

    path = f"{output_path}/{dataset}/{model_name}"
    with open('./dataset/boolq.csv', 'r', encoding="utf-8") as f:
        correct = f.readlines()
    all_files = [str(file) for file in Path(path).rglob('*') if file.is_file()]
    for file in all_files:
        count = 0
        mode = file.split("/")[-1].replace(".csv", "")
        with open(file, 'r', encoding="utf-8") as f:
            test = f.readlines()
        num = 0
        for i in range(len(test)):
            y_item = eval(correct[i].strip())
            pred_item = eval(test[i].strip())
            if "fixed" in pred_item.keys():
                y = str(y_item["passage"])
                pred = pred_item["fixed"]
                count += equal(y, pred)
                if count != 0:
                    num += 1
        print(f"{mode}: {count / num}")


def evaluate_input(dataset):
    path = f"./embedding/input/{dataset}"
    base_path = f"{path}/base.csv"
    file_names = []
    for filename in os.listdir(path):
        file_names.append(filename)
    for file in file_names:
        # 读取文件并将每一行转换为list
        def read_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                # 假设每行是一个list的字符串格式，使用eval将其转换为实际list
                data = [eval(line.strip()) for line in lines]
            return data

        # 计算两个列表之间的余弦相似度
        def cosine_similarity(list1, list2):
            # scipy的cosine计算的是余弦距离，余弦相似度是 1 - 余弦距离
            return 1 - cosine(list1, list2)

        # 计算两个文件对应行的余弦相似度的平均值
        def average_cosine_similarity(file1, file2):
            data1 = read_file(file1)
            data2 = read_file(file2)
            # 确保两个文件的行数相同
            assert len(data1) == len(data2), "两个文件的行数不同！"
            similarities = []
            for list1, list2 in zip(data1, data2):
                sim = cosine_similarity(list1, list2)
                similarities.append(sim)
            # 计算余弦相似度的平均值
            avg_similarity = np.mean(similarities)
            return avg_similarity

        # 示例：文件路径
        file1 = base_path
        file2 = f"{path}/{file}"

        # 计算平均余弦相似度
        avg_cos_sim = average_cosine_similarity(file1, file2)
        print(f"{file2}: {avg_cos_sim}")


def evaluate_output_summarize(model, dataset):
    path = f"./embedding/output_summarize/{dataset}/{model}"
    base_path = f"{path}/base_summarize.csv"
    file_names = []
    for filename in os.listdir(f"{path}/char"):
        file_names.append(filename)
    for file in file_names:
        # 读取文件并将每一行转换为list
        def read_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                # 假设每行是一个list的字符串格式，使用eval将其转换为实际list
                data = [eval(line.strip()) for line in lines]
            return data

        # 计算两个列表之间的余弦相似度
        def cosine_similarity(list1, list2):
            # scipy的cosine计算的是余弦距离，余弦相似度是 1 - 余弦距离
            return 1 - cosine(list1, list2)

        # 计算两个文件对应行的余弦相似度的平均值
        def average_cosine_similarity(file1, file2):
            data1 = read_file(file1)
            data2 = read_file(file2)
            # 确保两个文件的行数相同
            assert len(data1) == len(data2), "两个文件的行数不同！"
            similarities = []
            for list1, list2 in zip(data1, data2):
                sim = cosine_similarity(list1, list2)
                similarities.append(sim)
            # 计算余弦相似度的平均值
            avg_similarity = np.mean(similarities)
            return avg_similarity

        # 示例：文件路径
        file1 = base_path
        file2 = f"{path}/char/{file}"

        # 计算平均余弦相似度
        avg_cos_sim = average_cosine_similarity(file1, file2)
        print(f"{file2}: {avg_cos_sim}")


def evaluate_output_translate(model, dataset):
    path = f"./embedding/output_translate/{dataset}/{model}"
    base_path = f"{path}/base_translate.csv"
    file_names = []
    for filename in os.listdir(f"{path}/char"):
        file_names.append(filename)
    for file in file_names:
        # 读取文件并将每一行转换为list
        def read_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                # 假设每行是一个list的字符串格式，使用eval将其转换为实际list
                data = [eval(line.strip()) for line in lines]
            return data

        # 计算两个列表之间的余弦相似度
        def cosine_similarity(list1, list2):
            # scipy的cosine计算的是余弦距离，余弦相似度是 1 - 余弦距离
            return 1 - cosine(list1, list2)

        # 计算两个文件对应行的余弦相似度的平均值
        def average_cosine_similarity(file1, file2):
            data1 = read_file(file1)
            data2 = read_file(file2)
            # 确保两个文件的行数相同
            assert len(data1) == len(data2), "两个文件的行数不同！"
            similarities = []
            for list1, list2 in zip(data1, data2):
                sim = cosine_similarity(list1, list2)
                similarities.append(sim)
            # 计算余弦相似度的平均值
            avg_similarity = np.mean(similarities)
            return avg_similarity

        # 示例：文件路径
        file1 = base_path
        file2 = f"{path}/char/{file}"

        # 计算平均余弦相似度
        avg_cos_sim = average_cosine_similarity(file1, file2)
        print(f"{file2}: {avg_cos_sim}")


if __name__ == '__main__':
    model_names = [
        "llama-3.1-8b-instruct",
        "llama3.1-70b",
        "gemma-2-2b-it",
        "gemma-2-9b-it",
        "gemma2-27b",
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-4o-songshu"
    ]
    data = {}
    for model_name in model_names:
        data[model_name] = {}
    for model_name in model_names:
        print(model_name)
        # evaluate_gsm8k(model_name,"./output")
        # evaluate_bool_q(model_name, "./output")
        evaluate_squad(model_name, "./output")
        # evaluate_mbpp(model_name, "./output")
        # evaluate_csqa(model_name, "./output")
        # evaluate_token(model_name, "gsm8k", "./output", ["prompt_tokens", "completion_time"], "char_swap_random_int")
        # evaluate_fix(model_name, "boolq", "./output_fix")
        # evaluate_output_summarize(model_name, "boolq")
        # evaluate_output_translate(model_name, "boolq")
    # evaluate_input("mbpp")
