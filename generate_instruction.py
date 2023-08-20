"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time   # 引入时间模块
import json   # 引入json模块
import os     # 引入os模块
import random # 引入随机数模块
import re     # 引入正则表达式模块
import string # 引入字符串模块
from functools import partial # 引入偏函数模块
from multiprocessing import Pool # 引入多进程模块

import numpy as np
import tqdm
from rouge_score import rouge_scorer # 引入rouge评分器，用于文本相似度计算
import utils  # 引入自定义的工具模块

import fire  # 引入fire库，用于命令行参数解析


# 定义一个将多个提示指令编码成单一字符串的函数
def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    # 打开并读取提示文本文件
    prompt = open("./prompt.txt").read() + "\n"

    # 遍历提示指令，将其格式化并附加到提示字符串中
    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        # 对指令进行清洗
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        # 若无输入则标注为"<noinput>"
        input = "<noinput>" if input.lower() == "" else input
        # 格式化并添加指令、输入和输出到提示中
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"  # 添加下一个指令的前缀
    return prompt


# 定义一个对GPT-3响应进行后处理的函数，抽取生成的新指令
def post_process_gpt3_response(num_prompt_instructions, response):
    # 如果响应为空，则返回空列表
    if response is None:
        return []
    # 获取原始的指令文本
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    # 根据"###"切分原始指令
    raw_instructions = re.split("###", raw_instructions)
    # 初始化指令列表
    instructions = []
    # 对每个切分出的原始指令进行处理
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        # 如果解码由于长度停止，最后一个示例可能被截断，因此我们丢弃它
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        # 根据索引和"Instruction", "Input", "Output"关键字进行切分
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        # 如果切分结果不等于7，则继续下一轮循环
        if len(splitted_data) != 7:
            continue
        else:
            # 提取指令、输入、输出
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            # 对输入进行处理，如果是"<noinput>"，则替换为空字符串
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        # 过滤掉太短或太长的指令
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        # 根据不适合语言模型的关键词进行过滤
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        # 如果指令中存在黑名单中的词，则忽略该指令
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        # 模型倾向于为一些现有指令添加"编写程序"，这会导致很多这样的指令。
        # 这里过滤掉这类指令
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        # 过滤那些以标点符号开始的指令
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        # 过滤那些以非英语字符开始的指令
        if not inst[0].isascii():
            continue
        # 将处理后的指令添加到指令列表中
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


# 定义一个在字符串中查找单词的函数
def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


# 定义一个生成指令的函数
def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks.jsonl",
    num_instructions_to_generate=100,
    model_name="text-davinci-003",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    # 读取并解析种子任务
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    # 从种子任务中提取指令、输入和输出
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    # 加载LM生成的指令
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    # 初始化Rouge得分计算器
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    # 进度条，总数为要生成的指令数量
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    # 如果已有机器生成的指令，则更新进度条
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    # 首先，我们对所有的种子指令和生成的机器指令进行 tokenize
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    # 当机器指令数据的数量小于需要生成的指令数量时，持续生成
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            # 只从种子任务中采样
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            # 将多个提示指令编码成一个字符串
            prompt = encode_prompt(prompt_instructions)
            # 将编码的指令添加到批输入列表中
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],  # 当出现这些字符串时，生成停止
        )
        request_start = time.time()
        # 调用OpenAI API进行批量生成
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated  阻止特定token被生成
        )
        request_duration = time.time() - request_start

        # 开始后处理生成的结果
        process_start = time.time()
        instruction_data = []
        for result in results:
            # 对每个结果进行后处理，获取新的指令
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        # 对每一条新指令进行处理
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            # 如果最大相似度得分大于0.7，则认为该指令与已有指令过于相似，不予采纳
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            # 将新指令添加到机器生成的指令数据中
            machine_instruction_data.append(instruction_data_entry)
            # 将新指令添加到已有指令列表和已有指令标记列表中
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        # 将机器生成的指令数据保存到文件中
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)

# 使用fire库解析命令行参数，并调用函数
if __name__ == "__main__":
    fire.Fire(main)
