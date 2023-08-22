#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

# 定义一些全局变量，如特殊字符、提示模板等
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


# 这是Python中的装饰器，用于指示该类是一个数据类。数据类是一个专门用于存储数据的类
# 它为我们自动实现了一些基础方法，如__init__，__repr__，__eq__等
@dataclass
# 定义一个名为ModelArguments的数据类
class ModelArguments:
    # 定义一个名为model_name_or_path的实例变量，类型为Optional[str]，默认值为"facebook/opt-125m"
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
# 定义一个名为DataArguments的数据类
class DataArguments:
    # 定义一个名为data_path的实例变量，类型为str，默认值为None，额外的metadata提供了该变量的帮助信息
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
# 定义一个名为TrainingArguments的数据类，这个类继承了transformers库的TrainingArguments类
class TrainingArguments(transformers.TrainingArguments):
    # 定义一个名为cache_dir的实例变量，类型为Optional[str]，默认值为None 
    cache_dir: Optional[str] = field(default=None)
    # 定义一个名为optim的实例变量，类型为str，默认值为"adamw_torch"   
    optim: str = field(default="adamw_torch")
    # 定义一个名为model_max_length的实例变量，类型为int
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


# 调整分词器和词嵌入大小
# 定义一个函数，函数名为 smart_tokenizer_and_embedding_resize，输入包括一个字典（用于定义特殊词汇），一个分词器和一个预训练模型
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # 向分词器添加特殊词汇，返回新添加的词汇数量
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 将模型的嵌入层大小调整为与新的词汇表大小一致
    model.resize_token_embeddings(len(tokenizer))

    # 如果添加了新的词汇
    if num_new_tokens > 0:
        # 获取模型输入嵌入的权重数据
        input_embeddings = model.get_input_embeddings().weight.data
        # 获取模型输出嵌入的权重数据
        output_embeddings = model.get_output_embeddings().weight.data

        # 计算输入嵌入中旧词汇的平均向量
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # 计算输出嵌入中旧词汇的平均向量
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        # 将新添加的词汇的输入嵌入向量设置为旧词汇的平均输入嵌入向量
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        # 将新添加的词汇的输出嵌入向量设置为旧词汇的平均输出嵌入向量
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# 将字符串序列进行分词
# 函数定义，接受一个字符串序列和一个预训练的分词器，返回一个字典
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,                      # 对每个字符串进行分词处理
            return_tensors="pt",       # 返回PyTorch tensors
            padding="longest",         # padding策略为 "longest"，即填充到最长序列的长度
            max_length=tokenizer.model_max_length,  # 最大长度为分词器的最大长度
            truncation=True,           # 如果序列超过最大长度，则进行截断
        )
        for text in strings            # 遍历输入的每个字符串
    ]
    # 从分词结果中提取输入的ids和标签    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
 
    # 计算输入ids和标签的长度（不包括padding）                                                         
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
 
    # 返回一个字典，包含输入的ids、标签、输入的长度和标签的长度
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# 预处理数据，对源数据和目标数据进行分词
# 函数定义，接受源字符串、目标字符串和一个预训练的分词器，返回一个字典
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    # 将源字符串和目标字符串组合在一起
    examples = [s + t for s, t in zip(sources, targets)]
    # 对组合后的字符串和源字符串分别进行分词处理
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]    # 从组合后的分词结果中提取输入ID
    labels = copy.deepcopy(input_ids)              # 复制一份输入ID作为标签
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        # 对于标签，将源字符串部分的ID设置为忽略索引（IGNORE_INDEX）
        label[:source_len] = IGNORE_INDEX
    # 返回一个字典，包含输入ID和标签
    return dict(input_ids=input_ids, labels=labels)


# 定义SupervisedDataset 类(用于监督微调的数据集)，用于加载数据、格式化输入、进行分词等操作
# 定义一个用于监督学习微调的数据集类
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()      # 初始化父类
        logging.warning("Loading data...")             # 记录开始加载数据的日志
        list_data_dict = utils.jload(data_path)        # 加载数据
 
        logging.warning("Formatting inputs...")        # 记录开始格式化输入的日志
        # 从字典中获取有输入提示和无输入提示
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]                  
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            # 遍历每个例子，如果有输入则使用输入提示，否则使用无输入提示
            for example in list_data_dict      
        ]
        # 构造目标，每个目标是输出加上结束标记
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]  
 
        # 记录开始分词输入的日志
        logging.warning("Tokenizing inputs... This may take some time...")  
        data_dict = preprocess(sources, targets, tokenizer)  # 预处理源和目标
 
        self.input_ids = data_dict["input_ids"]              # 保存输入ID
        self.labels = data_dict["labels"]                    # 保存标签
 
    # 返回数据集的大小
    def __len__(self):
        return len(self.input_ids)  
 
    # 返回第i个样本，包含输入ID和标签
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])  


# 用于将数据集的实例整理为批次
# 定义一个用于监督学习微调的数据整理类
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    # 预训练的分词器
    tokenizer: transformers.PreTrainedTokenizer

    # 从实例中提取输入ID和标签
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 对输入ID进行填充，使它们具有相同的长度，填充值为分词器的填充标记ID
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # 对标签进行填充，使它们具有相同的长度，填充值为忽略索引（IGNORE_INDEX）
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # 返回一个字典，包含输入ID、标签和注意力掩码。注意力掩码用于指示哪些元素应该被模型关注（在这里是非填充的元素）
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# 用于创建监督学习任务的数据集和整理器
# 函数定义，接受一个预训练的分词器和数据参数，返回一个字典
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # 创建一个监督学习的微调数据集
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    # 创建一个数据整理器
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # 返回一个字典，包含训练数据集、评估数据集和数据整理器。在这里，评估数据集为None
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    # 使用transformers.HfArgumentParser 解析命令行参数，将它们分为模型参数、数据参数和训练参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # 此行代码调用parse_args_into_dataclasses方法，解析命令行参数，并将这些参数映射到相应的数据类上。该方法返回的顺序与在HfArgumentParser构造函数中指定的数据类的顺序相同
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 从预训练的模型检查点加载一个用于因果语言建模的模型​​​​​​​
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # 当调用from_pretrained方法下载预训练模型时，库首先检查cache_dir是否已经有需要的模型文件。如果在cache_dir中找到了，则直接从本地加载，避免了再次从互联网下载。如果没有找到，则从Hugging Face's model hub下载模型到cache_dir中，以备将来使用
        cache_dir=training_args.cache_dir,
    )

    # 从预训练模型创建一个自动化的分词器，其中包含了模型的名称或路径，缓存目录，模型的最大长度，填充的位置以及是否使用快速分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 如果分词器没有 special_tokens，那么添加，并重新设置模型的嵌入大小
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # 使用make_supervised_data_module 函数为监督学习任务创建数据集和整理器
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # 实例化transformers.Trainer 类，并传入模型、分词器、训练参数以及数据集，Trainer类负责管理训练过程
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # 调用​​​​​​​Trainer​​​​​​​类的train() 方法对模型进行微调，相当于链路就是：transformers库 → Trainer类 → train函数
    trainer.train()
    # 在训练完成后，调用Trainer.save_state() 方法保存模型的状态
    trainer.save_state()
    # 将训练器的模型安全地保存到磁盘
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
