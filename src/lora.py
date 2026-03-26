from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk

# 加载模型和分词器
model_name = "./DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 设置填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
    use_cache=False,  # 禁用缓存，与梯度检查点一起使用
)
# 启用梯度检查点
model.gradient_checkpointing_enable()
# 准备模型用于训练
model = prepare_model_for_kbit_training(model)

# 应用 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head"
    ],
    lora_dropout=0.1, 
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 加载数据
dataset = load_from_disk("./huatuo_subset_20k")


# 数据预处理
def format_instruction(sample):
    # 根据您的数据集结构，使用正确的字段名称
    question = sample['question'] if 'question' in sample else sample['questions']
    answer = sample['answer'] if 'answer' in sample else sample['answers']

    # 创建指令格式的文本
    text = f"### 指令:\n{question}\n\n### 回答:\n{answer}"
    return {"text": text}

dataset = dataset.map(format_instruction)


# Tokenization - 正确处理标签和填充
def tokenize_function(samples):
    # 对文本进行标记化，启用填充和截断
    tokenized = tokenizer(
        samples["text"],
        truncation=True,
        padding=True,  # 启用填充
        max_length=512,  # 减小最大长度
        return_tensors=None
    )

    # 创建标签 - 与输入相同但向右移动一位
    labels = []
    for input_ids in tokenized["input_ids"]:
        # 复制输入ID
        label = input_ids.copy()

        # 将标签向右移动一位
        label[:-1] = label[1:]
        label[-1] = -100  # 最后一个位置设为忽略

        labels.append(label)

    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./medical_finetune_output",
    per_device_train_batch_size=1,  # 根据GPU内存调整
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False,  # 确保不删除必要的列
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 不使用掩码语言建模
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model()
tokenizer.save_pretrained("./medical_finetune_output")