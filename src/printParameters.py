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
# 查看所有参数名称和形状
print("参数名称和形状:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 查看可训练参数
print("\n可训练参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.shape} (可训练)")

# 查看不可训练参数
print("\n不可训练参数:")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"{name}: {param.shape} (不可训练)")