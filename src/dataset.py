import json
import random
from datasets import Dataset, DatasetDict


# 从本地 JSONL 文件加载数据集
def load_dataset_from_jsonl(file_path):
    """
    从 JSONL 文件加载数据集
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))

    # 创建 Hugging Face Dataset 对象
    dataset = Dataset.from_list(data)
    return DatasetDict({'train': dataset})


# 加载本地数据集
dataset = load_dataset_from_jsonl("train_datasets.jsonl")

def select_diverse_subset(dataset, num_samples=20000):
    """
    选择多样化的数据子集
    """
    # 获取所有数据
    all_data = dataset['train']

    # 策略1: 按主题分层抽样
    medical_topics = {
        '内科': ['糖尿病', '高血压', '心脏病', '肺炎'],
        '外科': ['手术', '骨折', '创伤', '肿瘤'],
        '儿科': ['儿童', '婴儿', '生长发育', '疫苗'],
        '妇科': ['妇科', '产科', '月经', '妊娠'],
        '急诊': ['急救', '中毒', '休克', '创伤']
    }

    selected_indices = set()

    # 为每个主题分配样本
    samples_per_topic = num_samples // len(medical_topics)

    for topic, keywords in medical_topics.items():
        topic_indices = []
        for i, sample in enumerate(all_data):
            question = sample['questions'][0][0]
            answer = sample['answers'][0]
            text = question + ' ' + answer

            if any(keyword in text for keyword in keywords):
                topic_indices.append(i)

        # 随机选择该主题的样本
        if len(topic_indices) > samples_per_topic:
            selected = random.sample(topic_indices, samples_per_topic)
        else:
            selected = topic_indices

        selected_indices.update(selected)

    # 策略2: 补充随机样本以确保总数
    remaining_needed = num_samples - len(selected_indices)
    if remaining_needed > 0:
        all_indices = set(range(len(all_data)))
        available_indices = all_indices - selected_indices
        additional_indices = random.sample(list(available_indices),
                                           min(remaining_needed, len(available_indices)))
        selected_indices.update(additional_indices)

    # 创建子集
    subset = all_data.select(list(selected_indices))
    return subset


# 选择20,000个多样本
subset = select_diverse_subset(dataset, num_samples=20000)

# 保存选定的子集
subset.save_to_disk("./huatuo_subset_20k")