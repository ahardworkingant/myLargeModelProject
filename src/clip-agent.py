import torch
from transformers import CLIPProcessor, CLIPModel

# 加载CLIP的处理器和模型
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 准备输入数据：图像和文本
image_inputs = processor(images="cat.png", return_tensors="pt")
text_inputs = processor(text=["a photo of a cat"], padding=True, truncation=True, return_tensors="pt")

# 将图像和文本输入到模型中，获取它们的嵌入表示
with torch.no_grad():
    image_embeddings = model.get_image_features(**image_inputs)
    text_embeddings = model.get_text_features(**text_inputs)

# 计算图像和文本之间的余弦相似度
cosine_similarity = torch.cosine_similarity(image_embeddings, text_embeddings, dim=-1)

print("Cosine similarity:", cosine_similarity.item())