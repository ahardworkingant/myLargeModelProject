import torch

# 检查是否有可用GPU
if torch.cuda.is_available():
    # 当前GPU设备索引（默认0）
    device = torch.device("cuda:0")

    # 查看当前分配的显存（字节）
    allocated = torch.cuda.memory_allocated(device)
    # 查看当前缓存的显存（字节）
    cached = torch.cuda.memory_cached(device)
    # 查看GPU总显存（字节）
    total = torch.cuda.get_device_properties(device).total_memory

    # 转换为GB单位显示
    print(f"总显存: {total / 1024 ** 3:.2f} GB")
    print(f"已分配显存: {allocated / 1024 ** 3:.2f} GB")
    print(f"缓存显存: {cached / 1024 ** 3:.2f} GB")
    print(f"剩余显存: {(total - allocated) / 1024 ** 3:.2f} GB")
else:
    print("未检测到GPU")