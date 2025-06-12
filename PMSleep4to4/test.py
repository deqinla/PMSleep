import torch
import torch.nn.functional as F

def generate_prompts(input_tensor,pad_size):
    # 确定填充尺寸，左侧填充3个元素
    # 在最后一个维度左侧填充pad_size个0
    padded = F.pad(input_tensor, (pad_size, 0))
    # 使用unfold创建滑动窗口（窗口大小3，步长1）
    unfolded = padded.unfold(-1, pad_size, 1)
    # 截取前N个窗口（N为原始序列长度）
    prompts = unfolded[..., :input_tensor.size(-1), :]

    later1 = F.pad(input_tensor, (0, 1))[..., 1:]

    change_2_later = input_tensor != later1
    change_2_later = torch.tensor(change_2_later, dtype=torch.int)
    return prompts

# 示例使用
x = torch.tensor([1, 2, 3, 4, 55, 6, 6])
# 生成提示
prompts = generate_prompts(x,3)
print(prompts)