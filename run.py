import torch
import random
import time
import datetime
import argparse
from transformers import ViTForImageClassification

def main(args):
    # 1. 准备四张卡
    devices = [torch.device(f"cuda:{i}") for i in range(4)]

    # 2. 在每张卡上加载同一个 ViT 模型
    models = [
        ViTForImageClassification.from_pretrained(args.model_name).to(dev)
        for dev in devices
    ]
    # 切到 eval 模式（关闭 dropout）
    for model in models:
        model.eval()

    # 3. 根据输入大小参数准备随机输入张量
    if args.use_large_input == 1:
        batch_size = 240
    elif args.use_large_input == 0:
        batch_size = 180
    else:
        batch_size = 158

    inputs = [
        # transformers 的 ViT 要求输入形状为 (batch, 3, 224, 224)
        # 随机生成 [0,1) 之间的浮点数即可
        torch.rand(batch_size, 3, 224, 224, device=dev)
        for dev in devices
    ]

    # 4. 无限循环推理
    start_time = time.time()
    while True:
        for model, inp in zip(models, inputs):
            # 前向：输出包括 logits
            with torch.no_grad():
                outputs = model(pixel_values=inp)

        # 每隔 10 秒打印一次随机 Loss（仅示例用）
        if int(time.time() - start_time) % 10 == 0:
            loss = random.random()
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now} 伪 Loss: {loss:.4f}")

        # 如需减轻 CPU 负担，可解除下一行的注释
        # time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4 卡 ViT 推理示例")
    parser.add_argument(
        '--use_large_input', type=int, default=1,
        help='1→batch=240, 0→batch=180, 其他→batch=158'
    )
    parser.add_argument(
        '--model_name', type=str,
        default='google/vit-base-patch16-224',
        help='Hugging Face 模型名称，如 google/vit-base-patch16-224'
    )
    args = parser.parse_args()
    main(args)
