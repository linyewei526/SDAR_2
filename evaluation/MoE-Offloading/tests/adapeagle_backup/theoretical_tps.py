"""
计算不同MoE模型的理论TPS上限
假设：完美流水线，只计算PCIe传输时间
从config.json读取实际配置
"""

import json

def calculate_theoretical_tps(
    hidden_size,
    intermediate_size,
    num_layers,
    experts_per_layer,
    pcie_bandwidth_gb_s=11.5,
    has_bias=True
):
    """
    计算理论TPS上限

    Args:
        hidden_size: 隐藏层维度
        intermediate_size: 中间层维度
        num_layers: 层数
        experts_per_layer: 每层expert数量
        pcie_bandwidth_gb_s: PCIe带宽 (GB/s)
        has_bias: 是否有bias参数
    """
    # 计算每个expert的大小
    gate_up_params = hidden_size * intermediate_size * 2
    down_params = hidden_size * intermediate_size

    if has_bias:
        gate_up_bias_params = intermediate_size * 2
        down_bias_params = hidden_size
    else:
        gate_up_bias_params = 0
        down_bias_params = 0

    # 转换为bytes (bfloat16 = 2 bytes)
    gate_up_bytes = gate_up_params * 2
    gate_up_bias_bytes = gate_up_bias_params * 2
    down_bytes = down_params * 2
    down_bias_bytes = down_bias_params * 2

    # 单个expert总大小
    expert_total_bytes = gate_up_bytes + gate_up_bias_bytes + down_bytes + down_bias_bytes

    # 每个token需要加载的expert总数
    num_experts_per_token = num_layers * experts_per_layer
    total_bytes_per_token = num_experts_per_token * expert_total_bytes

    # PCIe传输时间
    pcie_bandwidth = pcie_bandwidth_gb_s * 1024**3  # GB/s to bytes/s
    transfer_time_per_token = total_bytes_per_token / pcie_bandwidth

    # 理论TPS
    theoretical_tps = 1.0 / transfer_time_per_token

    return {
        'expert_size_mb': expert_total_bytes / 1024 / 1024,
        'total_transfer_mb': total_bytes_per_token / 1024 / 1024,
        'transfer_time_ms': transfer_time_per_token * 1000,
        'theoretical_tps': theoretical_tps
    }

def load_qwen3moe_config(config_path):
    """从config.json加载Qwen3MoE配置"""
    with open(config_path) as f:
        config = json.load(f)

    return {
        'hidden_size': config['hidden_size'],
        'intermediate_size': config['moe_intermediate_size'],
        'num_layers': config['num_hidden_layers'],
        'experts_per_layer': config['num_experts_per_tok'],
        'has_bias': False
    }

def load_gptoss_config(config_path):
    """从config.json加载GPT-OSS配置"""
    with open(config_path) as f:
        config = json.load(f)

    return {
        'hidden_size': config['hidden_size'],
        'intermediate_size': config['intermediate_size'],
        'num_layers': config['num_hidden_layers'],
        'experts_per_layer': config['num_experts_per_tok'],
        'has_bias': True  # GPT-OSS有bias
    }

def main():
    # 定义模型路径
    QWEN3MOE_PATH = "/data/home/tianjianyang/models/moe/Qwen3-30B-A3B/config.json"
    GPTOSS_PATH = "/data/home/tianjianyang/models/moe/gpt-oss-20b-BF16/config.json"

    # Qwen3MoE配置
    print("📊 Qwen3MoE-30B")
    qwen_config = load_qwen3moe_config(QWEN3MOE_PATH)
    print(f"  配置文件: {QWEN3MOE_PATH}")
    print(f"  hidden_size: {qwen_config['hidden_size']}")
    print(f"  moe_intermediate_size: {qwen_config['intermediate_size']}")
    print(f"  num_hidden_layers: {qwen_config['num_layers']}")
    print(f"  num_experts_per_tok: {qwen_config['experts_per_layer']}")
    print()

    qwen_result = calculate_theoretical_tps(**qwen_config)
    print(f"  Expert大小:      {qwen_result['expert_size_mb']:.2f} MB")
    print(f"  每token传输量:   {qwen_result['total_transfer_mb']:.2f} MB")
    print(f"  传输时间:        {qwen_result['transfer_time_ms']:.2f} ms")
    print(f"  🚀 理论TPS上限:  {qwen_result['theoretical_tps']:.2f} tokens/s")
    print()

    # GPT-OSS配置
    print("📊 GPT-OSS-20B")
    oss_config = load_gptoss_config(GPTOSS_PATH)
    print(f"  配置文件: {GPTOSS_PATH}")
    print(f"  hidden_size: {oss_config['hidden_size']}")
    print(f"  intermediate_size: {oss_config['intermediate_size']}")
    print(f"  num_hidden_layers: {oss_config['num_layers']}")
    print(f"  num_experts_per_tok: {oss_config['experts_per_layer']}")
    print()

    oss_result = calculate_theoretical_tps(**oss_config)
    print(f"  Expert大小:      {oss_result['expert_size_mb']:.2f} MB")
    print(f"  每token传输量:   {oss_result['total_transfer_mb']:.2f} MB")
    print(f"  传输时间:        {oss_result['transfer_time_ms']:.2f} ms")
    print(f"  🚀 理论TPS上限:  {oss_result['theoretical_tps']:.2f} tokens/s")
    print()

    print("理论TPS总结")
    print(f"  假设PCIe传输速率(gpu3测试结果):  11.5 GB/s")
    print(f"  Qwen3MoE-30B:  {qwen_result['theoretical_tps']:.2f} tokens/s")
    print(f"  GPT-OSS-20B:   {oss_result['theoretical_tps']:.2f} tokens/s")
    print()
    print("注意：这是纯理论值，假设完美流水线且只计算PCIe传输时间")
    print("实际TPS会受以下因素影响：")
    print("  - PCIe带宽利用率")
    print("  - Prefetch准确率")

if __name__ == "__main__":
    main()
