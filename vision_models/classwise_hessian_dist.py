import json
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_class_features(json_file_path, num_classes=10, start_class=None, normalize=False):
    """
    从JSON文件中读取类别特征值，绘制多个类别的特征值分布柱状图
    
    参数:
    json_file_path: JSON文件路径
    num_classes: 要绘制的类别数量
    start_class: 起始类别索引，如果为None则随机选择
    """
    
    # 1. 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 检查数据格式并提取特征值
    if isinstance(data, dict):
        # 如果数据是字典格式，假设键是类别名，值是特征向量
        class_names = list(data.keys())
        features_list = list(data.values())
    elif isinstance(data, list):
        # 如果数据是列表格式，假设每个元素是一个类别的特征向量
        class_names = [f"Class_{i}" for i in range(len(data))]
        features_list = data
    else:
        raise ValueError("不支持的JSON格式")
    
    total_classes = len(features_list)
    
    # 3. 确定起始类别
    if start_class is None:
        # 随机选择起始类别
        max_start = total_classes - num_classes
        if max_start < 0:
            raise ValueError(f"类别数量不足{num_classes}个，当前只有{total_classes}个类别")
        start_class = random.randint(0, max_start)
    
    # 4. 提取选中的类别
    selected_indices = list(range(0, total_classes, 5))[:num_classes]  # 从0开始，每50个选一个，最多选num_classes个
    selected_classes = [class_names[i] for i in selected_indices]
    selected_features = [features_list[i] for i in selected_indices]
    
    # 5. 创建图表
    fig, axes = plt.subplots(1, num_classes, figsize=(3*num_classes, 5))
    if num_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    if normalize:
        all_features = np.concatenate([features for features in selected_features])
        global_max = np.max(all_features)
    
    # 6. 为每个类别绘制特征值分布柱状图
    for i, (class_name, features) in enumerate(zip(selected_classes, selected_features)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # 计算特征值的分布（频率）
        # 使用直方图来统计特征值在不同区间的出现频率
        n_bins = min(400, len(features))  # 箱子的数量，最多50个
        counts, bin_edges = np.histogram(features, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 计算每个箱子的中心位置
        
        # 绘制柱状图
        bars = ax.bar(bin_centers, counts, width=bin_edges[1]-bin_edges[0])
        
        # 设置子图属性
        ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')  # 添加这一行
        ax.grid(axis='y', alpha=0.3)
        
        if normalize:
            ax.set_xlim(right=global_max)
        
        # 如果特征值范围很大，使用对数坐标
        feature_range = max(features) - min(features)
        if feature_range > 1000:  # 如果特征值范围很大，使用对数坐标
            ax.set_xscale('log')
    
    # 7. 设置总标题
    plt.suptitle(f'Feature Value Distribution for Selected Classes\n'
                f'X-axis: Feature Value, Y-axis: Frequency', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('class_features_distribution_plot.png', dpi=200, bbox_inches='tight')
    
    # 8. 打印选中的类别信息
    print(f"选中的类别索引: {selected_indices}")
    print(f"类别名称: {selected_classes}")
    print(f"特征维度: {len(selected_features[0])}")
    

# 如果你想要将所有类别的特征放在同一个图中比较
def plot_features_comparison(json_file_path, num_classes=10, start_class=None):
    """
    在同一个图中比较多个类别的特征值
    """
    
    # 读取数据（代码同上）
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        class_names = list(data.keys())
        features_list = list(data.values())
    elif isinstance(data, list):
        class_names = [f"Class_{i}" for i in range(len(data))]
        features_list = data
    
    total_classes = len(features_list)
    
    if start_class is None:
        max_start = total_classes - num_classes
        if max_start < 0:
            raise ValueError(f"类别数量不足{num_classes}个")
        start_class = random.randint(0, max_start)
    
    end_class = start_class + num_classes
    selected_classes = class_names[start_class:end_class]
    selected_features = features_list[start_class:end_class]
    
    feature_dim = len(selected_features[0])
    
    # 创建图表
    plt.figure(figsize=(15, 8))
    
    # 设置x轴位置（特征索引）
    x_pos = np.arange(feature_dim)
    bar_width = 0.8 / num_classes  # 动态调整柱宽
    
    # 为每个类别绘制柱状图
    for i, (class_name, features) in enumerate(zip(selected_classes, selected_features)):
        offset = i * bar_width
        plt.bar(x_pos + offset, features, width=bar_width, 
               label=class_name, alpha=0.7)
    
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Feature Value', fontsize=12)
    plt.title(f'Feature Values Comparison for {num_classes} Classes\n'
             f'X-axis: Feature Index, Y-axis: Feature Value', 
             fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # 设置x轴刻度
    if feature_dim <= 20:
        plt.xticks(x_pos + bar_width * (num_classes - 1) / 2, 
                  [str(i) for i in range(feature_dim)])
    else:
        plt.xticks(x_pos + bar_width * (num_classes - 1) / 2, 
                  [str(i) if i % 5 == 0 else '' for i in range(feature_dim)])
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # sgd uniform
    p = "/libingheng/hessian-spectrum/vision_models/files/resnet_files/model-resnet18-opt-sgd-lr-0.5-momentum-0.95-wd-0.0-seed-32-dsmode-uniform_minibatch_True_bs_10240_m_10_v_1_ckpt_89/classwise_values_full.json"
    
    # sgd 1k
    # p = "/libingheng/hessian-spectrum/vision_models/files/resnet_files/model-resnet18-opt-sgd-lr-0.5-momentum-0.95-wd-0.0-seed-32-dsmode-1k_minibatch_True_bs_10240_m_10_v_1_ckpt_89/classwise_values_full.json"
    
    # adamw uniform
    # p = "/libingheng/hessian-spectrum/vision_models/files/resnet_files/model-resnet18-opt-adamw-lr-0.01-beta1-0.9-beta2-0.999-wd-0.0001-seed-32-dsmode-uniform_minibatch_True_bs_10240_m_10_v_1_ckpt_89/classwise_values_full.json"
    
    # adamw 1k
    # p = "/libingheng/hessian-spectrum/vision_models/files/resnet_files/model-resnet18-opt-adamw-lr-0.01-beta1-0.9-beta2-0.999-wd-0.0001-seed-32-dsmode-1k_minibatch_True_bs_10240_m_10_v_1_ckpt_89/classwise_values_full.json"
    
    
    # 方法1: 每个类别单独一个子图
    plot_class_features(json_file_path=p, num_classes=5, start_class=0, normalize=False)
    
    # 方法2: 所有类别在同一个图中比较
    # plot_features_comparison('class_features.json', num_classes=10)