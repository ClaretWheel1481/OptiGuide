import os
import yaml
from ultralytics import YOLO

# 将配置写入YAML文件
def setup_dataset_config(dataset_dir, yaml_path):
    """
    创建数据集配置文件
    """
    # 定义数据集配置
    dataset_config = {
        'path': dataset_dir,  # 数据集根目录
        'train': 'train/images',  # 训练图像相对路径
        'val': 'valid/images',  # 验证图像相对路径
        'test': 'test/images',  # 测试图像相对路径
        'names': {
            0: 'car',
        }
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f)

    print(f"数据集配置已保存至 {yaml_path}")
    return yaml_path


def train_model(data_yaml, model_name='yolov12n.pt', epochs=20, batch_size=16, imgsz=640):
    """
    训练YOLO模型
    """
    # 加载预训练模型
    model = YOLO(model_name)

    # 开始训练
    results = model.train(
        data=data_yaml,  # 数据集配置文件
        epochs=epochs,  # 总训练轮次
        batch=batch_size,  # 批次大小
        imgsz=imgsz,  # 图像大小
        patience=20,  # 早停耐心值
        save=True,  # 保存模型
        device='0',  # 使用GPU，如果无GPU设为'cpu'
        project='car_detection',  # 项目名称
        name='yolov12n_car',  # 实验名称
        exist_ok=True,  # 如果输出目录已存在则覆盖
        pretrained=True,  # 使用预训练权重
        optimizer='auto',  # 自动选择优化器
        plots=True,  # 生成训练图表
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3,  # 预热轮次
        warmup_momentum=0.8,  # 预热动量
        close_mosaic=10,  # 最后几轮关闭马赛克增强
        augment=True,  # 使用数据增强
        cache=False,  # 缓存图像以加速训练
    )

    return results


def validate_model(model_path):
    """
    验证模型
    """
    model = YOLO(model_path)
    results = model.val()
    return results


if __name__ == "__main__":
    # 数据集路径
    dataset_dir = "./datasets/car"

    # 创建配置文件
    yaml_path = "./datasets/car.yaml"
    setup_dataset_config(dataset_dir, yaml_path)

    # 训练模型
    print("开始训练模型...")
    train_results = train_model(yaml_path)

    # 获取最佳模型路径
    best_model_path = f"./car/yolov12n_car/weights/best.pt"

    # 验证模型
    print(f"验证最佳模型: {best_model_path}")
    if os.path.exists(best_model_path):
        val_results = validate_model(best_model_path)
        print(f"验证结果: {val_results}")
    else:
        print(f"找不到最佳模型: {best_model_path}")

    print("训练和验证完成!")