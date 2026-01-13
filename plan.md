# OccRWKV → 医学体素数据集快速迁移计划

## 目标
以最快速度完成项目迁移，优先跑通训练流程，后续再优化。

---

## 核心差异

| 项 | 原SemanticKITTI | 新医学数据集 |
|----|----------------|------------|
| 点云 | (N,4) | (N,3) → 填充0 → (N,4) |
| 类别 | 20 | 72 |
| 尺寸 | 256×256×32 | 动态 → 固定128×128×256 |
| 文件 | 多文件 | .npz |
| 单位 | 米 | 毫米 → 转为米 |

---

## 快速迁移路径 (3步)

### Step 1: 创建医学数据集类 ⭐️ 核心
**文件**: `datasets/medical_voxel.py` (新建)

**功能**:
1. 读取 `.npz` 文件
2. 点云 (N,3) → 填充0 → (N,4)
3. 坐标 毫米/1000 → 米
4. voxel_labels → pad/crop → (128,128,256)
5. 使用 `dataset_split.json`

**关键代码结构**:
```python
class MedicalVoxelDataset(Dataset):
    def __init__(self, data_root, split_file, split='train', target_size=(128,128,256), augmentation=False):
        # 加载划分
        with open(split_file) as f:
            splits = json.load(f)['splits'][split]
        self.files = [f"{data_root}/{s}.npz" for s in splits]
        self.target_size = target_size
        self.augmentation = augmentation

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        # 点云处理
        pc = data['sensor_pc'] / 1000.0  # mm → m
        pc = np.pad(pc, ((0,0), (0,1)), constant_values=0)  # (N,3)→(N,4)

        # 体素处理
        voxel = data['voxel_labels']
        voxel = self._pad_or_crop(voxel, self.target_size)

        # 增强
        if self.augmentation:
            pc, voxel = self._random_flip(pc, voxel)

        return {
            'points': torch.from_numpy(pc).float(),
            'label_1_1': torch.from_numpy(voxel).long(),
            # ... 其他字段按需添加
        }
```

---

### Step 2: 创建配置文件 ⭐️
**文件**: `cfgs/medical_voxel.yaml` (新建)

**策略**: 复制 `cfgs/template.yaml`，只改关键参数

```yaml
MODEL:
  TYPE: 'OccRWKV'

DATASET:
  TYPE: 'MedicalVoxel'
  DATA_ROOT: '/home/comp/25481568/code/Dataset/voxel_data'
  SPLIT_FILE: '/home/comp/25481568/code/OccRWKV/dataset_split.json'
  GRID_METERS: [0.004, 0.004, 0.004]  # 4mm
  NCLASS: 72  # ← 改
  SIZES: [128, 128, 256]  # ← 改

TRAIN:
  BATCH_SIZE: 1  # ← A100从1开始测试
  MAX_EPOCHS: 80

OPTIMIZER:
  BASE_LR: 0.001
```

---

### Step 3: 最小化代码修改 ⭐️

#### 3.1 注册数据集 (`utils/dataset.py`)
```python
def get_dataset(_cfg):
    if _cfg['DATASET']['TYPE'] == 'MedicalVoxel':
        from datasets.medical_voxel import MedicalVoxelDataset
        ds_train = MedicalVoxelDataset(
            data_root=_cfg['DATASET']['DATA_ROOT'],
            split_file=_cfg['DATASET']['SPLIT_FILE'],
            split='train',
            target_size=tuple(_cfg['DATASET']['SIZES']),
            augmentation=True
        )
        # ... val, test
    else:
        # 原有逻辑
```

#### 3.2 模型配置 (`networks/occrwkv.py`)
**修改**: 从配置读取参数（已支持，只需确认）
- `self.nbr_classes = cfg['DATASET']['NCLASS']` → 72
- `self.sizes = cfg['DATASET']['SIZES']` → [128,128,256]
- `self.n_height = self.sizes[-1]` → 256

**验证**: 检查输出reshape逻辑是否支持动态尺寸

#### 3.3 损失函数 (可选优化)
**快速方案**: 使用均匀权重，跳过类别频率统计
```python
# 临时使用均匀权重
class_weights = torch.ones(72)
```

---

## 非必要项 (后续优化)

以下可以在跑通后再添加:
- ❌ `data/medical-voxel.yaml` (标签映射) - 直接使用索引
- ❌ 类别频率统计脚本 - 先用均匀权重
- ❌ 多尺度标签 (1_2, 1_4, 1_8) - 先只用 1_1
- ❌ invalid/occupancy mask - 先跳过

---

## 实施顺序 (最快路径)

```
1. 创建 datasets/medical_voxel.py (30分钟)
   ├─ 基础加载逻辑
   ├─ pad/crop函数
   └─ 简单增强

2. 创建 cfgs/medical_voxel.yaml (5分钟)
   └─ 复制template.yaml + 改3个参数

3. 修改 utils/dataset.py (5分钟)
   └─ 添加分支逻辑

4. 验证数据加载 (10分钟)
   └─ python -c "test dataset loading"

5. 验证模型前向 (10分钟)
   └─ python -c "test model forward"

6. 启动训练 (5分钟)
   └─ python train.py --cfg cfgs/medical_voxel.yaml

总计: ~1小时代码 + 测试
```

---

## 潜在风险点

1. **显存**: 输出 [72,128,128,256] 较大
   - **解决**: batch_size=1，监控显存

2. **坐标系**: grid_world_min/max 可能需要调整
   - **解决**: 先忽略，如果报错再处理

3. **多尺度**: 原模型使用多尺度，新数据集只提供1_1
   - **解决**: 其他尺度填充dummy数据

---

## 测试命令

```bash
# 1. 测试数据加载
conda activate occ_rwkv
python -c "
from datasets.medical_voxel import MedicalVoxelDataset
ds = MedicalVoxelDataset('/home/comp/25481568/code/Dataset/voxel_data',
                          '/home/comp/25481568/code/OccRWKV/dataset_split.json',
                          'train')
batch = ds[0]
print('Points:', batch['points'].shape)
print('Voxel:', batch['label_1_1'].shape)
"

# 2. 测试训练
python train.py --cfg cfgs/medical_voxel.yaml --dset_root /home/comp/25481568/code/Dataset/voxel_data
```

---

## 成功标准

- [x] 数据加载无报错
- [x] 模型前向传播成功
- [x] 训练loss正常下降
- [x] 1 epoch完成无崩溃
