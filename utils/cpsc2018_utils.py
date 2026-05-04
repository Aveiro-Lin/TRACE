# -*- coding: utf-8 -*-

import os
import pickle
import json
from typing import Dict, Optional
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
try:
    from .my_ecg_process import resample, process_ecg, normalize
except:
    from my_ecg_process import resample, process_ecg, normalize

try:
    from .imagepreprocess_FA_exp3 import target_strategy
except Exception:
    try:
        from imagepreprocess_FA_exp3 import target_strategy
    except Exception:
        def target_strategy(num_classes):
            def _identity_target(labels):
                return labels
            return _identity_target, None, None
import scipy.io as sio
try:
    from . import transforms as T
except Exception:
    T = None

# 诊断类别到PTB-XL风格的映射字典
DIAGNOSTIC_MAP = {
    "Normal": {"diagnostic": "Normal ECG", "rhythm": "NSR"},
    "AF": {"diagnostic": "Atrial fibrillation", "rhythm": "AFIB"},
    "I-AVB": {"diagnostic": "First-degree AV block", "form": "AVB"},
    "LBBB": {"diagnostic": "Left bundle branch block", "form": "IVCD"},
    "RBBB": {"diagnostic": "Right bundle branch block", "form": "IVCD"},    
    "PAC": {"diagnostic": "Premature atrial contraction", "rhythm": "PAC"},
    "PVC": {"diagnostic": "Premature ventricular contraction", "rhythm": "PVC"},    
    "STD": {"diagnostic": "ST-segment depression", "form": "STC"},
    "STE": {"diagnostic": "ST-segment elevation", "form": "STC"}
}
DIAGNOSTIC_NAMES = list(DIAGNOSTIC_MAP.keys())

def mat_loader(path):
    # data = sio.loadmat(path)['ECG'][0][0][2]

    # return data.astype(np.float32)  # 12 leads (12, 5000)

    data = sio.loadmat(path)['val']
    return data.astype(np.float32)  # 12 leads (12, 5000)

def get_demo_info(record_id):
    """使用WFDB库提取年龄和性别"""
    record = wfdb.rdheader(record_id)
    age = np.nan
    sex = np.nan
    for comment in record.comments:
        comment = comment.lower()    
        value = comment.split(':')[1]
        if 'age' in comment:
            try:
                age = int(value)
            except:
                age = np.nan
        if 'sex' in comment:
            sex = value.strip()
    sex = sex if sex in ['male', 'female'] else np.nan
    return {'age': age, 'sex': sex}

class CPSC2018Utils:
    """
    管理 .hea 文件到子目录 (g1-g7) 的映射关系
    功能：
    - 自动扫描目录结构生成映射
    - 支持保存/加载映射字典
    - 提供验证功能
    """
    
    def __init__(self, base_dir: str):
        """
        初始化映射器
        :param base_dir: 包含 g1-g7 子目录的根目录
        """
        self.base_dir = base_dir
        self.mapping: Dict[str, str] = {}  # {record_id: subfolder}

        self.num_classes = 9
        self.target_transforms, _, _ = target_strategy(self.num_classes)
        
    def build_mapping(self) -> Dict[str, str]:
        """
        构建 .hea 文件到子目录的映射
        :return: 生成的映射字典
        """
        self.mapping = {}
        subfolders = [f"g{i}" for i in range(1, 8)]  # g1 到 g7
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.base_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
                
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.hea'):
                    record_id = os.path.splitext(filename)[0]
                    self.mapping[record_id] = subfolder
                    
        return self.mapping
    
    def save_mapping(self, output_file: str, format: str = 'pickle') -> None:
        """
        保存映射字典到文件
        :param output_file: 输出文件路径
        :param format: 保存格式 ('pickle' 或 'json')
        """
        if not self.mapping:
            raise ValueError("映射字典为空，请先调用 build_mapping()")
            
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if format == 'pickle':
            with open(output_file, 'wb') as f:
                pickle.dump(self.mapping, f)
        elif format == 'json':
            with open(output_file, 'w') as f:
                json.dump(self.mapping, f, indent=2)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @staticmethod
    def load_mapping(input_file: str) -> Dict[str, str]:
        """
        从文件加载映射字典
        :param input_file: 输入文件路径
        :return: 加载的映射字典
        """
        if input_file.endswith('.pkl') or input_file.endswith('.pickle'):
            with open(input_file, 'rb') as f:
                return pickle.load(f)
        elif input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                return json.load(f)
        else:
            raise ValueError("无法识别的文件格式，支持 .pkl/.pickle 或 .json")
    
    def validate_mapping(self, verbose: bool = False) -> bool:
        """
        验证所有映射关系是否正确
        :param verbose: 是否打印详细错误信息
        :return: 所有映射是否有效
        """
        if not self.mapping:
            raise ValueError("映射字典为空，请先调用 build_mapping()")
            
        all_valid = True
        for record_id, subfolder in self.mapping.items():
            hea_path = os.path.join(self.base_dir, subfolder, f"{record_id}.hea")
            if not os.path.exists(hea_path):
                if verbose:
                    print(f"错误: {record_id}.hea 应存在于 {subfolder} 但未找到")
                all_valid = False
                
        return all_valid
    
    def get_subfolder(self, record_id: str) -> Optional[str]:
        """
        获取指定记录ID所在的子目录
        :param record_id: 记录ID（不带.hea扩展名）
        :return: 子目录名 (如 'g3')，如果不存在则返回None
        """
        return self.mapping.get(record_id)

    def set_data_files(self, reference_path):
        self.data_files = []
        with open(reference_path, 'r') as f:
            records = [l.strip().split(",") for l in f][1:]
        for i, record in enumerate(records):
            file_name = record[0]
            # self.data_files.append(os.path.join(self.base_dir, self.get_subfolder(file_name), file_name + '.mat'))
            self.data_files.append(os.path.join(self.base_dir, file_name + '.mat'))

    def load_raw_data_cpsc2018(self, path, sampling_rate, seq_length=2000, target_rate=250, handle_via_trace=False):
        data = []
        self.set_data_files(path)
        if handle_via_trace:
            if T is None:
                raise ImportError("utils.transforms is required when handle_via_trace=True")
            resample_fun = T.Resample(target_fs=target_rate)
        for f in self.data_files:
            sample = mat_loader(f)
            if not handle_via_trace:
                sample = sample.T
                channel = []
                for col in range(sample.shape[1]):
                    ecg = sample[:, col]
                    if sampling_rate != target_rate:
                        ecg_resampled = resample(ecg, original_rate=sampling_rate, target_rate=target_rate)
                    ecg_cleaned = nk.ecg_clean(ecg_resampled, sampling_rate=target_rate)
                    channel.append(ecg_cleaned)
            else:
                if sampling_rate != target_rate:
                    sample = resample_fun(sample, sampling_rate)
                    sample = sample.T
                channel = []
                for col in range(sample.shape[1]):
                    ecg = sample[:, col]
                    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=target_rate)
                    channel.append(ecg_cleaned)
            
            X = np.array(channel, dtype=np.float32)
            X = process_ecg(X, seq_length=seq_length, sampling_rate=target_rate)
            norm = normalize(X)
            data.append(norm)

        data = np.array(data, dtype=np.float32)
        return data

    def load_labels_cpsc2018(self, reference_path):
        labels = []
        with open(reference_path, 'r') as f:
            records = [l.strip().split(",") for l in f][1:]
        for i, record in enumerate(records):
            label = {
                "diagnostic": {},
                "form": {}, 
                "rhythm": {},
                "label": None,
            }

            file_name = record[0]
            label.update(get_demo_info(os.path.join(self.base_dir, file_name)))

            for label_idx in record[1:]:
                if label_idx == '':
                    continue
                
                diagnostic_name = DIAGNOSTIC_NAMES[int(label_idx)-1]
                label["diagnostic"].update(dict([(diagnostic_name, 100)]))
                
                form = DIAGNOSTIC_MAP[diagnostic_name].get("form")
                if form is not None:
                    label["form"].update(dict([(form, 100)]))
                
                rhythm = DIAGNOSTIC_MAP[diagnostic_name].get("rhythm")
                if rhythm is not None:
                    label["rhythm"].update(dict([(rhythm, 100)]))
            
            label["diagnostic"] = str(label["diagnostic"])
            label["form"] = str(label["form"])
            label["rhythm"] = str(label["rhythm"])
            # label["label"] = self.target_transforms(record[1:])
            labels.append(label)
        df = pd.DataFrame(labels)
        return df


# 示例用法
if __name__ == '__main__':
    # # 1. 初始化并构建映射
    # BASE_DIR = 'path/to/cpsc2018/waveforms'
    # mapper = CPSC2018Utils(BASE_DIR)
    # mapping = mapper.build_mapping()
    # print(f"共找到 {len(mapping)} 条记录映射")
    
    # # 2. 保存映射 (两种格式)
    # mapping_path = 'data/hea_mapping.json'
    # # mapper.save_mapping('data/hea_mapping.pkl')  # 二进制格式
    # mapper.save_mapping(mapping_path, format='json')  # JSON格式
    
    # # 3. 验证映射
    # if mapper.validate_mapping(verbose=True):
    #     print("所有映射关系验证通过！")
    # else:
    #     print("警告: 存在无效的映射关系")
    
    # # 4. 加载映射 (示例)
    # # loaded_mapping = CPSC2018Utils.load_mapping('data/hea_mapping.pkl')
    # loaded_mapping = CPSC2018Utils.load_mapping(mapping_path)
    # print(f"从文件加载了 {len(loaded_mapping)} 条映射")
    
    # # 5. 查询示例
    # # sample_record = list(mapping.keys())[0] if mapping else 'A0723'
    # sample_record = 'A0723'
    # print(f"记录 '{sample_record}' 位于: {mapper.get_subfolder(sample_record)}")

    # # 6. 
    # reference_path = 'data/TRAIN_reference_cpsc2018.csv'
    # mapper.set_data_files(reference_path)

    # . 初始化并构建映射
    BASE_DIR = 'data/cpsc2018/Training_WFDB'  # 替换为实际路径
    reference_path = 'data/TRAIN_reference_cpsc2018.csv'
    mapper = CPSC2018Utils(BASE_DIR)

    # # .
    # sampling_rate = 500
    # batch_size = 32
    # seq_length = 500
    # patch_size = 25
    # latent_ratio = 0.5
    # channels = 12
    # codebook_size = 256
    # residual_levels = 2
    # total_length = int(seq_length * (1 + latent_ratio))
    # mapper.load_raw_data_cpsc2018(reference_path, sampling_rate, total_length)

    mapper.load_labels_cpsc2018(reference_path)
