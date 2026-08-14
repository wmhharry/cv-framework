import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Dict

class GlaSClassificationDataset(Dataset):
    """
    GlaS 分類資料集讀取器 (基於 CSV 檔案讀取 Benign / Malignant 標籤)
    """
    def __init__(
        self, 
        root_dir: str, 
        csv_path: str, 
        split: str = "train", 
        transform=None,
        class_mapping: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            root_dir: 影像檔案根目錄 (包含 train_1.bmp, testA_1.bmp 等)
            csv_path: 包含檔名與標籤的 CSV 檔案路徑
            split: 'train' 或 'test' (或 'testA' / 'testB')
            transform: Albumentations 影像增強 Pipeline
            class_mapping: 類別字串與整數的對照表
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split.lower()
        
        # 預設標籤對應表
        self.class_mapping = class_mapping or {
            "benign": 0,
            "malignant": 1
        }
        
        # 1. 讀取 CSV 檔
        df = pd.read_csv(csv_path)
        
        # 清理欄位名稱與字串空白 (常見欄位: 'name', 'grade' / 'patient', 'grade')
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 自動尋找影像檔名欄位與分級欄位
        name_col = [c for c in df.columns if any(k in c for k in ["name", "file", "image", "patient"])][0]
        grade_col = [c for c in df.columns if any(k in c for k in ["grade", "label", "class", "diagnosis"])][0]

        # 2. 根據 split 篩選資料 (例如檔名開頭為 'train' 或 'test')
        df = df[df[name_col].astype(str).str.lower().str.startswith(self.split)].copy()
        
        # 3. 建立影像路徑與標籤清單
        self.samples = []
        for _, row in df.iterrows():
            img_name = str(row[name_col]).strip()
            # 若 CSV 檔名沒有副檔名，自動補上 .bmp
            if not img_name.endswith(('.bmp', '.png', '.jpg', '.tif')):
                img_name += '.bmp'
                
            img_path = os.path.join(self.root_dir, img_name)
            
            # 解析標籤 (轉為小寫比對)
            grade_str = str(row[grade_col]).strip().lower()
            if grade_str in self.class_mapping:
                label = self.class_mapping[grade_str]
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                else:
                    print(f"找不到影像檔案: {img_path}，已略過。")
            else:
                print(f"未知的標籤類型: {grade_str} (檔案: {img_name})，已略過。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 讀取影像
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"無法讀取影像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 影像增強與正規化
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)