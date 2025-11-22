import torch
import joblib
import pandas as pd
import numpy as np
from transformers.models.bert.tokenization_bert import BertTokenizer
from predict.hybrid_model import HybridBertModel
import re

# 設定參數
MODEL_PATH = 'model\\fraud_detection_model.pth'
SCALER_PATH = 'model\\scaler.pkl'
MAX_LEN = 25

class FraudPredictor:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 載入 Scaler
        self.scaler = joblib.load(scaler_path)
        
        # 載入 Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        # 載入模型
        # 這裡假設 tabular_dim 是 12 (跟訓練時一樣)
        self.model = HybridBertModel(tabular_input_dim=12)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_input(self, data_dict):
        """
        將單筆輸入資料轉換為模型可接受的格式
        data_dict 必須包含訓練時的所有原始欄位
        """
        # 1. 簡單特徵工程 (與 select_features.py 邏輯一致)
        full_content = data_dict.get('full_content', '')
        
        # 如果沒有 full_content，嘗試從其他欄位組合
        if not full_content:
            full_content = (
                str(data_dict.get('職缺名稱 (Job Title)', '')) + ' ' + 
                str(data_dict.get('工作內容 (Job Description)', '')) + ' ' + 
                str(data_dict.get('其他條件 (Other Conditions)', '')) + ' ' +
                str(data_dict.get('公司名稱 (Company Name)', ''))
            )
            
        desc_length = len(full_content)
        
        line_pattern = r'(?i)line\s*(?:id|:|：|\s)|加\s*line'
        has_line = 1 if re.search(line_pattern, full_content) else 0
        
        wfh_pattern = r'在家工作|居家辦公|輕鬆|簡單|免經驗|日領|現領'
        has_risk = 1 if re.search(wfh_pattern, full_content) else 0
        
        # 2. 準備數值向量
        # 注意：這裡的順序必須跟訓練時完全一樣！
        # 'salary_min', 'salary_max', 'salary_type', 'capital_amount_cleaned', 'employees_cleaned', 
        # '供需人數 (應徵人數) (Number of Applicants)', '縣市 (City/County)', '工作經歷 (Work Experience)', 
        # '學歷要求 (Educational Requirements)', 'desc_length', 'has_line_keyword', 'has_high_risk_keywords'
        
        # 為了簡化預測流程，這裡假設輸入已經包含清洗好的數值
        # 在實際應用中，這裡應該要包含 parse_salary 等清洗邏輯
        # 這裡先做簡單處理
        
        features = [
            data_dict.get('salary_min', 0),
            data_dict.get('salary_max', 0),
            data_dict.get('salary_type', 0),
            data_dict.get('capital_amount_cleaned', 0),
            data_dict.get('employees_cleaned', 0),
            data_dict.get('供需人數 (應徵人數) (Number of Applicants)', 0),
            data_dict.get('縣市 (City/County)', 0), # 假設已編碼
            data_dict.get('工作經歷 (Work Experience)', 0), # 假設已編碼
            data_dict.get('學歷要求 (Educational Requirements)', 0), # 假設已編碼
            desc_length,
            has_line,
            has_risk
        ]
        
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        return full_content, torch.tensor(features_scaled, dtype=torch.float)

    def predict(self, data_dict):
        """
        sample_data = {
            'full_content': '急徵在家工作人員，日領薪水，加LINE ID: scam123',
            'salary_min': 50000,
            'salary_max': 100000,
            'salary_type': 1,
            'capital_amount_cleaned': 0,
            'employees_cleaned': 0,
            '供需人數 (應徵人數) (Number of Applicants)': 10,
            '縣市 (City/County)': 1,
            '工作經歷 (Work Experience)': 0,
            '學歷要求 (Educational Requirements)': 0
        }
        """
        text, tabular_tensor = self.preprocess_input(data_dict)
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        tabular_tensor = tabular_tensor.to(self.device)
        
        with torch.no_grad():
            probability = self.model(input_ids, attention_mask, tabular_tensor)
            score = probability.item()
            
        return {
            'is_fraud': score > 0.5,
            'confidence_score': score, # 0~1, 越接近 1 越像詐騙
            'risk_level': 'High' if score > 0.8 else 'Medium' if score > 0.5 else 'Low'
        }

# 測試用
if __name__ == "__main__":
    # 模擬一筆測試資料
    sample_data = {
        'full_content': '急徵在家工作人員，日領薪水，加LINE ID: scam123',
        'salary_min': 50000,
        'salary_max': 100000,
        'salary_type': 1,
        'capital_amount_cleaned': 0,
        'employees_cleaned': 0,
        '供需人數 (應徵人數) (Number of Applicants)': 10,
        '縣市 (City/County)': 1,
        '工作經歷 (Work Experience)': 0,
        '學歷要求 (Educational Requirements)': 0
    }
    
    try:
        predictor = FraudPredictor()
        result = predictor.predict(sample_data)
        print("Prediction Result:")
        print(result)
    except Exception as e:
        print(f"Model not found or error: {e}")
        print("Please run train.py first.")
