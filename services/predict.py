import torch
import joblib
import pandas as pd
import numpy as np
from transformers.models.bert.tokenization_bert import BertTokenizer
from services.hybrid_model import HybridBertModel
import re
import os
import json
import argparse

# 設定參數
MODEL_PATH = './model/fraud_detection_model.pth'
SCALER_PATH = './services/scaler.pkl'
MAPPING_PATH = './services/category_mappings.json'
MAX_LEN = 256

class FraudPredictor:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH, mapping_path=MAPPING_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 載入 Scaler
        self.scaler = joblib.load(scaler_path)
        
        # 載入 Category Mappings
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.mappings = json.load(f)
            print(f"Loaded category mappings from {mapping_path}")
        else:
            print(f"Warning: Mapping file not found at {mapping_path}. Categorical features might fail.")
            self.mappings = {}
        
        # 載入 Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        # 載入模型
        # 這裡假設 tabular_dim 是 12 (跟訓練時一樣)
        self.model = HybridBertModel(tabular_input_dim=12)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _apply_mapping(self, data_dict):
        """
        內部輔助方法：將字典中的文字類別轉換為數值 ID
        """
        # 需要轉換的三個主要欄位
        target_cols = [
            '縣市 (City/County)', 
            '工作經歷 (Work Experience)', 
            '學歷要求 (Educational Requirements)'
        ]

        for col in target_cols:
            if col in self.mappings:
                raw_val = data_dict.get(col)
                
                # 處理 NaN 或 None
                if pd.isna(raw_val) or raw_val is None:
                    raw_val = "Unknown"
                
                raw_val_str = str(raw_val).strip()
                mapping_dict = self.mappings[col]
                
                # 取得對應數值，如果找不到則使用該類別定義的 "Unknown" 值，若無則預設 0
                unknown_val = mapping_dict.get("Unknown", 0)
                mapped_val = mapping_dict.get(raw_val_str, unknown_val)
                
                # 更新字典
                data_dict[col] = mapped_val
        
        return data_dict

    def preprocess_input(self, data_dict):
        """
        將單筆輸入資料轉換為模型可接受的格式
        """
        # 1. 簡單特徵工程 (文字部分)
        full_content = data_dict.get('full_content', '')
        if not full_content or pd.isna(full_content):
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
        # 注意：這裡預期 '縣市', '工作經歷', '學歷要求' 已經在 _apply_mapping 中轉為 int 了
        try:
            features = [
                float(data_dict.get('salary_min', 0)),
                float(data_dict.get('salary_max', 0)),
                int(data_dict.get('salary_type', 0)),
                float(data_dict.get('capital_amount_cleaned', 0)),
                int(data_dict.get('employees_cleaned', 0)),
                int(data_dict.get('供需人數 (應徵人數) (Number of Applicants)', 0)),
                int(data_dict.get('縣市 (City/County)', 0)), 
                int(data_dict.get('工作經歷 (Work Experience)', 0)),
                int(data_dict.get('學歷要求 (Educational Requirements)', 0)),
                desc_length,
                has_line,
                has_risk
            ]
        except ValueError as e:
            print(f"Error converting features: {e}, Data: {data_dict}")
            # 發生轉換錯誤時回傳全 0 向量，避免崩潰
            features = [0] * 12

        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        return full_content, torch.tensor(features_scaled, dtype=torch.float)
    
    def predict(self, data_dict):
        # 在這裡預處理並轉 Tensor
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
            'confidence_score': round(score, 4), 
            'risk_level': 'High' if score > 0.8 else 'Medium' if score > 0.5 else 'Low'
        }
    
    def predict_csv(self, input_df, output_path='predictions.csv'):
        """
        接收 DataFrame，轉換類別欄位後進行預測
        """
        df = input_df.copy() # 避免修改到原始 df
        results = []
        
        print(f"Processing {len(df)} records...")
        
        for idx, row in df.iterrows():
            # 1. 轉為字典
            data_dict = row.to_dict()
            
            try:
                # 顯示進度
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df)} records")
                
                # 2. 【關鍵修改】在此處進行 Mapping 轉換 (String -> Int)
                # 這會將 '台北市' 轉為 47 (依據您的 JSON)
                data_dict = self._apply_mapping(data_dict)
                
                # 3. 進行預測 (傳入的 data_dict 已經是乾淨的數值了)
                pred = self.predict(data_dict)
                results.append(pred)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                # 印出有問題的資料以便除錯
                print(f"Problematic Data: {row.to_dict()}") 
                results.append({'is_fraud': False, 'confidence_score': 0, 'risk_level': 'Error'})
                
        # 將預測結果轉換為 DataFrame
        results_df = pd.DataFrame(results)
        
        # 合併原始資料與預測結果
        # reset_index 確保 concat 時不會因為索引不對齊而產生 NaN
        final_df = pd.concat([input_df.reset_index(drop=True), results_df], axis=1)
        
        # 儲存結果
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Prediction complete. Results saved to '{output_path}'")
        return final_df

# 測試區塊
if __name__ == "__main__":
    # 簡單測試用，模擬外部呼叫
    # 建立一個假的 DataFrame
    data = {
        'full_content': ['急徵在家工作打字員'],
        'salary_min': [30000],
        'salary_max': [40000],
        'salary_type': [1],
        'capital_amount_cleaned': [0],
        'employees_cleaned': [0],
        '供需人數 (應徵人數) (Number of Applicants)': [5],
        '縣市 (City/County)': ['台北市大安區'], # 測試 Mapping
        '工作經歷 (Work Experience)': ['不拘'],  # 測試 Mapping
        '學歷要求 (Educational Requirements)': ['高中'] # 測試 Mapping
    }
    test_df = pd.DataFrame(data)

    try:
        predictor = FraudPredictor()
        predictor.predict_csv(test_df, 'test_output.csv')
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Ensure 'category_mappings.json', 'scaler.pkl', and model file exist.")