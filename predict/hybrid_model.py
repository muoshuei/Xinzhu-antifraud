import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer

class HybridBertModel(nn.Module):
    def __init__(self, tabular_input_dim, dropout_rate=0.3):
        super(HybridBertModel, self).__init__()
        
        # 1. BERT Module (Text)
        # 使用中文 RoBERTa (hfl/chinese-roberta-wwm-ext)
        self.bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.bert_hidden_size = self.bert.config.hidden_size  # 通常是 768
        
        # 2. MLP Module (Tabular)
        # 處理數值與類別特徵
        self.mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 3. Fusion & Classifier
        # 結合 BERT 的 CLS token 輸出與 MLP 輸出
        fusion_dim = self.bert_hidden_size + 32
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 輸出 0~1 的機率值 (信心度)
        )

    def forward(self, input_ids, attention_mask, tabular_data):
        # Text Path
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取出 [CLS] token 的向量 (代表整句語意)
        text_embedding = bert_output.pooler_output  # Shape: (batch_size, 768)
        
        # Tabular Path
        tabular_embedding = self.mlp(tabular_data)  # Shape: (batch_size, 32)
        
        # Fusion
        combined = torch.cat((text_embedding, tabular_embedding), dim=1)
        
        # Classification
        probability = self.classifier(combined)
        
        return probability
