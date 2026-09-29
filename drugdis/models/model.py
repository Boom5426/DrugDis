import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIO(nn.Module):
    """
    *** 已优化 (Optimized "Two-Tower" Version) ***
    
    该模型现在是一个“双编码器 (Dual Encoder)”MLP。
    它首先使用两个独立的MLP塔 (towers) 分别处理药物和组学特征，
    将它们投影 (project) 到一个共同的低维空间。
    
    然后，它拼接这些低维嵌入，并通过一个联合MLP (joint MLP) 
    来预测最终的敏感性分数 (AAC)。
    """
    
    def __init__(self, 
                 drug_dim, 
                 gene_dim, 
                 projection_dim=512,  # 两个塔投影到的共同维度
                 hidden_dim1=1024,    # 联合MLP的第一个隐藏层
                 hidden_dim2=512,     # 联合MLP的第二个隐藏层
                 dropout_p=0.4):
        """
        初始化双编码器回归 MLP。
        
        参数:
        drug_dim (int): 
            原始药物特征维度 (例如 ECFP4=2048)
            
        gene_dim (int): 
            原始组学特征维度 (例如 baseline_mrna=15962)
            
        projection_dim (int): 
            两个塔投影到的共同维度 (例如 512)
            
        hidden_dim1 (int): 
            联合MLP的第一个隐藏层维度 (例如 1024)
            
        hidden_dim2 (int): 
            联合MLP的第二个隐藏层维度 (例如 512)
            
        dropout_p (float): Dropout 概率。
        """
        super(CLIO, self).__init__()
        
        # --- 1. 药物编码塔 (Drug Tower) ---
        # (drug_dim -> projection_dim)
        # 这是您要求的 "先用一层MLP分别处理"
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, projection_dim),
            nn.ReLU(),
            nn.BatchNorm1d(projection_dim),
            nn.Dropout(dropout_p)
        )
        
        # --- 2. 组学编码塔 (Genomics Tower) ---
        # (gene_dim -> projection_dim)
        # (我们对高维的基因组学使用一个稍微深一点的编码器
        #  以更好地将其压缩到低维空间)
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, 1024), # (15962 -> 1024)
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_p),
            nn.Linear(1024, projection_dim), # (1024 -> 512)
            nn.ReLU(),
            nn.BatchNorm1d(projection_dim),
            nn.Dropout(dropout_p)
        )

        # --- 3. 联合MLP头 (Joint Head) ---
        # 拼接后的维度将是 (projection_dim + projection_dim)
        self.joint_input_dim = projection_dim * 2
        
        self.joint_head = nn.Sequential(
            # (1024 -> 1024)
            nn.Linear(self.joint_input_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(dropout_p),
            
            # (1024 -> 512)
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(dropout_p),
            
            # (512 -> 1)
            nn.Linear(hidden_dim2, 1) # 回归
        )
        
        print(f"[Model Init] CLIO (Dual Encoder MLP) 已初始化：")
        print(f"  > 药物塔: {drug_dim} -> {projection_dim}")
        print(f"  > 组学塔: {gene_dim} -> 1024 -> {projection_dim}")
        print(f"  > 联合头: {self.joint_input_dim} -> {hidden_dim1} -> {hidden_dim2} -> 1")


    def forward(self, drug_batch, cell_batch):
        """
        模型的前向传播。
        """
        
        # 1. 分别编码 (Encode separately)
        drug_embedding = self.drug_encoder(drug_batch) # (B, projection_dim)
        gene_embedding = self.gene_encoder(cell_batch) # (B, projection_dim)
        
        # 2. 拼接低维嵌入
        # (B, projection_dim * 2)
        x = torch.cat([drug_embedding, gene_embedding], dim=1)
        
        # 3. 通过联合头进行预测
        return self.joint_head(x)