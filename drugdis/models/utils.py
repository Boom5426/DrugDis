import torch
import numpy as np
import random
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import gc

# 内存保护阈值 (超过这个数量的药物/细胞对将进行下采样计算结构指标)
SAMPLE_LIMIT = 5000 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed}")

def get_base_metrics(y_true, y_pred):
    """ 计算并返回回归任务的基础指标 (RMSE, PCC, Spearman) """
    try:
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)
        
        valid_idx = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.sum(valid_idx) < 2: return 0.0, 0.0, 0.0

        y_true_valid = y_true[valid_idx]
        y_pred_valid = y_pred[valid_idx]

        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        # 添加 1e-8 防止常数序列导致的除零警告
        if np.std(y_true_valid) < 1e-9 or np.std(y_pred_valid) < 1e-9:
            pcc, spearman = 0.0, 0.0
        else:
            pcc, _ = pearsonr(y_true_valid, y_pred_valid)
            spearman, _ = spearmanr(y_true_valid, y_pred_valid)
        
        return float(rmse), float(0.0 if np.isnan(pcc) else pcc), float(0.0 if np.isnan(spearman) else spearman)
    except ValueError:
        return 0.0, 0.0, 0.0 

# --- CKA 计算函数 ---
def linear_cka(X, Y):
    """ 计算 Linear CKA (Centered Kernel Alignment) """
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # Center the matrices
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    gram_X = X @ X.T
    gram_Y = Y @ Y.T
    
    def hsic(K, L):
        n = K.shape[0]
        H = np.eye(n, dtype=np.float32) - 1.0/n * np.ones((n, n), dtype=np.float32)
        return np.trace(K @ H @ L @ H)
    
    hsic_xy = hsic(gram_X, gram_Y)
    hsic_xx = hsic(gram_X, gram_X)
    hsic_yy = hsic(gram_Y, gram_Y)
    
    # [Fix] 增加 epsilon 防止除零
    denominator = np.sqrt(hsic_xx) * np.sqrt(hsic_yy)
    if denominator <= 1e-8: return 0.0
    
    return float(hsic_xy / denominator)

class SmartBiasBaseline:
    def __init__(self, task_type='general'):
        self.global_mean = 0.0
        self.drug_bias = {}
        self.cell_bias = {}
        self.task_type = task_type 

    def fit(self, train_df):
        target_col = 'Target_AAC' if 'Target_AAC' in train_df.columns else 'Sensitivity'
        self.global_mean = train_df[target_col].mean()
        
        drug_means = train_df.groupby('SMILES')[target_col].mean().astype(np.float32)
        self.drug_bias = (drug_means - self.global_mean).to_dict()
        
        cell_means = train_df.groupby('Sample_ID')[target_col].mean().astype(np.float32)
        self.cell_bias = (cell_means - self.global_mean).to_dict()
        
    def get_bias_maps(self):
        return {
            'global_mean': self.global_mean,
            'drug_bias': self.drug_bias,
            'cell_bias': self.cell_bias
        }

class InteractionEffectEvaluator:
    def __init__(self, df_context, split_type='general'):
        """
        :param df_context: 用于学习 Bias 的数据集 (通常是 Train Set)
        :param split_type: 任务类型 ('LSO', 'LCLO', 'general')
        """
        self.split_type = split_type
        
        self.bias_model = SmartBiasBaseline(split_type)
        self.bias_model.fit(df_context)
        self.bias_maps = self.bias_model.get_bias_maps()
        
        # [Fix] 显式暴露 bias map 属性，防止 AttributeError
        self.drug_bias_map = self.bias_maps['drug_bias']
        self.cell_bias_map = self.bias_maps['cell_bias']
        self.global_mean = self.bias_maps['global_mean']

    def get_interaction_effects(self, df_with_preds):
        """
        解耦 B部分 (互动效应) - 包含 Nature Methods 要求的动态去偏逻辑
        注意：此函数通常应在完整的 Validation/Test DataFrame 上调用，而不是在 mini-batch 上
        """
        # 1. 动态 Bias 计算 (防止 LSO/LCLO 下的指标虚高)
        if self.split_type == 'LCLO':
             # LCLO: 细胞是新的，必须从当前测试集动态计算 Cell Bias
             # 使用 transform 保持维度对齐
             cell_bias = df_with_preds.groupby('Sample_ID')['Target_AAC'].transform('mean') - self.global_mean
             drug_bias = df_with_preds['SMILES'].map(self.drug_bias_map).fillna(0.0)
        
        elif self.split_type == 'LSO':
             # LSO: 药物是新的，必须从当前测试集动态计算 Drug Bias
             current_drug_means = df_with_preds.groupby('SMILES')['Target_AAC'].transform('mean')
             drug_bias = current_drug_means - self.global_mean
             cell_bias = df_with_preds['Sample_ID'].map(self.cell_bias_map).fillna(0.0)
             
        else:
             # General/Random Split: 沿用训练集的 Bias
             drug_bias = df_with_preds['SMILES'].map(self.drug_bias_map).fillna(0.0)
             cell_bias = df_with_preds['Sample_ID'].map(self.cell_bias_map).fillna(0.0)

        # 2. 组装 Bias (A部分)
        system_score = self.global_mean + drug_bias + cell_bias
        
        # 3. 计算残差 (B部分: Interaction)
        # 核心逻辑：Interaction = Observed - (Mean + Bias_Drug + Bias_Cell)
        interaction_true = df_with_preds['Target_AAC'] - system_score
        interaction_pred = df_with_preds['Predicted_AAC'] - system_score
        
        df_with_preds['Interaction_True'] = interaction_true
        df_with_preds['Interaction_Pred'] = interaction_pred
        
        return interaction_true.values, interaction_pred.values

    def calculate_zscore_pcc(self, df_with_preds):
        try:
            target_col = 'Target_AAC'
            pred_col = 'Predicted_AAC'
            
            def z_score_group(group):
                std = group.std()
                if std == 0: return (group - group.mean())
                return (group - group.mean()) / std

            z_true = df_with_preds.groupby('SMILES')[target_col].transform(z_score_group).astype(np.float32)
            z_pred = df_with_preds.groupby('SMILES')[pred_col].transform(z_score_group).astype(np.float32)
            
            _, z_pcc, _ = get_base_metrics(z_true.values, z_pred.values)
            return z_pcc
        except:
            return 0.0

    def _compute_structural_metrics(self, df, col_true, col_pred):
        try:
            # Pivot (SMILES x Sample_ID)
            # 先聚合去重，防止重复对导致 Pivot 失败
            df_deduped = df.groupby(['SMILES', 'Sample_ID'])[[col_true, col_pred]].mean().reset_index()
            
            unique_drugs = df_deduped['SMILES'].unique()
            # 内存保护：如果药物太多，进行下采样
            if len(unique_drugs) > SAMPLE_LIMIT:
                selected_drugs = np.random.choice(unique_drugs, SAMPLE_LIMIT, replace=False)
                df_deduped = df_deduped[df_deduped['SMILES'].isin(selected_drugs)]
            
            # 填充缺失值为 -999，方便后续掩码处理
            true_pivot = df_deduped.pivot(index='SMILES', columns='Sample_ID', values=col_true).fillna(-999)
            pred_pivot = df_deduped.pivot(index='SMILES', columns='Sample_ID', values=col_pred).fillna(-999)
            
            del df_deduped; gc.collect()

            # 对齐行列
            all_drugs = true_pivot.index.intersection(pred_pivot.index)
            all_cells = true_pivot.columns.union(pred_pivot.columns)
            
            if len(all_drugs) < 10: return 0.0, 0.0
            
            true_mat = true_pivot.loc[all_drugs].reindex(columns=all_cells, fill_value=-999).values.astype(np.float32)
            pred_mat = pred_pivot.loc[all_drugs].reindex(columns=all_cells, fill_value=-999).values.astype(np.float32)
            
            del true_pivot, pred_pivot; gc.collect()

            # Metric 1: CKA
            # 将填充的 -999 替换为 0 用于计算 (假设缺失值的响应为中性/零)
            # 注意：如果数据已经去偏（Interaction），均值接近0，这种填充是合理的
            true_mat_cka = np.where(true_mat == -999, 0, true_mat)
            pred_mat_cka = np.where(pred_mat == -999, 0, pred_mat)
            cka_score = linear_cka(pred_mat_cka, true_mat_cka)
            
            # Metric 2: Retrieval Top-10 (ieR10)
            k = 10
            if len(all_drugs) < k: k = len(all_drugs)
            
            # 对每一列（细胞）进行排序
            topk_pred_indices = np.argsort(pred_mat, axis=0)[-k:, :]
            topk_true_indices = np.argsort(true_mat, axis=0)[-k:, :]
            
            n_cells = true_mat.shape[1]
            total_hits = 0
            for col in range(n_cells):
                pred_set = set(topk_pred_indices[:, col])
                true_set = set(topk_true_indices[:, col])
                total_hits += len(pred_set.intersection(true_set))
                
            ie_r10 = total_hits / (n_cells * k)
            
            del true_mat, pred_mat; gc.collect()
            return cka_score, ie_r10
            
        except Exception as e:
            print(f"Warning in structural metrics: {e}")
            return 0.0, 0.0

    def calculate_metrics(self, df_with_preds):
        """ 统一的指标计算入口 """
        # 兼容性处理
        if 'Target_AAC' not in df_with_preds.columns and 'Sensitivity' in df_with_preds.columns:
            df_with_preds['Target_AAC'] = df_with_preds['Sensitivity']
        if 'Predicted_AAC' not in df_with_preds.columns and 'Prediction' in df_with_preds.columns:
            df_with_preds['Predicted_AAC'] = df_with_preds['Prediction']

        y_true = df_with_preds['Target_AAC'].values.astype(np.float32)
        y_pred = df_with_preds['Predicted_AAC'].values.astype(np.float32)
        
        # 1. Global Metrics (Standard Evaluation)
        rmse, pcc, spearman = get_base_metrics(y_true, y_pred)
        
        # Ovchinnikova Z-score metric
        z_pcc = self.calculate_zscore_pcc(df_with_preds)
        
        # Global Structure (Based on raw values)
        global_cka, _ = self._compute_structural_metrics(df_with_preds, 'Target_AAC', 'Predicted_AAC')
        
        # 2. Interaction Metrics (CLIO Decoupling)
        # 获取去偏后的残差
        interaction_true, interaction_pred = self.get_interaction_effects(df_with_preds)
        
        # 在残差上计算指标
        _, ie_pcc, ie_spearman = get_base_metrics(interaction_true, interaction_pred)
        ie_cka, ie_r10 = self._compute_structural_metrics(df_with_preds, 'Interaction_True', 'Interaction_Pred')
        
        return {
            'RMSE': rmse, 
            'PCC': pcc, 
            'Spearman': spearman, 
            'Z_PCC': z_pcc, 
            'CKA': global_cka,
            'iePCC': ie_pcc, 
            'ieSpearman': ie_spearman, 
            'ieCKA': ie_cka, 
            'ieR10': ie_r10,
            'ieCCA': ie_r10 # Alias keeping for consistency
        }