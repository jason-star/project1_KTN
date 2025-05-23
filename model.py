import torch
from torch import nn
from torch.nn.parameter import Parameter

from utils.utils import *

class BertForModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        outputs = self.backbone(**X, output_hidden_states=True)
        CLSEmbedding = torch.mean(outputs.hidden_states[-1][:, 1:], dim=1)
        CLSEmbedding = self.dropout(CLSEmbedding)

        logits = self.classifier(CLSEmbedding)

        return CLSEmbedding, logits

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, y)
        return output 


class BertForOT(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForOT, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels, bias=False)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        outputs = self.backbone(**X, output_hidden_states=True)
        CLSEmbedding = torch.mean(outputs.hidden_states[-1][:, 1:], dim=1)
        CLSEmbedding = self.dropout(CLSEmbedding)

        logits = self.classifier(F.normalize(CLSEmbedding, dim=1))

        return CLSEmbedding, logits


#余弦相似度计算
    def loss_contrast(self, emb_i, emb_j, temperature):		
        z_i = F.normalize(emb_i, dim=1)    # 对emb_i进行归一化处理，使其在第1维度上的每个向量的长度为1
        z_j = F.normalize(emb_j, dim=1)    

        representations = torch.cat([z_i, z_j], dim=0)    # 将z_i和z_j沿着第0维度拼接起来      
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # 计算所有表示之间的余弦相似度矩阵
        
        sim_ij = torch.diag(similarity_matrix, len(emb_i))       # 提取对角线上的相似度值（即相同表示之间的相似度）  
        sim_ji = torch.diag(similarity_matrix, -len(emb_i))        
        positives = torch.cat([sim_ij, sim_ji], dim=0)          # 将sim_ij和sim_ji拼接起来，形成正样本的相似度矩阵        
        negatives_mask = (~torch.eye(len(emb_i) * 2, len(emb_i) * 2, dtype=bool).to(emb_i.device)).float()   # 创建一个掩码矩阵，用于标记负样本

        nominator = torch.exp(positives / temperature)             
        denominator = negatives_mask * torch.exp(similarity_matrix / temperature)      # 计算负样本的相似度得分       
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))       # 计算部分损失
        loss = torch.sum(loss_partial) / (2 * len(emb_i))   # 计算总损失
        return loss
    