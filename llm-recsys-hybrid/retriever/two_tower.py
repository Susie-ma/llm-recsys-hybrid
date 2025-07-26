# Simplified Two-Tower model
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, user_vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(user_vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, user_ids):
        x = self.embedding(user_ids)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, item_vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(item_vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, item_ids):
        x = self.embedding(item_ids)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size, embed_dim=64):
        super().__init__()
        self.user_tower = UserTower(user_vocab_size, embed_dim)
        self.item_tower = ItemTower(item_vocab_size, embed_dim)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_tower(user_ids)
        item_vec = self.item_tower(item_ids)
        # 余弦相似度作为预测评分
        scores = (user_vec * item_vec).sum(dim=1)
        return scores
