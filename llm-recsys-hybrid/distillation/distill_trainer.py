# Knowledge distillation trainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Dummy Dataset (user_ids, item_ids, labels)
class RecDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.labels = labels

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]

# Student Model (Two-Tower, simplified)
class StudentModel(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size, embed_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(user_vocab_size, embed_dim)
        self.item_embedding = nn.Embedding(item_vocab_size, embed_dim)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        scores = (user_vec * item_vec).sum(dim=1)
        return scores

# Teacher Model — assume pretrained, here simplified as bigger embeddings
class TeacherModel(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size, embed_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(user_vocab_size, embed_dim)
        self.item_embedding = nn.Embedding(item_vocab_size, embed_dim)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        scores = (user_vec * item_vec).sum(dim=1)
        return scores

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    KL divergence between softened outputs
    """
    student_soft = nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = nn.functional.softmax(teacher_logits / temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(student_soft, teacher_soft) * (temperature ** 2)
    return loss

def train_kd(student, teacher, dataloader, epochs=5, lr=1e-3, temperature=2.0):
    optimizer = optim.Adam(student.parameters(), lr=lr)
    teacher.eval()

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for user_ids, item_ids, _ in dataloader:
            user_ids = user_ids.long()
            item_ids = item_ids.long()

            with torch.no_grad():
                teacher_logits = teacher(user_ids, item_ids).unsqueeze(1)  # shape [batch, 1]

            student_logits = student(user_ids, item_ids).unsqueeze(1)

            # Here logits shape: [batch, 1] so for softmax over batch dim, reshape is needed
            # For demo, treat batch dim as classes for KLDivLoss — or you can adapt accordingly.

            loss = distillation_loss(student_logits.T, teacher_logits.T, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    # Simulated data (user_ids, item_ids, labels)
    user_ids = torch.randint(0, 100, (1000,))
    item_ids = torch.randint(0, 500, (1000,))
    labels = torch.ones_like(user_ids)  # dummy labels, unused here

    dataset = RecDataset(user_ids, item_ids, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    student = StudentModel(user_vocab_size=100, item_vocab_size=500, embed_dim=32)
    teacher = TeacherModel(user_vocab_size=100, item_vocab_size=500, embed_dim=64)

    train_kd(student, teacher, dataloader)
