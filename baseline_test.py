import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import config
from dataloader import load_corpus_file, bytes_to_one_hot

# ----------------------------------------------------
# 1. Standard LSTM Baseline
# ----------------------------------------------------
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [Batch, Seq, Features]
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

# ----------------------------------------------------
# 2. Standard Transformer Baseline
# ----------------------------------------------------
class BaselineTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, nhead=4):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.transformer(x)
        return self.fc(out)

# ----------------------------------------------------
# 3. Training & Evaluation Loop
# ----------------------------------------------------
def train_and_eval_baseline(model, name, train_chunks, test_chunks, device, epochs=20, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*50}")
    print(f"[*] Training Baseline: {name}")
    print(f"{'='*50}")
    print("EPOCH | Train Loss | Train ACC | Test Loss | Test ACC | Time")
    
    start_time = time.time()
    
    max_train_acc = 0.0
    max_train_ep = 0
    max_test_acc = 0.0
    max_test_ep = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # --- TRAIN PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for chunk_in, chunk_target in train_chunks:
            # Format: [Seq_len, Batch(1), Dim] -> [Batch(1), Seq_len, Dim]
            x = bytes_to_one_hot(chunk_in, device).transpose(0, 1)
            target = bytes_to_one_hot(chunk_target, device).transpose(0, 1)
            target_idx = torch.argmax(target, dim=2).view(-1)
            
            optimizer.zero_grad()
            output = model(x)
            
            # Loss計算は全系列で行う
            loss = criterion(output.view(-1, config.NODE_OUTPUT_DIM), target_idx)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 💡 修正: 評価は「系列の最後の文字 ([:, -1, :])」のみ行う
            pred_idx = torch.argmax(output[:, -1, :], dim=1)
            true_idx = torch.argmax(target[:, -1, :], dim=1)
            train_correct += (pred_idx == true_idx).sum().item()
            train_total += 1
            
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # --- TEST PHASE ---
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        
        with torch.no_grad():
            for chunk_in, chunk_target in test_chunks:
                x = bytes_to_one_hot(chunk_in, device).transpose(0, 1)
                target = bytes_to_one_hot(chunk_target, device).transpose(0, 1)
                target_idx = torch.argmax(target, dim=2).view(-1)
                
                output = model(x)
                loss = criterion(output.view(-1, config.NODE_OUTPUT_DIM), target_idx)
                
                test_loss += loss.item()
                
                pred_idx = torch.argmax(output[:, -1, :], dim=1)
                true_idx = torch.argmax(target[:, -1, :], dim=1)
                test_correct += (pred_idx == true_idx).sum().item()
                test_total += 1
                
        test_acc = test_correct / test_total if test_total > 0 else 0.0
        
        # --- Update Max Values ---
        if train_acc > max_train_acc:
            max_train_acc = train_acc
            max_train_ep = epoch + 1
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_test_ep = epoch + 1
            
        # --- Display ---
        elapsed = time.time() - epoch_start
        print(f"{epoch+1:2d}/{epochs} | {train_loss/len(train_chunks):.4f}     | {train_acc*100:6.2f}%   | {test_loss/len(test_chunks):.4f}    | {test_acc*100:6.2f}%  | {elapsed:.1f}s")
        
    total_time = time.time() - start_time
    print(f"\n>>> [{name}] SUMMARY")
    print(f"TRAIN MAX {max_train_acc*100:.2f}% @ Ep {max_train_ep}")
    print(f"TEST  MAX {max_test_acc*100:.2f}% @ Ep {max_test_ep}")
    print(f"Total Time: {total_time:.1f}s\n")
    
    return max_train_acc, max_test_acc

if __name__ == "__main__":
    device = config.DEVICE
    print(f"Loading Base Corpus (Train) and Extrapolation Corpus (Test)...")
    
    # 💡 ここに、比較したいコーパスのファイル名を指定してください
    train_chunks = load_corpus_file("train_corpus_base.txt")
    test_chunks = load_corpus_file("test_corpus_extrapolate.txt")
    
    # 1. LSTM
    lstm_model = BaselineLSTM(config.NODE_INPUT_DIM, 256, config.NODE_OUTPUT_DIM)
    train_and_eval_baseline(lstm_model, "Standard LSTM", train_chunks, test_chunks, device)
    
    # 2. Transformer
    transformer_model = BaselineTransformer(config.NODE_INPUT_DIM, 256, config.NODE_OUTPUT_DIM)
    train_and_eval_baseline(transformer_model, "Standard Transformer", train_chunks, test_chunks, device)