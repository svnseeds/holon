# dataloader.py
import os
import numpy as np
import torch
import config

def load_corpus_file(filename, chunk_size=config.INPUT_BYTES_SIZE):
    """
    Reads a raw text file and extracts seamless chunks of raw bytes.
    Generates (Input, Target) pairs where Target is shifted by 1 byte into the future.
    """
    if not os.path.exists(filename):
        print(f"Warning: Could not find {filename}. Operation Halted.")

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        
    print(f"Corpus preview (first 100 chars): {text[0:100]}\n")

    byte_array = text.encode('utf-8')
    
    # Pad if the corpus is too small
    if len(byte_array) <= chunk_size:
        byte_array += b'\x00' * (chunk_size + 1 - len(byte_array))

    chunks = []
    # Sliding window extraction over the raw byte stream
    for i in range(0, len(byte_array) - chunk_size, 1):
        chunk_in = byte_array[i : i + chunk_size]
        chunk_target = byte_array[i + 1 : i + 1 + chunk_size] # 1 byte into the future
        chunks.append((chunk_in, chunk_target))
        
    return chunks

def bytes_to_one_hot(byte_chunk, device=config.DEVICE):
    """Converts a byte chunk to a [Nodes, Batch(1), 256] one-hot float32 tensor."""
    node_size = len(byte_chunk)
    one_hot = np.zeros((node_size, 1, config.NODE_INPUT_DIM), dtype=np.float32)
    
    for idx, b in enumerate(byte_chunk):
        one_hot[idx, 0, b] = 1.0
        
    if device != "cpu":
        return torch.from_numpy(one_hot).to(device)
    return one_hot