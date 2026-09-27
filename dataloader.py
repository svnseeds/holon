# dataloader.py
# Zero-Overhead Hardware-Native Byte Stream Loader
import torch

class ByteStreamLoader:
    """
    Memory-mapped byte stream loader converting raw files directly into
    device-resident tensor tables with zero per-step CPU allocation overhead.
    """
    @torch.no_grad()
    def __init__(self, file_path, config):
        self.device = config.DEVICE
        self.io_dim = config.IO_DIM
        
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        print(f"[*] Loaded corpus: {file_path} ({len(raw_data)} bytes)")
        
        # Resident tensor on target compute device
        self.data = torch.tensor(list(raw_data), dtype=torch.long, device=self.device)
        self.size = len(self.data)
        self.ptr = 0
        
        # Precomputed identity matrix for zero-overhead one-hot lookup
        self.one_hot_table = torch.eye(self.io_dim, dtype=config.DTYPE, device=self.device)

    @torch.no_grad()
    def next_token(self):
        """Fetch next single-byte token as a one-hot vector and scalar index."""
        if self.ptr >= self.size:
            self.ptr = 0
        
        token = self.data[self.ptr]
        one_hot = self.one_hot_table[token]
        self.ptr += 1
        return one_hot, token

    @torch.no_grad()
    def reset(self):
        """Reset stream pointer to beginning of corpus."""
        self.ptr = 0