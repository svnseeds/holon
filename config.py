# config.py
# Holon v14.0 Canonical Configuration & Hyperparameters
# Hardware-Native 1-N Modular Cortical Network
# holon_v14.0.2
import math
import os
import random
import numpy as np
import torch

class Config:
    """
    Holon v14.0 Unified Hyperparameters & Constant Registry
    - Fully vectorized 1-N multi-column cortical array (Parallel IP-Core Clones)
    - Zero handcrafted heuristics: Uniform initial credit (0.50) & uniform standby floor (0.10)
    - True Winner-Take-All Top-2 lateral inhibition with extrinsic climb
    - Zero update on uncorrelated inputs (E_1st >= 1.000)
    - Deterministic world-universal Haar Dejima ROM interface
    """
    def __init__(self, seed=42):
        # 1. Hardware & Compute Environment Recognition (CUDA, MPS, CPU)
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.DEVICE = torch.device("mps")
        else:
            self.DEVICE = torch.device("cpu")
        self.DTYPE = torch.float32

        # 2. Reproducibility & Deterministic Seed Protocol
        self.SEED = seed
        self.set_seed(self.SEED)

        # 3. Network Dimensions & Physical Modular Sizing
        self.IO_DIM = 256
        self.LATENT_DIM = 256
        self.INV_SQRT_D = 1.0 / math.sqrt(self.LATENT_DIM)

        self.NUM_NODES_L0 = 1
        self.NUM_NODES_L1 = 4
        self.NUM_NODES_L2 = 4
        self.NUM_SUPER_COLUMNS = 3

        # 4. Timescale Hierarchy (Identical IP-Core Spectrum)
        self.LEAK_L0 = 0.50
        self.LEAKS_L1 = [0.2500, 1.0 / 6.0, 0.1250, 0.1000]
        self.LEAKS_L2 = [0.1250, 1.0 / 18.0, 1.0 / 32.0, 0.0200]
        self.SPECTRAL_RADIUS = 0.90
        self.POWER_ITER_STEPS = 15

        # 5. Holographic Latent Space & Normalization Standards
        self.TARGET_RMS = 1.000
        self.EPSILON = 1e-6
        self.E_BASELINE = 1.000           # Physical error baseline for uncorrelated noise

        # 6. Universal Deterministic Haar Dejima Protocol
        self.SEED_SENSOR_PROTOCOL = 42

        # 7. LFSR Pacemaker Clock ("Donkama" Tick)
        self.CLOCK_LFSR_TAPS = (0, 2, 5, 10)
        self.CLOCK_SMOOTH_ALPHA = 0.50
        self.CLOCK_BUDGET = 0.500         # Pacemaker ratio to prevent residual vanishing
        self.CLOCK_NORM_CONST = math.sqrt((2.0 - self.CLOCK_SMOOTH_ALPHA) / self.CLOCK_SMOOTH_ALPHA)

        # 8. Deterministic Seed Offsets (Silicon IP-Core Instantiation)
        self.CLOCK_SEED_BASE = 100
        self.CLOCK_SEED_COL_STEP = 10
        self.SEED_OFFSET_RES_L0 = 10
        self.SEED_OFFSET_RES_L1 = 20
        self.SEED_OFFSET_RES_L2 = 30
        self.SEED_OFFSET_WOUT_L0 = 60
        self.SEED_OFFSET_WOUT_L1 = 70
        self.SEED_OFFSET_WOUT_L2 = 80

        # 9. Hyper-Local Correlation Delta Plasticity & Output Readouts
        self.ETA_DELTA = 0.001
        self.W_OUT_NORM_MAX = 3.0
        self.W_OUT_INIT_STD = 0.005
        # Somatic Synaptic Turnover / Leaky Readout Decay (Oja-style Regularization)
        # Prevents memorization overfitting during prolonged stream exposure without manual early stopping.
        # Hardware shift equivalent: ~ 2^(-16) (Maps to >> 16 bit-shift subtraction)
        self.W_OUT_LEAK = 1e-5
        

        # 10. Intra-Column Credit Dynamics (W_td)
        self.ETA_TD = 0.05
        self.W_TD_INIT = 0.50
        self.W_TD_MIN = 0.10
        self.W_TD_MAX = 1.00
        self.W_TD_TEST_GAIN = 0.01

        # 11. Inter-Column Credit Metabolism (W_col: Pure Symmetric Baseline)
        self.ETA_COL_DROP = 0.50          # Penalty rate for defeated modules
        self.ETA_COL_RISE = 0.50          # Ascent rate for matching winner
        self.W_COL_INIT = 0.50            # Uniform initial credit for all columns
        self.W_COL_MIN = 0.10             # Uniform dormant standby floor
        self.W_COL_MAX = 1.00
        self.W_COL_POWER = 4.0            # Fourth-power contrastive WTA exponent
        self.PLASTICITY_THRESHOLD = 5e-4  # Instant plasticity gating threshold
        # Sliding Window Persistence Gating (M-of-N Window)
        # Window length N=6 (Hardware 6-bit shift register, maps to 1x LUT6 on FPGA)
        # Minimum win threshold M=4 (4 wins out of last 6 steps = 66.7% majority)
        self.WINDOW_N = 6
        self.WINDOW_M = 4

        # 12. Somatic Homeostasis
        self.H_ACT_TARGET = 0.55
        self.GAIN_LEARNING_RATE = 0.05
        self.GAIN_MIN = 0.10
        self.GAIN_MAX = 5.00

        # 13. ASCII Byte Code Standards
        self.CHAR_ROUND_OPEN = ord('(')
        self.CHAR_ROUND_CLOSE = ord(')')
        self.CHAR_SQUARE_OPEN = ord('[')
        self.CHAR_SQUARE_CLOSE = ord(']')
        self.CHAR_A = ord('a')
        self.CHAR_B = ord('b')
        self.CHAR_C = ord('c')           
        self.CHAR_D = ord('d')           
        self.TRIGGER_MIRROR = ord('@')
        self.TRIGGER_FIFO = ord('>')
        self.CHAR_SPACE = ord(' ')

        # 14. Benchmark Task Registry & Bayes Optimal Limits
        self.TASKS = {
            'A': {
                'name': 'Pure V2D2 Dyck-2',
                'train_corpus': 'train_v2d2.txt',
                'test_corpus': 'test_v2d2.txt',
                'bayes_limit': 70.00,
                'eval_margin': 2.0,
                'epochs': 40,
            },
            'B': {
                'name': 'V2L2-Mirror',
                'train_corpus': 'train_v2l2_mirror.txt',
                'test_corpus': 'test_v2l2_mirror.txt',
                'bayes_limit': 83.33,
                'eval_margin': 1.5,
                'epochs': 40,
            },
            'C': {
                'name': 'V2L2-FIFO',
                'train_corpus': 'train_v2l2_fifo.txt',
                'test_corpus': 'test_v2l2_fifo.txt',
                'bayes_limit': 83.33,
                'eval_margin': 1.5,
                'epochs': 40,
            }
        }

        # 15. Autonomous Closed-Loop Oscillation Bounds
        self.GEN_PATTERN_MAX = 20
        self.GEN_STEP_MAX = 500
        self.GEN_PREVIEW_LEN = 80

        # 16. Two-Tier Dynamics Visualization Standards
        self.MACRO_SAMPLE_INTERVAL = 50   # Sampling decimation for macro lifetime plot
        self.MICRO_ZOOM_STEPS = 50        # Step resolution for microscopic bifurcation zoom
        self.PLOT_DPI = 300

    def set_seed(self, seed):
        """Strict deterministic seed configuration across OS, CPU, and GPU runtimes."""
        self.SEED = seed
        os.environ["PYTHONHASHSEED"] = str(self.SEED)
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.SEED)

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass