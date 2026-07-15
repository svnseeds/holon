# config.py
import torch

# ==========================================
# 0. System Settings
# ==========================================
SEED = 42

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

TRAIN_FILE = "train_corpus.txt"
TEST_FILE = "test_corpus.txt"
EPOCHS = 20

# ==========================================
# 1. Network Topology (Fractal Architecture)
# ==========================================
TOP_NODES = 4           
NUM_LAYERS = 4          

LAYER_SIZES = []
_current = TOP_NODES
for _ in range(NUM_LAYERS):
    LAYER_SIZES.insert(0, _current)
    _current *= 2

# Number of nodes at the lowest layer (Sliding window size for raw bytes)
INPUT_BYTES_SIZE = LAYER_SIZES[0]  

# Node Dimensions (Uniform representation space)
NODE_INPUT_DIM = 256                
NODE_HIDDEN_DIM = 2048               
NODE_OUTPUT_DIM = 256               

# ==========================================
# 2. Learning Dynamics & Metabolism
# ==========================================
# Base Learning Rates (Awake Phase)
LR_ONLINE_BASE = 0.0050          
LR_ONLINE = LR_ONLINE_BASE      
LR_ONLINE_MIN = 0.000001         

# Adaptive Annealing (Ratchet mechanism for plateau detection)
LR_ONLINE_SHARPNESS = 30.0      
LR_ONLINE_SIGMOIDFACTOR = 0.90  
EMA_RANGE = 3
FLUCT_THRESHOLD = 0.02
MIN_ACC = 0.45
TARGET_CONSTANT = 0.98

# Sleep Consolidation Phase
LR_SLEEP_BASE = 0.0010                  
LR_SLEEP = LR_SLEEP_BASE               
LR_SLEEP_MIN = LR_ONLINE_MIN
LR_SLEEP_BELL_CENTER = 0.70             
LR_SLEEP_BELL_WIDTH = 0.025              
BASE_SLEEP_STEPS = 10           
BONUS_SLEEP_STEPS = 30          

# Synaptic Metabolism (Pruning & Neurogenesis)
PRUNING_THRESHOLD = 0.0005       
METABOLISM_SCALE = PRUNING_THRESHOLD * 3        
PRINT_METABOLISM = True         

# ==========================================
# 3. Neuromodulation (Biologically Plausible Rules)
# ==========================================
# Horizontal Recurrent Connection Strength
LR_HORIZ_SCALE = 0.50           
# Spatial decay for wave propagation (Left 1 to Left 4)
HORIZ_DECAY_RATE = 0.80         

# Precision-weighted Top-down Predictive Coding
TOPDOWN_BETA = 0.05             

# Leaky Integration (Memory retention over time)
LEAK_BASE = 0.05               
# Noradrenaline-driven Surprise Reset (High = Flush old context)
SURPRISE_RESET_ALPHA = 10.0    
LEAK_ADAPT_ALPHA = 2.0         

# Spike-based Saturation (Non-linear signal amplification)
SPINDLE_GAIN = 5.0            

# Spike Frequency Adaptation (Neuron fatigue to enforce Synfire Chains)
FATIGUE_RATE = 0.3        
FATIGUE_RECOVERY = 0.1    

# ==========================================
# 4. Stochasticity (Dropout & Noise)
# ==========================================
HORIZ_DROPOUT = 0.01            
HORIZ_DROPOUT_SLEEPFACTOR = 2.0 
VERTICAL_DROPOUT = 0.01            
VERTICAL_DROPOUT_SLEEPFACTOR = 2.0 

VAL_NOISE = 0.01                
VAL_NOISE_BASE = 0.01           
VAL_NOISE_SLEEPFACTOR = 2.0     
VAL_MULT_NOISE_MIN = 0.95        
VAL_MULT_NOISE_MAX = 1.05