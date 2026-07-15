# main.py
import time
from datetime import datetime
import sys
import os
import random
import numpy as np
import torch
import config
from dataloader import load_corpus_file, bytes_to_one_hot
from model import LivingNetwork

def set_seed(seed=config.SEED):
    """Fixes random seeds across CPU and GPU for complete determinism."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_phase(network, chunks, is_train=True, record_vis=False):
    """Executes a single phase (Train or Test). Ensures clean internal states before start."""
    # Reset internal working memory (S) and fatigue (F) to prevent leakage across phases
    network.reset_states()
    
    total_loss, total_acc, total_sparsity = 0.0, 0.0, 0.0
    steps = len(chunks)
    if steps == 0:
        return 0, 0, 0

    vis_history_S = {f"L{l}": [] for l in range(network.num_layers)}
    vis_history_inputs = []

    for chunk_in, chunk_target in chunks:
        X_t = bytes_to_one_hot(chunk_in)
        Target_t = bytes_to_one_hot(chunk_target)
        
        top_output = network.step(X_t, target_one_hot=Target_t, is_sleeping=False, is_train=is_train)

        if record_vis:
            char_ascii = int(chunk_in[0])
            vis_history_inputs.append(char_ascii)
            for l in range(network.num_layers):
                if network.device == "cpu":
                    s_state = network.states[f"L{l}_S"][0, 0, :].copy() 
                else:
                    s_state = network.states[f"L{l}_S"][0, 0, :].cpu().numpy().copy() 
                vis_history_S[f"L{l}"].append(s_state)

        loss, acc, sparsity = network.calculate_metrics(top_output, Target_t)
        total_loss += loss
        total_acc += acc
        total_sparsity += sparsity

    if record_vis and len(vis_history_inputs) > 0:
        save_dict = {f"L{l}_S": np.array(vis_history_S[f"L{l}"]) for l in range(network.num_layers)}
        save_dict["inputs"] = np.array(vis_history_inputs)
        save_dict["num_layers"] = np.array(network.num_layers)
        np.savez("vis_data.npz", **save_dict)
        print(f"\n[*] Saved brain states for all {network.num_layers} layers to 'vis_data.npz'.")
        
    return total_loss / steps, total_acc / steps, total_sparsity / steps

def main():
    # Continual Learning Mode (Extrapolation test)
    continue_learning = len(sys.argv) > 1 and sys.argv[1] == "-c"
    
    set_seed(config.SEED)

    # Variables for Autonomous Ratchet Annealing
    last_acc_history = []
    adapted_sigmoid_target = None  
    max_acc_velocity = 0.0         
    sleep_bell_center = config.LR_SLEEP_BELL_CENTER 
    previous_acc = None            

    start_dt = datetime.now()
    print(f"\nSTART TIME: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DEVICE: {config.DEVICE} TOPOLOGY: {config.LAYER_SIZES}\n")
    global_start_time = time.time()

    train_chunks = load_corpus_file(config.TRAIN_FILE)
    test_chunks = load_corpus_file(config.TEST_FILE)
    
    print("Chnks SEED  Noise HorDO LR_OLBs LR_OLShp LR_OLSig LR_SLBs LR_SLBct LR_SLBwd LR_HRScl VMNois PRThrsld METScl")
    print(f"{len(train_chunks):5} {config.SEED:4} {config.VAL_NOISE:5.4} {config.HORIZ_DROPOUT:5.4} {config.LR_ONLINE_BASE:7.4} {config.LR_ONLINE_SHARPNESS:8.4} {config.LR_ONLINE_SIGMOIDFACTOR:8.4} {config.LR_SLEEP_BASE:7.4} {config.LR_SLEEP_BELL_CENTER:8.4} {config.LR_SLEEP_BELL_WIDTH:8.4} {config.LR_HORIZ_SCALE:8.4} {(config.VAL_MULT_NOISE_MAX-1)*100:5.4}% {config.PRUNING_THRESHOLD:8.4} {config.METABOLISM_SCALE:6.4}\n")
    
    network = LivingNetwork()
    if continue_learning:
        network.load_checkpoint("holon_brain.npz")

    last_acc = None
    train_acc_max, train_acc_max_ep = 0.0, 0
    test_acc_max, test_acc_max_ep = 0.0, 0
    ep10_train_acc, ep10_test_acc = 0.0, 0.0
    ep20_train_acc, ep20_test_acc = 0.0, 0.0
    
    if config.PRINT_METABOLISM:
        print("EPOCH TrainL  ACC%    SPRSTY TestL   ACC%    SPRSTY LR_ONLINE LR_SLEEP SLStp MbW_in   MbW_feed TIME")
    else:
        print("EPOCH TrainL  ACC%    SPRSTY TestL   ACC%    SPRSTY LR_ONLINE LR_SLEEP SLStp TIME")
    
    for epoch in range(config.EPOCHS):
        # --- Autonomous Ratchet Annealing ---
        if last_acc is not None:
            # 1. Lock sleep bell center at the point of maximum growth velocity
            if previous_acc is not None:
                velocity = last_acc - previous_acc
                if velocity > max_acc_velocity:
                    max_acc_velocity = velocity
                    sleep_bell_center = last_acc
                    print(f" [*] Max growth velocity detected: Sleep bell center shifted to {sleep_bell_center*100:.2f}%.")
            previous_acc = last_acc

            # 2. Detect plateau and apply Ratchet mechanism (irreversible brake wall)
            last_acc_history.append(last_acc)
            if len(last_acc_history) > config.EMA_RANGE:
                last_acc_history.pop(0)

            sigmoid_target = config.LR_ONLINE_SIGMOIDFACTOR
            if len(last_acc_history) == config.EMA_RANGE:
                fluctuation = max(last_acc_history) - min(last_acc_history)
                if fluctuation < config.FLUCT_THRESHOLD and max(last_acc_history) > config.MIN_ACC:
                    new_target = max(last_acc_history) * config.TARGET_CONSTANT
                    print(f" [*] Autonomous plateau detection: Brake wall shifted to {new_target*100:.2f}%.")
                    
                    if adapted_sigmoid_target is None:
                        adapted_sigmoid_target = new_target
                    else:
                        adapted_sigmoid_target = max(adapted_sigmoid_target, new_target)

            if adapted_sigmoid_target is not None:
                sigmoid_target = adapted_sigmoid_target

            # Calculate dynamic learning rates
            sigmoid_factor = 1.0 / (1.0 + np.exp(config.LR_ONLINE_SHARPNESS * (last_acc - sigmoid_target)))
            bell_power = -((last_acc - sleep_bell_center) ** 2) / (2 * (config.LR_SLEEP_BELL_WIDTH ** 2))
            bell_factor_sleep = np.exp(bell_power)

            config.LR_ONLINE = max(config.LR_ONLINE_BASE * sigmoid_factor, config.LR_ONLINE_MIN)
            config.LR_SLEEP = max(config.LR_SLEEP_BASE * bell_factor_sleep, config.LR_SLEEP_MIN)
        else:
            config.LR_ONLINE = config.LR_ONLINE_BASE
            config.LR_SLEEP = config.LR_SLEEP_MIN

        # --- Phase 1: Awake (Online Learning) ---
        train_loss, train_acc, train_sparsity = run_phase(network, train_chunks, is_train=True)
            
        # --- Phase 2: Sleep (Memory Consolidation) ---
        base_sleep_steps = config.BASE_SLEEP_STEPS
        bonus_steps = int(last_acc * config.BONUS_SLEEP_STEPS) if last_acc is not None else 0
        dynamic_sleep_steps = min(base_sleep_steps + bonus_steps, len(train_chunks))

        mb_count_in, mb_count_feed = 0, 0
        if len(train_chunks) > 0 and dynamic_sleep_steps > 0:
            sleep_samples = random.choices(train_chunks, k=dynamic_sleep_steps)
            
            for chunk_in, chunk_target in sleep_samples:
                sleep_in = bytes_to_one_hot(chunk_in)
                sleep_target = bytes_to_one_hot(chunk_target)
                
                if config.DEVICE != "cpu":
                    sleep_in = torch.from_numpy(sleep_in).to(config.DEVICE) if not isinstance(sleep_in, torch.Tensor) else sleep_in.to(config.DEVICE)
                    sleep_target = torch.from_numpy(sleep_target).to(config.DEVICE) if not isinstance(sleep_target, torch.Tensor) else sleep_target.to(config.DEVICE)
                    
                _ = network.step(sleep_in, target_one_hot=sleep_target, is_sleeping=True)

            # Trigger synaptic metabolism (pruning & neurogenesis) after sleep
            network.metabolize()  
            if hasattr(network, 'metabolism_count_in'):
                mb_count_in = network.metabolism_count_in
                mb_count_feed = network.metabolism_count_feed

        # --- Phase 3: Test (Evaluation) ---
        record_vis = (epoch == config.EPOCHS - 1)
        test_loss, test_acc, test_sparsity = run_phase(network, test_chunks, is_train=False, record_vis=record_vis)

        last_acc = (train_acc + test_acc) / 2

        # --- Dashboard ---
        elapsed_sec = time.time() - global_start_time
        h, m, s = int(elapsed_sec // 3600), int((elapsed_sec % 3600) // 60), int(elapsed_sec % 60)
        time_str = f"{h}:{m:02d}:{s:02d}"

        if config.PRINT_METABOLISM:
            print(f"{epoch+1:2}/{config.EPOCHS:2} {train_loss:.4f} {train_acc*100:6.2f}% {train_sparsity*100:6.2f}% {test_loss:6.4f} {test_acc*100:6.2f}% {test_sparsity*100:6.2f}% {config.LR_ONLINE:9.6f} {config.LR_SLEEP:8.6f} {dynamic_sleep_steps:5} {mb_count_in:8} {mb_count_feed:8} {time_str}")
        else: 
            print(f"{epoch+1:2}/{config.EPOCHS:2} {train_loss:.4f} {train_acc*100:6.2f}% {train_sparsity*100:6.2f}% {test_loss:6.4f} {test_acc*100:6.2f}% {test_sparsity*100:6.2f}% {config.LR_ONLINE:9.6f} {config.LR_SLEEP:8.6f} {dynamic_sleep_steps:5} {time_str}")

        if train_acc > train_acc_max: train_acc_max, train_acc_max_ep = train_acc, epoch+1
        if test_acc > test_acc_max: test_acc_max, test_acc_max_ep = test_acc, epoch+1
        if epoch+1 == 10: ep10_train_acc, ep10_test_acc = train_acc, test_acc
        if epoch+1 == 20: ep20_train_acc, ep20_test_acc = train_acc, test_acc

    # SUMMARY
    print(f"\nTRAIN MAX {train_acc_max*100:5.2f}% @ {train_acc_max_ep} / TEST MAX {test_acc_max*100:5.2f}% @ {test_acc_max_ep}")
    print(f"Ep10: {ep10_train_acc*100:5.2f}% {ep10_test_acc*100:5.2f}% / Ep20: {ep20_train_acc*100:5.2f}% {ep20_test_acc*100:5.2f}%")
    
    end_dt = datetime.now()
    print(f"\nEND TIME: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DURATION: {str(end_dt - start_dt)}")

    print("\n=========================================================")
    network.save_checkpoint("holon_brain.npz")
    print("=========================================================")

if __name__ == "__main__":
    main()