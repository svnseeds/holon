# model.py 
import numpy as np
import numba
import torch
import config
import sys

# =====================================================================
# 1. Numba Core Engine (Ultra-fast Local Learning for CPU)
# =====================================================================
@numba.njit(parallel=True, cache=True)
def numba_node_forward_and_learn(
    X, feed_combined, W_in, W_feed, W_out, 
    lr, target_l, noise_period, val_noise, val_noise_base, 
    lr_horiz_scale, mult_noise_mask,
    F_layer, fatigue_rate, fatigue_recovery
):
    N, B_raw, In_Dim = X.shape
    H_Dim = W_in.shape[1]     
    In_Dim = W_in.shape[2]
    Feed_Dim = feed_combined.shape[2]
    Out_Dim = W_out.shape[1]
    
    B = min(B_raw, feed_combined.shape[1], target_l.shape[1])
    H = np.zeros((N, B, H_Dim), dtype=np.float32)
    
    # --- 1. Hidden Layer Forward (Spike Generation) ---
    for n in numba.prange(N):
        if n >= X.shape[0] or n >= feed_combined.shape[0] or n >= W_in.shape[0] or n >= W_feed.shape[0]:
            continue
        for b in range(B):
            if b >= X.shape[1] or b >= feed_combined.shape[1]:
                continue
            for h in range(H_Dim):
                if h >= W_in.shape[1] or h >= W_feed.shape[1]:
                    continue
                val = 0.0
                for i in range(In_Dim):
                    if i >= X.shape[2] or i >= W_in.shape[2]:
                        continue
                    val += X[n, b, i] * W_in[n, h, i]
                for f in range(Feed_Dim):
                    if f >= feed_combined.shape[2] or f >= W_feed.shape[2]:
                        break
                    val += feed_combined[n, b, f] * W_feed[n, h, f]
                
                if val < 0.0:
                    val = 0.0
                else:
                    if val_noise > 0.0:
                        neuron_id = n * 31 + h * 97
                        if neuron_id % noise_period == 0:
                            val += val_noise_base
                    
                    val *= mult_noise_mask[n, b, h]
                
                if val > 1.0:
                    val = 1.0
                
                # Spike Frequency Adaptation (Fatigue)
                val_discounted = val - F_layer[n, b, h]
                if val_discounted < 0.0:
                    val_discounted = 0.0

                H[n, b, h] = val
            
            # Winner-Take-All (WTA) Normalization
            max_val = 0.0
            for h in range(H_Dim):
                if H[n, b, h] > max_val:
                    max_val = H[n, b, h]
            if max_val > 0.0:
                threshold = max_val * 0.5
                for h in range(H_Dim):
                    v = H[n, b, h]
                    if v < threshold:
                        H[n, b, h] = 0.0
                    else:
                        H[n, b, h] = v / max_val
                        
            # Update Neuron Fatigue
            for h in range(H_Dim):
                F_layer[n, b, h] = F_layer[n, b, h] * (1.0 - fatigue_recovery) + H[n, b, h] * fatigue_rate

    # --- 2. Output Layer Forward ---
    Y = np.zeros((N, B, Out_Dim), dtype=np.float32)
    for n in numba.prange(N):
        if n >= H.shape[0] or n >= W_out.shape[0] or n >= Y.shape[0]:
            continue
        for b in range(B):
            if b >= H.shape[1] or b >= Y.shape[1]:
                continue
            for o in range(Out_Dim):
                if o >= W_out.shape[1] or o >= Y.shape[2]:
                    continue
                val = 0.0
                for h in range(H_Dim):
                    if h >= H.shape[2] or h >= W_out.shape[2]:
                        continue
                    val += H[n, b, h] * W_out[n, o, h]
                Y[n, b, o] = val
                            
    Y = np.clip(Y, 0.0, 1.0)

    # --- 3. Local Hebbian Learning (BP-Free) ---
    for n in numba.prange(N):
        if n >= target_l.shape[0] or n >= Y.shape[0] or n >= W_out.shape[0] or n >= W_in.shape[0] or n >= X.shape[0]:
            continue
        for b in range(B):
            if b >= target_l.shape[1] or b >= Y.shape[1] or b >= H.shape[1] or b >= X.shape[1]:
                continue
            # Output Layer Update (Delta Rule)
            for o in range(Out_Dim):
                if o >= target_l.shape[2] or o >= Y.shape[2] or o >= W_out.shape[1]:
                    continue
                error_o = target_l[n, b, o] - Y[n, b, o]
                for h in range(H_Dim):
                    if h >= H.shape[2] or h >= W_out.shape[2]:
                        continue
                    W_out[n, o, h] += lr * error_o * H[n, b, h]
                    
            # Input Layer Update (Oja's Rule + Novelty Gating)
            for h in range(H_Dim):
                if h >= H.shape[2] or h >= W_in.shape[1]:
                    continue
                h_val = H[n, b, h]
                if h_val > 0.0:
                    sum_err = 0.0
                    for o in range(Out_Dim):
                        sum_err += abs(target_l[n, b, o] - Y[n, b, o])
                    node_err = sum_err / Out_Dim
                    
                    lr_dynamic = lr * node_err
                    for i in range(In_Dim):
                        if i >= X.shape[2] or i >= W_in.shape[2]:
                            continue
                        W_in[n, h, i] += lr_dynamic * h_val * (X[n, b, i] - h_val * W_in[n, h, i])

                    # Horizontal Layer Update (Hebbian Gating to prevent decay)
                    lr_horiz = lr * lr_horiz_scale * node_err
                    for f in range(Feed_Dim):
                        if f >= feed_combined.shape[2] or f >= W_feed.shape[2]:
                            continue
                        S_val = feed_combined[n, b, f]
                        if S_val > 0.01:
                            W_feed[n, h, f] += lr_horiz * h_val * (S_val - h_val * W_feed[n, h, f])

    # --- 4. Synaptic Homeostasis (L2 Normalization) ---
    eps = 1e-8
    for n in numba.prange(N):
        if n >= W_in.shape[0] or n >= W_feed.shape[0]:
            continue
        for h in range(H_Dim):
            if h >= W_in.shape[1] or h >= W_feed.shape[1]:
                continue
            
            sum_sq_in = eps
            for i in range(In_Dim):
                sum_sq_in += W_in[n, h, i] * W_in[n, h, i]
            norm_in = np.sqrt(sum_sq_in)
            if norm_in > 1.0:
                for i in range(In_Dim):
                    W_in[n, h, i] /= norm_in
                    
            sum_sq_feed = eps
            for f in range(Feed_Dim):
                sum_sq_feed += W_feed[n, h, f] * W_feed[n, h, f]
            norm_feed = np.sqrt(sum_sq_feed)
            if norm_feed > 1.0:
                for f in range(Feed_Dim):
                    W_feed[n, h, f] /= norm_feed
                                
    return Y, W_in, W_feed, W_out

# =====================================================================
# 2. Living Network (Hierarchical Architecture)
# =====================================================================
class LivingNetwork:
    def __init__(self):
        self.device = config.DEVICE
        self.layer_sizes = config.LAYER_SIZES
        self.num_layers = config.NUM_LAYERS
        
        self.weights = {}
        self.states = {}

        rng = np.random.default_rng(config.SEED)
        
        for idx, size in enumerate(self.layer_sizes):
            self.weights[f"L{idx}_W_in"] = np.random.randn(size, config.NODE_HIDDEN_DIM, config.NODE_INPUT_DIM).astype(np.float32) * 0.05
            self.weights[f"L{idx}_W_feed"] = np.random.randn(size, config.NODE_HIDDEN_DIM, config.NODE_OUTPUT_DIM * 5).astype(np.float32) * 0.05
            self.weights[f"L{idx}_W_out"] = np.random.randn(size, config.NODE_OUTPUT_DIM, config.NODE_HIDDEN_DIM).astype(np.float32) * 0.05
            self.states[f"L{idx}_Y"] = np.zeros((size, 1, config.NODE_OUTPUT_DIM), dtype=np.float32)
            self.states[f"L{idx}_S"] = np.zeros((size, 1, config.NODE_OUTPUT_DIM), dtype=np.float32)
            self.states[f"L{idx}_F"] = np.zeros((size, 1, config.NODE_HIDDEN_DIM), dtype=np.float32)
            self.states[f"L{idx}_E"] = np.ones((size, 1, 1), dtype=np.float32)

        if self.device != "cpu":
            self._to_pytorch_gpu()

    def _to_pytorch_gpu(self):
        for k in self.weights.keys():
            self.weights[k] = torch.from_numpy(self.weights[k]).to(self.device)
        for k in self.states.keys():
            self.states[k] = torch.from_numpy(self.states[k]).to(self.device)

    def _get_horizontal_states(self, layer_idx, Y_current, is_train=True, is_sleeping=False):
        """Constructs temporal context from adjacent spatial nodes (Synfire wave)."""
        horiz_dropout = 0.0 if not is_train else (config.HORIZ_DROPOUT * config.HORIZ_DROPOUT_SLEEPFACTOR if is_sleeping else config.HORIZ_DROPOUT)
        
        decay_1 = config.HORIZ_DECAY_RATE
        decay_2 = config.HORIZ_DECAY_RATE * config.HORIZ_DECAY_RATE 
        
        if self.device == "cpu":
            Y_self = Y_current
            Y_L1 = np.roll(Y_current, shift=1, axis=0) * decay_1
            Y_L2 = np.roll(Y_current, shift=2, axis=0) * decay_2
            Y_L3 = np.roll(Y_current, shift=3, axis=0) * (decay_1 * decay_2)
            Y_L4 = np.roll(Y_current, shift=4, axis=0) * (decay_2 * decay_2)
            combined = np.concatenate([Y_self, Y_L1, Y_L2, Y_L3, Y_L4], axis=2).copy()

            if is_train and horiz_dropout > 0.0:
                N, B, C = combined.shape
                mask = np.ones_like(combined)
                period = int(1.0 / horiz_dropout)
                for n in range(N):
                    for c in range(C):
                        if (n * 17 + c * 79 + layer_idx * 13) % period == 0:
                            mask[n, :, c] = 0.0
                combined *= mask
            
            return combined
        else:
            Y_self = Y_current
            Y_L1 = torch.roll(Y_current, shifts=1, dims=0) * decay_1
            Y_L2 = torch.roll(Y_current, shifts=2, dims=0) * decay_2
            Y_L3 = torch.roll(Y_current, shifts=3, dims=0) * (decay_1 * decay_2)
            Y_L4 = torch.roll(Y_current, shifts=4, dims=0) * (decay_2 * decay_2)
            combined = torch.cat([Y_self, Y_L1, Y_L2, Y_L3, Y_L4], dim=2) 

            if is_train and horiz_dropout > 0.0:
                N, B, C = combined.shape
                period = int(1.0 / horiz_dropout)
                n_idx = torch.arange(N, device=self.device).view(N, 1, 1)
                c_idx = torch.arange(C, device=self.device).view(1, 1, C)
                drop_mask = ((n_idx * 17 + c_idx * 79 + layer_idx * 13) % period == 0)
                mask = torch.ones_like(combined)
                mask[drop_mask] = 0.0
                combined = combined * mask
                
            return combined
    
    def step(self, raw_bytes_one_hot, target_one_hot=None, is_sleeping=False, is_train=True):
        # Freeze learning (weights updates) during inference
        lr = 0.0 if not is_train else (config.LR_SLEEP if is_sleeping else config.LR_ONLINE)
        current_input = raw_bytes_one_hot 

        # 1. Build Bottom-Up Target Pyramid
        targets_pyramid = {}
        t_curr = target_one_hot if target_one_hot is not None else raw_bytes_one_hot
        targets_pyramid[0] = t_curr
        for l in range(1, self.num_layers):
            min_len = (t_curr.shape[0] // 2) * 2
            val = (t_curr[0:min_len:2, :, :] + t_curr[1:min_len:2, :, :]) * 0.5
            t_curr = val.copy() if self.device == "cpu" else val.clone()
            targets_pyramid[l] = t_curr

        vert_dropout = 0.0 if not is_train else (config.VERTICAL_DROPOUT * config.VERTICAL_DROPOUT_SLEEPFACTOR if is_sleeping else config.VERTICAL_DROPOUT)

        # 2. Top-down Precision-Weighted Predictive Coding (Hallucination injection)
        for l in reversed(range(0, self.num_layers - 1)):
            Y_parent = self.states[f"L{l+1}_Y"] 
            t_parent = targets_pyramid[l+1]
            
            if self.device == "cpu":
                pred_idx = np.argmax(Y_parent, axis=2)
                true_idx = np.argmax(t_parent, axis=2)
                Pi_parent = np.mean(pred_idx == true_idx, axis=1, keepdims=True).reshape(-1, 1, 1)
                Y_parent_expanded = np.repeat(Y_parent, repeats=2, axis=0)
                Pi_parent_expanded = np.repeat(Pi_parent, repeats=2, axis=0)
            else:
                pred_idx = torch.argmax(Y_parent, dim=2)
                true_idx = torch.argmax(t_parent, dim=2)
                Pi_parent = torch.mean((pred_idx == true_idx).float(), dim=1, keepdim=True).unsqueeze(2)
                Y_parent_expanded = torch.repeat_interleave(Y_parent, repeats=2, dim=0)
                Pi_parent_expanded = torch.repeat_interleave(Pi_parent, repeats=2, dim=0)
                
            beta_dynamic = config.TOPDOWN_BETA * Pi_parent_expanded
            targets_pyramid[l] = targets_pyramid[l] * (1.0 - beta_dynamic) + Y_parent_expanded * beta_dynamic

        # 3. Bottom-Up Local Learning Phase
        for l in range(self.num_layers):
            S_prev = self.states[f"L{l}_S"]
            
            # Spike-based Soft Saturation (Amplifier)
            if self.device == "cpu":
                S_quantized = np.tanh(S_prev * config.SPINDLE_GAIN).astype(np.float32)
            else:
                S_quantized = torch.tanh(S_prev * config.SPINDLE_GAIN)
            
            feed_combined = self._get_horizontal_states(l, S_quantized, is_train=is_train, is_sleeping=is_sleeping)
            t_layer = targets_pyramid[l]
                
            if self.device == "cpu":
                current_noise_prob = config.VAL_NOISE * config.VAL_NOISE_SLEEPFACTOR if is_sleeping else config.VAL_NOISE
                noise_period = int(1.0 / current_noise_prob) if 0.0 < current_noise_prob < 1.0 else (1 if current_noise_prob >= 1.0 else 999999)

                N_size, B_size = current_input.shape[0], min(current_input.shape[1], feed_combined.shape[1], t_layer.shape[1])
                
                if is_sleeping:
                    mult_noise_mask = np.random.uniform(config.VAL_MULT_NOISE_MIN, config.VAL_MULT_NOISE_MAX, size=(N_size, B_size, config.NODE_HIDDEN_DIM)).astype(np.float32)
                else:
                    mult_noise_mask = np.ones((N_size, B_size, config.NODE_HIDDEN_DIM), dtype=np.float32)

                Y_new, W_in_new, W_feed_new, W_out_new = numba_node_forward_and_learn(
                    current_input, feed_combined, 
                    self.weights[f"L{l}_W_in"], self.weights[f"L{l}_W_feed"], self.weights[f"L{l}_W_out"], 
                    lr, t_layer, noise_period, config.VAL_NOISE, config.VAL_NOISE_BASE,
                    config.LR_HORIZ_SCALE, mult_noise_mask,
                    self.states[f"L{l}_F"], config.FATIGUE_RATE, config.FATIGUE_RECOVERY
                )
                self.weights[f"L{l}_W_in"], self.weights[f"L{l}_W_feed"], self.weights[f"L{l}_W_out"] = W_in_new, W_feed_new, W_out_new
                self.states[f"L{l}_Y"] = Y_new

                # Calculate L1 Surprise (Error)
                err_instant = np.mean(np.abs(t_layer - Y_new), axis=(1, 2), keepdims=True)
                self.states[f"L{l}_E"] = self.states[f"L{l}_E"] * 0.9 + err_instant * 0.1 

                # Hierarchical Surprise Reset (Gradient across layers)
                layer_ratio = l / max(1, self.num_layers - 1)
                leak_base = config.LEAK_BASE * (2 ** (self.num_layers - 1 - l))
                E_layer = self.states[f"L{l}_E"]
                leak_base_dynamic = leak_base * np.exp(-config.LEAK_ADAPT_ALPHA * E_layer)
                
                flush_power = 1.0 - np.exp(-config.SURPRISE_RESET_ALPHA * err_instant)
                leak_dynamic = leak_base_dynamic + (1.0 - leak_base_dynamic) * flush_power * layer_ratio

                self.states[f"L{l}_S"] = (1.0 - leak_dynamic) * self.states[f"L{l}_S"] + leak_dynamic * Y_new

                if l < self.num_layers - 1:
                    min_len = (Y_new.shape[0] // 2) * 2
                    val = ((Y_new[0:min_len:2, :, :] + Y_new[1:min_len:2, :, :]) * 0.5)

                    if is_train and vert_dropout > 0.0:
                        N_v, B_v, C_v = val.shape
                        period = int(1.0 / vert_dropout)
                        mask = np.ones_like(val)
                        for n in range(N_v):
                            for c in range(C_v):
                                if (n * 23 + c * 61 + l * 19) % period == 0:
                                    mask[n, :, c] = 0.0
                        val *= mask
                    current_input = val.copy()
            else:
                # --- GPU (PyTorch) Path ---
                with torch.no_grad():
                    h_in_x = torch.bmm(self.weights[f"L{l}_W_in"], current_input.transpose(1, 2))
                    h_in_feed = torch.bmm(self.weights[f"L{l}_W_feed"], feed_combined.transpose(1, 2))
                    h_layer_format = (h_in_x + h_in_feed).transpose(1, 2)
                    h = torch.clamp(h_layer_format, min=0.0, max=1.0)

                    if config.VAL_NOISE > 0.0:
                        N_size, B_size, H_size = h.shape
                        current_noise_prob = config.VAL_NOISE * config.VAL_NOISE_SLEEPFACTOR if is_sleeping else config.VAL_NOISE
                        period = max(1, int(1.0 / current_noise_prob)) if current_noise_prob < 1.0 else 1
                        n_idx = torch.arange(N_size, device=self.device).unsqueeze(1)
                        h_idx = torch.arange(H_size, device=self.device).unsqueeze(0)
                        neuron_id = n_idx * 31 + h_idx * 97
                        noise_mask = ((neuron_id % period) == 0).float().unsqueeze(1)
                        h += (noise_mask * config.VAL_NOISE_BASE) * (h > 0.0).float()

                        if is_sleeping:
                            mult_mask = torch.zeros_like(h).uniform_(config.VAL_MULT_NOISE_MIN, config.VAL_MULT_NOISE_MAX)
                            h = torch.where(h > 0.0, h * mult_mask, h)

                    # Fatigue (SFA)
                    F_layer = self.states[f"L{l}_F"]
                    h = h - F_layer
                    h = torch.clamp(h, min=0.0)

                    # WTA Normalization
                    h_max, _ = torch.max(h, dim=2, keepdim=True)
                    h = torch.where(h < h_max * 0.5, torch.zeros_like(h), h)
                    h = torch.where(h_max > 0.0, h / (h_max + 1e-8), h)

                    self.states[f"L{l}_F"] = F_layer * (1.0 - config.FATIGUE_RECOVERY) + h * config.FATIGUE_RATE

                    Y_new = torch.bmm(self.weights[f"L{l}_W_out"], h.transpose(1, 2)).transpose(1, 2)
                    Y_new = torch.clamp(Y_new, min=0.0, max=1.0)
                
                    error = t_layer - Y_new
                    self.weights[f"L{l}_W_out"] += lr * torch.bmm(error.transpose(1, 2), h)

                    h_un = h.unsqueeze(3)
                    h_sq_un = (h * h).unsqueeze(3)
                    
                    err_instant = torch.mean(torch.abs(t_layer - Y_new), dim=(1, 2), keepdim=True)

                    # Novelty Gated Oja's Rule for W_in
                    X_un = current_input.unsqueeze(2)
                    W_in_un = self.weights[f"L{l}_W_in"].unsqueeze(1)
                    oja_delta_all = h_un * X_un - h_sq_un * W_in_un
                    oja_delta = torch.sum(oja_delta_all, dim=1) 
                    self.weights[f"L{l}_W_in"] += lr * (err_instant * oja_delta)
                    
                    # Hebbian Gated Oja's Rule for W_feed
                    feed_un = feed_combined.unsqueeze(2)
                    W_feed_un = self.weights[f"L{l}_W_feed"].unsqueeze(1)
                    feed_gate = (feed_un > 0.01).float()
                    feed_delta_all = (h_un * feed_un - h_sq_un * W_feed_un) * feed_gate
                    feed_delta = torch.sum(feed_delta_all, dim=1) 
                    self.weights[f"L{l}_W_feed"] += (lr * config.LR_HORIZ_SCALE) * (err_instant * feed_delta)

                    # L2 Scaling
                    eps = 1e-8
                    norm_in = torch.norm(self.weights[f"L{l}_W_in"], p=2, dim=2, keepdim=True) + eps
                    scale_in = torch.clamp(norm_in, min=1.0)
                    self.weights[f"L{l}_W_in"] /= scale_in

                    norm_feed = torch.norm(self.weights[f"L{l}_W_feed"], p=2, dim=2, keepdim=True) + eps
                    scale_feed = torch.clamp(norm_feed, min=1.0)
                    self.weights[f"L{l}_W_feed"] /= scale_feed
                    
                    self.states[f"L{l}_Y"] = Y_new

                    err_instant = torch.mean(torch.abs(t_layer - Y_new), dim=(1, 2), keepdim=True)
                    self.states[f"L{l}_E"] = self.states[f"L{l}_E"] * 0.9 + err_instant * 0.1 

                    layer_ratio = l / max(1, self.num_layers - 1)
                    leak_base = config.LEAK_BASE * (2 ** (self.num_layers - 1 - l))
                    E_layer = self.states[f"L{l}_E"]
                    leak_base_dynamic = leak_base * torch.exp(-config.LEAK_ADAPT_ALPHA * E_layer)
                    
                    flush_power = 1.0 - torch.exp(-config.SURPRISE_RESET_ALPHA * err_instant)
                    leak_dynamic = leak_base_dynamic + (1.0 - leak_base_dynamic) * flush_power * layer_ratio

                    self.states[f"L{l}_S"] = (1.0 - leak_dynamic) * self.states[f"L{l}_S"] + leak_dynamic * Y_new

                    if l < self.num_layers - 1:
                        min_len = (Y_new.shape[0] // 2) * 2
                        val = (Y_new[0:min_len:2] + Y_new[1:min_len:2]) * 0.5
                        
                        if is_train and vert_dropout > 0.0:
                            N_v, B_v, C_v = val.shape
                            period = int(1.0 / vert_dropout)
                            n_idx = torch.arange(N_v, device=self.device).view(N_v, 1, 1)
                            c_idx = torch.arange(C_v, device=self.device).view(1, 1, C_v)
                            drop_mask = ((n_idx * 23 + c_idx * 61 + l * 19) % period == 0)
                            mask = torch.ones_like(val)
                            mask[drop_mask] = 0.0
                            val = val * mask
                            
                        current_input = val
                        
        return self.states[f"L{self.num_layers-1}_Y"]

    def metabolize(self):
        """Synaptic Metabolism: Prunes weak connections and generates new random synapses."""
        self.metabolism_count_in = 0
        self.metabolism_count_feed = 0
        
        for l in range(self.num_layers):
            w_in_key, w_feed_key = f"L{l}_W_in", f"L{l}_W_feed"
    
            if self.device == "cpu":
                w_in = self.weights[w_in_key]
                prune_mask_in = np.abs(w_in) < config.PRUNING_THRESHOLD
                if np.any(prune_mask_in):
                    self.metabolism_count_in += np.sum(prune_mask_in)
                    w_in[prune_mask_in] = 0.0
                    new_buds = np.random.randn(*w_in.shape).astype(np.float32) * config.METABOLISM_SCALE
                    w_in = np.where(prune_mask_in, new_buds, w_in)
                    self.weights[w_in_key] = w_in
        
                w_feed = self.weights[w_feed_key]
                prune_mask_feed = np.abs(w_feed) < config.PRUNING_THRESHOLD
                if np.any(prune_mask_feed):
                    self.metabolism_count_feed += np.sum(prune_mask_feed)
                    w_feed[prune_mask_feed] = 0.0
                    new_buds_feed = np.random.randn(*w_feed.shape).astype(np.float32) * config.METABOLISM_SCALE
                    w_feed = np.where(prune_mask_feed, new_buds_feed, w_feed)
                    self.weights[w_feed_key] = w_feed
            else:
                with torch.no_grad():
                    w_in = self.weights[w_in_key]
                    prune_mask_in = torch.abs(w_in) < config.PRUNING_THRESHOLD
                    if torch.any(prune_mask_in):
                        self.metabolism_count_in += int(torch.sum(prune_mask_in).item())
                        w_in[prune_mask_in] = 0.0
                        new_buds = torch.randn_like(w_in) * config.METABOLISM_SCALE
                        self.weights[w_in_key] = torch.where(prune_mask_in, new_buds, w_in)
            
                    w_feed = self.weights[w_feed_key]
                    prune_mask_feed = torch.abs(w_feed) < config.PRUNING_THRESHOLD
                    if torch.any(prune_mask_feed):
                        self.metabolism_count_feed += int(torch.sum(prune_mask_feed).item())
                        w_feed[prune_mask_feed] = 0.0
                        new_buds_feed = torch.randn_like(w_feed) * config.METABOLISM_SCALE
                        self.weights[w_feed_key] = torch.where(prune_mask_feed, new_buds_feed, w_feed)

    def reset_states(self):
        """Clears working memory (S), fatigue (F), and error (E) across phase transitions."""
        for l in range(self.num_layers):
            if self.device == "cpu":
                self.states[f"L{l}_Y"].fill(0.0)
                self.states[f"L{l}_S"].fill(0.0)
                self.states[f"L{l}_F"].fill(0.0)
                self.states[f"L{l}_E"].fill(1.0) 
            else:
                self.states[f"L{l}_Y"].zero_()
                self.states[f"L{l}_S"].zero_()
                self.states[f"L{l}_F"].zero_()
                self.states[f"L{l}_E"].fill_(1.0)

    def calculate_metrics(self, top_output, target_one_hot):
        """Evaluates raw L0 predictions to prevent spatial cheating."""
        current_output = self.states["L0_Y"]  
        current_target = target_one_hot       
        
        if self.device == "cpu":
            loss = np.mean((current_output - current_target) ** 2)
            pred_idx = np.argmax(current_output[-1:, :, :], axis=2)
            true_idx = np.argmax(current_target[-1:, :, :], axis=2)
            accuracy = np.mean(pred_idx == true_idx)
            
            total_elements = sum(self.states[f"L{l}_Y"].size for l in range(self.num_layers))
            active_elements = sum(np.sum(self.states[f"L{l}_Y"] > 0.1) for l in range(self.num_layers))
            sparsity = active_elements / total_elements
        else:
            with torch.no_grad():
                loss = torch.mean((current_output - current_target) ** 2).item()
                pred_idx = torch.argmax(current_output[-1:, :, :], dim=2)
                true_idx = torch.argmax(current_target[-1:, :, :], dim=2)
                accuracy = torch.mean((pred_idx == true_idx).float()).item()
                
                total_elements = sum(self.states[f"L{l}_Y"].numel() for l in range(self.num_layers))
                active_elements = sum(torch.sum(self.states[f"L{l}_Y"] > 0.1).item() for l in range(self.num_layers))
                sparsity = active_elements / total_elements
                
        return loss, accuracy, sparsity

    def save_checkpoint(self, filepath="holon_brain.npz"):
        save_data = {}
        for k, v in self.weights.items():
            save_data[k] = v.cpu().numpy() if self.device != "cpu" else v
        for k, v in self.states.items():
            save_data[k] = v.cpu().numpy() if self.device != "cpu" else v
        np.savez(filepath, **save_data)
        print(f"[*] Brain state saved to '{filepath}'.")

    def load_checkpoint(self, filepath="holon_brain.npz"):
        import os
        if not os.path.exists(filepath):
            print(f"[!] Warning: '{filepath}' not found. Initializing a new brain.")
            return False
            
        with np.load(filepath) as data:
            for k in self.weights.keys():
                if k in data:
                    self.weights[k] = torch.from_numpy(data[k]).to(self.device) if self.device != "cpu" else data[k].copy()
            for k in self.states.keys():
                if k in data:
                    self.states[k] = torch.from_numpy(data[k]).to(self.device) if self.device != "cpu" else data[k].copy()
                    
        print(f"[*] Brain state fully restored from '{filepath}'.")
        return True