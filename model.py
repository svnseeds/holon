# model.py
# Holon v14.0 Core Neural Architecture
# 1-N Multi-Column Cortical Network under Pure Local Dynamics
import math
import os
import torch

class SensorInterface:
    """
    Universal Deterministic Haar Dejima Gateway (Shared ROM Interface)
    Projects raw 256-byte inputs into an isometric, energy-conserving latent wave.
    """
    _shared_w_sensor = None

    @torch.no_grad()
    def __init__(self, config):
        self.dim = config.LATENT_DIM
        self.device = config.DEVICE
        self.cfg = config

        if SensorInterface._shared_w_sensor is None:
            SensorInterface._shared_w_sensor = self._build_universal_haar_matrix(config)

        self.w_sensor = SensorInterface._shared_w_sensor.to(dtype=config.DTYPE, device=self.device)

    @torch.no_grad()
    def _build_universal_haar_matrix(self, config):
        """Synthesize deterministic orthogonal projection matrix via QR decomposition."""
        total_elements = self.dim * self.dim
        state = config.SEED_SENSOR_PROTOCOL & 0xFFFFFFFFFFFFFFFF
        gamma = 0x9E3779B97F4A7C15
        u_vals = []
        for _ in range(total_elements):
            state = (state + gamma) & 0xFFFFFFFFFFFFFFFF
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
            u = (z + 1.0) / 18446744073709551617.0
            u_vals.append(u)

        u_tensor = torch.tensor(u_vals, dtype=torch.float64)
        u1 = u_tensor[0::2]
        u2 = u_tensor[1::2]

        r_part = torch.sqrt(-2.0 * torch.log(u1))
        theta_part = 2.0 * math.pi * u2
        g1 = r_part * torch.cos(theta_part)
        g2 = r_part * torch.sin(theta_part)

        gaussian_seq = torch.empty(total_elements, dtype=torch.float64)
        gaussian_seq[0::2] = g1
        gaussian_seq[1::2] = g2
        gaussian_mat = gaussian_seq.view(self.dim, self.dim)

        q, r = torch.linalg.qr(gaussian_mat)
        diag_sign = torch.diag(r).sign()
        diag_sign[diag_sign == 0] = 1.0
        return q * diag_sign.unsqueeze(0)

    @torch.no_grad()
    def encode(self, x):
        """Normalize raw input and project onto universal latent space (RMS ≡ 1.000)."""
        in_rms = torch.sqrt(torch.mean(x ** 2) + self.cfg.EPSILON)
        return torch.mv(self.w_sensor, x / in_rms)

    @torch.no_grad()
    def decode(self, pred_latent):
        """Decode latent wave back into physical byte probability space."""
        scaled_pred = torch.mv(self.w_sensor.t(), pred_latent) * self.cfg.INV_SQRT_D
        return torch.clamp(scaled_pred, 0.0, 1.0)


class ParallelLFSRClockArray:
    """
    Skeletal Pacemaker Clock Generator ("Donkama" Tick)
    Prevents residual vanishing by driving internal states with an invariant temporal carrier.
    """
    @torch.no_grad()
    def __init__(self, config):
        self.num_cols = config.NUM_SUPER_COLUMNS
        self.dim = config.LATENT_DIM
        self.device = config.DEVICE
        self.alpha = config.CLOCK_SMOOTH_ALPHA
        self.norm_const = config.CLOCK_NORM_CONST
        self.taps = config.CLOCK_LFSR_TAPS

        init_list = []
        for k in range(self.num_cols):
            clock_seed = config.SEED + config.CLOCK_SEED_BASE + k * config.CLOCK_SEED_COL_STEP
            gen = torch.Generator(device="cpu").manual_seed(clock_seed)
            raw_bits = torch.randint(0, 2, (self.dim,), generator=gen, dtype=config.DTYPE)
            init_list.append((raw_bits * 2.0) - 1.0)

        self.init_bits = torch.stack(init_list, dim=0).to(self.device)
        self.bits = self.init_bits.clone()
        self.c_smooth = torch.zeros(self.num_cols, self.dim, dtype=config.DTYPE, device=self.device)

    @torch.no_grad()
    def step(self):
        """Advance LFSR register and return exponentially smoothed clock wave."""
        t0, t1, t2, t3 = self.taps
        fb = self.bits[:, t0] * self.bits[:, t1] * self.bits[:, t2] * self.bits[:, t3]
        self.bits = torch.roll(self.bits, shifts=1, dims=1)
        self.bits[:, 0] = fb
        self.c_smooth.mul_(1.0 - self.alpha).add_(self.bits, alpha=self.alpha)
        return self.c_smooth * self.norm_const

    @torch.no_grad()
    def reset(self):
        self.bits.copy_(self.init_bits)
        self.c_smooth.zero_()


class ParallelReservoirBank:
    """
    Tensor-Vectorized Reservoir Bank
    Executes multiple column layers simultaneously across hardware batch dimensions.
    """
    @torch.no_grad()
    def __init__(self, config, num_nodes, base_leaks, clock_budget, seed_res_offset, seed_wout_offset):
        self.cfg = config
        self.num_cols = config.NUM_SUPER_COLUMNS
        self.num_nodes = num_nodes
        self.dim = config.LATENT_DIM
        self.device = config.DEVICE
        self.clock_budget = clock_budget

        leaks_tensor = torch.tensor(base_leaks, dtype=config.DTYPE, device=self.device)
        self.leak = leaks_tensor.unsqueeze(0).unsqueeze(2).expand(self.num_cols, num_nodes, 1)
        self.eta = (config.ETA_DELTA * torch.sqrt(self.leak)).unsqueeze(3)

        self.h = torch.zeros(self.num_cols, num_nodes, self.dim, dtype=config.DTYPE, device=self.device)
        self.h_prev = torch.zeros(self.num_cols, num_nodes, self.dim, dtype=config.DTYPE, device=self.device)
        self.gain = torch.ones(self.num_cols, num_nodes, 1, dtype=config.DTYPE, device=self.device)

        # Recurrent reservoir weights normalized to target spectral radius
        w_res_list = []
        for k in range(self.num_cols):
            gen_r = torch.Generator(device="cpu").manual_seed(config.SEED + seed_res_offset + k * 100)
            w_k = torch.randn(num_nodes, self.dim, self.dim, generator=gen_r, dtype=config.DTYPE)
            for m in range(num_nodes):
                v = torch.randn(self.dim, dtype=config.DTYPE)
                for _ in range(config.POWER_ITER_STEPS):
                    v = torch.mv(w_k[m], v)
                    v = v / (torch.norm(v) + config.EPSILON)
                rad = torch.norm(torch.mv(w_k[m], v))
                w_k[m].mul_(config.SPECTRAL_RADIUS / (rad.item() + config.EPSILON))
            w_res_list.append(w_k)

        self.w_res = torch.stack(w_res_list, dim=0).to(self.device)

        # Trainable readout synapses
        w_out_list = []
        for k in range(self.num_cols):
            gen_w = torch.Generator(device="cpu").manual_seed(config.SEED + seed_wout_offset + k * 100)
            raw_w = torch.randn(num_nodes, self.dim, self.dim, generator=gen_w, dtype=config.DTYPE)
            w_out_list.append(raw_w * config.W_OUT_INIT_STD)

        self.w_out = torch.stack(w_out_list, dim=0).to(self.device)

    @torch.no_grad()
    def update_w_out(self, err_feedback, h_state, plasticity_gates):
        """Hyper-local correlation Delta rule with instant plasticity gating."""
        mask = (plasticity_gates > self.cfg.PLASTICITY_THRESHOLD).float()
        grad = torch.matmul(err_feedback.unsqueeze(3), h_state.unsqueeze(2))
        effective_eta = self.eta * plasticity_gates * mask
        self.w_out.add_(effective_eta * grad)

        # Enforce maximum row norm boundary
        row_norms = torch.norm(self.w_out, dim=3, keepdim=True)
        scales = torch.clamp(self.cfg.W_OUT_NORM_MAX / (row_norms + self.cfg.EPSILON), max=1.0)
        self.w_out.mul_(scales)

    @torch.no_grad()
    def step_reservoir(self, u_in, u_clock=None):
        """Execute somatic integration, homeostatic gain adaptation, and leaky integration."""
        self.h_prev.copy_(self.h)

        in_rms = torch.sqrt(torch.mean(u_in ** 2, dim=2, keepdim=True) + self.cfg.EPSILON)
        scale_limit = torch.clamp(self.cfg.TARGET_RMS / (in_rms + self.cfg.EPSILON), max=1.0)
        u_in_clamped = u_in * scale_limit
        u_in_scaled = self.leak * u_in_clamped

        ratio_in = torch.clamp(in_rms / self.cfg.TARGET_RMS, max=1.0)
        beta = (1.0 - self.leak) + self.leak * (1.0 - ratio_in)

        v_rec = beta * torch.matmul(self.w_res, self.h_prev.unsqueeze(3)).squeeze(3)
        u_somatic = u_in_scaled + v_rec

        if u_clock is not None and self.clock_budget > 0.0:
            u_somatic = u_somatic + self.clock_budget * u_clock.unsqueeze(1)

        new_h = torch.tanh(u_somatic * self.gain)
        activity = torch.mean(torch.abs(new_h), dim=2, keepdim=True)
        gain_err = self.cfg.H_ACT_TARGET - activity
        self.gain.add_(gain_err, alpha=self.cfg.GAIN_LEARNING_RATE).clamp_(self.cfg.GAIN_MIN, self.cfg.GAIN_MAX)

        self.h = (1.0 - self.leak) * self.h_prev + self.leak * new_h
        return torch.matmul(self.w_out, self.h.unsqueeze(3)).squeeze(3)

    @torch.no_grad()
    def reset_state(self):
        self.h.zero_()
        self.h_prev.zero_()


class SuperColumnArray:
    """
    Canonical 1-4-4 Cortical Column Array
    Handles multi-column hierarchical inference and top-down credit metabolism.
    """
    @torch.no_grad()
    def __init__(self, config):
        self.cfg = config
        self.num_cols = config.NUM_SUPER_COLUMNS
        self.dim = config.LATENT_DIM
        self.device = config.DEVICE

        self.clock = ParallelLFSRClockArray(config)

        self.l0 = ParallelReservoirBank(config, config.NUM_NODES_L0, [config.LEAK_L0], 0.0,
                                        config.SEED_OFFSET_RES_L0, config.SEED_OFFSET_WOUT_L0)
        self.l1 = ParallelReservoirBank(config, config.NUM_NODES_L1, config.LEAKS_L1, config.CLOCK_BUDGET,
                                        config.SEED_OFFSET_RES_L1, config.SEED_OFFSET_WOUT_L1)
        self.l2 = ParallelReservoirBank(config, config.NUM_NODES_L2, config.LEAKS_L2, config.CLOCK_BUDGET,
                                        config.SEED_OFFSET_RES_L2, config.SEED_OFFSET_WOUT_L2)

        # Top-down credit weights between timescale banks
        self.w_td0 = torch.full((self.num_cols, config.NUM_NODES_L1), config.W_TD_INIT,
                                dtype=config.DTYPE, device=self.device)
        self.w_td1 = torch.full((self.num_cols, config.NUM_NODES_L1, config.NUM_NODES_L2), config.W_TD_INIT,
                                dtype=config.DTYPE, device=self.device)

        self.eta_td0 = (config.ETA_TD * config.LEAK_L0)
        self.eta_td1 = (config.ETA_TD * self.l1.leak.squeeze(2))

        # Memory registers for delayed causal credit assignment
        self.prev_h0 = torch.zeros(self.num_cols, config.NUM_NODES_L0, self.dim, dtype=config.DTYPE, device=self.device)
        self.prev_h1 = torch.zeros(self.num_cols, config.NUM_NODES_L1, self.dim, dtype=config.DTYPE, device=self.device)
        self.prev_h2 = torch.zeros(self.num_cols, config.NUM_NODES_L2, self.dim, dtype=config.DTYPE, device=self.device)

        self.prev_pred_local0 = torch.zeros(self.num_cols, config.NUM_NODES_L0, self.dim, dtype=config.DTYPE, device=self.device)
        self.prev_pred_local1 = torch.zeros(self.num_cols, config.NUM_NODES_L1, self.dim, dtype=config.DTYPE, device=self.device)
        self.prev_pred_local2 = torch.zeros(self.num_cols, config.NUM_NODES_L2, self.dim, dtype=config.DTYPE, device=self.device)

        self.prev_b0_total = torch.zeros(self.num_cols, 1, self.dim, dtype=config.DTYPE, device=self.device)
        self.prev_b1_total = torch.zeros(self.num_cols, config.NUM_NODES_L1, self.dim, dtype=config.DTYPE, device=self.device)

        self.prev_td1 = torch.zeros(self.num_cols, config.NUM_NODES_L1, self.dim, dtype=config.DTYPE, device=self.device)
        self.prev_pred_mod0 = torch.zeros(self.num_cols, self.dim, dtype=config.DTYPE, device=self.device)

        self.step_count = 0

    @torch.no_grad()
    def forward_phase1_learning(self, u_in0, learn=True, plasticity_gates=None):
        """Phase 1: Local credit assignment and synaptic updates based on observed error."""
        inv_sqrt_d = self.cfg.INV_SQRT_D
        u_in_all = u_in0.unsqueeze(0).expand(self.num_cols, -1)

        if self.step_count > 0:
            err_0 = u_in_all - self.prev_pred_mod0
            err_local0 = u_in_all - self.prev_pred_local0.squeeze(1)
            norm_local0 = (torch.norm(err_local0, dim=1) * inv_sqrt_d).unsqueeze(1)

            gate_fb0 = 1.0 + torch.tanh(self.prev_b0_total.squeeze(1))
            err_to_l1 = (err_0 * gate_fb0).unsqueeze(1).expand(-1, self.cfg.NUM_NODES_L1, -1)

            if learn and plasticity_gates is not None:
                p_mask = (plasticity_gates > self.cfg.PLASTICITY_THRESHOLD).float().unsqueeze(1)
                gates_4d = plasticity_gates.view(self.num_cols, 1, 1, 1)

                b0_test = self.cfg.W_TD_TEST_GAIN * self.prev_td1
                gate_b0_test = 1.0 + torch.tanh(b0_test)
                pred_mod0_ports = self.prev_pred_local0 + b0_test * gate_b0_test
                err_mod0_ports = u_in_all.unsqueeze(1) - pred_mod0_ports
                norm_mod0_ports = torch.norm(err_mod0_ports, dim=2) * inv_sqrt_d

                credit0 = norm_local0 * (norm_local0 - norm_mod0_ports)
                credit0_rel = credit0 - torch.mean(credit0, dim=1, keepdim=True)
                self.w_td0.add_(self.eta_td0 * credit0_rel * plasticity_gates.unsqueeze(1) * p_mask).clamp_(self.cfg.W_TD_MIN, self.cfg.W_TD_MAX)
                self.l0.update_w_out(err_0.unsqueeze(1), self.prev_h0, gates_4d)

                err_local1 = err_to_l1 - self.prev_pred_local1
                norm_local1 = torch.norm(err_local1, dim=2) * inv_sqrt_d

                b1_test = self.cfg.W_TD_TEST_GAIN * self.prev_pred_local2.unsqueeze(1)
                gate_b1_test = 1.0 + torch.tanh(b1_test)
                pred_mod1_ports = self.prev_pred_local1.unsqueeze(2) + b1_test * gate_b1_test
                err_mod1_ports = err_to_l1.unsqueeze(2) - pred_mod1_ports
                norm_mod1_ports = torch.norm(err_mod1_ports, dim=3) * inv_sqrt_d

                credit1 = norm_local1.unsqueeze(2) * (norm_local1.unsqueeze(2) - norm_mod1_ports)
                credit1_rel = credit1 - torch.mean(credit1, dim=2, keepdim=True)
                self.w_td1.add_(self.eta_td1.unsqueeze(2) * credit1_rel * plasticity_gates.view(self.num_cols, 1, 1) * p_mask.unsqueeze(2)).clamp_(self.cfg.W_TD_MIN, self.cfg.W_TD_MAX)
                self.l1.update_w_out(err_to_l1, self.prev_h1, gates_4d)

                gate_fb1 = 1.0 + torch.tanh(self.prev_b1_total)
                err_to_l2_nodes = err_to_l1 * gate_fb1
                err_to_l2 = torch.mean(err_to_l2_nodes, dim=1, keepdim=True).expand(-1, self.cfg.NUM_NODES_L2, -1)
                self.l2.update_w_out(err_to_l2, self.prev_h2, gates_4d)
        else:
            err_0 = torch.zeros(self.num_cols, self.dim, dtype=self.cfg.DTYPE, device=self.device)
            err_to_l1 = torch.zeros(self.num_cols, self.cfg.NUM_NODES_L1, self.dim, dtype=self.cfg.DTYPE, device=self.device)

        return err_0, err_to_l1

    @torch.no_grad()
    def forward_phase2_dynamics(self, u_in0, err_0, err_to_l1):
        """Phase 2: Forward leaky integration and hierarchical top-down prediction synthesis."""
        u_clk = self.clock.step()

        u_in_l0 = u_in0.unsqueeze(0).unsqueeze(1).expand(self.num_cols, self.cfg.NUM_NODES_L0, -1)
        pred_local0 = self.l0.step_reservoir(u_in_l0, u_clock=None)

        u_bu1 = err_0.unsqueeze(1).expand(-1, self.cfg.NUM_NODES_L1, -1)
        pred_local1 = self.l1.step_reservoir(u_bu1, u_clock=u_clk)

        err_mod1 = err_to_l1 - self.prev_td1
        u_bu2 = torch.mean(err_mod1, dim=1, keepdim=True).expand(-1, self.cfg.NUM_NODES_L2, -1)
        pred_local2 = self.l2.step_reservoir(u_bu2, u_clock=u_clk)

        td2 = pred_local2
        w1_ratio = self.w_td1 / (torch.sum(self.w_td1, dim=2, keepdim=True) + self.cfg.EPSILON)
        b1_ports = self.w_td1.unsqueeze(3) * td2.unsqueeze(1)
        b1_total = torch.sum(w1_ratio.unsqueeze(3) * b1_ports, dim=2)
        td1 = pred_local1 + b1_total * (1.0 + torch.tanh(b1_total))

        w0_ratio = self.w_td0 / (torch.sum(self.w_td0, dim=1, keepdim=True) + self.cfg.EPSILON)
        b0_ports = self.w_td0.unsqueeze(2) * td1
        b0_total = torch.sum(w0_ratio.unsqueeze(2) * b0_ports, dim=1, keepdim=True)
        pred_mod0 = (pred_local0 + b0_total * (1.0 + torch.tanh(b0_total))).squeeze(1)

        self.prev_h0.copy_(self.l0.h)
        self.prev_h1.copy_(self.l1.h)
        self.prev_h2.copy_(self.l2.h)

        self.prev_pred_local0.copy_(pred_local0)
        self.prev_pred_local1.copy_(pred_local1)
        self.prev_pred_local2.copy_(pred_local2)

        self.prev_b0_total.copy_(b0_total)
        self.prev_b1_total.copy_(b1_total)

        self.prev_td1.copy_(td1)
        self.prev_pred_mod0.copy_(pred_mod0)

        self.step_count += 1
        return pred_mod0

    @torch.no_grad()
    def reset_state(self):
        self.l0.reset_state()
        self.l1.reset_state()
        self.l2.reset_state()
        self.clock.reset()

        self.prev_h0.zero_()
        self.prev_h1.zero_()
        self.prev_h2.zero_()
        self.prev_pred_local0.zero_()
        self.prev_pred_local1.zero_()
        self.prev_pred_local2.zero_()
        self.prev_b0_total.zero_()
        self.prev_b1_total.zero_()
        self.prev_td1.zero_()
        self.prev_pred_mod0.zero_()
        self.step_count = 0


class HolonMultiColumnNetwork:
    """
    Holon v14.0 Multi-Column Cortical Network
    Coordinates autonomous column specialization via Top-2 competitive lateral inhibition.
    """
    @torch.no_grad()
    def __init__(self, config):
        self.cfg = config
        self.num_super = config.NUM_SUPER_COLUMNS

        self.sensor = SensorInterface(config)
        self.array = SuperColumnArray(config)

        # Uniform initial credit across all cloned columns
        self.w_col = torch.full((self.num_super,), config.W_COL_INIT, dtype=config.DTYPE, device=config.DEVICE)

        # Uniform standby floor
        self.w_col_min = torch.full((self.num_super,), config.W_COL_MIN, dtype=config.DTYPE, device=config.DEVICE)

        d = config.LATENT_DIM
        dev = config.DEVICE
        dt = config.DTYPE

        self.prev_td_cols = torch.zeros(self.num_super, d, dtype=dt, device=dev)
        self.prev_b_top = torch.zeros(d, dtype=dt, device=dev)
        self.step_count = 0

    @torch.no_grad()
    def step(self, x_curr, learn=True):
        """Execute single online stream step."""
        inv_sqrt_d = self.cfg.INV_SQRT_D
        u_in0 = self.sensor.encode(x_curr)

        # Top-2 Competitive Lateral Inhibition
        if self.step_count > 0:
            err_cols = u_in0.unsqueeze(0) - self.prev_td_cols
            norm_errs = torch.norm(err_cols, dim=1) * inv_sqrt_d

            top_vals, top_indices = torch.topk(norm_errs, k=2, largest=False)
            e_1st, _ = top_vals[0], top_vals[1]
            win_idx = top_indices[0]

            # Update credit only if winner correlates with extrinsic environment
            if e_1st < self.cfg.E_BASELINE:
                credit_rel = e_1st - norm_errs
                credit_rel[win_idx] = self.cfg.E_BASELINE - e_1st

                eta_vec = torch.full_like(self.w_col, self.cfg.ETA_COL_DROP)
                eta_vec[win_idx] = self.cfg.ETA_COL_RISE

                updated_w = torch.clamp(self.w_col + eta_vec * credit_rel, max=self.cfg.W_COL_MAX)
                self.w_col = torch.maximum(updated_w, self.w_col_min)

        # Fourth-power contrastive Winner-Take-All weighting
        w_powered = torch.pow(self.w_col, self.cfg.W_COL_POWER)
        w_ratio = w_powered / (torch.sum(w_powered) + self.cfg.EPSILON)

        gates = w_ratio if learn else torch.zeros_like(w_ratio)

        err_0, err_l1 = self.array.forward_phase1_learning(u_in0, learn=learn, plasticity_gates=gates)

        eval_pred = self.sensor.decode(self.prev_b_top) if self.step_count > 0 else torch.zeros_like(x_curr)
        residual_err = x_curr - eval_pred

        td_stack = self.array.forward_phase2_dynamics(u_in0, err_0, err_l1)
        b_top = torch.sum(w_ratio.unsqueeze(1) * td_stack, dim=0)

        self.prev_td_cols.copy_(td_stack)
        self.prev_b_top.copy_(b_top)

        next_pred = self.sensor.decode(b_top)
        self.step_count += 1

        return {
            'eval_pred': eval_pred,
            'next_pred': next_pred,
            'residual_err': residual_err,
            'l0_err_norm': torch.norm(residual_err).item() * inv_sqrt_d,
            'w_col': self.w_col.clone(),
            'w_ratio': w_ratio.clone()
        }

    @torch.no_grad()
    def save(self, path, metadata=None):
        """Serialize full physical model state and metadata."""
        state = {
            'num_super': self.num_super,
            'w_col': self.w_col,
            'w_col_min': self.w_col_min,
            'sensor': self.sensor.w_sensor,
            'l0_res': self.array.l0.w_res, 'l0_out': self.array.l0.w_out, 'l0_gain': self.array.l0.gain,
            'l1_res': self.array.l1.w_res, 'l1_out': self.array.l1.w_out, 'l1_gain': self.array.l1.gain,
            'l2_res': self.array.l2.w_res, 'l2_out': self.array.l2.w_out, 'l2_gain': self.array.l2.gain,
            'w_td0': self.array.w_td0, 'w_td1': self.array.w_td1,
            'metadata': metadata or {}
        }
        torch.save(state, path)

    @torch.no_grad()
    def load(self, path):
        """Restore physical model state and adjust column layout dynamically."""
        if not os.path.exists(path):
            print(f" [!] Checkpoint not found: {path}")
            return False, {}
        state = torch.load(path, map_location=self.cfg.DEVICE)
        self.w_col.copy_(state['w_col'])
        if 'w_col_min' in state:
            self.w_col_min.copy_(state['w_col_min'])
        self.sensor.w_sensor.copy_(state['sensor'])

        self.array.l0.w_res.copy_(state['l0_res'])
        self.array.l0.w_out.copy_(state['l0_out'])
        self.array.l0.gain.copy_(state['l0_gain'])

        self.array.l1.w_res.copy_(state['l1_res'])
        self.array.l1.w_out.copy_(state['l1_out'])
        self.array.l1.gain.copy_(state['l1_gain'])

        self.array.l2.w_res.copy_(state['l2_res'])
        self.array.l2.w_out.copy_(state['l2_out'])
        self.array.l2.gain.copy_(state['l2_gain'])

        self.array.w_td0.copy_(state['w_td0'])
        self.array.w_td1.copy_(state['w_td1'])

        meta = state.get('metadata', {})
        print(f" [*] Restored Holon state ({self.num_super} columns): {path}")
        return True, meta

    @torch.no_grad()
    def reset_state(self):
        self.array.reset_state()
        self.prev_td_cols.zero_()
        self.prev_b_top.zero_()
        self.step_count = 0