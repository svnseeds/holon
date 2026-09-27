# train_continual.py
# Holon v14.0 Continual Lifelong Learning Runner
# Autonomous Column Specialization & Real-Time Dynamics Visualization
import argparse
import time
import os
import matplotlib
matplotlib.use('Agg')  # Headless server compatible
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from config import Config
from dataloader import ByteStreamLoader
from model import HolonMultiColumnNetwork
from eval_continual import run_evaluation

def format_vec(vec):
    """Format float vector into slash-separated string."""
    return "/".join([f"{v:.2f}" for v in vec])

def format_ratio(vec):
    """Format ratio vector into percentage string."""
    return "/".join([f"{v*100:4.1f}%" for v in vec])

def plot_two_tier_dynamics(cfg, task_order, macro_steps, macro_ratios,
                           transition_steps, micro_zooms, prefix=""):
    """
    Two-Tier Dynamics Visualizer
    - Top: Full lifetime macro overview (stability across 1,000,000 steps)
    - Bottom: Microscopic bifurcation zoom (first 40 steps from task switch)
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(15, 10), dpi=cfg.PLOT_DPI)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    num_cols = cfg.NUM_SUPER_COLUMNS
    col_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:num_cols]
    task_colors = {'A': '#e6f2ff', 'B': '#e8f8e8', 'C': '#ffebee'}

    # =========================================================================
    # Top Panel: Full Lifetime Macro Overview
    # =========================================================================
    ax_macro = plt.subplot(gs[0, :])
    macro_ratios_arr = np.array(macro_ratios) * 100.0

    for k in range(num_cols):
        ax_macro.plot(macro_steps, macro_ratios_arr[:, k],
                      label=f"Column {k+1}", color=col_palette[k], linewidth=2.0, alpha=0.9)

    total_steps = macro_steps[-1] if len(macro_steps) > 0 else 1
    boundaries = transition_steps + [total_steps]
    for idx, task_key in enumerate(task_order):
        start_s = boundaries[idx]
        end_s = boundaries[idx + 1]
        t_name = cfg.TASKS[task_key]['name']
        bg_c = task_colors.get(task_key, '#f5f5f5')
        ax_macro.axvspan(start_s, end_s, color=bg_c, alpha=0.6, zorder=0)

        mid_s = (start_s + end_s) / 2
        ax_macro.text(mid_s, 103.5, f"Phase {idx+1}: Task {task_key}\n({t_name})",
                      horizontalalignment='center', verticalalignment='bottom',
                      fontsize=11, fontweight='bold', color='#333333')

        if idx > 0:
            ax_macro.axvline(start_s, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)

    ax_macro.set_title("Holon v14.0 Cortical Macro-Dynamics (Full Lifetime Stability)",
                       fontsize=14, fontweight='bold', pad=25)
    ax_macro.set_xlabel("Global Lifetime Steps (Tokens)", fontsize=11)
    ax_macro.set_ylabel("Dominance Ratio $w_{ratio}$ (%)", fontsize=11)
    ax_macro.set_ylim(-2.0, 105.0)
    ax_macro.set_xlim(0, total_steps)
    ax_macro.legend(loc='center right', frameon=True, framealpha=0.9, fontsize=10)

    # =========================================================================
    # Bottom Panel: Microscopic Bifurcation Zoom
    # =========================================================================
    for t_idx, micro_data in enumerate(micro_zooms):
        if t_idx >= 2:
            break
        ax_micro = plt.subplot(gs[1, t_idx])
        micro_arr = np.array(micro_data['ratios']) * 100.0
        steps_local = np.arange(len(micro_arr))

        for k in range(num_cols):
            ax_micro.plot(steps_local, micro_arr[:, k],
                          label=f"C{k+1}", color=col_palette[k], linewidth=2.2, alpha=0.95)

        t_from = task_order[t_idx]
        t_to = task_order[t_idx + 1]
        ax_micro.set_title(f"Transition {t_idx+1}: Task {t_from} $\\to$ Task {t_to} (First {len(micro_arr)} Steps)",
                           fontsize=12, fontweight='bold')
        ax_micro.set_xlabel("Steps from Task Switch", fontsize=10)
        ax_micro.set_ylabel("Dominance Ratio $w_{ratio}$ (%)", fontsize=10)
        ax_micro.set_ylim(-2.0, 105.0)
        ax_micro.set_xlim(0, len(micro_arr) - 1)
        ax_micro.legend(loc='center right', frameon=True, fontsize=9)

    png_path = f"training_dynamics_{prefix}.png"
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print(f" [*] Saved two-tier dynamics visualization: {png_path}")

    # Dual save raw numerical trajectory (.npz)
    npz_path = f"training_dynamics_{prefix}.npz"
    np.savez_compressed(
        npz_path,
        macro_steps=macro_steps,
        macro_ratios=macro_ratios_arr,
        transition_steps=transition_steps,
        task_order=task_order,
        micro_zooms=micro_zooms
    )
    print(f" [*] Saved compressed raw trajectory data: {npz_path}\n")

@torch.no_grad()
def run_continual_training(task_order_str="ABC", seed=42, num_cols=3):
    cfg = Config(seed=seed)
    cfg.NUM_SUPER_COLUMNS = num_cols

    holon = HolonMultiColumnNetwork(cfg)

    task_order = list(task_order_str.upper())
    prefix = f"{task_order_str.upper()}_s{seed}" if num_cols == 3 else f"{task_order_str.upper()}_s{seed}_col{num_cols}"
    ckpt_path = f"checkpoint_v14_{prefix}.pth"

    print(f"\n=========================================================================================")
    print(f"  Holon v14.0 Multi-Column (1-{cfg.NUM_SUPER_COLUMNS}) Continual Learning")
    print(f"  Task Order: {' -> '.join(task_order)} | Seed: {seed} | Columns: {cfg.NUM_SUPER_COLUMNS}")
    print(f"  Initial W_col: [{format_vec(holon.w_col.cpu().numpy())}] | Standby Floor: [{format_vec(holon.w_col_min.cpu().numpy())}]")
    print(f"  Routing Rates -> Drop: {cfg.ETA_COL_DROP} | Rise: {cfg.ETA_COL_RISE}")
    print(f"  Device: {cfg.DEVICE} | Latent Dim: {cfg.LATENT_DIM}")
    print(f"=========================================================================================\n")

    global_step = 0
    macro_steps = []
    macro_ratios = []
    transition_steps = []
    micro_zooms = []

    for phase_idx, task_key in enumerate(task_order):
        task_info = cfg.TASKS[task_key]
        loader = ByteStreamLoader(task_info['train_corpus'], cfg)
        epochs = task_info['epochs']
        bayes_limit = task_info['bayes_limit']

        transition_steps.append(global_step)
        is_tracking_micro = (phase_idx > 0)
        curr_micro_ratios = []

        print(f"\n>>> [PHASE {phase_idx+1}] Task {task_key} ({task_info['name']}) Learning Started (Epochs: {epochs}) ...")
        print(f" EPOCH |   ACC%   | L0_Err |  FUV   | W_col (C1..Cn) | Ratio (C1..Cn) | TIME     | STATUS")
        print("-" * 92)

        for epoch in range(epochs):
            start_time = time.time()
            loader.reset()
            holon.reset_state()

            epoch_correct = 0
            epoch_l0_err = 0.0
            total_err_sq = torch.zeros((), dtype=cfg.DTYPE, device=cfg.DEVICE)
            total_in_sq = torch.zeros((), dtype=cfg.DTYPE, device=cfg.DEVICE)
            sum_w_col = torch.zeros(cfg.NUM_SUPER_COLUMNS, dtype=cfg.DTYPE, device=cfg.DEVICE)

            for step_idx in range(loader.size):
                x, target_idx = loader.next_token()
                stats = holon.step(x, learn=True)
                target_val = target_idx.item()

                w_ratio_np = stats['w_ratio'].cpu().numpy()

                # Microscopic bifurcation zoom logging (first 40 steps after task switch)
                if is_tracking_micro and len(curr_micro_ratios) < cfg.MICRO_ZOOM_STEPS:
                    curr_micro_ratios.append(w_ratio_np)

                # Macro lifelong overview decimation logging
                if global_step % cfg.MACRO_SAMPLE_INTERVAL == 0:
                    macro_steps.append(global_step)
                    macro_ratios.append(w_ratio_np)

                if step_idx > 0:
                    pred_val = torch.argmax(stats['eval_pred']).item()
                    if pred_val == target_val:
                        epoch_correct += 1

                    epoch_l0_err += stats['l0_err_norm']
                    total_err_sq.add_(torch.sum(stats['residual_err'] ** 2))
                    total_in_sq.add_(torch.sum(x ** 2))

                sum_w_col.add_(stats['w_col'])
                global_step += 1

            elapsed = time.time() - start_time
            eval_steps = max(1, loader.size - 1)
            acc = (epoch_correct / eval_steps) * 100.0
            avg_l0_err = epoch_l0_err / eval_steps
            fuv = (total_err_sq / max(cfg.EPSILON, total_in_sq.item())).item()
            avg_w = (sum_w_col / loader.size).cpu().numpy()
            w_pow = avg_w ** cfg.W_COL_POWER
            w_rat = w_pow / (np.sum(w_pow) + cfg.EPSILON)

            status = "[★BAYES]" if acc >= bayes_limit else ""
            print(f" {epoch+1:03d}/{epochs:03d} | {acc:6.2f}% | {avg_l0_err:6.4f} | {fuv:6.4f} |  {format_vec(avg_w)}  |  {format_ratio(w_rat)}  | {elapsed:4.2f}s | {status:10s}")

        if is_tracking_micro:
            micro_zooms.append({
                'transition_idx': phase_idx,
                'from_task': task_order[phase_idx - 1],
                'to_task': task_key,
                'ratios': curr_micro_ratios
            })

    # Save checkpoint with full architectural metadata
    metadata = {
        'task_order': task_order,
        'seed': seed,
        'num_super': num_cols
    }
    holon.save(ckpt_path, metadata=metadata)
    print("-" * 92)
    print(f" [*] Continual learning finished. Model saved: {ckpt_path}\n")

    # Render and save two-tier dynamics visualizer
    plot_two_tier_dynamics(cfg, task_order, macro_steps, macro_ratios,
                           transition_steps, micro_zooms, prefix=prefix)

    # Automatically transition to zero-forgetting evaluation on unseen test data
    print(f"==========================================================================")
    print(f"  Transitioning directly to Unseen Test Zero-Forgetting Benchmark ...")
    print(f"==========================================================================")
    run_evaluation(holon, cfg, task_order, ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holon v14.0 Continual Learning Runner")
    parser.add_argument("--order", type=str, default="ABC", help="Task sequence (e.g. ABC, CBA, BAC)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (e.g. 42, 100, 777)")
    parser.add_argument("--cols", type=int, default=3, help="Number of cortical columns (e.g. 3, 4, 5)")
    args = parser.parse_args()

    run_continual_training(task_order_str=args.order, seed=args.seed, num_cols=args.cols)