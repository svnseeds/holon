# generate_learning_animation.py
# Holon v14.0 Real-Time Mechanistic Self-Organization Animation Generator
# Renders synchronized multi-panel GIF: Accuracy trajectory, error dissipation,
# circuit wiring formation, and emergent block-diagonal synaptic matrix.
# holon_v14.0.2
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import torch
from config import Config
from dataloader import ByteStreamLoader
from model import HolonMultiColumnNetwork

# Animation configuration constants
TOTAL_EPOCHS = 40                 # 40 epochs (up to Bayes limit breakthrough)
SAMPLES_PER_EPOCH = 2             # 2 samples per epoch (80 frames total)
FRAME_DURATION_MS = 80            # 80ms per frame (~12.5 fps / ~6.4s loop)
ANIM_DPI = 100                    # Web-optimized resolution

NODE_RADIUS = 0.22
LINE_WIDTH_MIN = 0.6
LINE_WIDTH_MAX = 5.0
ALPHA_MIN = 0.15
ALPHA_MAX = 0.95

# Unified timescale pastel palette
TAU_COLORS = {
    2.0: '#1976d2', 4.0: '#4fc3f7', 6.0: '#29b6f6',
    8.0: '#7e57c2', 10.0: '#ab47bc', 18.0: '#ffb74d',
    32.0: '#ffa726', 50.0: '#66bb6a'
}

def draw_circuit_frame(ax, w_td0, w_td1, tau_l0, tau_l1, tau_l2):
    """Render single frame of 1-4-4 neural circuit wiring."""
    ax.clear()
    ax.set_xlim(-0.8, 3.8)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')

    x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
    y_l2, y_l1, y_l0 = 2.0, 1.0, 0.0
    x_l0 = 1.5
    cmap = plt.cm.plasma

    # L2 -> L1 Top-down edges
    for j in range(4):
        for i in range(4):
            w = float(w_td1[i, j])
            norm_w = np.clip((w - 0.10) / (1.00 - 0.10 + 1e-6), 0.0, 1.0)
            lw = LINE_WIDTH_MIN + (LINE_WIDTH_MAX - LINE_WIDTH_MIN) * norm_w
            alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (norm_w ** 1.5)
            color = cmap(norm_w) if norm_w > 0.05 else '#cccccc'
            ax.plot([x_nodes[j], x_nodes[i]], [y_l2, y_l1], color=color, linewidth=lw, alpha=alpha, zorder=1)

    # L1 -> L0 Top-down edges
    for i in range(4):
        w = float(w_td0[i])
        norm_w = np.clip((w - 0.10) / (1.00 - 0.10 + 1e-6), 0.0, 1.0)
        lw = LINE_WIDTH_MIN + (LINE_WIDTH_MAX - LINE_WIDTH_MIN) * norm_w
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (norm_w ** 1.5)
        color = cmap(norm_w) if norm_w > 0.05 else '#cccccc'
        ax.plot([x_nodes[i], x_l0], [y_l1, y_l0], color=color, linewidth=lw, alpha=alpha, zorder=1)

    # Node glyphs with timescale palette
    for j in range(4):
        c = TAU_COLORS.get(round(tau_l2[j], 1), '#ffffff')
        circle = plt.Circle((x_nodes[j], y_l2), NODE_RADIUS, color=c, ec='#333333', lw=1.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x_nodes[j], y_l2, f"$\\tau$={int(tau_l2[j])}", ha='center', va='center', fontsize=8.5, fontweight='bold', zorder=4)

    for i in range(4):
        c = TAU_COLORS.get(round(tau_l1[i], 1), '#ffffff')
        circle = plt.Circle((x_nodes[i], y_l1), NODE_RADIUS, color=c, ec='#333333', lw=1.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x_nodes[i], y_l1, f"$\\tau$={int(tau_l1[i])}", ha='center', va='center', fontsize=8.5, fontweight='bold', zorder=4)

    c_l0 = TAU_COLORS.get(round(tau_l0, 1), '#1976d2')
    circle_l0 = plt.Circle((x_l0, y_l0), NODE_RADIUS * 1.15, color='#e6f2ff', ec=c_l0, lw=2.2, zorder=3)
    ax.add_patch(circle_l0)
    ax.text(x_l0, y_l0, f"$\\tau$={tau_l0:.1f}", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_l0, zorder=4)

    ax.text(-0.6, y_l2, "L2 Deep", ha='right', va='center', fontsize=9, fontweight='bold', color='#555555')
    ax.text(-0.6, y_l1, "L1 Mid", ha='right', va='center', fontsize=9, fontweight='bold', color='#555555')
    ax.text(-0.6, y_l0, "L0 Base", ha='right', va='center', fontsize=9, fontweight='bold', color='#1f77b4')
    ax.set_title("Self-Organizing Neural Circuit (1-4-4)", fontsize=11, fontweight='bold', pad=8)

def draw_heatmap_frame(ax, w_td1, tau_l1, tau_l2):
    """Render single frame of W_td1 synaptic matrix heatmap."""
    ax.clear()
    im = ax.imshow(w_td1, cmap='plasma', vmin=0.10, vmax=1.00, aspect='auto')

    for i in range(4):
        for j in range(4):
            val = w_td1[i, j]
            tc = '#ffffff' if val > 0.55 else '#111111'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8.5, fontweight='bold', color=tc)

    ax.set_xticks(range(4))
    ax.set_xticklabels([f"$\\tau$={int(t)}" for t in tau_l2], fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"$\\tau$={int(t)}" for t in tau_l1], fontsize=8)
    ax.set_xlabel("L2 Source Node (Long $\\tau$)", fontsize=8.5)
    ax.set_ylabel("L1 Target Node (Mid $\\tau$)", fontsize=8.5)
    ax.set_title("$W_{td1}$ Synaptic Matrix (Emergent Modularity)", fontsize=10, fontweight='bold', pad=6)

@torch.no_grad()
def main():
    cfg = Config(seed=42)
    holon = HolonMultiColumnNetwork(cfg)
    corpus_path = cfg.TASKS['B']['train_corpus']

    if not os.path.exists(corpus_path):
        print(f"[!] Corpus not found: {corpus_path}. Run generate_data.py first.")
        return

    loader = ByteStreamLoader(corpus_path, cfg)
    bayes_limit = cfg.TASKS['B']['bayes_limit']

    tau_l0 = 1.0 / cfg.LEAK_L0
    tau_l1 = [1.0 / l for l in cfg.LEAKS_L1]
    tau_l2 = [1.0 / l for l in cfg.LEAKS_L2]

    sample_interval = loader.size // SAMPLES_PER_EPOCH

    history = {
        'steps': [],
        'epochs': [],
        'acc': [],
        'err': [],
        'w_td0': [],
        'w_td1': []
    }

    print(f"\n==========================================================================")
    print(f"  Holon v14.0 Real-Time Learning Dynamics Animation Generator")
    print(f"  Task: V2L2-mirror | Epochs: {TOTAL_EPOCHS} | Target: Bayes Limit {bayes_limit:.2f}%")
    print(f"==========================================================================\n")
    print(f"[*] 1. Executing training and logging frame snapshots ...")

    global_step = 0

    for epoch in range(TOTAL_EPOCHS):
        loader.reset()
        holon.reset_state()

        epoch_correct = 0
        epoch_l0_err = 0.0

        for step_idx in range(loader.size):
            x, target_idx = loader.next_token()
            stats = holon.step(x, learn=True)
            target_val = target_idx.item()

            if step_idx > 0:
                pred_val = torch.argmax(stats['eval_pred']).item()
                if pred_val == target_val:
                    epoch_correct += 1
                epoch_l0_err += stats['l0_err_norm']

            global_step += 1

            if (step_idx + 1) % sample_interval == 0:
                eval_steps = max(1, step_idx)
                curr_acc = (epoch_correct / eval_steps) * 100.0
                curr_err = epoch_l0_err / eval_steps

                win_col = torch.argmax(holon.w_col).item()
                w0 = holon.array.w_td0[win_col].cpu().numpy().copy()
                w1 = holon.array.w_td1[win_col].cpu().numpy().copy()

                history['steps'].append(global_step)
                history['epochs'].append(epoch + (step_idx + 1) / loader.size)
                history['acc'].append(curr_acc)
                history['err'].append(curr_err)
                history['w_td0'].append(w0)
                history['w_td1'].append(w1)

        print(f"    Epoch {epoch+1:02d}/{TOTAL_EPOCHS:02d} Complete | ACC: {curr_acc:5.2f}% | L0_Err: {curr_err:6.4f}")

    total_frames = len(history['steps'])
    print(f"\n[*] 2. Rendering animation frames ({total_frames} frames total) ...")

    frames = []
    fig = plt.figure(figsize=(15, 8.5), dpi=ANIM_DPI)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    ax_acc = plt.subplot(gs[0, 0])
    ax_err = plt.subplot(gs[1, 0])
    ax_circuit = plt.subplot(gs[0, 1])
    ax_heat = plt.subplot(gs[1, 1])

    for f_idx in range(total_frames):
        # 1. Top-Left: Accuracy Trajectory with Bayes Limit Benchmark
        ax_acc.clear()
        steps_so_far = history['epochs'][:f_idx + 1]
        acc_so_far = history['acc'][:f_idx + 1]

        ax_acc.plot(steps_so_far, acc_so_far, color='#1f77b4', linewidth=2.5, label='ACC (%)')
        ax_acc.axhline(bayes_limit, color='#d62728', linestyle='--', linewidth=1.8,
                       label=f'Bayes Limit ({bayes_limit:.2f}%)')

        ax_acc.set_xlim(0, TOTAL_EPOCHS)
        ax_acc.set_ylim(40.0, 90.0)
        ax_acc.set_xlabel("Epoch", fontsize=9.5)
        ax_acc.set_ylabel("Accuracy (%)", fontsize=9.5)
        curr_acc_val = history['acc'][f_idx]
        is_bayes = curr_acc_val >= bayes_limit
        acc_title_color = '#d62728' if is_bayes else '#111111'
        bayes_tag = " [BAYES OPTIMAL REACHED]" if is_bayes else ""
        ax_acc.set_title(f"Accuracy Trajectory: {curr_acc_val:.2f}%{bayes_tag}",
                         fontsize=11, fontweight='bold', color=acc_title_color)
        ax_acc.legend(loc='lower right', frameon=True, fontsize=8.5)
        ax_acc.grid(True, linestyle=':', alpha=0.6)

        # 2. Bottom-Left: Prediction Error Dissipation
        ax_err.clear()
        err_so_far = history['err'][:f_idx + 1]
        ax_err.plot(steps_so_far, err_so_far, color='#2ca02c', linewidth=2.2, label='L0 Error')
        ax_err.set_xlim(0, TOTAL_EPOCHS)
        ax_err.set_ylim(0.025, 0.055)
        ax_err.set_xlabel("Epoch", fontsize=9.5)
        ax_err.set_ylabel("L0 Prediction Error", fontsize=9.5)
        ax_err.set_title(f"Error Dissipation: {history['err'][f_idx]:.4f}", fontsize=10.5, fontweight='bold')
        ax_err.grid(True, linestyle=':', alpha=0.6)

        # 3. Top-Right: Self-Organizing Neural Circuit
        draw_circuit_frame(ax_circuit, history['w_td0'][f_idx], history['w_td1'][f_idx], tau_l0, tau_l1, tau_l2)

        # 4. Bottom-Right: Emergent Synaptic Matrix
        draw_heatmap_frame(ax_heat, history['w_td1'][f_idx], tau_l1, tau_l2)

        curr_ep = history['epochs'][f_idx]
        plt.suptitle(f"Holon v14.0 Mechanistic Self-Organization | Task: V2L2-mirror (Epoch: {curr_ep:.1f}/{TOTAL_EPOCHS})",
                     fontsize=13, fontweight='bold', y=0.98)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))

        if (f_idx + 1) % 10 == 0 or f_idx == total_frames - 1:
            print(f"    Rendering progress: {f_idx + 1:02d}/{total_frames:02d} frames complete ...")

    plt.close()

    out_gif = "holon_v14_learning_dynamics.gif"
    print(f"\n[*] 3. Compiling GIF animation: {out_gif} ...")
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True
    )

    file_size_mb = os.path.getsize(out_gif) / (1024 * 1024)
    print(f"==========================================================================")
    print(f" [*] GIF animation saved: {out_gif} ({file_size_mb:.2f} MB)")
    print(f"==========================================================================\n")

if __name__ == "__main__":
    main()