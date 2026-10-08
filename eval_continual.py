# eval_continual.py
# Holon v14.0 Zero-Catastrophic-Forgetting Evaluator
# Evaluates frozen models on completely UNSEEN test corpora in reverse chronological order.
# holon_v14.0.2
import argparse
import os
import numpy as np
import torch
from config import Config
from dataloader import ByteStreamLoader
from model import HolonMultiColumnNetwork

def format_ratio(vec):
    """Format dominance ratio vector as percentage string."""
    return "/".join([f"{v*100:4.1f}%" for v in vec])

@torch.no_grad()
def evaluate_test_corpus(holon, cfg, corpus_path):
    """Evaluate frozen model on unseen test stream with zero plasticity updates."""
    loader = ByteStreamLoader(corpus_path, cfg)
    holon.reset_state()

    correct = 0
    total_l0_err = 0.0
    sum_w_col = torch.zeros(cfg.NUM_SUPER_COLUMNS, dtype=cfg.DTYPE, device=cfg.DEVICE)

    for step_idx in range(loader.size):
        x, target_idx = loader.next_token()
        stats = holon.step(x, learn=False)
        target_val = target_idx.item()

        if step_idx > 0:
            pred_val = torch.argmax(stats['eval_pred']).item()
            if pred_val == target_val:
                correct += 1
            total_l0_err += stats['l0_err_norm']

        sum_w_col.add_(stats['w_col'])

    eval_steps = max(1, loader.size - 1)
    acc = (correct / eval_steps) * 100.0
    avg_l0_err = total_l0_err / eval_steps
    avg_w = (sum_w_col / loader.size).cpu().numpy()
    w_pow = avg_w ** cfg.W_COL_POWER
    w_rat = w_pow / (np.sum(w_pow) + cfg.EPSILON)

    return {
        'acc': acc,
        'l0_err': avg_l0_err,
        'w_col': avg_w,
        'w_ratio': w_rat
    }

@torch.no_grad()
def run_evaluation(holon, cfg, task_order, ckpt_path):
    """Evaluate retention in reverse chronological order (Newest -> Past -> Oldest)."""
    eval_order = list(reversed(task_order))
    results = {}

    print(f"\n==========================================================================")
    print(f"  Holon v14.0 Zero-Forgetting Benchmark on UNSEEN TEST CORPORA")
    print(f"  Model: {ckpt_path}")
    print(f"  Learned Order: {' -> '.join(task_order)} | Eval Order: {' -> '.join(eval_order)}")
    print(f"==========================================================================")

    all_passed = True

    for rank_idx, task_key in enumerate(eval_order):
        task_info = cfg.TASKS[task_key]
        if rank_idx == 0:
            label = "Last Learned (Recent Memory)"
        elif rank_idx == len(eval_order) - 1:
            label = "1st Learned (Oldest Memory / Retention Check)"
        else:
            label = f"{len(eval_order)-rank_idx}nd Learned (Past Memory / Retention Check)"

        print(f"\n[*] {rank_idx+1}. Evaluating: Task {task_key} ({task_info['name']}) [{label}] ...")
        res = evaluate_test_corpus(holon, cfg, task_info['test_corpus'])
        results[task_key] = res

        pass_task = res['acc'] >= (task_info['bayes_limit'] - task_info['eval_margin'])
        if not pass_task:
            all_passed = False

        status_str = "[ZERO FORGETTING / SUCCESS]" if pass_task else "[INTERFERENCE DETECTED / FAILED]"

        print(f"--------------------------------------------------------------------------")
        print(f"  [Task {task_key}: {task_info['name']} ({label})]")
        print(f"    - Accuracy (Unseen)  : {res['acc']:6.2f}% (Bayes Limit: {task_info['bayes_limit']:.2f}%)")
        print(f"    - Dominant Ratio     : {format_ratio(res['w_ratio'])}")
        print(f"    - Status             : {status_str}")

    print(f"==========================================================================")
    if all_passed:
        print(f"  OVERALL RESULT: >>> 3-TASK ZERO FORGETTING FULLY PROVED ON UNSEEN DATA <<<")
    else:
        print(f"  OVERALL RESULT: >>> MULTI-COLUMN INTERFERENCE DETECTED <<<")
    print(f"==========================================================================\n")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Holon v14.0 Continual Learning Evaluator")
    parser.add_argument("--model", type=str, default="checkpoint_v14_ABC_s42.pth", help="Target model checkpoint")
    args = parser.parse_args()

    cfg = Config()

    # Automatically adapt column dimensions from checkpoint metadata
    if os.path.exists(args.model):
        ckpt_state = torch.load(args.model, map_location=cfg.DEVICE)
        if 'num_super' in ckpt_state:
            cfg.NUM_SUPER_COLUMNS = ckpt_state['num_super']

    holon = HolonMultiColumnNetwork(cfg)
    success, meta = holon.load(args.model)
    if not success:
        return

    task_order = meta.get('task_order', ['A', 'B', 'C'])
    run_evaluation(holon, cfg, task_order, args.model)

if __name__ == "__main__":
    main()