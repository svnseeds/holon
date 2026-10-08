# generate_data.py
# One-Stop Benchmark Corpora Generator for Tasks A, B, and C
# Supports custom random seed configuration via CLI arguments.
# holon_v14.0.2
import argparse
import random as rd
from config import Config

DEFAULT_SEED = 42
NUM_PATTERNS_TRAIN = 1500
NUM_PATTERNS_TEST = 1500

# =========================================================================
# Task A: Pure V2D2 Dyck-2 (Monte Carlo Rejection Sampling)
# =========================================================================
def generate_random_walk_bracket(max_depth, pairs):
    stack = []
    result = ""
    max_reached_depth = 1

    open_char, close_char = rd.choice(pairs)
    stack.append(close_char)
    result += open_char

    while len(stack) > 0:
        current_depth = len(stack)
        if current_depth > max_reached_depth:
            max_reached_depth = current_depth

        if current_depth >= max_depth:
            char = stack.pop()
            result += char
        else:
            if rd.random() < 0.5:
                open_char, close_char = rd.choice(pairs)
                stack.append(close_char)
                result += open_char
            else:
                char = stack.pop()
                result += char

    return result, max_reached_depth

def build_dyck_corpus(filename, num_patterns, target_depth, pairs):
    with open(filename, "w", encoding="utf-8") as f:
        accepted_count = 0
        while accepted_count < num_patterns:
            pattern, reached_depth = generate_random_walk_bracket(target_depth, pairs)
            if reached_depth == target_depth and len(pattern) > 0:
                f.write(pattern + " ")
                accepted_count += 1
    print(f"  [+] Generated {filename:<28} ({num_patterns} patterns | depth: {target_depth})")

# =========================================================================
# Task B: V2L2-Mirror (Time-Reversal LIFO Buffer on ['a', 'b'])
# =========================================================================
def build_mirror_corpus(filename, num_patterns, length, char_set, trigger):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_patterns):
            seq = [rd.choice(char_set) for _ in range(length)]
            pattern = "".join(seq) + trigger + "".join(reversed(seq)) + " "
            f.write(pattern)
    print(f"  [+] Generated {filename:<28} ({num_patterns} patterns | length: {length})")

# =========================================================================
# Task C: V2L2-FIFO (Phase-Delay Queue on ['c', 'd'])
# =========================================================================
def build_fifo_corpus(filename, num_patterns, length, char_set, trigger):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_patterns):
            seq = [rd.choice(char_set) for _ in range(length)]
            pattern = "".join(seq) + trigger + "".join(seq) + " "
            f.write(pattern)
    print(f"  [+] Generated {filename:<28} ({num_patterns} patterns | length: {length})")

def resolve_filename(base_filename, seed, default_seed):
    """Append _s{seed} suffix only when an alternative non-default seed is specified."""
    if seed == default_seed:
        return base_filename
    parts = base_filename.rsplit('.', 1)
    return f"{parts[0]}_s{seed}.{parts[1]}" if len(parts) == 2 else f"{base_filename}_s{seed}"

def main():
    parser = argparse.ArgumentParser(description="Holon Benchmark Corpora Generator")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Deterministic seed (default: {DEFAULT_SEED})")
    parser.add_argument("--train-patterns", type=int, default=NUM_PATTERNS_TRAIN, help="Number of training patterns")
    parser.add_argument("--test-patterns", type=int, default=NUM_PATTERNS_TEST, help="Number of test patterns")
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    print("\n==========================================================================")
    print(f"  Holon v14.0.2 Benchmark Corpora Generator (Target Seed: {args.seed})")
    print(f"  Train: {args.train_patterns} patterns | Test: {args.test_patterns} patterns")
    print("==========================================================================\n")

    # 1. Task A: Pure Dyck-2
    print(f"[*] Generating Task A ({cfg.TASKS['A']['name']}) Corpora ...")
    rd.seed(args.seed)
    pairs = [("(", ")"), ("[", "]")]
    train_a = resolve_filename(cfg.TASKS['A']['train_corpus'], args.seed, DEFAULT_SEED)
    test_a = resolve_filename(cfg.TASKS['A']['test_corpus'], args.seed, DEFAULT_SEED)
    build_dyck_corpus(train_a, args.train_patterns, target_depth=2, pairs=pairs)
    build_dyck_corpus(test_a, args.test_patterns, target_depth=2, pairs=pairs)

    # 2. Task B: V2L2-Mirror
    print(f"\n[*] Generating Task B ({cfg.TASKS['B']['name']}) Corpora ...")
    rd.seed(args.seed)
    char_set_b = ["a", "b"]
    trigger_mirror = chr(cfg.TRIGGER_MIRROR)
    train_b = resolve_filename(cfg.TASKS['B']['train_corpus'], args.seed, DEFAULT_SEED)
    test_b = resolve_filename(cfg.TASKS['B']['test_corpus'], args.seed, DEFAULT_SEED)
    build_mirror_corpus(train_b, args.train_patterns, length=2, char_set=char_set_b, trigger=trigger_mirror)
    build_mirror_corpus(test_b, args.test_patterns, length=2, char_set=char_set_b, trigger=trigger_mirror)

    # 3. Task C: V2L2-FIFO
    print(f"\n[*] Generating Task C ({cfg.TASKS['C']['name']}) Corpora ...")
    rd.seed(args.seed)
    char_set_c = ["c", "d"]
    trigger_fifo = chr(cfg.TRIGGER_FIFO)
    train_c = resolve_filename(cfg.TASKS['C']['train_corpus'], args.seed, DEFAULT_SEED)
    test_c = resolve_filename(cfg.TASKS['C']['test_corpus'], args.seed, DEFAULT_SEED)
    build_fifo_corpus(train_c, args.train_patterns, length=2, char_set=char_set_c, trigger=trigger_fifo)
    build_fifo_corpus(test_c, args.test_patterns, length=2, char_set=char_set_c, trigger=trigger_fifo)

    print("\n==========================================================================")
    print(" [*] Benchmark corpora generated successfully.")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()