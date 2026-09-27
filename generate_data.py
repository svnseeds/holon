# generate_data.py
# One-Stop Benchmark Corpora Generator for Tasks A, B, and C
# Generates balanced Train (1500 patterns) and Test (300 patterns) datasets.
import random as rd
from config import Config

NUM_PATTERNS_TRAIN = 1500
NUM_PATTERNS_TEST = 300

SEED_TASK_A = 42
SEED_TASK_B = 100
SEED_TASK_C = 200

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
    print(f"  [+] Generated {filename:<24} ({num_patterns} patterns | depth: {target_depth})")

# =========================================================================
# Task B: V2L2-Mirror (Time-Reversal LIFO Buffer)
# =========================================================================
def build_mirror_corpus(filename, num_patterns, length, char_set, trigger):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_patterns):
            seq = [rd.choice(char_set) for _ in range(length)]
            pattern = "".join(seq) + trigger + "".join(reversed(seq)) + " "
            f.write(pattern)
    print(f"  [+] Generated {filename:<24} ({num_patterns} patterns | length: {length})")

# =========================================================================
# Task C: V2L2-FIFO (Phase-Delay Queue)
# =========================================================================
def build_fifo_corpus(filename, num_patterns, length, char_set, trigger):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_patterns):
            seq = [rd.choice(char_set) for _ in range(length)]
            pattern = "".join(seq) + trigger + "".join(seq) + " "
            f.write(pattern)
    print(f"  [+] Generated {filename:<24} ({num_patterns} patterns | length: {length})")

def main():
    cfg = Config()
    print("\n==========================================================================")
    print("  Holon v14.0 Benchmark Corpora Generator (1-Stop Balanced Synthesis)")
    print(f"  Train: {NUM_PATTERNS_TRAIN} patterns | Test: {NUM_PATTERNS_TEST} patterns")
    print("==========================================================================\n")

    # 1. Task A: Pure Dyck-2
    print("[*] Generating Task A (Pure V2D2 Dyck-2) Corpora ...")
    rd.seed(SEED_TASK_A)
    pairs = [("(", ")"), ("[", "]")]
    build_dyck_corpus(cfg.TASKS['A']['train_corpus'], NUM_PATTERNS_TRAIN, target_depth=2, pairs=pairs)
    build_dyck_corpus(cfg.TASKS['A']['test_corpus'], NUM_PATTERNS_TEST, target_depth=2, pairs=pairs)

    # 2. Task B: V2L2-Mirror
    print("\n[*] Generating Task B (V2L2-Mirror) Corpora ...")
    rd.seed(SEED_TASK_B)
    char_set = ["a", "b"]
    trigger_mirror = chr(cfg.TRIGGER_MIRROR)
    build_mirror_corpus(cfg.TASKS['B']['train_corpus'], NUM_PATTERNS_TRAIN, length=2,
                        char_set=char_set, trigger=trigger_mirror)
    build_mirror_corpus(cfg.TASKS['B']['test_corpus'], NUM_PATTERNS_TEST, length=2,
                        char_set=char_set, trigger=trigger_mirror)

    # 3. Task C: V2L2-FIFO
    print("\n[*] Generating Task C (V2L2-FIFO) Corpora ...")
    rd.seed(SEED_TASK_C)
    trigger_fifo = chr(cfg.TRIGGER_FIFO)
    build_fifo_corpus(cfg.TASKS['C']['train_corpus'], NUM_PATTERNS_TRAIN, length=2,
                      char_set=char_set, trigger=trigger_fifo)
    build_fifo_corpus(cfg.TASKS['C']['test_corpus'], NUM_PATTERNS_TEST, length=2,
                      char_set=char_set, trigger=trigger_fifo)

    print("\n==========================================================================")
    print(" [*] All 6 corpora generated successfully.")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()