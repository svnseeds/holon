# generate_brackets.py (モンテカルロ棄却サンプリング版)
import random as rd

TRAIN_PATTERNS = 400       
TEST_PATTERNS = 80         

TRAIN_FILE = "train_corpus.txt"
TEST_FILE = "test_corpus.txt"

PAIRS = [
    ("(", ")"),
    ("[", "]"),
#    ("{", "}"),
#    ("<", ">")
]

TRAIN_MIN_DEPTH = 3
TRAIN_MAX_DEPTH = 3         

TEST_MIN_DEPTH = 3
TEST_MAX_DEPTH = 3          

def generate_random_walk_bracket(max_depth, pairs):
    """
    純粋な 50/50 のランダムウォーク。
    生成と同時に、そのパターンが「到達した最大の深さ（max_reached_depth）」も記録して返す。
    """
    stack = []
    result = ""
    max_reached_depth = 1  # 記録用
    
    open_char, close_char = rd.choice(pairs)
    stack.append(close_char)
    result += open_char
    
    while len(stack) > 0:
        current_depth = len(stack)
        
        # 今回のウォークでの最大深度を更新
        if current_depth > max_reached_depth:
            max_reached_depth = current_depth
            
        if current_depth >= max_depth:
            # これ以上は潜れないので強制的に閉じる
            char = stack.pop()
            result += char
        else:
            # 💡【純粋確率】50%で開き、50%で閉じる
            if rd.random() < 0.5:
                open_char, close_char = rd.choice(pairs)
                stack.append(close_char)
                result += open_char
            else:
                char = stack.pop()
                result += char
                
    return result, max_reached_depth

def make_bracket_corpus(filename, num_patterns, min_depth, max_depth, pairs):
    with open(filename, "w", encoding="utf-8") as f:
        accepted_count = 0
        
        while accepted_count < num_patterns:
            target_max = rd.randint(min_depth, max_depth)
            
            # 純粋なランダムウォークでパターンを生成
            pattern, reached_depth = generate_random_walk_bracket(target_max, pairs)
            
            # 💡【モンテカルロ棄却法】
            # 生成されたパターンの「実際の最大深度」が、要求された min_depth を満たしているかチェック
            # 満たしていなければ、破棄してもう一度作り直す（確率は一切歪めない！）
            if min_depth <= reached_depth <= max_depth:
                if len(pattern) > 0:
                    f.write(pattern + " ")
                    accepted_count += 1
                
    print(f"[*] {filename} を作成しました (到達深度: {min_depth}〜{max_depth} | パターン数: {num_patterns})")

# データを自動生成
make_bracket_corpus(TRAIN_FILE, TRAIN_PATTERNS, TRAIN_MIN_DEPTH, TRAIN_MAX_DEPTH, PAIRS)
make_bracket_corpus(TEST_FILE, TEST_PATTERNS, TEST_MIN_DEPTH, TEST_MAX_DEPTH, PAIRS)