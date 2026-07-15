# generate_test.py (逆順ミラー・コピー課題版)
import random as rd

TRAIN_PATTERNS = 400       
TEST_PATTERNS = 80         

TRAIN_FILE = "train_corpus.txt"
TEST_FILE = "test_corpus.txt"

# 💡 使用するアルファベットの種類（a〜eの5文字）
#CHAR_SET = ["a", "b", "c", "d", "e"]
CHAR_SET = ["a", "b", "c"]
TRIGGER_CHAR = "@"  # 反転のトリガー記号

# 💡 訓練とテストで、文字列の「長さ」を完全に分離（外挿テスト）
# Train: abc@cba (長さ3) | Test: abcde@edcba (長さ5)
TRAIN_MIN_LEN = 2
TRAIN_MAX_LEN = 2         # 訓練時は 1〜3文字の長さしか見せない

TEST_MIN_LEN = 2
TEST_MAX_LEN = 2         # テスト時は 4〜5文字の未知の長い反転を解かせる

def generate_mirror_pattern(length, char_set, trigger):
    """
    "abc@cba " のような、完全情報ミラーパターンを生成する
    """
    # 指定された長さのランダムな文字列を生成
    sequence = [rd.choice(char_set) for _ in range(length)]
    
    # 逆順の文字列を作成
    reversed_sequence = sequence[::-1]
    
    # ドッキングして、最後にスペース
    pattern = "".join(sequence) + trigger + "".join(reversed_sequence) + " "
    return pattern

def make_mirror_corpus(filename, num_patterns, min_len, max_len, char_set, trigger):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_patterns):
            target_len = rd.randint(min_len, max_len)
            pattern = generate_mirror_pattern(target_len, char_set, trigger)
            f.write(pattern)
                
    print(f"[*] {filename} を作成しました (長さ: {min_len}〜{max_len} | パターン数: {num_patterns})")

# データを自動生成
make_mirror_corpus(TRAIN_FILE, TRAIN_PATTERNS, TRAIN_MIN_LEN, TRAIN_MAX_LEN, CHAR_SET, TRIGGER_CHAR)
make_mirror_corpus(TEST_FILE, TEST_PATTERNS, TEST_MIN_LEN, TEST_MAX_LEN, CHAR_SET, TRIGGER_CHAR)