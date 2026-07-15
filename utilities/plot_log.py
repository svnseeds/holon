import matplotlib.pyplot as plt
import numpy as np
import re
import os

# ==========================================
# 設定：読み込むログファイルのパス
# (同じフォルダに log.txt という名前でログをコピペして保存してください)
# ==========================================
LOG_FILE = "log.txt"

def parse_logs(file_path):
    """
    ログファイルを読み込み、Epoch, ACC, 代謝(Metabolism)のデータを抽出する
    Phase2（-cによる連続学習）のEpochリセットも自動検知して通し番号にする
    """
    epochs = []
    train_acc = []
    test_acc = []
    mb_in = []
    mb_feed = []
    
    current_epoch = 0
    phase_split_epoch = None  # 外挿（Phase 2）が始まったエポックを記録

    if not os.path.exists(file_path):
        print(f"[!] {file_path} が見つかりません。テスト用のダミーデータを生成します。")
        return [1,2,3,4,5], [20,40,60,70,75], [25,35,55,65,70], [250000, 60000, 10000, 5000, 1000], [2200000, 600000, 100000, 50000, 10000], 3

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # "1/20 0.0072 23.48% ..." のような行を正規表現で探す
            match = re.search(r"^\s*(\d+)/\d+\s+[\d\.]+\s+([\d\.]+)%\s+[\d\.]+%?\s+[\d\.]+\s+([\d\.]+)%.*?\s+(\d+)\s+(\d+)\s+\d+:\d+:\d+", line)
            
            if match:
                ep_str = int(match.group(1))
                
                # Epochが1に戻ったら、Phase2（外挿）が始まったと判定
                if ep_str == 1 and current_epoch > 0:
                    phase_split_epoch = current_epoch + 1
                
                current_epoch += 1
                
                epochs.append(current_epoch)
                train_acc.append(float(match.group(2)))
                test_acc.append(float(match.group(3)))
                
                # 代謝量の取得（対数スケールで見やすくするため +1 しておく）
                mb_in.append(int(match.group(4)) + 1)
                mb_feed.append(int(match.group(5)) + 1)

    return epochs, train_acc, test_acc, mb_in, mb_feed, phase_split_epoch

def plot_holon_dynamics(epochs, train_acc, test_acc, mb_in, mb_feed, phase_split_epoch):
    """美しい学術的なグラフを描画・保存する"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- 左軸：Accuracy (折れ線グラフ) ---
    ax1.plot(epochs, train_acc, label='Train ACC', color='tab:blue', linewidth=2.5, marker='o', markersize=4)
    ax1.plot(epochs, test_acc, label='Test ACC', color='tab:orange', linewidth=2.5, marker='s', markersize=4)
    ax1.set_xlabel('Epochs (Continual Learning Timeline)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='black')
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='black')

    # --- 右軸：Metabolism (棒グラフ・対数スケール) ---
    ax2 = ax1.twinx()
    # Mb_feed (水平結合の代謝) を背面に描画
    ax2.bar(epochs, mb_feed, label='Synaptic Metabolism (W_feed)', color='tab:green', alpha=0.3, width=0.6)
    ax2.set_ylabel('Synaptic Metabolism Count (Log Scale)', fontsize=12, fontweight='bold', color='tab:green')
    ax2.set_yscale('log')  # 代謝は爆発するので対数スケールが美しい
    ax2.set_ylim(1, max(mb_feed) * 10)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # --- 外挿フェーズの境界線 (Vertical Line) ---
    if phase_split_epoch is not None:
        ax1.axvline(x=phase_split_epoch - 0.5, color='red', linestyle='--', linewidth=2)
        # テキストの追加
        ax1.text(phase_split_epoch, 10, ' Extrapolation\n (OOD Phase 2)', color='red', fontsize=11, fontweight='bold', verticalalignment='bottom')
        ax1.text(1, 10, ' Base Training\n (Phase 1)', color='black', fontsize=11, fontweight='bold', verticalalignment='bottom')

    # --- タイトルと凡例 ---
    plt.title('HOLON Dynamics: Continual Learning & Synaptic Metabolism', fontsize=15, fontweight='bold')
    
    # 凡例をまとめる
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('holon_dynamics.png', dpi=300)
    print("[*] グラフを 'holon_dynamics.png' として保存しました！")
    plt.show()

if __name__ == "__main__":
    epochs, train_acc, test_acc, mb_in, mb_feed, phase_split_epoch = parse_logs(LOG_FILE)
    if len(epochs) > 0:
        plot_holon_dynamics(epochs, train_acc, test_acc, mb_in, mb_feed, phase_split_epoch)