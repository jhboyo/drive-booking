"""
Phase 3 결과 시각화 스크립트

통합 시스템 학습 결과를 시각화하여 저장함.

생성되는 그래프:
    1. 학습 곡선 (Phase 1, Phase 2, 통합 학습)
    2. 성능 비교 (베이스라인 vs Phase 3)
    3. 시너지 보너스 분석
    4. End-to-End 성공률 추이
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_learning_curves(history: dict, save_path: Path):
    """
    학습 곡선 시각화

    3단계 학습 과정의 보상 변화를 시각화함.

    Args:
        history: 학습 히스토리 딕셔너리
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === 1. Phase 1 사전 학습 ===
    ax1 = axes[0, 0]
    rewards = history.get('phase1_pretrain_rewards', [])
    if rewards:
        # 이동 평균 계산
        window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
        ax1.plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Phase 1 사전 학습 (Q-Learning)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === 2. Phase 2 사전 학습 ===
    ax2 = axes[0, 1]
    rewards = history.get('phase2_pretrain_rewards', [])
    if rewards:
        window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax2.plot(rewards, alpha=0.3, color='green', label='Raw')
        ax2.plot(range(window-1, len(rewards)), smoothed, color='green', linewidth=2, label='Smoothed')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Phase 2 사전 학습 (DQN)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # === 3. 통합 학습 ===
    ax3 = axes[1, 0]
    rewards = history.get('integrated_rewards', [])
    if rewards:
        window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax3.plot(rewards, alpha=0.3, color='red', label='Raw')
        ax3.plot(range(window-1, len(rewards)), smoothed, color='red', linewidth=2, label='Smoothed')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Reward (R1 + R2 + Synergy)')
    ax3.set_title('Phase 3 통합 학습')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === 4. End-to-End 성공률 ===
    ax4 = axes[1, 1]
    success = history.get('end_to_end_success', [])
    if success:
        # 이동 평균 성공률
        window = min(100, len(success) // 5) if len(success) > 10 else 1
        smoothed = np.convolve(success, np.ones(window)/window, mode='valid')
        ax4.plot(range(window-1, len(success)), smoothed * 100, color='purple', linewidth=2)
        ax4.axhline(y=np.mean(success) * 100, color='gray', linestyle='--', label=f'Mean: {np.mean(success)*100:.1f}%')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('End-to-End 성공률 추이')
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'phase3_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path / 'phase3_learning_curves.png'}")


def plot_performance_comparison(eval_results: dict, baseline_results: dict, save_path: Path):
    """
    성능 비교 시각화

    Phase 3 통합 시스템과 베이스라인 비교.

    Args:
        eval_results: Phase 3 평가 결과
        baseline_results: 베이스라인 평가 결과들
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    agents = ['Random', 'Individual\n(No Synergy)', 'Phase 3\n(Integrated)']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    # === 1. 총 보상 비교 ===
    ax1 = axes[0]
    rewards = [
        baseline_results.get('random', {}).get('mean_total_reward', 0),
        baseline_results.get('individual', {}).get('mean_total_reward', 0),
        eval_results.get('mean_total_reward', 0)
    ]
    stds = [
        baseline_results.get('random', {}).get('std_total_reward', 0),
        baseline_results.get('individual', {}).get('std_total_reward', 0),
        eval_results.get('std_total_reward', 0)
    ]
    bars = ax1.bar(agents, rewards, color=colors, yerr=stds, capsize=5)
    ax1.set_ylabel('Total Reward')
    ax1.set_title('총 보상 비교')
    ax1.grid(True, alpha=0.3, axis='y')
    # 값 표시
    for bar, reward in zip(bars, rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{reward:.1f}', ha='center', va='bottom', fontsize=10)

    # === 2. 성공률 비교 ===
    ax2 = axes[1]
    success_rates = [
        baseline_results.get('random', {}).get('end_to_end_success_rate', 0) * 100,
        baseline_results.get('individual', {}).get('end_to_end_success_rate', 0) * 100,
        eval_results.get('end_to_end_success_rate', 0) * 100
    ]
    bars = ax2.bar(agents, success_rates, color=colors)
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('End-to-End 성공률')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    # === 3. 효율성 비교 (상호작용 횟수) ===
    ax3 = axes[2]
    interactions = [
        baseline_results.get('random', {}).get('mean_total_interactions', 0),
        baseline_results.get('individual', {}).get('mean_total_interactions', 0),
        eval_results.get('mean_total_interactions', 0)
    ]
    bars = ax3.bar(agents, interactions, color=colors)
    ax3.set_ylabel('Total Interactions')
    ax3.set_title('평균 상호작용 횟수')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, inter in zip(bars, interactions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{inter:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path / 'phase3_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path / 'phase3_performance_comparison.png'}")


def plot_synergy_analysis(history: dict, eval_results: dict, save_path: Path):
    """
    시너지 보너스 분석 시각화

    Args:
        history: 학습 히스토리
        eval_results: 평가 결과
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # === 1. 시너지 보너스 히스토리 ===
    ax1 = axes[0]
    synergy = history.get('synergy_bonuses', [])
    if synergy:
        window = min(50, len(synergy) // 5) if len(synergy) > 10 else 1
        smoothed = np.convolve(synergy, np.ones(window)/window, mode='valid')
        ax1.plot(synergy, alpha=0.3, color='orange', label='Raw')
        ax1.plot(range(window-1, len(synergy)), smoothed, color='orange', linewidth=2, label='Smoothed')
        ax1.axhline(y=np.mean(synergy), color='red', linestyle='--', label=f'Mean: {np.mean(synergy):.2f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Synergy Bonus')
    ax1.set_title('시너지 보너스 추이')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === 2. 보상 구성 분석 (파이 차트) ===
    ax2 = axes[1]
    p1_reward = eval_results.get('mean_phase1_reward', 0)
    p2_reward = eval_results.get('mean_phase2_reward', 0)
    synergy_bonus = eval_results.get('mean_synergy_bonus', 0)

    labels = ['Phase 1\n(추천)', 'Phase 2\n(스케줄링)', 'Synergy\n(통합 보너스)']
    sizes = [max(0, p1_reward), max(0, p2_reward), max(0, synergy_bonus)]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    explode = (0, 0, 0.1)  # 시너지 강조

    if sum(sizes) > 0:
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax2.set_title(f'총 보상 구성\n(Total: {sum(sizes):.2f})')
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path / 'phase3_synergy_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path / 'phase3_synergy_analysis.png'}")


def plot_detailed_metrics(eval_results: dict, save_path: Path):
    """
    세부 지표 시각화

    Args:
        eval_results: 평가 결과
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # === 1. 보상 분포 ===
    ax1 = axes[0, 0]
    metrics = ['Phase 1\nReward', 'Phase 2\nReward', 'Synergy\nBonus', 'Total\nReward']
    values = [
        eval_results.get('mean_phase1_reward', 0),
        eval_results.get('mean_phase2_reward', 0),
        eval_results.get('mean_synergy_bonus', 0),
        eval_results.get('mean_total_reward', 0)
    ]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    bars = ax1.bar(metrics, values, color=colors)
    ax1.set_ylabel('Reward')
    ax1.set_title('평균 보상 분포')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # === 2. 성공률 지표 ===
    ax2 = axes[0, 1]
    metrics = ['End-to-End\n성공률', '선호 시간\n매칭률']
    values = [
        eval_results.get('end_to_end_success_rate', 0) * 100,
        eval_results.get('preferred_time_match_rate', 0) * 100
    ]
    colors = ['#9467bd', '#8c564b']
    bars = ax2.bar(metrics, values, color=colors)
    ax2.set_ylabel('Rate (%)')
    ax2.set_title('성공률 지표')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # === 3. 효율성 지표 ===
    ax3 = axes[1, 0]
    metrics = ['평균\n질문 수', '평균\n스케줄링 시도', '총\n상호작용']
    values = [
        eval_results.get('mean_questions', 0),
        eval_results.get('mean_attempts', 0),
        eval_results.get('mean_total_interactions', 0)
    ]
    colors = ['#17becf', '#bcbd22', '#7f7f7f']
    bars = ax3.bar(metrics, values, color=colors)
    ax3.set_ylabel('Count')
    ax3.set_title('효율성 지표')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # === 4. 종합 요약 (텍스트) ===
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    ═══════════════════════════════════════
    Phase 3 통합 시스템 평가 요약
    ═══════════════════════════════════════

    [보상]
      총 보상: {eval_results.get('mean_total_reward', 0):.2f} ± {eval_results.get('std_total_reward', 0):.2f}
      Phase 1 보상: {eval_results.get('mean_phase1_reward', 0):.2f}
      Phase 2 보상: {eval_results.get('mean_phase2_reward', 0):.2f}
      시너지 보너스: {eval_results.get('mean_synergy_bonus', 0):.2f}

    [성공률]
      End-to-End 성공률: {eval_results.get('end_to_end_success_rate', 0)*100:.1f}%
      선호 시간 매칭률: {eval_results.get('preferred_time_match_rate', 0)*100:.1f}%

    [효율성]
      평균 질문 수: {eval_results.get('mean_questions', 0):.2f}
      평균 스케줄링 시도: {eval_results.get('mean_attempts', 0):.2f}
      총 상호작용: {eval_results.get('mean_total_interactions', 0):.2f}

    ═══════════════════════════════════════
    """
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path / 'phase3_detailed_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {save_path / 'phase3_detailed_metrics.png'}")


def run_and_visualize():
    """
    Phase 3 학습 실행 및 시각화
    """
    from src.integrated_system import IntegratedSystem, train_integrated, evaluate_integrated
    from src.evaluate_phase3 import evaluate_random_baseline, evaluate_individual_phases

    print("=" * 70)
    print("Phase 3: 통합 시스템 학습 및 시각화")
    print("=" * 70)

    # 결과 저장 디렉토리 생성
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. 통합 시스템 학습
    # =========================================================================
    print("\n[1/4] 통합 시스템 학습 중...")
    system = IntegratedSystem(seed=42)

    history = train_integrated(
        system=system,
        n_episodes=500,        # 통합 학습
        phase1_pretrain=300,   # Phase 1 사전 학습
        phase2_pretrain=300,   # Phase 2 사전 학습
        log_interval=100,
        verbose=True
    )

    # =========================================================================
    # 2. Phase 3 평가
    # =========================================================================
    print("\n[2/4] Phase 3 평가 중...")
    eval_results = evaluate_integrated(
        system=system,
        n_episodes=100,
        verbose=True
    )

    # =========================================================================
    # 3. 베이스라인 평가
    # =========================================================================
    print("\n[3/4] 베이스라인 평가 중...")

    print("  Random Baseline 평가...")
    random_results = evaluate_random_baseline(n_episodes=100, seed=42)

    print("  Individual Phases 평가...")
    individual_results = evaluate_individual_phases(n_episodes=100, seed=42)

    baseline_results = {
        'random': random_results,
        'individual': individual_results
    }

    # =========================================================================
    # 4. 시각화
    # =========================================================================
    print("\n[4/4] 결과 시각화 중...")

    # 학습 곡선
    plot_learning_curves(history, figures_dir)

    # 성능 비교
    plot_performance_comparison(eval_results, baseline_results, figures_dir)

    # 시너지 분석
    plot_synergy_analysis(history, eval_results, figures_dir)

    # 세부 지표
    plot_detailed_metrics(eval_results, figures_dir)

    # =========================================================================
    # 결과 저장 (JSON)
    # =========================================================================
    all_results = {
        "training_history": {
            "phase1_pretrain_rewards": [float(r) for r in history['phase1_pretrain_rewards']],
            "phase2_pretrain_rewards": [float(r) for r in history['phase2_pretrain_rewards']],
            "integrated_rewards": [float(r) for r in history['integrated_rewards']],
            "synergy_bonuses": [float(r) for r in history['synergy_bonuses']],
            "end_to_end_success": history['end_to_end_success']
        },
        "evaluation": eval_results,
        "baselines": {
            "random": random_results,
            "individual": individual_results
        }
    }

    results_path = results_dir / "phase3_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {results_path}")

    # =========================================================================
    # 최종 요약
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 3 실험 완료!")
    print("=" * 70)

    print(f"\n[저장된 파일]")
    print(f"  - {figures_dir / 'phase3_learning_curves.png'}")
    print(f"  - {figures_dir / 'phase3_performance_comparison.png'}")
    print(f"  - {figures_dir / 'phase3_synergy_analysis.png'}")
    print(f"  - {figures_dir / 'phase3_detailed_metrics.png'}")
    print(f"  - {results_path}")

    # 개선율 계산
    random_reward = random_results['mean_total_reward']
    phase3_reward = eval_results['mean_total_reward']
    improvement = ((phase3_reward - random_reward) / abs(random_reward)) * 100 if random_reward != 0 else 0

    print(f"\n[핵심 결과]")
    print(f"  총 보상: {eval_results['mean_total_reward']:.2f}")
    print(f"  End-to-End 성공률: {eval_results['end_to_end_success_rate']*100:.1f}%")
    print(f"  시너지 보너스: {eval_results['mean_synergy_bonus']:.2f}")
    print(f"  Random 대비 개선율: {improvement:+.1f}%")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    run_and_visualize()
