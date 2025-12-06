"""
실험 1: 하이퍼파라미터 튜닝 실험

현재 설정 vs 개선 설정 비교:
- Phase 1 사전학습: 300 → 1000 에피소드
- Phase 2 사전학습: 300 → 1000 에피소드
- 통합 학습: 500 → 1000 에피소드
- DQN Hidden Dim: 128 → 256
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.integrated_system import IntegratedSystem, train_integrated, evaluate_integrated
from src.agents.q_learning_agent import QLearningAgent
from src.agents.scheduling_agent import DQNAgent


def run_baseline_config(seed: int = 42) -> dict:
    """
    기존 설정으로 실험 (Baseline)

    설정:
        - Phase 1 사전학습: 300 에피소드
        - Phase 2 사전학습: 300 에피소드
        - 통합 학습: 500 에피소드
        - DQN Hidden Dim: 128
    """
    print("\n" + "=" * 60)
    print("🔵 Baseline 설정 실험")
    print("=" * 60)
    print("Phase 1 사전학습: 300 에피소드")
    print("Phase 2 사전학습: 300 에피소드")
    print("통합 학습: 500 에피소드")
    print("DQN Hidden Dim: 128")
    print("=" * 60)

    start_time = time.time()

    # 기본 설정으로 통합 시스템 생성
    system = IntegratedSystem(seed=seed)

    # 학습 실행
    history = train_integrated(
        system=system,
        phase1_pretrain=300,
        phase2_pretrain=300,
        n_episodes=500,
        verbose=True
    )

    # 평가 실행
    results = evaluate_integrated(
        system=system,
        n_episodes=100,
        verbose=False
    )

    elapsed = time.time() - start_time
    results['training_time'] = elapsed
    results['config'] = 'baseline'

    print(f"\n⏱️ 총 소요 시간: {elapsed:.1f}초")

    return results


def run_tuned_config(seed: int = 42) -> dict:
    """
    튜닝된 설정으로 실험

    설정:
        - Phase 1 사전학습: 1000 에피소드
        - Phase 2 사전학습: 1000 에피소드
        - 통합 학습: 1000 에피소드
        - DQN Hidden Dim: 256
    """
    print("\n" + "=" * 60)
    print("🟢 튜닝된 설정 실험")
    print("=" * 60)
    print("Phase 1 사전학습: 1000 에피소드")
    print("Phase 2 사전학습: 1000 에피소드")
    print("통합 학습: 1000 에피소드")
    print("DQN Hidden Dim: 256")
    print("=" * 60)

    start_time = time.time()

    # 튜닝된 DQN 에이전트 생성 (Hidden Dim 256)
    tuned_phase2_agent = DQNAgent(
        state_dim=159,
        action_dim=6,
        hidden_dim=256,  # 128 → 256으로 증가
        learning_rate=1e-3,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10,
        seed=seed
    )

    # 통합 시스템 생성 (튜닝된 에이전트 사용)
    system = IntegratedSystem(
        phase2_agent=tuned_phase2_agent,
        seed=seed
    )

    # 학습 실행 (증가된 에피소드)
    history = train_integrated(
        system=system,
        phase1_pretrain=1000,   # 300 → 1000
        phase2_pretrain=1000,   # 300 → 1000
        n_episodes=1000,        # 500 → 1000
        verbose=True
    )

    # 평가 실행
    results = evaluate_integrated(
        system=system,
        n_episodes=100,
        verbose=False
    )

    elapsed = time.time() - start_time
    results['training_time'] = elapsed
    results['config'] = 'tuned'

    print(f"\n⏱️ 총 소요 시간: {elapsed:.1f}초")

    return results


def compare_results(baseline: dict, tuned: dict) -> dict:
    """
    두 실험 결과 비교
    """
    print("\n" + "=" * 60)
    print("📊 실험 결과 비교")
    print("=" * 60)

    # evaluate_integrated 반환값과 일치하는 키 이름 사용
    metrics = [
        ('mean_total_reward', '총 보상'),
        ('end_to_end_success_rate', 'End-to-End 성공률'),
        ('preferred_time_match_rate', '선호 시간 매칭률'),
        ('mean_synergy_bonus', '시너지 보너스'),
        ('mean_questions', '평균 질문 수'),
        ('mean_attempts', '평균 스케줄링 시도'),
    ]

    comparison = {}

    print(f"\n{'지표':<25} {'Baseline':>12} {'Tuned':>12} {'개선':>12}")
    print("-" * 65)

    for key, name in metrics:
        baseline_val = baseline.get(key, 0)
        tuned_val = tuned.get(key, 0)

        if key in ['end_to_end_success_rate', 'preferred_time_match_rate']:
            # 백분율로 표시
            baseline_str = f"{baseline_val * 100:.1f}%"
            tuned_str = f"{tuned_val * 100:.1f}%"
            diff = (tuned_val - baseline_val) * 100
            diff_str = f"{diff:+.1f}%p"
        else:
            baseline_str = f"{baseline_val:.2f}"
            tuned_str = f"{tuned_val:.2f}"
            if baseline_val != 0:
                diff = ((tuned_val - baseline_val) / abs(baseline_val)) * 100
                diff_str = f"{diff:+.1f}%"
            else:
                diff_str = "N/A"

        print(f"{name:<25} {baseline_str:>12} {tuned_str:>12} {diff_str:>12}")

        comparison[key] = {
            'baseline': baseline_val,
            'tuned': tuned_val,
            'improvement': tuned_val - baseline_val
        }

    # 학습 시간 비교
    print("-" * 65)
    print(f"{'학습 시간':<25} {baseline['training_time']:>10.1f}s {tuned['training_time']:>10.1f}s")

    return comparison


def main():
    """
    하이퍼파라미터 튜닝 실험 메인 함수
    """
    print("\n" + "=" * 60)
    print("🧪 실험 1: 하이퍼파라미터 튜닝")
    print("=" * 60)

    seed = 42

    # 1. Baseline 실험
    baseline_results = run_baseline_config(seed=seed)

    # 2. 튜닝된 설정 실험
    tuned_results = run_tuned_config(seed=seed)

    # 3. 결과 비교
    comparison = compare_results(baseline_results, tuned_results)

    # 4. 결과 저장
    results_dir = project_root / "results" / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'hyperparameter_tuning',
        'baseline': baseline_results,
        'tuned': tuned_results,
        'comparison': comparison
    }

    output_path = results_dir / "experiment1_hyperparams.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ 결과 저장: {output_path}")

    # 5. 최종 요약
    print("\n" + "=" * 60)
    print("📈 최종 요약")
    print("=" * 60)

    baseline_reward = baseline_results['mean_total_reward']
    tuned_reward = tuned_results['mean_total_reward']
    improvement = ((tuned_reward - baseline_reward) / baseline_reward) * 100

    print(f"Baseline 총 보상: {baseline_reward:.2f}")
    print(f"Tuned 총 보상: {tuned_reward:.2f}")
    print(f"개선율: {improvement:+.1f}%")

    if improvement > 0:
        print("\n🎉 하이퍼파라미터 튜닝으로 성능이 개선되었습니다!")
    else:
        print("\n⚠️ 추가 튜닝이 필요합니다.")

    return output


if __name__ == "__main__":
    results = main()
