"""
실험 2: 정교한 시너지 보너스 실험

현재 설정 (단순 조건 기반):
    - 즉시 예약 성공: +5 (고정)
    - 추천-스케줄 매칭: +3 (고정)

개선 설정 (연속적 보상):
    - 질문 효율성: max(0, 5 - questions) (0~5점)
    - 시도 효율성: max(0, 5 - attempts) * 1.5 (0~7.5점)
    - 시간 근접도: (1 - time_diff/max_diff) * 3 (0~3점)
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.integrated_system import IntegratedSystem, train_integrated, evaluate_integrated
from src.agents.q_learning_agent import QLearningAgent
from src.agents.scheduling_agent import DQNAgent


class ImprovedSynergySystem(IntegratedSystem):
    """
    개선된 시너지 보너스를 사용하는 통합 시스템

    기존 IntegratedSystem을 상속하고,
    _calculate_synergy_bonus 메서드만 오버라이드하여
    더 정교한 보상 함수 사용.
    """

    def _calculate_synergy_bonus(
        self,
        phase1_result: dict,
        phase2_result: dict
    ) -> float:
        """
        개선된 시너지 보너스 계산

        기존: 고정 보너스 (5 + 3 = 최대 8점)
        개선: 연속적 보너스 (최대 ~15점)

        구성요소:
            1. 질문 효율성 (0~5점): 질문이 적을수록 높은 점수
            2. 시도 효율성 (0~7.5점): 스케줄링 시도가 적을수록 높은 점수
            3. 즉시 예약 보너스 (0~3점): 첫 시도 성공 시
            4. 선호 시간 매칭 (0~2점): 고객 선호 시간에 예약 성공 시
        """
        if phase2_result is None:
            return 0.0

        synergy = 0.0

        # =====================================================================
        # 1. 질문 효율성 보너스 (0~5점)
        # =====================================================================
        # 질문 수가 적을수록 높은 보너스
        # 0개 질문 = 5점, 5개 이상 = 0점
        questions_count = phase1_result.get('questions_count', 0)
        question_efficiency = max(0, 5 - questions_count)
        synergy += question_efficiency

        # =====================================================================
        # 2. 시도 효율성 보너스 (0~7.5점)
        # =====================================================================
        # 스케줄링 시도가 적을수록 높은 보너스
        # 1회 시도 = 6점, 5회 이상 = 0점
        attempt_count = phase2_result.get('attempt_count', 1)
        attempt_efficiency = max(0, 5 - attempt_count) * 1.5
        synergy += attempt_efficiency

        # =====================================================================
        # 3. 즉시 예약 보너스 (0~3점)
        # =====================================================================
        # 첫 시도에 예약 성공 시 추가 보너스
        if attempt_count == 1 and phase2_result.get('booking_success', False):
            synergy += 3.0

        # =====================================================================
        # 4. 선호 시간 매칭 보너스 (0~2점)
        # =====================================================================
        # 고객이 원하는 시간에 예약 성공 시
        if phase2_result.get('preferred_time_match', False):
            synergy += 2.0

        return synergy


def run_baseline_synergy(seed: int = 42) -> dict:
    """
    기존 시너지 보너스로 실험 (Baseline)
    """
    print("\n" + "=" * 60)
    print("🔵 기존 시너지 보너스 실험")
    print("=" * 60)
    print("시너지 구성:")
    print("  - 즉시 예약 성공: +5 (고정)")
    print("  - 추천-스케줄 매칭: +3 (고정)")
    print("  - 최대 시너지: 8점")
    print("=" * 60)

    start_time = time.time()

    # 기본 설정으로 통합 시스템 생성
    system = IntegratedSystem(seed=seed)

    # 튜닝된 하이퍼파라미터로 학습 (실험 1 결과 반영)
    history = train_integrated(
        system=system,
        phase1_pretrain=1000,
        phase2_pretrain=1000,
        n_episodes=1000,
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
    results['config'] = 'baseline_synergy'

    print(f"\n⏱️ 총 소요 시간: {elapsed:.1f}초")

    return results


def run_improved_synergy(seed: int = 42) -> dict:
    """
    개선된 시너지 보너스로 실험
    """
    print("\n" + "=" * 60)
    print("🟢 개선된 시너지 보너스 실험")
    print("=" * 60)
    print("시너지 구성:")
    print("  - 질문 효율성: max(0, 5 - questions) (0~5점)")
    print("  - 시도 효율성: max(0, 5 - attempts) * 1.5 (0~7.5점)")
    print("  - 즉시 예약 보너스: +3 (첫 시도 성공 시)")
    print("  - 선호 시간 매칭: +2 (선호 시간 예약 시)")
    print("  - 최대 시너지: ~17.5점")
    print("=" * 60)

    start_time = time.time()

    # 개선된 시너지 시스템 생성
    system = ImprovedSynergySystem(seed=seed)

    # 튜닝된 하이퍼파라미터로 학습
    history = train_integrated(
        system=system,
        phase1_pretrain=1000,
        phase2_pretrain=1000,
        n_episodes=1000,
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
    results['config'] = 'improved_synergy'

    print(f"\n⏱️ 총 소요 시간: {elapsed:.1f}초")

    return results


def compare_results(baseline: dict, improved: dict) -> dict:
    """
    두 실험 결과 비교
    """
    print("\n" + "=" * 60)
    print("📊 실험 결과 비교")
    print("=" * 60)

    metrics = [
        ('mean_total_reward', '총 보상'),
        ('end_to_end_success_rate', 'End-to-End 성공률'),
        ('preferred_time_match_rate', '선호 시간 매칭률'),
        ('mean_synergy_bonus', '시너지 보너스'),
        ('mean_questions', '평균 질문 수'),
        ('mean_attempts', '평균 스케줄링 시도'),
    ]

    comparison = {}

    print(f"\n{'지표':<25} {'기존':>12} {'개선':>12} {'변화':>12}")
    print("-" * 65)

    for key, name in metrics:
        baseline_val = baseline.get(key, 0)
        improved_val = improved.get(key, 0)

        if key in ['end_to_end_success_rate', 'preferred_time_match_rate']:
            baseline_str = f"{baseline_val * 100:.1f}%"
            improved_str = f"{improved_val * 100:.1f}%"
            diff = (improved_val - baseline_val) * 100
            diff_str = f"{diff:+.1f}%p"
        else:
            baseline_str = f"{baseline_val:.2f}"
            improved_str = f"{improved_val:.2f}"
            if baseline_val != 0:
                diff = ((improved_val - baseline_val) / abs(baseline_val)) * 100
                diff_str = f"{diff:+.1f}%"
            else:
                diff_str = "N/A"

        print(f"{name:<25} {baseline_str:>12} {improved_str:>12} {diff_str:>12}")

        comparison[key] = {
            'baseline': baseline_val,
            'improved': improved_val,
            'improvement': improved_val - baseline_val
        }

    print("-" * 65)
    print(f"{'학습 시간':<25} {baseline['training_time']:>10.1f}s {improved['training_time']:>10.1f}s")

    return comparison


def main():
    """
    시너지 보너스 개선 실험 메인 함수
    """
    print("\n" + "=" * 60)
    print("🧪 실험 2: 정교한 시너지 보너스")
    print("=" * 60)

    seed = 42

    # 1. 기존 시너지 보너스 실험
    baseline_results = run_baseline_synergy(seed=seed)

    # 2. 개선된 시너지 보너스 실험
    improved_results = run_improved_synergy(seed=seed)

    # 3. 결과 비교
    comparison = compare_results(baseline_results, improved_results)

    # 4. 결과 저장
    results_dir = project_root / "results" / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'improved_synergy_bonus',
        'baseline': baseline_results,
        'improved': improved_results,
        'comparison': comparison
    }

    output_path = results_dir / "experiment2_synergy.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ 결과 저장: {output_path}")

    # 5. 최종 요약
    print("\n" + "=" * 60)
    print("📈 최종 요약")
    print("=" * 60)

    baseline_reward = baseline_results['mean_total_reward']
    improved_reward = improved_results['mean_total_reward']
    improvement = ((improved_reward - baseline_reward) / baseline_reward) * 100

    baseline_synergy = baseline_results['mean_synergy_bonus']
    improved_synergy = improved_results['mean_synergy_bonus']
    synergy_change = ((improved_synergy - baseline_synergy) / baseline_synergy) * 100

    print(f"기존 총 보상: {baseline_reward:.2f} (시너지: {baseline_synergy:.2f})")
    print(f"개선 총 보상: {improved_reward:.2f} (시너지: {improved_synergy:.2f})")
    print(f"총 보상 변화: {improvement:+.1f}%")
    print(f"시너지 변화: {synergy_change:+.1f}%")

    if improvement > 0:
        print("\n🎉 시너지 보너스 개선으로 성능이 향상되었습니다!")
    else:
        print("\n⚠️ 추가 조정이 필요합니다.")

    return output


if __name__ == "__main__":
    results = main()
