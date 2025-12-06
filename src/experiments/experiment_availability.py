"""
실험 3: Phase 1에 가용성 정보 추가

현재 Phase 1 상태 (69차원):
    - 고객 프로필 + 질문 응답 + 차량 점수

개선 Phase 1 상태 (92차원):
    - 고객 프로필 + 질문 응답 + 차량 점수 + 차량별 가용성(23차원)

이 개선으로 Phase 1이 가용한 차량을 우선 추천하도록 유도.
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
from src.env.recommendation_env import VehicleRecommendationEnv
from src.env.scheduling_env import SchedulingEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.scheduling_agent import DQNAgent


class AvailabilityAwareSystem(IntegratedSystem):
    """
    가용성 정보를 Phase 1에 전달하는 통합 시스템

    핵심 변경:
        1. Phase 2 환경에서 차량 가용성 정보 추출
        2. Phase 1 상태에 가용성 정보 추가
        3. Phase 1이 가용한 차량을 우선 추천하도록 유도
    """

    def __init__(
        self,
        phase1_agent: Optional[QLearningAgent] = None,
        phase2_agent: Optional[DQNAgent] = None,
        seed: int = 42
    ):
        """초기화"""
        super().__init__(phase1_agent, phase2_agent, seed)

        # 차량 수 (가용성 정보 차원)
        self.n_vehicles = 23

    def _get_vehicle_availability(self) -> np.ndarray:
        """
        Phase 2 환경에서 차량 가용성 정보 추출

        Returns:
            차량별 가용성 (23차원, 0~1 범위)
            1.0: 가용 가능, 0.0: 가용 불가
        """
        # Phase 2 환경 초기화하여 가용성 확인
        self.phase2_env.reset()

        # 각 차량의 가용 슬롯 비율 계산
        # schedule_state: (21일, 9슬롯, 23차량)
        availability = np.zeros(self.n_vehicles)

        for vid in range(self.n_vehicles):
            # 해당 차량의 가용 슬롯 수 (schedule_state > 0이면 가용)
            available_slots = np.sum(self.phase2_env.schedule_state[:, :, vid] > 0)
            total_slots = self.phase2_env.schedule_state.shape[0] * self.phase2_env.schedule_state.shape[1]
            availability[vid] = available_slots / total_slots

        return availability

    def _run_phase1(
        self,
        training: bool = False,
        customer_idx: Optional[int] = None
    ) -> dict:
        """
        가용성 정보를 포함하여 Phase 1 실행 (부모 메서드 오버라이드)

        Phase 1 추천 시 차량 가용성 정보를 보상에 반영하여
        가용한 차량을 우선 추천하도록 유도.
        """
        # 차량 가용성 정보 추출
        availability = self._get_vehicle_availability()

        # Phase 1 환경 초기화
        options = {'customer_idx': customer_idx} if customer_idx is not None else None
        obs, info = self.phase1_env.reset(
            seed=self.seed + self.episode_count,
            options=options
        )

        episode_reward = 0.0
        steps = 0
        done = False

        while not done:
            # 액션 선택
            action = self.phase1_agent.select_action(obs, training=training)

            # 환경 스텝
            next_obs, reward, terminated, truncated, info = self.phase1_env.step(action)

            # 추천 액션인 경우 가용성 보너스 추가
            if action >= 8:  # 추천 액션 (8-11)
                top_candidates = info.get('top_candidates', [])
                if top_candidates:
                    vehicle_name = top_candidates[0][0]
                    # 차량 인덱스 찾기
                    vehicle_idx = self._get_vehicle_index(vehicle_name)
                    if vehicle_idx is not None and vehicle_idx < self.n_vehicles:
                        # 가용성 높은 차량 추천 시 보너스
                        availability_bonus = availability[vehicle_idx] * 2.0
                        reward += availability_bonus

            # 학습 모드에서 Q-table 업데이트
            if training:
                self.phase1_agent.update(obs, action, reward, next_obs, terminated, truncated)

            obs = next_obs
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # 에피소드 종료 처리
        if training:
            self.phase1_agent.end_episode()

        # 추천 차량 추출
        top_candidates = info.get('top_candidates', [])
        recommended_vehicle = top_candidates[0][0] if top_candidates else None
        vehicle_id = self._get_vehicle_id(recommended_vehicle)

        return {
            'success': terminated,
            'reward': episode_reward,
            'questions_count': steps - 1 if steps > 0 else 0,  # 마지막 추천 제외
            'recommended_vehicle': vehicle_id,
            'vehicle_name': recommended_vehicle,
            'customer_satisfaction': info.get('customer_satisfaction', 0),
            'steps': steps
        }

    def _get_vehicle_index(self, vehicle_name: str) -> Optional[int]:
        """차량 이름으로 인덱스 반환"""
        if vehicle_name is None:
            return None
        vehicles = self.phase1_env.vehicles
        for idx, v in enumerate(vehicles):
            if v['name'] == vehicle_name:
                return idx
        return None

    def _calculate_synergy_bonus(
        self,
        phase1_result: dict,
        phase2_result: dict
    ) -> float:
        """
        개선된 시너지 보너스 (실험 2의 결과 반영)
        """
        if phase2_result is None:
            return 0.0

        synergy = 0.0

        # 1. 질문 효율성 보너스 (0~5점)
        questions_count = phase1_result.get('questions_count', 0)
        synergy += max(0, 5 - questions_count)

        # 2. 시도 효율성 보너스 (0~7.5점)
        attempt_count = phase2_result.get('attempt_count', 1)
        synergy += max(0, 5 - attempt_count) * 1.5

        # 3. 즉시 예약 보너스 (0~3점)
        if attempt_count == 1 and phase2_result.get('booking_success', False):
            synergy += 3.0

        # 4. 선호 시간 매칭 보너스 (0~2점)
        if phase2_result.get('preferred_time_match', False):
            synergy += 2.0

        # 5. 가용성 활용 보너스 (0~2점) - 새로 추가
        # Phase 1이 가용성 정보를 활용했고, 예약이 성공하면 보너스
        if phase1_result.get('availability_used', False) and phase2_result.get('booking_success', False):
            synergy += 2.0

        return synergy


def run_baseline(seed: int = 42) -> dict:
    """
    기존 방식 (가용성 정보 없음)
    """
    print("\n" + "=" * 60)
    print("🔵 기존 방식 (가용성 정보 없음)")
    print("=" * 60)
    print("Phase 1 상태: 69차원 (고객 프로필 + 질문 응답 + 차량 점수)")
    print("가용성 반영: 없음")
    print("=" * 60)

    start_time = time.time()

    system = IntegratedSystem(seed=seed)

    history = train_integrated(
        system=system,
        phase1_pretrain=1000,
        phase2_pretrain=1000,
        n_episodes=1000,
        verbose=True
    )

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


def run_availability_aware(seed: int = 42) -> dict:
    """
    개선된 방식 (가용성 정보 활용)
    """
    print("\n" + "=" * 60)
    print("🟢 개선된 방식 (가용성 정보 활용)")
    print("=" * 60)
    print("Phase 1 상태: 69차원 + 가용성 보너스")
    print("가용성 반영: 추천 시 가용한 차량에 보너스 부여")
    print("추가 시너지: 가용성 활용 보너스 (+2점)")
    print("=" * 60)

    start_time = time.time()

    system = AvailabilityAwareSystem(seed=seed)

    history = train_integrated(
        system=system,
        phase1_pretrain=1000,
        phase2_pretrain=1000,
        n_episodes=1000,
        verbose=True
    )

    results = evaluate_integrated(
        system=system,
        n_episodes=100,
        verbose=False
    )

    elapsed = time.time() - start_time
    results['training_time'] = elapsed
    results['config'] = 'availability_aware'

    print(f"\n⏱️ 총 소요 시간: {elapsed:.1f}초")

    return results


def compare_results(baseline: dict, improved: dict) -> dict:
    """결과 비교"""
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
            'improved': improved_val
        }

    print("-" * 65)
    print(f"{'학습 시간':<25} {baseline['training_time']:>10.1f}s {improved['training_time']:>10.1f}s")

    return comparison


def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("🧪 실험 3: Phase 1에 가용성 정보 추가")
    print("=" * 60)

    seed = 42

    # 1. 기존 방식 실험
    baseline_results = run_baseline(seed=seed)

    # 2. 가용성 정보 활용 실험
    improved_results = run_availability_aware(seed=seed)

    # 3. 결과 비교
    comparison = compare_results(baseline_results, improved_results)

    # 4. 결과 저장
    results_dir = project_root / "results" / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'phase1_availability',
        'baseline': baseline_results,
        'improved': improved_results,
        'comparison': comparison
    }

    output_path = results_dir / "experiment3_availability.json"
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

    baseline_success = baseline_results['end_to_end_success_rate']
    improved_success = improved_results['end_to_end_success_rate']

    print(f"기존 총 보상: {baseline_reward:.2f} (성공률: {baseline_success*100:.1f}%)")
    print(f"개선 총 보상: {improved_reward:.2f} (성공률: {improved_success*100:.1f}%)")
    print(f"총 보상 변화: {improvement:+.1f}%")
    print(f"성공률 변화: {(improved_success-baseline_success)*100:+.1f}%p")

    if improvement > 0:
        print("\n🎉 가용성 정보 활용으로 성능이 향상되었습니다!")
    else:
        print("\n⚠️ 추가 조정이 필요합니다.")

    return output


if __name__ == "__main__":
    results = main()
