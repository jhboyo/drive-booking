"""
Phase 3 평가 스크립트

통합 시스템(Phase 1 + Phase 2)의 성능을 평가하고 베이스라인과 비교함.
학습된 모델의 최종 성능을 다양한 관점에서 분석하기 위한 스크립트임.

평가 지표:
    - End-to-End 성공률: 추천 → 스케줄링 → 예약 확정까지 전체 성공
    - 총 보상: R1 + R2 + Synergy Bonus
    - 효율성: 총 상호작용 횟수 (질문 수 + 스케줄링 시도)
    - 선호 시간 매칭률: 고객이 원하는 시간에 예약 성공

베이스라인 비교:
    - Random Baseline: 모든 액션을 무작위로 선택
    - Individual Phases: 시너지 보너스 없이 개별 Phase만 순차 실행

사용법:
    # 기본 평가
    python -m src.evaluate_phase3

    # 저장된 모델로 평가
    python -m src.evaluate_phase3 --model-dir checkpoints/integrated

    # 베이스라인 비교 포함
    python -m src.evaluate_phase3 --compare-baselines --save-results
"""

import argparse
import json
from pathlib import Path

import numpy as np

# 통합 시스템 및 평가 함수
from src.integrated_system import IntegratedSystem, evaluate_integrated


def parse_args():
    """
    커맨드라인 인자 파싱

    평가 실험에 필요한 설정을 커맨드라인으로 받음.

    Returns:
        파싱된 인자 객체
    """
    parser = argparse.ArgumentParser(
        description="Phase 3 통합 시스템 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m src.evaluate_phase3                        # 기본 평가
  python -m src.evaluate_phase3 --model-dir checkpoints/    # 저장된 모델 평가
  python -m src.evaluate_phase3 --compare-baselines    # 베이스라인 비교
        """
    )

    # =========================================================================
    # 평가 설정
    # =========================================================================
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="평가 에피소드 수 (기본값: 100). 통계적으로 유의한 결과를 위해 충분한 수 필요."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본값: 42). 재현 가능한 평가를 위해 사용."
    )

    # =========================================================================
    # 모델 로드 옵션
    # =========================================================================
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="학습된 모델 디렉토리 경로. None이면 새로 학습 후 평가."
    )
    parser.add_argument(
        "--train-episodes", type=int, default=1000,
        help="모델이 없을 때 학습 에피소드 수 (기본값: 1000)"
    )

    # =========================================================================
    # 결과 저장 옵션
    # =========================================================================
    parser.add_argument(
        "--save-results", action="store_true",
        help="평가 결과를 JSON으로 저장"
    )
    parser.add_argument(
        "--compare-baselines", action="store_true",
        help="베이스라인과 성능 비교 (Random, Individual Phases)"
    )

    return parser.parse_args()


def evaluate_random_baseline(n_episodes: int = 100, seed: int = 42) -> dict:
    """
    랜덤 베이스라인 평가

    Phase 1, Phase 2 모두 랜덤 액션을 선택하는 가장 단순한 베이스라인.
    강화학습 에이전트의 성능 개선을 입증하기 위한 기준점.

    랜덤 정책:
        - Phase 1: 8개 질문 + 4개 추천 중 무작위 선택
        - Phase 2: 6개 액션 중 무작위 선택

    Args:
        n_episodes: 평가 에피소드 수
        seed: 랜덤 시드

    Returns:
        평가 결과 딕셔너리
    """
    # Phase 1, Phase 2 환경 import
    from src.env.recommendation_env import VehicleRecommendationEnv
    from src.env.scheduling_env import SchedulingEnv

    # 랜덤 생성기 초기화
    rng = np.random.default_rng(seed)

    # 결과 수집용
    results = {
        'total_rewards': [],
        'end_to_end_success': [],
        'total_interactions': []
    }

    # 에피소드 반복
    for ep in range(n_episodes):
        # =====================================================================
        # Phase 1: 랜덤 추천
        # =====================================================================
        env1 = VehicleRecommendationEnv()
        obs, info = env1.reset(seed=seed + ep)
        p1_reward = 0.0
        p1_steps = 0
        done = False

        while not done:
            # 무작위 액션 선택
            action = rng.integers(0, env1.action_space.n)
            obs, reward, terminated, truncated, info = env1.step(action)
            p1_reward += reward
            p1_steps += 1
            done = terminated or truncated

        # 추천 차량 추출
        top_candidates = info.get('top_candidates', [])
        recommended = top_candidates[0][0] if top_candidates else None
        # 차량 이름을 ID로 변환
        vehicle_id = recommended.lower().replace(" ", "_") if recommended else "avante"

        # =====================================================================
        # Phase 2: 랜덤 스케줄링
        # =====================================================================
        env2 = SchedulingEnv()
        obs, info = env2.reset(seed=seed + ep + 10000, options={'vehicle_id': vehicle_id})
        p2_reward = 0.0
        p2_steps = 0
        done = False

        while not done:
            # 무작위 액션 선택
            action = rng.integers(0, env2.action_space.n)
            obs, reward, terminated, truncated, info = env2.step(action)
            p2_reward += reward
            p2_steps += 1
            done = terminated or truncated

        # 결과 기록
        results['total_rewards'].append(p1_reward + p2_reward)
        results['end_to_end_success'].append(1 if terminated and p2_reward > 0 else 0)
        results['total_interactions'].append(p1_steps + p2_steps)

    # 통계 계산 및 반환
    return {
        'agent': 'Random Baseline',
        'mean_total_reward': float(np.mean(results['total_rewards'])),
        'std_total_reward': float(np.std(results['total_rewards'])),
        'end_to_end_success_rate': float(np.mean(results['end_to_end_success'])),
        'mean_total_interactions': float(np.mean(results['total_interactions']))
    }


def evaluate_individual_phases(n_episodes: int = 100, seed: int = 42) -> dict:
    """
    개별 Phase 순차 실행 평가 (시너지 보너스 없음)

    Phase 1, Phase 2를 개별 학습 후 순차 실행하되,
    통합 학습 없이 Synergy Bonus를 제외한 성능 측정.

    이 베이스라인의 의미:
        - 통합 학습(Phase 3)의 효과를 입증하기 위한 기준
        - 시너지 보너스의 기여도 측정
        - 개별 최적화 vs 전역 최적화 비교

    Args:
        n_episodes: 평가 에피소드 수
        seed: 랜덤 시드

    Returns:
        평가 결과 딕셔너리
    """
    from src.integrated_system import IntegratedSystem, train_integrated

    # 개별 학습만 수행 (통합 학습 없음)
    system = IntegratedSystem(seed=seed)

    # 사전 학습만 수행, 통합 학습 없음 (n_episodes=0)
    train_integrated(
        system=system,
        n_episodes=0,          # 통합 학습 없음 (핵심!)
        phase1_pretrain=300,   # Phase 1만 개별 학습
        phase2_pretrain=300,   # Phase 2만 개별 학습
        verbose=False
    )

    # 결과 수집용
    results = {
        'total_rewards': [],
        'phase1_rewards': [],
        'phase2_rewards': [],
        'end_to_end_success': [],
        'total_interactions': []
    }

    # 평가 에피소드 실행
    for ep in range(n_episodes):
        result = system.run_episode(training=False)

        # 시너지 보너스 제외한 보상 (핵심!)
        # 개별 Phase만 순차 실행했을 때의 순수 성능
        reward_without_synergy = result['phase1_reward'] + result['phase2_reward']
        results['total_rewards'].append(reward_without_synergy)
        results['phase1_rewards'].append(result['phase1_reward'])
        results['phase2_rewards'].append(result['phase2_reward'])
        results['end_to_end_success'].append(1 if result['end_to_end_success'] else 0)
        results['total_interactions'].append(result['total_interactions'])

    # 통계 계산 및 반환
    return {
        'agent': 'Individual Phases (No Synergy)',
        'mean_total_reward': float(np.mean(results['total_rewards'])),
        'std_total_reward': float(np.std(results['total_rewards'])),
        'mean_phase1_reward': float(np.mean(results['phase1_rewards'])),
        'mean_phase2_reward': float(np.mean(results['phase2_rewards'])),
        'end_to_end_success_rate': float(np.mean(results['end_to_end_success'])),
        'mean_total_interactions': float(np.mean(results['total_interactions']))
    }


def main():
    """
    메인 평가 함수

    전체 평가 파이프라인을 실행함:
        1. 모델 로드 또는 학습
        2. Phase 3 통합 시스템 평가
        3. (옵션) 베이스라인과 비교
        4. (옵션) 결과 저장

    Returns:
        Phase 3 평가 결과 딕셔너리
    """
    args = parse_args()

    print("=" * 70)
    print("Phase 3: 통합 시스템 평가")
    print("=" * 70)

    # =========================================================================
    # 통합 시스템 준비 (모델 로드 또는 학습)
    # =========================================================================
    system = IntegratedSystem(seed=args.seed)

    if args.model_dir and Path(args.model_dir).exists():
        # 저장된 모델 로드
        print(f"\n모델 로드: {args.model_dir}")
        system.load(args.model_dir)
    else:
        # 모델이 없으면 새로 학습
        print(f"\n모델 학습 시작 ({args.train_episodes} 에피소드)")
        from src.integrated_system import train_integrated

        train_integrated(
            system=system,
            n_episodes=args.train_episodes,
            phase1_pretrain=300,
            phase2_pretrain=300,
            log_interval=200,
            verbose=True
        )

    # =========================================================================
    # Phase 3 통합 시스템 평가
    # =========================================================================
    print("\n" + "-" * 70)
    print("Phase 3 통합 시스템 평가")
    print("-" * 70)

    phase3_results = evaluate_integrated(
        system=system,
        n_episodes=args.episodes,
        verbose=True
    )

    # =========================================================================
    # 베이스라인 비교 (옵션)
    # =========================================================================
    if args.compare_baselines:
        print("\n" + "-" * 70)
        print("베이스라인 비교")
        print("-" * 70)

        # -----------------------------------------------------------------
        # 베이스라인 1: Random (가장 단순한 기준)
        # -----------------------------------------------------------------
        print("\n[1] Random Baseline 평가 중...")
        random_results = evaluate_random_baseline(
            n_episodes=args.episodes,
            seed=args.seed
        )

        # -----------------------------------------------------------------
        # 베이스라인 2: Individual Phases (시너지 없이 개별 학습만)
        # -----------------------------------------------------------------
        print("\n[2] Individual Phases (No Synergy) 평가 중...")
        individual_results = evaluate_individual_phases(
            n_episodes=args.episodes,
            seed=args.seed
        )

        # =================================================================
        # 비교 결과 출력
        # =================================================================
        print("\n" + "=" * 70)
        print("성능 비교 결과")
        print("=" * 70)

        # 표 헤더
        print(f"\n{'에이전트':<35} {'총 보상':>12} {'성공률':>10} {'상호작용':>10}")
        print("-" * 70)

        # Random Baseline
        print(f"{'Random Baseline':<35} "
              f"{random_results['mean_total_reward']:>12.2f} "
              f"{random_results['end_to_end_success_rate']:>9.1%} "
              f"{random_results['mean_total_interactions']:>10.2f}")

        # Individual Phases (No Synergy)
        print(f"{'Individual Phases (No Synergy)':<35} "
              f"{individual_results['mean_total_reward']:>12.2f} "
              f"{individual_results['end_to_end_success_rate']:>9.1%} "
              f"{individual_results['mean_total_interactions']:>10.2f}")

        # Phase 3 Integrated (With Synergy) - 핵심 결과
        print(f"{'Phase 3 Integrated (With Synergy)':<35} "
              f"{phase3_results['mean_total_reward']:>12.2f} "
              f"{phase3_results['end_to_end_success_rate']:>9.1%} "
              f"{phase3_results['mean_total_interactions']:>10.2f}")

        print("-" * 70)

        # =================================================================
        # 개선율 계산 (연구 결과의 핵심)
        # =================================================================
        random_reward = random_results['mean_total_reward']
        phase3_reward = phase3_results['mean_total_reward']

        # Random 대비 개선율
        if random_reward != 0:
            improvement_vs_random = ((phase3_reward - random_reward) / abs(random_reward)) * 100
        else:
            improvement_vs_random = 0

        # 시너지 보너스 기여도
        individual_reward = individual_results['mean_total_reward']
        synergy_contribution = phase3_reward - individual_reward

        # 시너지 비율 (총 보상 중 시너지가 차지하는 비율)
        synergy_ratio = phase3_results['mean_synergy_bonus'] / phase3_reward * 100 if phase3_reward > 0 else 0

        print(f"\n[개선율]")
        print(f"  vs Random Baseline: {improvement_vs_random:+.1f}%")
        print(f"  시너지 보너스 기여: {synergy_contribution:+.2f}")
        print(f"  시너지 비율: {synergy_ratio:.1f}%")

    # =========================================================================
    # 결과 저장 (옵션)
    # =========================================================================
    if args.save_results:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 모든 결과를 하나의 JSON으로 저장
        all_results = {
            "phase3_integrated": phase3_results,
        }

        # 베이스라인 결과 추가 (있는 경우)
        if args.compare_baselines:
            all_results["random_baseline"] = random_results
            all_results["individual_phases"] = individual_results

        results_path = results_dir / "phase3_evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장 완료: {results_path}")

    # =========================================================================
    # 완료 메시지
    # =========================================================================
    print("\n" + "=" * 70)
    print("평가 완료")
    print("=" * 70)

    return phase3_results


# =============================================================================
# 스크립트 실행 진입점
# =============================================================================
if __name__ == "__main__":
    main()
