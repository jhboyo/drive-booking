"""
Phase 3 학습 스크립트

Phase 1(차량 추천)과 Phase 2(스케줄링)를 통합한 End-to-End 파이프라인 학습.
이 스크립트는 연구 실험의 메인 엔트리포인트로 사용됨.

학습 전략 (3단계):
    1단계. Phase 1 사전 학습: Q-Learning으로 차량 추천 정책 학습
    2단계. Phase 2 사전 학습: DQN으로 스케줄링 정책 학습
    3단계. 통합 미세조정: Synergy Bonus를 포함한 End-to-End 최적화

핵심 기여:
    - Two-Phase RL 파이프라인 구조 제안
    - Synergy Bonus를 통한 협업 효과 측정
    - 개별 학습 → 통합 학습 전략으로 안정적인 수렴

사용법:
    # 기본 학습 (2000 에피소드)
    python -m src.train_phase3

    # 커스텀 설정
    python -m src.train_phase3 --episodes 3000 --seed 123 --save-model

    # 모든 옵션
    python -m src.train_phase3 --episodes 2000 --phase1-pretrain 500 \\
        --phase2-pretrain 500 --seed 42 --save-model --save-results
"""

import argparse
import json
import time
from pathlib import Path

# 통합 시스템 및 학습/평가 함수
from src.integrated_system import IntegratedSystem, train_integrated, evaluate_integrated


def parse_args():
    """
    커맨드라인 인자 파싱

    학습 실험에 필요한 다양한 설정을 커맨드라인으로 받음.
    기본값은 논문 실험에서 사용된 값으로 설정됨.

    Returns:
        파싱된 인자 객체
    """
    parser = argparse.ArgumentParser(
        description="Phase 3 통합 시스템 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m src.train_phase3                          # 기본 학습
  python -m src.train_phase3 --episodes 3000          # 에피소드 수 변경
  python -m src.train_phase3 --save-model --save-results  # 결과 저장
        """
    )

    # =========================================================================
    # 학습 설정
    # =========================================================================
    parser.add_argument(
        "--episodes", type=int, default=2000,
        help="통합 학습 에피소드 수 (기본값: 2000). "
             "Phase 1 → Phase 2 파이프라인을 몇 번 실행할지 결정."
    )
    parser.add_argument(
        "--phase1-pretrain", type=int, default=500,
        help="Phase 1 사전 학습 에피소드 수 (기본값: 500). "
             "차량 추천 정책을 독립적으로 학습하는 횟수."
    )
    parser.add_argument(
        "--phase2-pretrain", type=int, default=500,
        help="Phase 2 사전 학습 에피소드 수 (기본값: 500). "
             "스케줄링 정책을 독립적으로 학습하는 횟수."
    )

    # =========================================================================
    # 시드 및 로깅
    # =========================================================================
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본값: 42). 재현 가능한 실험을 위해 사용."
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="로그 출력 간격 (기본값: 100). 몇 에피소드마다 진행 상황을 출력할지."
    )

    # =========================================================================
    # 저장 옵션
    # =========================================================================
    parser.add_argument(
        "--save-model", action="store_true",
        help="학습된 모델 저장 (Phase 1 Q-table + Phase 2 DQN)"
    )
    parser.add_argument(
        "--save-results", action="store_true",
        help="학습 결과를 JSON으로 저장 (학습 곡선, 평가 결과 포함)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="checkpoints/integrated",
        help="모델 저장 디렉토리 (기본값: checkpoints/integrated)"
    )

    # =========================================================================
    # 평가 옵션
    # =========================================================================
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="평가 에피소드 수 (기본값: 100). 학습 후 성능 측정에 사용."
    )

    return parser.parse_args()


def main():
    """
    메인 학습 함수

    전체 학습 파이프라인을 실행함:
        1. 설정 출력 및 확인
        2. IntegratedSystem 생성
        3. 3단계 학습 수행
        4. 평가 및 결과 출력
        5. (옵션) 모델/결과 저장

    Returns:
        평가 결과 딕셔너리
    """
    args = parse_args()

    # =========================================================================
    # 설정 출력
    # =========================================================================
    print("=" * 70)
    print("Phase 3: 통합 시스템 학습")
    print("=" * 70)
    print(f"\n[학습 설정]")
    print(f"  Phase 1 사전 학습: {args.phase1_pretrain} 에피소드")
    print(f"  Phase 2 사전 학습: {args.phase2_pretrain} 에피소드")
    print(f"  통합 학습: {args.episodes} 에피소드")
    print(f"  평가: {args.eval_episodes} 에피소드")
    print(f"  시드: {args.seed}")
    print("=" * 70)

    # =========================================================================
    # 통합 시스템 생성
    # =========================================================================
    # Phase 1 환경/에이전트 + Phase 2 환경/에이전트를 파이프라인으로 연결
    system = IntegratedSystem(seed=args.seed)

    # =========================================================================
    # 학습 실행 (3단계 학습 전략)
    # =========================================================================
    # 1단계: Phase 1 사전 학습 (Q-Learning)
    # 2단계: Phase 2 사전 학습 (DQN)
    # 3단계: 통합 학습 (End-to-End + Synergy Bonus)
    start_time = time.time()

    history = train_integrated(
        system=system,
        n_episodes=args.episodes,
        phase1_pretrain=args.phase1_pretrain,
        phase2_pretrain=args.phase2_pretrain,
        log_interval=args.log_interval,
        verbose=True
    )

    train_time = time.time() - start_time
    print(f"\n학습 시간: {train_time:.1f}초")

    # =========================================================================
    # 평가 실행
    # =========================================================================
    # 학습된 정책으로 성능 측정 (탐험 없이 greedy 정책)
    print("\n" + "=" * 70)
    print("통합 시스템 평가")
    print("=" * 70)

    eval_results = evaluate_integrated(
        system=system,
        n_episodes=args.eval_episodes,
        verbose=True
    )

    # =========================================================================
    # 모델 저장
    # =========================================================================
    if args.save_model:
        model_dir = Path(args.model_dir)
        system.save(str(model_dir))
        print(f"\n모델 저장 완료: {model_dir}")

    # =========================================================================
    # 결과 저장 (JSON)
    # =========================================================================
    if args.save_results:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 학습 히스토리 + 평가 결과 + 설정을 하나의 JSON으로 저장
        results = {
            "config": {
                "episodes": args.episodes,
                "phase1_pretrain": args.phase1_pretrain,
                "phase2_pretrain": args.phase2_pretrain,
                "seed": args.seed
            },
            "training": {
                # Phase 1 사전 학습 보상 히스토리
                "phase1_pretrain_rewards": [float(r) for r in history['phase1_pretrain_rewards']],
                # Phase 2 사전 학습 보상 히스토리
                "phase2_pretrain_rewards": [float(r) for r in history['phase2_pretrain_rewards']],
                # 통합 학습 총 보상 히스토리 (R1 + R2 + Synergy)
                "integrated_rewards": [float(r) for r in history['integrated_rewards']],
                # 시너지 보너스 히스토리
                "synergy_bonuses": [float(r) for r in history['synergy_bonuses']],
                # End-to-End 성공 여부 히스토리
                "end_to_end_success": history['end_to_end_success']
            },
            # 최종 평가 결과
            "evaluation": eval_results,
            # 학습 시간
            "train_time_seconds": train_time
        }

        results_path = results_dir / "phase3_training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"결과 저장 완료: {results_path}")

    # =========================================================================
    # 최종 요약 출력
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 3 학습 최종 요약")
    print("=" * 70)

    # 학습 통계
    print(f"\n[학습 결과]")
    total_episodes = args.phase1_pretrain + args.phase2_pretrain + args.episodes
    print(f"  총 학습 에피소드: {total_episodes}")
    print(f"  학습 시간: {train_time:.1f}초")

    # 성능 지표 (핵심 결과)
    print(f"\n[성능 지표]")
    print(f"  End-to-End 성공률: {eval_results['end_to_end_success_rate']:.1%}")
    print(f"  총 보상: {eval_results['mean_total_reward']:.2f} ± {eval_results['std_total_reward']:.2f}")
    print(f"  시너지 보너스: {eval_results['mean_synergy_bonus']:.2f}")

    # 효율성 지표
    print(f"\n[효율성]")
    print(f"  평균 질문 수: {eval_results['mean_questions']:.2f}")
    print(f"  평균 스케줄링 시도: {eval_results['mean_attempts']:.2f}")
    print(f"  총 상호작용: {eval_results['mean_total_interactions']:.2f}")
    print("=" * 70)

    return eval_results


# =============================================================================
# 스크립트 실행 진입점
# =============================================================================
if __name__ == "__main__":
    main()
