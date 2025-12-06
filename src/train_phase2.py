"""
Phase 2 학습 스크립트: 시승 스케줄링

DQN 에이전트를 학습하고 베이스라인과 성능을 비교합니다.

사용법:
    python src/train_phase2.py --episodes 1000
    python src/train_phase2.py --episodes 500 --eval-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from env.scheduling_env import SchedulingEnv
from agents.scheduling_agent import DQNAgent, train_dqn, evaluate_dqn
from baselines.scheduling_baselines import (
    RandomSchedulingBaseline,
    FCFSSchedulingBaseline,
    GreedySchedulingBaseline,
    evaluate_scheduling_baseline
)


def train_and_evaluate(
    n_episodes: int = 1000,
    eval_episodes: int = 100,
    seed: int = 42,
    save_model: bool = True
) -> dict:
    """
    DQN 학습 및 베이스라인 비교 평가

    Args:
        n_episodes: 학습 에피소드 수
        eval_episodes: 평가 에피소드 수
        seed: 랜덤 시드
        save_model: 모델 저장 여부

    Returns:
        전체 결과 딕셔너리
    """
    env = SchedulingEnv()

    print("=" * 60)
    print("Phase 2: 시승 스케줄링 학습")
    print("=" * 60)
    print(f"State dim: {env.observation_space.shape[0]}")
    print(f"Action dim: {env.action_space.n}")
    print(f"학습 에피소드: {n_episodes}")
    print(f"평가 에피소드: {eval_episodes}")
    print("=" * 60)

    # === DQN 학습 ===
    print("\n[1/2] DQN 에이전트 학습 중...")
    start_time = time.time()

    agent, history = train_dqn(
        env,
        n_episodes=n_episodes,
        seed=seed,
        verbose=True,
        log_interval=max(50, n_episodes // 10)
    )

    train_time = time.time() - start_time
    print(f"학습 완료: {train_time:.1f}초")

    # 모델 저장
    if save_model:
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "dqn_scheduling.pth"
        agent.save(str(model_path))
        print(f"모델 저장: {model_path}")

    # === 평가 ===
    print("\n[2/2] 성능 평가 중...")
    print("-" * 60)

    results = {}

    # DQN 평가
    print("DQN 평가...")
    results['DQN'] = evaluate_dqn(env, agent, n_episodes=eval_episodes, seed=seed)

    # 베이스라인 평가
    print("Random 베이스라인 평가...")
    random_agent = RandomSchedulingBaseline(seed=seed)
    results['Random'] = evaluate_scheduling_baseline(env, random_agent, "Random", eval_episodes, seed)

    print("FCFS 베이스라인 평가...")
    fcfs_agent = FCFSSchedulingBaseline(seed=seed)
    results['FCFS'] = evaluate_scheduling_baseline(env, fcfs_agent, "FCFS", eval_episodes, seed)

    print("Greedy 베이스라인 평가...")
    greedy_agent = GreedySchedulingBaseline(seed=seed)
    results['Greedy'] = evaluate_scheduling_baseline(env, greedy_agent, "Greedy", eval_episodes, seed)

    # 결과 출력
    print("\n" + "=" * 60)
    print("성능 비교 결과")
    print("=" * 60)
    print(f"{'에이전트':<10} {'평균보상':>10} {'예약성사율':>12} {'선호시간':>10} {'평균시도':>10}")
    print("-" * 60)

    for name, r in results.items():
        success_rate = r.get('booking_success_rate', 0)
        pref_rate = r.get('preferred_time_match_rate', 0)
        attempts = r.get('mean_attempts', 0)
        print(f"{name:<10} {r['mean_reward']:>10.2f} {success_rate:>11.1%} {pref_rate:>9.1%} {attempts:>10.2f}")

    # DQN 개선율 계산
    print("\n" + "-" * 60)
    print("DQN 개선율")
    print("-" * 60)

    dqn_reward = results['DQN']['mean_reward']
    for name in ['Random', 'FCFS', 'Greedy']:
        baseline_reward = results[name]['mean_reward']
        if baseline_reward != 0:
            improvement = (dqn_reward - baseline_reward) / abs(baseline_reward) * 100
            print(f"vs {name}: {improvement:+.1f}%")

    # 결과 저장
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 학습 히스토리 저장
    history_path = results_dir / "phase2_training_history.json"
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

    # 평가 결과 저장
    eval_path = results_dir / "phase2_evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n결과 저장: {results_dir}")

    return {
        'results': results,
        'history': history,
        'train_time': train_time
    }


def evaluate_only(model_path: str, eval_episodes: int = 100, seed: int = 42):
    """저장된 모델 평가만 수행"""
    env = SchedulingEnv()

    print("=" * 60)
    print("Phase 2: 저장된 모델 평가")
    print("=" * 60)

    # 모델 로드
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        seed=seed
    )
    agent.load(model_path)
    print(f"모델 로드: {model_path}")
    print(f"학습된 에피소드: {agent.episode_count}")

    # 평가
    results = evaluate_dqn(env, agent, n_episodes=eval_episodes, seed=seed)

    print("\n=== 평가 결과 ===")
    print(f"평균 보상: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"예약 성사율: {results['booking_success_rate']:.1%}")
    print(f"선호 시간 매칭률: {results['preferred_time_match_rate']:.1%}")
    print(f"평균 시도 횟수: {results['mean_attempts']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 2 학습 스크립트')
    parser.add_argument('--episodes', type=int, default=1000, help='학습 에피소드 수')
    parser.add_argument('--eval-episodes', type=int, default=100, help='평가 에피소드 수')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--eval-only', action='store_true', help='평가만 수행')
    parser.add_argument('--model-path', type=str, default=None, help='평가할 모델 경로')

    args = parser.parse_args()

    if args.eval_only:
        model_path = args.model_path or str(Path(__file__).parent.parent / "models" / "dqn_scheduling.pth")
        evaluate_only(model_path, args.eval_episodes, args.seed)
    else:
        train_and_evaluate(args.episodes, args.eval_episodes, args.seed)


if __name__ == "__main__":
    main()
