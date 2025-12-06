"""
Phase 1 평가 스크립트

학습된 모델과 베이스라인의 성능을 평가하고 비교합니다.

사용법:
    # 모든 에이전트 평가
    python -m src.evaluate

    # 저장된 모델 로드하여 평가
    python -m src.evaluate --model models/q_learning_model.json

    # 상세 출력
    python -m src.evaluate --verbose --episodes 200
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.agents.q_learning_agent import QLearningAgent, train_q_learning
from src.baselines.random_baseline import RandomBaseline
from src.baselines.rule_based import AdaptiveRuleBasedBaseline, RuleBasedBaseline
from src.env.recommendation_env import VehicleRecommendationEnv


def detailed_evaluation(env, agent, agent_name: str, n_episodes: int = 100, seed: int = 42) -> dict:
    """
    상세 평가 수행

    에피소드별 세부 정보를 수집하여 분석합니다.

    Args:
        env: 평가 환경
        agent: 평가할 에이전트
        agent_name: 에이전트 이름
        n_episodes: 평가 에피소드 수
        seed: 랜덤 시드

    Returns:
        상세 평가 결과 딕셔너리
    """
    episode_data = []  # 에피소드별 상세 데이터

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)  # 환경 초기화

        # Rule-based 에이전트는 에피소드마다 리셋 필요
        if hasattr(agent, 'reset'):
            agent.reset()

        episode_reward = 0  # 에피소드 총 보상
        steps = 0
        actions_taken = []  # 수행한 액션 기록

        done = False
        while not done:
            # 액션 선택 (Q-Learning은 training=False로 greedy 선택)
            if isinstance(agent, QLearningAgent):
                action = agent.select_action(obs, training=False)
            else:
                action = agent.select_action(obs)

            actions_taken.append(action)

            obs, reward, terminated, truncated, info = env.step(action)  # 환경 스텝
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # 에피소드 결과 저장
        episode_data.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': steps,
            'questions': info.get('questions_count', 0),
            'terminated': terminated,
            'actions': actions_taken,
            'top_candidates': info.get('top_candidates', [])
        })

    # === 통계 계산 ===
    rewards = [ep['reward'] for ep in episode_data]
    questions = [ep['questions'] for ep in episode_data]
    steps_list = [ep['steps'] for ep in episode_data]
    successes = sum(1 for ep in episode_data if ep['terminated'])

    results = {
        'agent': agent_name,
        'n_episodes': n_episodes,
        'statistics': {  # 기본 통계
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'median_reward': float(np.median(rewards)),
            'mean_questions': float(np.mean(questions)),
            'std_questions': float(np.std(questions)),
            'mean_steps': float(np.mean(steps_list)),
            'success_rate': successes / n_episodes
        },
        'percentiles': {  # 백분위수
            '25th': float(np.percentile(rewards, 25)),
            '50th': float(np.percentile(rewards, 50)),
            '75th': float(np.percentile(rewards, 75)),
            '90th': float(np.percentile(rewards, 90))
        },
        'episode_data': episode_data if n_episodes <= 20 else None  # 소량만 저장
    }

    return results


def print_evaluation_report(results: list):
    """
    평가 결과 리포트 출력

    Args:
        results: 에이전트별 평가 결과 리스트
    """
    print("\n" + "=" * 80)
    print("평가 리포트")
    print("=" * 80)

    # 요약 테이블
    print("\n[성능 요약]")
    print(f"{'에이전트':<25} {'평균 보상':>12} {'표준편차':>10} {'질문 수':>10} {'성공률':>10}")
    print("-" * 70)

    for r in results:
        stats = r['statistics']
        print(f"{r['agent']:<25} {stats['mean_reward']:>12.3f} {stats['std_reward']:>10.3f} "
              f"{stats['mean_questions']:>10.2f} {stats['success_rate']:>9.1%}")

    # 보상 분포 (백분위수)
    print("\n[보상 분포 (백분위수)]")
    print(f"{'에이전트':<25} {'최소':>10} {'25%':>10} {'중앙값':>10} {'75%':>10} {'최대':>10}")
    print("-" * 80)

    for r in results:
        stats = r['statistics']
        pct = r['percentiles']
        print(f"{r['agent']:<25} {stats['min_reward']:>10.3f} {pct['25th']:>10.3f} "
              f"{pct['50th']:>10.3f} {pct['75th']:>10.3f} {stats['max_reward']:>10.3f}")

    # 최고/최저 성능 분석
    best_agent = max(results, key=lambda x: x['statistics']['mean_reward'])
    baseline_agent = min(results, key=lambda x: x['statistics']['mean_reward'])

    print("\n[분석]")
    print(f"  최고 성능: {best_agent['agent']} (평균 보상: {best_agent['statistics']['mean_reward']:.3f})")
    print(f"  최저 성능: {baseline_agent['agent']} (평균 보상: {baseline_agent['statistics']['mean_reward']:.3f})")

    # 개선율 계산
    improvement = best_agent['statistics']['mean_reward'] - baseline_agent['statistics']['mean_reward']
    improvement_pct = improvement / abs(baseline_agent['statistics']['mean_reward']) * 100
    print(f"  성능 향상: {improvement:.3f} ({improvement_pct:+.1f}%)")


def run_evaluation(args):
    """
    메인 평가 실행 함수
    """
    # === 설정 출력 ===
    print("=" * 70)
    print("Phase 1: 차량 추천 시스템 평가")
    print("=" * 70)
    print(f"평가 에피소드: {args.episodes}")
    print(f"시드: {args.seed}")
    print("=" * 70)

    env = VehicleRecommendationEnv()  # 평가 환경 생성
    results = []

    # === [1/4] Random Baseline 평가 ===
    print("\n[1/4] Random Baseline 평가 중...")
    random_agent = RandomBaseline(n_actions=env.action_space.n, seed=args.seed)
    random_results = detailed_evaluation(env, random_agent, "Random Baseline", args.episodes, args.seed)
    results.append(random_results)

    # === [2/4] Rule-based Baseline 평가 ===
    print("[2/4] Rule-based Baseline 평가 중...")
    rule_agent = RuleBasedBaseline(n_actions=env.action_space.n, seed=args.seed)
    rule_results = detailed_evaluation(env, rule_agent, "Rule-based Baseline", args.episodes, args.seed)
    results.append(rule_results)

    # === [3/4] Adaptive Rule-based 평가 ===
    print("[3/4] Adaptive Rule-based Baseline 평가 중...")
    adaptive_agent = AdaptiveRuleBasedBaseline(n_actions=env.action_space.n, seed=args.seed)
    adaptive_results = detailed_evaluation(env, adaptive_agent, "Adaptive Rule-based", args.episodes, args.seed)
    results.append(adaptive_results)

    # === [4/4] Q-Learning 평가 ===
    print("[4/4] Q-Learning 평가 중...")

    if args.model:
        # 저장된 모델 로드
        print(f"  모델 로드: {args.model}")
        q_agent = QLearningAgent(n_actions=env.action_space.n, seed=args.seed)
        q_agent.load(args.model)
    else:
        # 모델이 없으면 새로 학습
        print(f"  {args.train_episodes} 에피소드 학습 중...")
        q_agent, _ = train_q_learning(env, n_episodes=args.train_episodes, seed=args.seed, verbose=False)

    q_results = detailed_evaluation(env, q_agent, "Q-Learning", args.episodes, args.seed)
    results.append(q_results)

    # === 리포트 출력 ===
    print_evaluation_report(results)

    # === 결과 저장 (옵션) ===
    if args.save_results:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)

        # 에피소드 데이터 제거 (파일 크기 감소)
        save_results = []
        for r in results:
            save_r = r.copy()
            save_r['episode_data'] = None
            save_results.append(save_r)

        results_path = results_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {results_path}")

    return results


def main():
    """커맨드라인 인터페이스"""
    parser = argparse.ArgumentParser(
        description='Phase 1 차량 추천 시스템 평가',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 평가 설정 인자
    parser.add_argument('--episodes', type=int, default=100,
                        help='평가 에피소드 수')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (재현성)')

    # 모델 설정 인자
    parser.add_argument('--model', type=str, default=None,
                        help='로드할 Q-Learning 모델 경로')
    parser.add_argument('--train-episodes', type=int, default=1000,
                        help='모델이 없을 때 학습할 에피소드 수')

    # 출력 옵션 인자
    parser.add_argument('--verbose', action='store_true',
                        help='상세 출력')
    parser.add_argument('--save-results', action='store_true',
                        help='결과를 JSON 파일로 저장')

    args = parser.parse_args()  # 인자 파싱
    run_evaluation(args)  # 평가 실행


if __name__ == "__main__":
    main()
