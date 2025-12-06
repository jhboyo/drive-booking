"""
Phase 1 학습 스크립트

차량 추천 환경에서 Q-Learning 에이전트를 학습하고,
베이스라인과 성능을 비교합니다.

사용법:
    python -m src.train_phase1 --episodes 1000 --seed 42
    python -m src.train_phase1 --episodes 2000 --save-model
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.agents.q_learning_agent import QLearningAgent, evaluate_q_learning
from src.baselines.random_baseline import evaluate_random_baseline
from src.baselines.rule_based import evaluate_rule_based_baseline
from src.env.recommendation_env import VehicleRecommendationEnv
from src.visualization.plot_results import (
    plot_learning_curve,
    plot_performance_comparison,
    plot_reward_distribution,
)


def train_agent(
    env,
    agent: QLearningAgent,
    n_episodes: int,
    seed: int = 42,
    log_interval: int = 100,
    verbose: bool = True
) -> dict:
    """
    Q-Learning 에이전트 학습

    Args:
        env: 학습 환경
        agent: Q-Learning 에이전트
        n_episodes: 학습 에피소드 수
        seed: 랜덤 시드
        log_interval: 로그 출력 간격
        verbose: 상세 출력 여부

    Returns:
        학습 히스토리 딕셔너리
    """
    # 학습 과정 기록용 딕셔너리
    history = {
        'episode_rewards': [],      # 에피소드별 총 보상
        'episode_lengths': [],      # 에피소드별 스텝 수
        'questions_counts': [],     # 에피소드별 질문 횟수
        'epsilons': [],             # 탐험률 변화
        'q_table_sizes': []         # Q-table 크기 변화
    }

    for episode in range(n_episodes):  # 에피소드 반복
        obs, info = env.reset(seed=seed + episode)  # 환경 초기화, 새 고객 생성
        episode_reward = 0  # 에피소드 누적 보상 초기화
        steps = 0  # 스텝 카운터

        done = False
        while not done:  # 에피소드 종료까지 반복
            action = agent.select_action(obs, training=True)  # ε-greedy로 액션 선택
            next_obs, reward, terminated, truncated, info = env.step(action)  # 환경에 액션 적용
            agent.update(obs, action, reward, next_obs, terminated, truncated)  # Q-값 업데이트

            obs = next_obs  # 상태 전이
            episode_reward += reward  # 보상 누적
            steps += 1
            done = terminated or truncated  # 종료 조건 확인

        agent.end_episode()  # 에피소드 종료 처리 (ε 감소)

        # 학습 기록 저장
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(steps)
        history['questions_counts'].append(info.get('questions_count', 0))
        history['epsilons'].append(agent.epsilon)
        history['q_table_sizes'].append(len(agent.q_table))

        # 주기적 로그 출력
        if verbose and (episode + 1) % log_interval == 0:
            recent_rewards = history['episode_rewards'][-log_interval:]  # 최근 N개 보상
            recent_questions = history['questions_counts'][-log_interval:]  # 최근 N개 질문 수
            print(f"Episode {episode + 1:5d}/{n_episodes} | "
                  f"Reward: {np.mean(recent_rewards):6.2f} ± {np.std(recent_rewards):5.2f} | "
                  f"Questions: {np.mean(recent_questions):.2f} | "
                  f"Q-table: {len(agent.q_table):5d} | "
                  f"ε: {agent.epsilon:.3f}")

    return history


def run_training(args):
    """
    메인 학습 실행 함수
    """
    # === 설정 출력 ===
    print("=" * 70)
    print("Phase 1: 차량 추천 시스템 학습")
    print("=" * 70)
    print(f"에피소드: {args.episodes}")
    print(f"시드: {args.seed}")
    print(f"학습률: {args.lr}")
    print(f"할인율: {args.gamma}")
    print("=" * 70)

    # === 환경 및 에이전트 초기화 ===
    env = VehicleRecommendationEnv()  # Gymnasium 환경 생성

    agent = QLearningAgent(
        n_actions=env.action_space.n,       # 액션 수 (12개: 질문 8 + 추천 4)
        learning_rate=args.lr,              # 학습률 α
        discount_factor=args.gamma,         # 할인율 γ
        epsilon_start=1.0,                  # 초기 탐험률 (100% 랜덤)
        epsilon_end=0.05,                   # 최소 탐험률 (5%)
        epsilon_decay=args.epsilon_decay,   # 탐험률 감소율
        seed=args.seed
    )

    # === [1/4] 학습 수행 ===
    print("\n[1/4] Q-Learning 학습 시작...")
    history = train_agent(
        env, agent, args.episodes,
        seed=args.seed,
        log_interval=args.log_interval,
        verbose=True
    )

    # === [2/4] 평가 수행 ===
    print("\n[2/4] 에이전트 평가 중...")
    eval_episodes = 100  # 평가용 에피소드 수

    # 각 에이전트 평가 (동일 시드로 공정한 비교)
    q_results = evaluate_q_learning(env, agent, n_episodes=eval_episodes, seed=args.seed)
    print(f"  Q-Learning: 평균 보상 {q_results['mean_reward']:.3f} ± {q_results['std_reward']:.3f}")

    random_results = evaluate_random_baseline(env, n_episodes=eval_episodes, seed=args.seed)
    print(f"  Random: 평균 보상 {random_results['mean_reward']:.3f} ± {random_results['std_reward']:.3f}")

    rule_results = evaluate_rule_based_baseline(env, n_episodes=eval_episodes, seed=args.seed)
    print(f"  Rule-based: 평균 보상 {rule_results['mean_reward']:.3f} ± {rule_results['std_reward']:.3f}")

    adaptive_results = evaluate_rule_based_baseline(env, n_episodes=eval_episodes, seed=args.seed, adaptive=True)
    print(f"  Adaptive: 평균 보상 {adaptive_results['mean_reward']:.3f} ± {adaptive_results['std_reward']:.3f}")

    # === [3/4] 결과 저장 ===
    print("\n[3/4] 결과 저장 중...")
    project_root = Path(__file__).parent.parent  # 프로젝트 루트 경로
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)  # 디렉토리 생성 (없으면)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 타임스탬프 (파일명용)

    # 학습 결과 딕셔너리 구성
    results = {
        'timestamp': timestamp,
        'config': {  # 학습 설정
            'episodes': args.episodes,
            'seed': args.seed,
            'learning_rate': args.lr,
            'discount_factor': args.gamma,
            'epsilon_decay': args.epsilon_decay
        },
        'training_history': {  # 학습 과정 기록
            'episode_rewards': history['episode_rewards'],
            'questions_counts': history['questions_counts'],
            'epsilons': history['epsilons'],
            'q_table_sizes': history['q_table_sizes']
        },
        'evaluation': {  # 평가 결과
            'q_learning': {
                'mean_reward': float(q_results['mean_reward']),
                'std_reward': float(q_results['std_reward']),
                'mean_questions': float(q_results['mean_questions']),
                'success_rate': float(q_results['success_rate'])
            },
            'random': {
                'mean_reward': float(random_results['mean_reward']),
                'std_reward': float(random_results['std_reward']),
                'mean_questions': float(random_results['mean_questions']),
                'success_rate': float(random_results['success_rate'])
            },
            'rule_based': {
                'mean_reward': float(rule_results['mean_reward']),
                'std_reward': float(rule_results['std_reward']),
                'mean_questions': float(rule_results['mean_questions']),
                'success_rate': float(rule_results['success_rate'])
            },
            'adaptive_rule_based': {
                'mean_reward': float(adaptive_results['mean_reward']),
                'std_reward': float(adaptive_results['std_reward']),
                'mean_questions': float(adaptive_results['mean_questions']),
                'success_rate': float(adaptive_results['success_rate'])
            }
        }
    }

    # JSON 파일로 저장
    results_path = results_dir / f"training_results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  결과 저장: {results_path}")

    # 모델 저장 (옵션)
    if args.save_model:
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"q_learning_{timestamp}.json"
        agent.save(str(model_path))  # Q-table을 JSON으로 저장
        print(f"  모델 저장: {model_path}")

    # === [4/4] 시각화 ===
    print("\n[4/4] 그래프 생성 중...")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_performance_comparison(str(figures_dir / "performance_comparison.png"))  # 성능 비교 막대 그래프
    plot_learning_curve(str(figures_dir / "learning_curve.png"))  # 학습 곡선
    plot_reward_distribution(str(figures_dir / "reward_distribution.png"))  # 보상 분포 박스플롯

    # === 최종 결과 출력 ===
    print("\n" + "=" * 70)
    print("학습 완료!")
    print("=" * 70)

    # 성능 비교 테이블 출력
    print("\n[성능 비교]")
    print(f"{'에이전트':<20} {'평균 보상':>10} {'질문 수':>10} {'성공률':>10}")
    print("-" * 50)
    print(f"{'Random':<20} {random_results['mean_reward']:>10.3f} {random_results['mean_questions']:>10.2f} {random_results['success_rate']:>9.1%}")
    print(f"{'Rule-based':<20} {rule_results['mean_reward']:>10.3f} {rule_results['mean_questions']:>10.2f} {rule_results['success_rate']:>9.1%}")
    print(f"{'Adaptive Rule-based':<20} {adaptive_results['mean_reward']:>10.3f} {adaptive_results['mean_questions']:>10.2f} {adaptive_results['success_rate']:>9.1%}")
    print(f"{'Q-Learning':<20} {q_results['mean_reward']:>10.3f} {q_results['mean_questions']:>10.2f} {q_results['success_rate']:>9.1%}")

    # 개선율 계산 및 출력
    improvement_vs_random = (q_results['mean_reward'] - random_results['mean_reward']) / random_results['mean_reward'] * 100
    improvement_vs_rule = (q_results['mean_reward'] - rule_results['mean_reward']) / rule_results['mean_reward'] * 100

    print(f"\n[Q-Learning 개선율]")
    print(f"  Random 대비: {improvement_vs_random:+.1f}%")
    print(f"  Rule-based 대비: {improvement_vs_rule:+.1f}%")

    print(f"\n저장 위치: {results_dir}")

    return results


def main():
    """커맨드라인 인터페이스"""
    parser = argparse.ArgumentParser(
        description='Phase 1 차량 추천 시스템 학습',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 학습 설정 인자
    parser.add_argument('--episodes', type=int, default=1000,
                        help='학습 에피소드 수')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (재현성)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='로그 출력 간격')

    # Q-Learning 하이퍼파라미터 인자
    parser.add_argument('--lr', type=float, default=0.1,
                        help='학습률 (alpha)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='할인율 (gamma)')
    parser.add_argument('--epsilon-decay', type=float, default=0.998,
                        help='탐험률 감소율')

    # 저장 옵션 인자
    parser.add_argument('--save-model', action='store_true',
                        help='학습된 모델 저장')

    args = parser.parse_args()  # 인자 파싱
    run_training(args)  # 학습 실행


if __name__ == "__main__":
    main()
