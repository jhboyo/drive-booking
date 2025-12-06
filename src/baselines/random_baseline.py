"""
Random Baseline

랜덤하게 질문하거나 추천하는 가장 단순한 베이스라인입니다.
성능의 하한선(lower bound)을 제공합니다.
"""

import numpy as np


class RandomBaseline:
    """
    랜덤 베이스라인 에이전트

    모든 액션을 균등한 확률로 랜덤 선택합니다.
    학습 없이 동작하며, 다른 에이전트의 성능 비교 기준이 됩니다.
    """

    def __init__(self, n_actions: int, seed: int = None):
        """
        Args:
            n_actions: 가능한 액션 수 (환경의 action_space.n)
            seed: 랜덤 시드 (재현성을 위해)
        """
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray) -> int:
        """
        랜덤하게 액션 선택

        Args:
            observation: 현재 상태 (사용하지 않음)

        Returns:
            랜덤하게 선택된 액션 (0 ~ n_actions-1)
        """
        return self.rng.integers(0, self.n_actions)

    def update(self, *args, **kwargs):
        """학습 없음 (인터페이스 호환용)"""
        pass


def evaluate_random_baseline(env, n_episodes: int = 100, seed: int = 42) -> dict:
    """
    Random Baseline 성능 평가

    Args:
        env: VehicleRecommendationEnv 환경
        n_episodes: 평가할 에피소드 수
        seed: 랜덤 시드

    Returns:
        평가 결과 딕셔너리
    """
    agent = RandomBaseline(n_actions=env.action_space.n, seed=seed)

    # 결과 저장용
    total_rewards = []
    episode_lengths = []
    questions_counts = []
    successful_recommendations = 0

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        steps = 0

        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        questions_counts.append(info.get('questions_count', 0))

        # 추천으로 종료했으면 성공
        if terminated:
            successful_recommendations += 1

    # 통계 계산
    results = {
        'agent': 'Random Baseline',
        'n_episodes': n_episodes,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_questions': np.mean(questions_counts),
        'success_rate': successful_recommendations / n_episodes,
    }

    return results


if __name__ == "__main__":
    from src.env.recommendation_env import VehicleRecommendationEnv

    # 환경 생성
    env = VehicleRecommendationEnv()

    # 평가 실행
    results = evaluate_random_baseline(env, n_episodes=100)

    # 결과 출력
    print("\n=== Random Baseline 평가 결과 ===")
    print(f"평균 보상: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"보상 범위: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")
    print(f"평균 에피소드 길이: {results['mean_episode_length']:.2f}")
    print(f"평균 질문 수: {results['mean_questions']:.2f}")
    print(f"추천 성공률: {results['success_rate']:.1%}")
