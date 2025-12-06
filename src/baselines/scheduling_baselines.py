"""
Phase 2 스케줄링 베이스라인

Random과 FCFS(First Come First Serve) 베이스라인을 제공합니다.
DQN 에이전트의 성능 비교 기준이 됩니다.
"""

import numpy as np


class RandomSchedulingBaseline:
    """
    랜덤 스케줄링 베이스라인

    모든 액션을 균등한 확률로 랜덤 선택합니다.
    성능의 하한선(lower bound)을 제공합니다.
    """

    def __init__(self, n_actions: int = 6, seed: int = None):
        """
        Args:
            n_actions: 가능한 액션 수 (기본 6개)
            seed: 랜덤 시드
        """
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray) -> int:
        """랜덤하게 액션 선택"""
        return self.rng.integers(0, self.n_actions)

    def reset(self):
        """에피소드 리셋 (인터페이스 호환용)"""
        pass


class FCFSSchedulingBaseline:
    """
    FCFS (First Come First Serve) 스케줄링 베이스라인

    고객이 요청한 시간이 가용하면 즉시 확정하고,
    불가능하면 순서대로 대안을 제시합니다.

    전략:
        1. 요청 시간 가용 → 확정 (action=0)
        2. 불가 → 같은 날 대안 (action=1)
        3. 불가 → 다음 날 대안 (action=2)
        4. 불가 → 평일 대안 (action=3)
        5. 불가 → 다음 주 대안 (action=4)
    """

    def __init__(self, seed: int = None):
        """
        Args:
            seed: 랜덤 시드 (대안 선택 시 사용)
        """
        self.rng = np.random.default_rng(seed)
        self.attempt_count = 0

    def select_action(self, observation: np.ndarray) -> int:
        """
        FCFS 전략으로 액션 선택

        Args:
            observation: 현재 상태 벡터

        Returns:
            선택된 액션 (0-5)
        """
        # 첫 번째 시도: 항상 확정 시도
        if self.attempt_count == 0:
            self.attempt_count += 1
            return 0  # 확정 시도

        # 대안 순서: 같은날 → 다음날 → 평일 → 다음주
        action_sequence = [1, 2, 3, 4]

        if self.attempt_count <= len(action_sequence):
            action = action_sequence[self.attempt_count - 1]
            self.attempt_count += 1
            return action

        # 모든 대안 시도 후에도 실패하면 랜덤
        return self.rng.integers(1, 5)

    def reset(self):
        """에피소드 리셋"""
        self.attempt_count = 0


class GreedySchedulingBaseline:
    """
    Greedy 스케줄링 베이스라인

    현재 상태에서 가장 유리해 보이는 액션을 선택합니다.
    - 슬롯이 가용하면 확정
    - 아니면 가장 가까운 대안 제시
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray) -> int:
        """
        Greedy 전략으로 액션 선택

        observation의 슬롯 가용성 정보를 활용하여
        가용한 슬롯이 있으면 확정, 없으면 대안 제시
        """
        # observation 구조: [고객정보(6) + 슬롯가용성(126) + 차량가용성(23) + 메타(4)]
        # 현재 요청 슬롯이 가용한지 확인 (간단히 메타 정보 활용)

        # 메타 정보에서 시도 횟수 추출 (마지막 4개 중 첫 번째)
        attempt_ratio = observation[-4]

        if attempt_ratio < 0.1:  # 첫 시도
            return 0  # 확정 시도

        # 이미 시도했으면 대안 제시 (확률적으로 선택)
        # 같은날 대안이 가장 수락률 높을 것으로 가정
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # action 1-5
        action = self.rng.choice([1, 2, 3, 4, 5], p=weights)
        return action

    def reset(self):
        """에피소드 리셋"""
        pass


def evaluate_scheduling_baseline(
    env,
    agent,
    agent_name: str,
    n_episodes: int = 100,
    seed: int = 42
) -> dict:
    """
    스케줄링 베이스라인 성능 평가

    Args:
        env: SchedulingEnv 환경
        agent: 평가할 에이전트
        agent_name: 에이전트 이름
        n_episodes: 평가할 에피소드 수
        seed: 랜덤 시드

    Returns:
        평가 결과 딕셔너리
    """
    total_rewards = []
    booking_success = 0
    alternative_accepted = 0
    preferred_time_match = 0
    attempt_counts = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)

        if hasattr(agent, 'reset'):
            agent.reset()

        episode_reward = 0
        done = False

        # 선호 시간 저장
        preferred_day = info['requested_day']
        preferred_slot = info['requested_slot']

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        attempt_counts.append(info.get('attempt_count', 0))

        # 예약 성공 여부
        if terminated and episode_reward > 0:
            booking_success += 1

            # 선호 시간 매칭 여부
            if (info['requested_day'] == preferred_day and
                info['requested_slot'] == preferred_slot):
                preferred_time_match += 1
            elif info.get('attempt_count', 0) > 0:
                alternative_accepted += 1

    # 통계 계산
    results = {
        'agent': agent_name,
        'n_episodes': n_episodes,
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'min_reward': float(np.min(total_rewards)),
        'max_reward': float(np.max(total_rewards)),
        'booking_success_rate': booking_success / n_episodes,
        'preferred_time_match_rate': preferred_time_match / n_episodes,
        'alternative_accept_rate': alternative_accepted / max(1, booking_success - preferred_time_match),
        'mean_attempts': float(np.mean(attempt_counts)),
    }

    return results


if __name__ == "__main__":
    from src.env.scheduling_env import SchedulingEnv

    env = SchedulingEnv()
    n_episodes = 100

    print("=" * 60)
    print("Phase 2 스케줄링 베이스라인 평가")
    print("=" * 60)

    # Random Baseline
    print("\n[1/3] Random Baseline 평가 중...")
    random_agent = RandomSchedulingBaseline(seed=42)
    random_results = evaluate_scheduling_baseline(env, random_agent, "Random", n_episodes)

    # FCFS Baseline
    print("[2/3] FCFS Baseline 평가 중...")
    fcfs_agent = FCFSSchedulingBaseline(seed=42)
    fcfs_results = evaluate_scheduling_baseline(env, fcfs_agent, "FCFS", n_episodes)

    # Greedy Baseline
    print("[3/3] Greedy Baseline 평가 중...")
    greedy_agent = GreedySchedulingBaseline(seed=42)
    greedy_results = evaluate_scheduling_baseline(env, greedy_agent, "Greedy", n_episodes)

    # 결과 출력
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"{'에이전트':<15} {'평균보상':>10} {'예약성사율':>12} {'선호시간':>10} {'평균시도':>10}")
    print("-" * 60)

    for r in [random_results, fcfs_results, greedy_results]:
        print(f"{r['agent']:<15} {r['mean_reward']:>10.2f} "
              f"{r['booking_success_rate']:>11.1%} "
              f"{r['preferred_time_match_rate']:>9.1%} "
              f"{r['mean_attempts']:>10.2f}")
