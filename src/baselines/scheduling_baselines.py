"""
Phase 2 스케줄링 베이스라인

Random과 FCFS(First Come First Serve) 베이스라인을 제공합니다.
DQN 에이전트의 성능 비교 기준이 됩니다.
"""

import numpy as np  # 수치 연산 라이브러리


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
        self.n_actions = n_actions  # 액션 공간 크기 저장
        self.rng = np.random.default_rng(seed)  # 재현 가능한 랜덤 생성기 초기화

    def select_action(self, observation: np.ndarray) -> int:
        """랜덤하게 액션 선택"""
        return self.rng.integers(0, self.n_actions)  # 0 ~ n_actions-1 범위에서 랜덤 정수 반환

    def reset(self):
        """에피소드 리셋 (인터페이스 호환용)"""
        pass  # 랜덤 에이전트는 상태가 없으므로 아무 작업 없음


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
        self.rng = np.random.default_rng(seed)  # 랜덤 생성기 초기화
        self.attempt_count = 0  # 현재 에피소드의 시도 횟수 추적

    def select_action(self, observation: np.ndarray) -> int:
        """
        FCFS 전략으로 액션 선택

        Args:
            observation: 현재 상태 벡터

        Returns:
            선택된 액션 (0-5)
        """
        # 첫 번째 시도: 항상 확정 시도
        if self.attempt_count == 0:  # 아직 시도한 적 없으면
            self.attempt_count += 1  # 시도 횟수 증가
            return 0  # 확정 시도 (action=0)

        # 대안 순서: 같은날 → 다음날 → 평일 → 다음주
        action_sequence = [1, 2, 3, 4]  # 순차적으로 시도할 대안 액션들

        if self.attempt_count <= len(action_sequence):  # 아직 시도할 대안이 남았으면
            action = action_sequence[self.attempt_count - 1]  # 현재 순서의 대안 선택
            self.attempt_count += 1  # 시도 횟수 증가
            return action  # 선택한 대안 반환

        # 모든 대안 시도 후에도 실패하면 랜덤
        return self.rng.integers(1, 5)  # 1~4 범위에서 랜덤 선택 (확정 제외)

    def reset(self):
        """에피소드 리셋"""
        self.attempt_count = 0  # 시도 횟수 초기화


class GreedySchedulingBaseline:
    """
    Greedy 스케줄링 베이스라인

    현재 상태에서 가장 유리해 보이는 액션을 선택합니다.
    - 슬롯이 가용하면 확정
    - 아니면 가장 가까운 대안 제시
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)  # 랜덤 생성기 초기화

    def select_action(self, observation: np.ndarray) -> int:
        """
        Greedy 전략으로 액션 선택

        observation의 슬롯 가용성 정보를 활용하여
        가용한 슬롯이 있으면 확정, 없으면 대안 제시
        """
        # observation 구조: [고객정보(6) + 슬롯가용성(126) + 차량가용성(23) + 메타(4)]
        # 메타 정보에서 시도 횟수 추출 (마지막 4개 중 첫 번째)
        attempt_ratio = observation[-4]  # 시도 횟수 / 최대 시도 횟수 비율

        if attempt_ratio < 0.1:  # 첫 시도 (비율이 낮으면)
            return 0  # 확정 시도

        # 이미 시도했으면 대안 제시 (확률적으로 선택)
        # 같은날 대안이 가장 수락률 높을 것으로 가정
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # action 1-5 각각의 선택 확률
        action = self.rng.choice([1, 2, 3, 4, 5], p=weights)  # 가중치 기반 랜덤 선택
        return action  # 선택된 대안 반환

    def reset(self):
        """에피소드 리셋"""
        pass  # Greedy는 내부 상태가 없으므로 아무 작업 없음


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
    total_rewards = []  # 에피소드별 총 보상 저장
    booking_success = 0  # 예약 성공 횟수
    alternative_accepted = 0  # 대안 수락 횟수
    preferred_time_match = 0  # 선호 시간 매칭 횟수
    attempt_counts = []  # 에피소드별 시도 횟수 저장

    for episode in range(n_episodes):  # 지정된 에피소드 수만큼 반복
        obs, info = env.reset(seed=seed + episode)  # 환경 초기화 (시드로 재현성 보장)

        if hasattr(agent, 'reset'):  # 에이전트에 reset 메서드가 있으면
            agent.reset()  # 에이전트 상태 초기화

        episode_reward = 0  # 현재 에피소드 보상 초기화
        done = False  # 에피소드 종료 플래그

        # 선호 시간 저장 (나중에 매칭 여부 확인용)
        preferred_day = info['requested_day']  # 고객이 선호하는 날짜
        preferred_slot = info['requested_slot']  # 고객이 선호하는 시간 슬롯

        while not done:  # 에피소드가 끝날 때까지 반복
            action = agent.select_action(obs)  # 에이전트가 액션 선택
            obs, reward, terminated, truncated, info = env.step(action)  # 환경에 액션 적용
            episode_reward += reward  # 보상 누적
            done = terminated or truncated  # 종료 조건 확인

        total_rewards.append(episode_reward)  # 에피소드 보상 기록
        attempt_counts.append(info.get('attempt_count', 0))  # 시도 횟수 기록

        # 예약 성공 여부 판단
        if terminated and episode_reward > 0:  # 정상 종료 + 양수 보상이면 성공
            booking_success += 1  # 성공 횟수 증가

            # 선호 시간 매칭 여부 확인
            if (info['requested_day'] == preferred_day and
                info['requested_slot'] == preferred_slot):  # 선호 시간에 예약됨
                preferred_time_match += 1  # 선호 시간 매칭 횟수 증가
            elif info.get('attempt_count', 0) > 0:  # 대안을 제시해서 성공한 경우
                alternative_accepted += 1  # 대안 수락 횟수 증가

    # 통계 계산
    results = {
        'agent': agent_name,  # 에이전트 이름
        'n_episodes': n_episodes,  # 평가 에피소드 수
        'mean_reward': float(np.mean(total_rewards)),  # 평균 보상
        'std_reward': float(np.std(total_rewards)),  # 보상 표준편차
        'min_reward': float(np.min(total_rewards)),  # 최소 보상
        'max_reward': float(np.max(total_rewards)),  # 최대 보상
        'booking_success_rate': booking_success / n_episodes,  # 예약 성공률
        'preferred_time_match_rate': preferred_time_match / n_episodes,  # 선호 시간 매칭률
        'alternative_accept_rate': alternative_accepted / max(1, booking_success - preferred_time_match),  # 대안 수락률
        'mean_attempts': float(np.mean(attempt_counts)),  # 평균 시도 횟수
    }

    return results  # 평가 결과 반환


if __name__ == "__main__":
    from src.env.scheduling_env import SchedulingEnv  # 스케줄링 환경 임포트

    env = SchedulingEnv()  # 환경 인스턴스 생성
    n_episodes = 100  # 평가할 에피소드 수 설정

    print("=" * 60)
    print("Phase 2 스케줄링 베이스라인 평가")
    print("=" * 60)

    # Random Baseline 평가
    print("\n[1/3] Random Baseline 평가 중...")
    random_agent = RandomSchedulingBaseline(seed=42)  # 랜덤 에이전트 생성
    random_results = evaluate_scheduling_baseline(env, random_agent, "Random", n_episodes)  # 평가 실행

    # FCFS Baseline 평가
    print("[2/3] FCFS Baseline 평가 중...")
    fcfs_agent = FCFSSchedulingBaseline(seed=42)  # FCFS 에이전트 생성
    fcfs_results = evaluate_scheduling_baseline(env, fcfs_agent, "FCFS", n_episodes)  # 평가 실행

    # Greedy Baseline 평가
    print("[3/3] Greedy Baseline 평가 중...")
    greedy_agent = GreedySchedulingBaseline(seed=42)  # Greedy 에이전트 생성
    greedy_results = evaluate_scheduling_baseline(env, greedy_agent, "Greedy", n_episodes)  # 평가 실행

    # 결과 출력
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"{'에이전트':<15} {'평균보상':>10} {'예약성사율':>12} {'선호시간':>10} {'평균시도':>10}")
    print("-" * 60)

    for r in [random_results, fcfs_results, greedy_results]:  # 각 결과 순회
        print(f"{r['agent']:<15} {r['mean_reward']:>10.2f} "  # 에이전트명, 평균보상 출력
              f"{r['booking_success_rate']:>11.1%} "  # 예약 성사율 출력
              f"{r['preferred_time_match_rate']:>9.1%} "  # 선호 시간 매칭률 출력
              f"{r['mean_attempts']:>10.2f}")  # 평균 시도 횟수 출력
