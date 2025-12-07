"""
Q-Learning Agent

테이블 기반 Q-Learning으로 학습하는 에이전트입니다.
연속적인 상태 공간을 이산화(discretization)하여 테이블 기반 학습이 가능하게 합니다.

핵심 개념:
    - Q(s, a): 상태 s에서 액션 a를 수행했을 때 기대되는 누적 보상
    - 학습 규칙: Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
    - ε-greedy: 탐험(exploration)과 활용(exploitation) 균형
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class QLearningAgent:
    """
    Q-Learning 에이전트

    테이블 기반 Q-Learning을 사용하여 최적 정책을 학습합니다.
    연속적인 observation을 이산 상태로 변환하여 Q-table에 저장합니다.

    상태 표현 (State Representation):
        - 질문 응답 상태: 각 질문별로 "미응답(0)", "응답완료(1~5)"
        - 질문 횟수: 0, 1, 2, 3, 4, 5
        - 고객 유형: 나이대(3), 관심차량유무(2) → 6가지 조합

    이산화된 상태 공간 크기:
        - 질문 응답: 6^8 = 1,679,616 (각 질문당 6가지 상태)
        - 질문 횟수: 6가지
        - 고객 유형: 6가지
        - 총: 매우 크지만, 실제로는 방문한 상태만 저장 (sparse)
    """

    def __init__(
        self,
        n_actions: int = 12,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = None
    ):
        """
        Args:
            n_actions: 가능한 액션 수 (12 = 8질문 + 4추천)
            learning_rate: 학습률 (α), 새로운 정보 반영 비율
            discount_factor: 할인율 (γ), 미래 보상의 현재 가치
            epsilon_start: 초기 탐험률
            epsilon_end: 최소 탐험률
            epsilon_decay: 에피소드당 탐험률 감소 비율
            seed: 랜덤 시드
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # ε-greedy 파라미터
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: defaultdict으로 희소 저장
        # 키: (상태 튜플), 값: 액션별 Q값 배열
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        # 랜덤 생성기
        self.rng = np.random.default_rng(seed)

        # 학습 통계
        self.episode_count = 0
        self.total_updates = 0

    def _discretize_state(self, observation: np.ndarray) -> tuple:
        """
        연속적인 observation을 이산 상태로 변환

        Args:
            observation: 69차원 연속 벡터
                [0-4]: 고객 정보 (나이, 성별, 외국인, 직장, 관심차량)
                [5-44]: 질문 응답 (8질문 x 5옵션, one-hot)
                [45]: 질문 횟수 비율
                [46-68]: 차량 점수 (23개)

        Returns:
            이산 상태 튜플
        """
        state_parts = []

        # === 1. 고객 유형 (나이대 × 관심차량) ===
        # 나이 정규화값(-1~1)을 3구간으로 이산화
        normalized_age = observation[0]
        if normalized_age < -0.3:
            age_bin = 0  # 젊은층 (~35세)
        elif normalized_age < 0.3:
            age_bin = 1  # 중년층 (35~55세)
        else:
            age_bin = 2  # 장년층 (55세~)

        # 관심차량 유무
        has_interest = 1 if observation[4] > 0 else 0

        customer_type = age_bin * 2 + has_interest
        state_parts.append(customer_type)

        # === 2. 질문 응답 상태 (8개 질문) ===
        # 각 질문에 대해: 미응답(0) 또는 응답값+1(1~5)
        for q_id in range(8):
            start_idx = 5 + q_id * 5
            question_response = observation[start_idx:start_idx + 5]

            # one-hot에서 응답 인덱스 추출
            if np.max(question_response) > 0.5:  # 응답 있음
                answer_idx = np.argmax(question_response) + 1  # 1~5
            else:
                answer_idx = 0  # 미응답

            state_parts.append(answer_idx)

        # === 3. 질문 횟수 (0~5) ===
        question_ratio = observation[45]  # 0~1 범위
        question_count = min(5, int(question_ratio * 6))
        state_parts.append(question_count)

        return tuple(state_parts)

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        ε-greedy 정책으로 액션 선택

        Args:
            observation: 현재 상태 벡터
            training: 학습 모드 여부 (False면 greedy)

        Returns:
            선택된 액션 (0 ~ n_actions-1)
        """
        state = self._discretize_state(observation)

        # ε-greedy: 탐험 vs 활용
        if training and self.rng.random() < self.epsilon:
            # 탐험: 랜덤 액션
            return self.rng.integers(0, self.n_actions)
        else:
            # 활용: Q값이 가장 높은 액션
            q_values = self.q_table[state]

            # Q값이 동일한 액션들 중 랜덤 선택 (tie-breaking)
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return self.rng.choice(best_actions)

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool
    ):
        """
        Q-Learning 업데이트 수행

        Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]

        Args:
            observation: 현재 상태
            action: 수행한 액션
            reward: 받은 보상
            next_observation: 다음 상태
            terminated: 정상 종료 여부
            truncated: 강제 종료 여부
        """
        state = self._discretize_state(observation)
        next_state = self._discretize_state(next_observation)

        # 현재 Q값
        current_q = self.q_table[state][action]

        # TD 타겟 계산
        if terminated or truncated:
            # 종료 상태: 미래 보상 없음
            td_target = reward
        else:
            # 비종료 상태: 벨만 방정식
            max_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.discount_factor * max_next_q

        # Q값 업데이트
        td_error = td_target - current_q
        self.q_table[state][action] = current_q + self.learning_rate * td_error

        self.total_updates += 1

    def end_episode(self):
        """
        에피소드 종료 시 호출

        ε 감소 및 통계 업데이트
        """
        self.episode_count += 1

        # ε 감소 (지수적 감소)
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

    def get_stats(self) -> dict:
        """
        학습 통계 반환

        Returns:
            에피소드 수, 업데이트 수, Q-table 크기, 현재 ε
        """
        return {
            'episodes': self.episode_count,
            'total_updates': self.total_updates,
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }

    def save(self, path: str):
        """
        학습된 Q-table 저장

        Args:
            path: 저장 경로 (JSON 형식)
        """
        # defaultdict를 일반 dict로 변환 (numpy 타입을 Python 기본 타입으로)
        q_table_serializable = {
            str(k): v.tolist() for k, v in self.q_table.items()
        }

        data = {
            'q_table': q_table_serializable,
            'epsilon': float(self.epsilon),
            'episode_count': int(self.episode_count),
            'total_updates': int(self.total_updates),
            'hyperparameters': {
                'learning_rate': float(self.learning_rate),
                'discount_factor': float(self.discount_factor),
                'epsilon_end': float(self.epsilon_end),
                'epsilon_decay': float(self.epsilon_decay),
                'n_actions': int(self.n_actions)
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """
        저장된 Q-table 로드

        Args:
            path: 로드 경로
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Q-table 복원
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for k, v in data['q_table'].items():
            # 문자열 키를 튜플로 변환
            key = eval(k)  # "(0, 1, 2, ...)" → (0, 1, 2, ...)
            self.q_table[key] = np.array(v)

        self.epsilon = data['epsilon']
        self.episode_count = data['episode_count']
        self.total_updates = data['total_updates']


def train_q_learning(
    env,
    n_episodes: int = 1000,
    seed: int = 42,
    verbose: bool = True,
    log_interval: int = 100
) -> tuple:
    """
    Q-Learning 에이전트 학습

    Args:
        env: VehicleRecommendationEnv 환경
        n_episodes: 학습 에피소드 수
        seed: 랜덤 시드
        verbose: 학습 과정 출력 여부
        log_interval: 로그 출력 간격

    Returns:
        (agent, training_history) 튜플
    """
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,
        seed=seed
    )

    # 학습 기록
    training_history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'questions_counts': [],
        'epsilons': []
    }

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        steps = 0

        done = False
        while not done:
            # 액션 선택
            action = agent.select_action(obs, training=True)

            # 환경 스텝
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Q-Learning 업데이트
            agent.update(obs, action, reward, next_obs, terminated, truncated)

            obs = next_obs
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # 에피소드 종료 처리
        agent.end_episode()

        # 기록 저장
        training_history['episode_rewards'].append(episode_reward)
        training_history['episode_lengths'].append(steps)
        training_history['questions_counts'].append(info.get('questions_count', 0))
        training_history['epsilons'].append(agent.epsilon)

        # 로그 출력
        if verbose and (episode + 1) % log_interval == 0:
            recent_rewards = training_history['episode_rewards'][-log_interval:]
            recent_questions = training_history['questions_counts'][-log_interval:]
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {np.mean(recent_rewards):6.2f} | "
                  f"Avg Questions: {np.mean(recent_questions):.2f} | "
                  f"Q-table size: {len(agent.q_table):5d} | "
                  f"ε: {agent.epsilon:.3f}")

    return agent, training_history


def evaluate_q_learning(env, agent, n_episodes: int = 100, seed: int = 42) -> dict:
    """
    학습된 Q-Learning 에이전트 평가

    Args:
        env: 평가용 환경
        agent: 학습된 QLearningAgent
        n_episodes: 평가 에피소드 수
        seed: 랜덤 시드

    Returns:
        평가 결과 딕셔너리
    """
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
            # 평가 시에는 greedy (training=False)
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        questions_counts.append(info.get('questions_count', 0))

        if terminated:
            successful_recommendations += 1

    results = {
        'agent': 'Q-Learning',
        'n_episodes': n_episodes,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_questions': np.mean(questions_counts),
        'success_rate': successful_recommendations / n_episodes,
        'q_table_size': len(agent.q_table),
        'total_training_episodes': agent.episode_count
    }

    return results


if __name__ == "__main__":
    from src.env.recommendation_env import VehicleRecommendationEnv

    # 환경 생성
    env = VehicleRecommendationEnv()

    print("=" * 60)
    print("Q-Learning 에이전트 학습 시작")
    print("=" * 60)

    # 학습 수행
    agent, history = train_q_learning(
        env,
        n_episodes=1000,
        seed=42,
        verbose=True,
        log_interval=100
    )

    print("\n" + "=" * 60)
    print("학습 완료! 평가 수행 중...")
    print("=" * 60)

    # 평가 수행
    results = evaluate_q_learning(env, agent, n_episodes=100)

    print("\n=== Q-Learning 평가 결과 ===")
    print(f"평균 보상: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"보상 범위: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")
    print(f"평균 에피소드 길이: {results['mean_episode_length']:.2f}")
    print(f"평균 질문 수: {results['mean_questions']:.2f}")
    print(f"추천 성공률: {results['success_rate']:.1%}")
    print(f"Q-table 크기: {results['q_table_size']} 상태")

    # 모델 저장
    save_path = Path(__file__).parent.parent.parent / "checkpoints" / "standalone" / "q_learning_model.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_path))
    print(f"\n모델 저장: {save_path}")
