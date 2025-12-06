"""
Rule-based Baseline

휴리스틱 규칙에 따라 질문하고 추천하는 베이스라인입니다.
간단한 규칙으로 Random보다 나은 성능을 보여주며,
학습된 에이전트의 비교 기준이 됩니다.
"""

import numpy as np


class RuleBasedBaseline:
    """
    규칙 기반 베이스라인 에이전트

    전략:
        1. 중요한 질문을 우선순위에 따라 순차적으로 수행
           - 예산(3) → 사용 용도(0) → 차체 타입(6)
        2. 지정된 수의 질문 후 Top 1 추천
        3. 이미 질문한 내용은 건너뜀
    """

    # 질문 우선순위 (예산, 용도, 차체타입, 연료, 크기, 가족, 우선순위, 색상)
    QUESTION_PRIORITY = [3, 0, 6, 1, 5, 2, 4, 7]

    def __init__(
        self,
        n_questions: int = 8,
        n_actions: int = 12,
        questions_before_recommend: int = 3,
        seed: int = None
    ):
        """
        Args:
            n_questions: 환경의 총 질문 수
            n_actions: 환경의 총 액션 수
            questions_before_recommend: 추천 전 수행할 질문 수 (기본 3개)
            seed: 랜덤 시드 (재현성을 위해)
        """
        self.n_questions = n_questions
        self.n_actions = n_actions
        self.questions_before_recommend = questions_before_recommend
        self.rng = np.random.default_rng(seed)

        # 에피소드별 상태 (reset에서 초기화)
        self.asked_questions = set()
        self.question_count = 0

    def reset(self):
        """에피소드 시작 시 상태 초기화"""
        self.asked_questions = set()
        self.question_count = 0

    def select_action(self, observation: np.ndarray) -> int:
        """
        규칙에 따라 액션 선택

        Args:
            observation: 현재 상태 벡터 (환경에서 제공)

        Returns:
            선택된 액션 (0 ~ n_actions-1)

        규칙:
            1. questions_before_recommend 수만큼 질문
            2. 우선순위 순서대로 아직 안 한 질문 선택
            3. 목표 질문 수에 도달하면 Top 1 추천 (action=8)
        """
        # 목표 질문 수에 도달하면 Top 1 추천
        if self.question_count >= self.questions_before_recommend:
            return 8  # Top 1 추천

        # 우선순위 순서대로 다음 질문 선택
        for q_id in self.QUESTION_PRIORITY:
            if q_id not in self.asked_questions:
                self.asked_questions.add(q_id)
                self.question_count += 1
                return q_id

        # 모든 질문을 했으면 (이론상 도달 불가) Top 1 추천
        return 8

    def update(self, *args, **kwargs):
        """학습 없음 (인터페이스 호환용)"""
        pass


class AdaptiveRuleBasedBaseline:
    """
    적응형 규칙 기반 베이스라인

    고객 기본 정보(나이, 직장유무 등)를 활용하여
    더 적합한 질문을 먼저 수행하는 개선된 규칙 기반 에이전트입니다.

    전략:
        - 고객 나이에 따라 질문 우선순위 조정
        - 관심 차량이 있는 고객은 적은 질문 후 추천
        - 신규 고객은 더 많은 질문 수행
    """

    # 기본 질문 우선순위
    DEFAULT_PRIORITY = [3, 0, 6, 1, 5, 2, 4, 7]

    # 젊은 고객 (30세 미만): 연료타입과 디자인 우선
    YOUNG_PRIORITY = [1, 0, 4, 3, 6, 5, 2, 7]

    # 가족 고객 (관심차량 있음): 크기와 가족구성 우선
    FAMILY_PRIORITY = [2, 5, 0, 6, 3, 1, 4, 7]

    def __init__(
        self,
        n_questions: int = 8,
        n_actions: int = 12,
        seed: int = None
    ):
        """
        Args:
            n_questions: 환경의 총 질문 수
            n_actions: 환경의 총 액션 수
            seed: 랜덤 시드
        """
        self.n_questions = n_questions
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        # 에피소드별 상태
        self.asked_questions = set()
        self.question_count = 0
        self.question_priority = self.DEFAULT_PRIORITY.copy()
        self.target_questions = 3  # 기본 목표 질문 수

    def reset(self):
        """에피소드 시작 시 상태 초기화"""
        self.asked_questions = set()
        self.question_count = 0
        self.question_priority = self.DEFAULT_PRIORITY.copy()
        self.target_questions = 3

    def select_action(self, observation: np.ndarray) -> int:
        """
        고객 정보를 고려하여 액션 선택

        Args:
            observation: 현재 상태 벡터 (69차원)
                - [0]: 나이 (정규화됨, -1~1)
                - [4]: 관심 차량 유무 (1.0 또는 -1.0)

        Returns:
            선택된 액션
        """
        # 첫 액션일 때 고객 정보 기반으로 전략 설정
        if self.question_count == 0:
            self._set_strategy(observation)

        # 목표 질문 수에 도달하면 Top 1 추천
        if self.question_count >= self.target_questions:
            return 8

        # 현재 우선순위에 따라 질문 선택
        for q_id in self.question_priority:
            if q_id not in self.asked_questions:
                self.asked_questions.add(q_id)
                self.question_count += 1
                return q_id

        return 8

    def _set_strategy(self, observation: np.ndarray):
        """
        고객 정보 기반으로 질문 전략 설정

        Args:
            observation: 상태 벡터
        """
        # 나이 추출 (정규화 해제: -1~1 → 20~70)
        normalized_age = observation[0]
        age = int(normalized_age * 25 + 45)

        # 관심 차량 유무
        has_interest = observation[4] > 0

        # 전략 설정
        if age < 30:
            # 젊은 고객: 연료타입, 디자인 중시
            self.question_priority = self.YOUNG_PRIORITY.copy()
            self.target_questions = 3
        elif has_interest:
            # 관심 차량 있는 고객: 적은 질문으로 빠르게 추천
            self.question_priority = self.FAMILY_PRIORITY.copy()
            self.target_questions = 2
        else:
            # 기본 전략
            self.question_priority = self.DEFAULT_PRIORITY.copy()
            self.target_questions = 3

    def update(self, *args, **kwargs):
        """학습 없음 (인터페이스 호환용)"""
        pass


def evaluate_rule_based_baseline(
    env,
    n_episodes: int = 100,
    seed: int = 42,
    adaptive: bool = False
) -> dict:
    """
    Rule-based Baseline 성능 평가

    Args:
        env: VehicleRecommendationEnv 환경
        n_episodes: 평가할 에피소드 수
        seed: 랜덤 시드
        adaptive: True면 AdaptiveRuleBasedBaseline 사용

    Returns:
        평가 결과 딕셔너리
    """
    if adaptive:
        agent = AdaptiveRuleBasedBaseline(
            n_questions=8,
            n_actions=env.action_space.n,
            seed=seed
        )
        agent_name = "Adaptive Rule-based Baseline"
    else:
        agent = RuleBasedBaseline(
            n_questions=8,
            n_actions=env.action_space.n,
            questions_before_recommend=3,
            seed=seed
        )
        agent_name = "Rule-based Baseline"

    # 결과 저장용
    total_rewards = []
    episode_lengths = []
    questions_counts = []
    successful_recommendations = 0

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        agent.reset()  # 에피소드 시작 시 에이전트 상태 초기화

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
        'agent': agent_name,
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

    print("=" * 60)

    # 기본 Rule-based 평가
    results = evaluate_rule_based_baseline(env, n_episodes=100, adaptive=False)
    print("\n=== Rule-based Baseline 평가 결과 ===")
    print(f"평균 보상: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"보상 범위: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")
    print(f"평균 에피소드 길이: {results['mean_episode_length']:.2f}")
    print(f"평균 질문 수: {results['mean_questions']:.2f}")
    print(f"추천 성공률: {results['success_rate']:.1%}")

    print("=" * 60)

    # Adaptive Rule-based 평가
    results_adaptive = evaluate_rule_based_baseline(env, n_episodes=100, adaptive=True)
    print("\n=== Adaptive Rule-based Baseline 평가 결과 ===")
    print(f"평균 보상: {results_adaptive['mean_reward']:.3f} ± {results_adaptive['std_reward']:.3f}")
    print(f"보상 범위: [{results_adaptive['min_reward']:.3f}, {results_adaptive['max_reward']:.3f}]")
    print(f"평균 에피소드 길이: {results_adaptive['mean_episode_length']:.2f}")
    print(f"평균 질문 수: {results_adaptive['mean_questions']:.2f}")
    print(f"추천 성공률: {results_adaptive['success_rate']:.1%}")
