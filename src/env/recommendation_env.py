"""
차량 추천 환경 (Phase 1)

강화학습 에이전트가 고객과 대화하며 최적의 시승 차량을 추천하는 환경입니다.
Gymnasium 인터페이스를 따릅니다.

핵심 개념:
    - 에이전트는 고객의 "숨겨진 선호도"를 모르는 상태에서 시작
    - 질문을 통해 선호도를 파악하고, 적절한 시점에 차량 추천
    - 최소 질문으로 높은 만족도를 달성하는 것이 목표
"""

import json
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class VehicleRecommendationEnv(gym.Env):
    """
    차량 추천 강화학습 환경

    에이전트는 고객에게 질문을 하거나 차량을 추천할 수 있습니다.
    목표: 최소한의 질문으로 고객에게 적합한 차량을 추천하여 만족도를 최대화

    Action Space (Discrete(12)):
        - 0~7: 질문하기 (questions.json의 8개 질문)
            0: 사용 용도 (출퇴근/가족여행/업무용/레저)
            1: 연료 타입 (가솔린/하이브리드/전기차/상관없음)
            2: 가족 구성 (1명/2명/3-4명/5명이상)
            3: 예산 (3000만원 이하/3000-4500/4500-6000/6000이상)
            4: 우선순위 (안전성/연비/성능/디자인)
            5: 차량 크기 (소형/중형/대형/상관없음)
            6: 차체 타입 (세단/SUV/MPV/상관없음)
            7: 외장 색상 (화이트/블랙/그레이/기타/상관없음)
        - 8: Top 1 차량 추천 (가장 높은 점수 1개)
        - 9: Top 2 차량 추천 (상위 2개)
        - 10: Top 3 차량 추천 (상위 3개)
        - 11: 추가 질문 요청 (에피소드 계속, 페널티 -0.1)

    Observation Space (Box, 69차원):
        - 고객 기본 정보 (5차원): 나이, 성별, 외국인여부, 직장유무, 관심차량유무
        - 질문 응답 상태 (40차원): 8개 질문 x 5개 옵션 (one-hot, 미응답은 0)
        - 질문 횟수 (1차원): 현재 질문 수 / 최대 질문 수
        - 차량 매칭 점수 (23차원): 각 차량별 현재 매칭 점수

    Reward 구조:
        - 질문 시: -0.2 (기본 페널티) + 정보 획득 보너스
        - 중복 질문: -0.5
        - 추천 시: 만족도(~10) + 효율성 보너스(~2) + 정확도 보너스(~3)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # =========================================================================
    # 보상 관련 상수
    # =========================================================================
    REWARD_QUESTION_PENALTY = -0.2      # 질문 1회당 기본 페널티
    REWARD_DUPLICATE_PENALTY = -0.5     # 중복 질문 페널티
    REWARD_EXTRA_QUESTION = -0.1        # 추가 질문 요청 페널티
    REWARD_MAX_SATISFACTION = 10.0      # 최대 만족도 보상
    REWARD_EFFICIENCY_BONUS_HIGH = 2.0  # 2개 이하 질문 보너스
    REWARD_EFFICIENCY_BONUS_MID = 1.0   # 3개 이하 질문 보너스
    REWARD_EXACT_MATCH_BONUS = 3.0      # 1순위 정확히 맞춤 보너스
    REWARD_TOO_MANY_RECOMMEND = -0.5    # 3개 이상 추천 페널티

    def __init__(
        self,
        vehicles_path: str = None,
        questions_path: str = None,
        customers_path: str = None,
        max_questions: int = 5,
        render_mode: Optional[str] = None
    ):
        """
        환경 초기화

        Args:
            vehicles_path: 차량 데이터 JSON 경로 (None이면 기본 경로 사용)
            questions_path: 질문 목록 JSON 경로 (None이면 기본 경로 사용)
            customers_path: 고객 프로필 JSON 경로 (None이면 기본 경로 사용)
            max_questions: 최대 질문 횟수 (초과 시 truncated=True)
            render_mode: 렌더링 모드 ("human": 콘솔 출력, "ansi": 문자열 반환)
        """
        super().__init__()

        # === 데이터 경로 설정 ===
        # 경로가 지정되지 않으면 프로젝트 루트의 data 폴더 사용
        project_root = Path(__file__).parent.parent.parent
        vehicles_path = vehicles_path or str(project_root / "data" / "vehicles.json")
        questions_path = questions_path or str(project_root / "data" / "questions.json")
        customers_path = customers_path or str(project_root / "data" / "customer_profiles.json")

        # === 데이터 로드 ===
        self.vehicles = self._load_json(vehicles_path, 'vehicles')
        self.questions = self._load_json(questions_path, 'questions')
        self.customers = self._load_json(customers_path, 'profiles')

        # === 환경 설정 ===
        self.max_questions = max_questions
        self.render_mode = render_mode
        self.n_vehicles = len(self.vehicles)
        self.n_questions = len(self.questions)

        # === Action Space 정의 ===
        # 질문(0 ~ n_questions-1) + 추천 액션(4개)
        self.action_space = spaces.Discrete(self.n_questions + 4)

        # === Observation Space 정의 ===
        # 구성: 고객정보(5) + 질문응답(8*5=40) + 질문횟수(1) + 차량점수(n_vehicles)
        obs_dim = 5 + (self.n_questions * 5) + 1 + self.n_vehicles
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # === 에피소드 상태 변수 (reset에서 초기화) ===
        self.current_customer = None        # 현재 에피소드의 고객 정보
        self.asked_questions = set()        # 이미 질문한 질문 ID 집합
        self.question_answers = {}          # 질문별 응답 {question_id: answer_idx}
        self.questions_count = 0            # 현재까지 질문한 횟수
        self.candidate_scores = {}          # 차량별 현재 매칭 점수
        self.hidden_preferences = {}        # 고객의 숨겨진 선호도 (에이전트는 접근 불가)

    def _load_json(self, path: str, key: str) -> list:
        """
        JSON 파일에서 데이터 로드

        Args:
            path: JSON 파일 경로
            key: 추출할 최상위 키 (예: 'vehicles', 'questions', 'profiles')

        Returns:
            해당 키의 데이터 리스트
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get(key, [])

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """
        환경 초기화 및 새 에피소드 시작

        새로운 고객을 선택하고, 해당 고객의 숨겨진 선호도를 생성합니다.
        에이전트는 질문을 통해 이 선호도를 파악해야 합니다.

        Args:
            seed: 랜덤 시드 (재현성을 위해 사용)
            options: 추가 옵션
                - customer_idx: 특정 고객 인덱스 지정 (테스트용)

        Returns:
            observation: 초기 상태 벡터 (69차원)
            info: 디버깅/로깅용 추가 정보
        """
        super().reset(seed=seed)

        # === 고객 선택 ===
        # options에서 지정하거나 랜덤 선택
        if options and 'customer_idx' in options:
            customer_idx = options['customer_idx']
        else:
            customer_idx = self.np_random.integers(0, len(self.customers))

        self.current_customer = self.customers[customer_idx].copy()

        # === 에피소드 상태 초기화 ===
        self.asked_questions = set()
        self.question_answers = {}
        self.questions_count = 0

        # === 고객의 숨겨진 선호도 생성 ===
        # 이 정보는 에이전트가 직접 접근할 수 없으며, 질문을 통해 파악해야 함
        self._generate_hidden_preferences()

        # === 초기 차량 점수 계산 ===
        # 질문 전이므로 모든 차량이 동일한 기본 점수(0.5)를 가짐
        self._update_candidate_scores()

        return self._get_observation(), self._get_info()

    def _generate_hidden_preferences(self):
        """
        고객의 숨겨진 선호도 생성

        고객의 나이, 직장 유무 등 기본 정보를 바탕으로
        각 질문에 대한 "실제" 선호도를 확률적으로 생성합니다.

        이 선호도는:
        1. 에이전트가 질문하면 해당 값을 응답으로 반환
        2. 추천 시 보상 계산의 기준이 됨

        생성되는 선호도 (questions.json의 attribute와 매칭):
        - usage: 사용 용도 (0=출퇴근, 1=가족여행, 2=업무용, 3=레저)
        - fuel_type: 연료 타입 (0=가솔린, 1=하이브리드, 2=전기차, 3=상관없음)
        - family_size: 가족 구성 (0=1명, 1=2명, 2=3-4명, 3=5명이상)
        - budget: 예산 (0=3000이하, 1=3000-4500, 2=4500-6000, 3=6000이상)
        - priority: 우선순위 (0=안전성, 1=연비, 2=성능, 3=디자인)
        - size: 차량 크기 (0=소형, 1=중형, 2=대형, 3=상관없음)
        - body_type: 차체 타입 (0=세단, 1=SUV, 2=MPV, 3=상관없음)
        - color: 외장 색상 (0=화이트, 1=블랙, 2=그레이, 3=기타, 4=상관없음)
        """
        customer = self.current_customer
        age = customer.get('age', 35)
        has_workplace = customer.get('has_workplace', True)

        # --- 사용 용도 (usage) ---
        # 나이별 사용 패턴 가정:
        #   20대: 출퇴근(50%), 레저(30%) 중심
        #   30-44세: 가족여행(40%) 중심
        #   45세+: 고른 분포
        if age < 30:
            usage_weights = [0.5, 0.1, 0.1, 0.3]
        elif age < 45:
            usage_weights = [0.3, 0.4, 0.2, 0.1]
        else:
            usage_weights = [0.2, 0.3, 0.3, 0.2]
        self.hidden_preferences['usage'] = self.np_random.choice(4, p=usage_weights)

        # --- 연료 타입 (fuel_type) ---
        # 젊은 층은 전기차 선호, 나이 들수록 가솔린 선호 가정
        if age < 35:
            fuel_weights = [0.2, 0.3, 0.4, 0.1]
        else:
            fuel_weights = [0.4, 0.3, 0.2, 0.1]
        self.hidden_preferences['fuel_type'] = self.np_random.choice(4, p=fuel_weights)

        # --- 가족 구성 (family_size) ---
        # 직장인은 1-2인 가구 비율 높음, 비직장인은 다인 가구 비율 높음
        if has_workplace:
            family_weights = [0.3, 0.3, 0.3, 0.1]
        else:
            family_weights = [0.2, 0.2, 0.4, 0.2]
        self.hidden_preferences['family_size'] = self.np_random.choice(4, p=family_weights)

        # --- 예산 (budget) ---
        # 나이에 따른 경제력 가정:
        #   20대: 저예산 중심
        #   30-49세: 중간 예산
        #   50세+: 고예산 가능
        if age < 30:
            budget_weights = [0.5, 0.3, 0.15, 0.05]
        elif age < 50:
            budget_weights = [0.2, 0.35, 0.3, 0.15]
        else:
            budget_weights = [0.15, 0.25, 0.35, 0.25]
        self.hidden_preferences['budget'] = self.np_random.choice(4, p=budget_weights)

        # --- 우선순위 (priority) ---
        # 모든 연령대에서 비교적 균등 분포
        priority_weights = [0.3, 0.25, 0.2, 0.25]
        self.hidden_preferences['priority'] = self.np_random.choice(4, p=priority_weights)

        # --- 차량 크기 (size) ---
        # 가족 구성에 따라 결정 (family_size와 연계)
        family = self.hidden_preferences['family_size']
        if family >= 2:     # 3-4명 또는 5명 이상
            size_weights = [0.05, 0.3, 0.5, 0.15]
        elif family == 1:   # 2명
            size_weights = [0.2, 0.4, 0.2, 0.2]
        else:               # 1명
            size_weights = [0.4, 0.3, 0.1, 0.2]
        self.hidden_preferences['size'] = self.np_random.choice(4, p=size_weights)

        # --- 차체 타입 (body_type) ---
        # 사용 용도에 따라 결정 (usage와 연계)
        usage = self.hidden_preferences['usage']
        if usage in [1, 3]:  # 가족여행 또는 레저
            body_weights = [0.1, 0.6, 0.2, 0.1]
        else:                # 출퇴근 또는 업무용
            body_weights = [0.4, 0.3, 0.1, 0.2]
        self.hidden_preferences['body_type'] = self.np_random.choice(4, p=body_weights)

        # --- 외장 색상 (color) ---
        # 일반적인 색상 선호도 분포
        color_weights = [0.3, 0.25, 0.2, 0.1, 0.15]
        self.hidden_preferences['color'] = self.np_random.choice(5, p=color_weights)

        # === 이상적인 차량 목록 생성 ===
        # 숨겨진 선호도 기반으로 가장 적합한 차량 순위 결정 (보상 계산용)
        self._calculate_ideal_vehicles()

    def _calculate_ideal_vehicles(self):
        """
        고객에게 가장 적합한 차량 순위 계산

        숨겨진 선호도를 기반으로 모든 차량의 매칭 점수를 계산하고,
        상위 3개 차량을 ideal_vehicles로 저장합니다.
        이는 추천 보상 계산의 기준이 됩니다.
        """
        scores = []
        for vehicle in self.vehicles:
            score = self._calculate_vehicle_match_score(vehicle, use_hidden=True)
            scores.append((vehicle['name'], score))

        # 점수 내림차순 정렬
        scores.sort(key=lambda x: x[1], reverse=True)

        # 상위 3개 차량 저장 (보상 계산 시 사용)
        self.hidden_preferences['ideal_vehicles'] = [name for name, _ in scores[:3]]

        # 전체 차량 점수 저장 (세부 보상 계산용)
        self.hidden_preferences['ideal_scores'] = {name: score for name, score in scores}

    def _calculate_vehicle_match_score(
        self,
        vehicle: dict,
        use_hidden: bool = False
    ) -> float:
        """
        차량과 고객 선호도의 매칭 점수 계산

        Args:
            vehicle: 차량 정보 딕셔너리 (vehicles.json의 한 항목)
            use_hidden: True면 숨겨진 선호도 사용 (보상 계산용)
                       False면 질문 응답 기반 (observation용)

        Returns:
            매칭 점수 (0.0 ~ 1.0)

        점수 구성:
            - 기본 점수: 0.5
            - 각 속성 매칭 시 가/감점
            - 최종 점수는 0~1 범위로 클리핑
        """
        score = 0.5  # 기본 점수 (정보 없을 때)

        # 선호도 소스 결정
        if use_hidden:
            prefs = self.hidden_preferences
        else:
            # 질문 응답에서 선호도 추출
            prefs = {}
            for q_id, answer in self.question_answers.items():
                attribute = self.questions[q_id]['attribute']
                prefs[attribute] = answer

        # --- 연료 타입 매칭 ---
        if 'fuel_type' in prefs:
            fuel_pref = prefs['fuel_type']
            vehicle_fuel = vehicle.get('fuel_type', '')

            if fuel_pref == 3:  # "상관없음"
                score += 0.1
            elif (fuel_pref == 0 and vehicle_fuel == 'gasoline') or \
                 (fuel_pref == 1 and vehicle_fuel == 'hybrid') or \
                 (fuel_pref == 2 and vehicle_fuel == 'electric'):
                score += 0.2  # 정확히 매칭
            else:
                score -= 0.1  # 불일치

        # --- 차량 크기 매칭 ---
        if 'size' in prefs:
            size_pref = prefs['size']
            vehicle_size = vehicle.get('size', '')

            # 차량 크기를 인덱스로 변환 (mini/compact=0, mid=1, full=2)
            size_map = {'mini': 0, 'compact': 0, 'mid': 1, 'full': 2}
            vehicle_size_idx = size_map.get(vehicle_size, 1)

            if size_pref == 3:  # "상관없음"
                score += 0.05
            elif size_pref == vehicle_size_idx:
                score += 0.15  # 정확히 매칭
            elif abs(size_pref - vehicle_size_idx) == 1:
                score += 0.05  # 인접 크기

        # --- 차체 타입 매칭 ---
        if 'body_type' in prefs:
            body_pref = prefs['body_type']
            vehicle_category = vehicle.get('category', '')

            if body_pref == 3:  # "상관없음"
                score += 0.05
            elif (body_pref == 0 and vehicle_category == 'sedan') or \
                 (body_pref == 1 and vehicle_category in ['suv', 'electric']) or \
                 (body_pref == 2 and vehicle_category == 'mpv'):
                score += 0.15

        # --- 예산 매칭 ---
        if 'budget' in prefs:
            budget_pref = prefs['budget']
            price_range = vehicle.get('price_range', {})
            avg_price = (price_range.get('min', 3000) + price_range.get('max', 5000)) / 2

            # 예산 구간 정의 (만원 단위)
            budget_ranges = [
                (0, 3000),      # 0: 3000만원 이하
                (3000, 4500),   # 1: 3000-4500만원
                (4500, 6000),   # 2: 4500-6000만원
                (6000, 10000)   # 3: 6000만원 이상
            ]

            if budget_pref < len(budget_ranges):
                min_budget, max_budget = budget_ranges[budget_pref]
                if min_budget <= avg_price <= max_budget:
                    score += 0.2   # 예산 범위 내
                elif avg_price < min_budget:
                    score += 0.1   # 예산보다 저렴 (긍정적)
                else:
                    score -= 0.15  # 예산 초과 (부정적)

        # --- 우선순위 매칭 ---
        if 'priority' in prefs:
            priority_pref = prefs['priority']
            features = vehicle.get('features', {})

            # 우선순위 인덱스를 feature 키로 변환
            priority_to_feature = {
                0: 'safety',
                1: 'fuel_efficiency',
                2: 'performance',
                3: 'design'
            }

            if priority_pref in priority_to_feature:
                feature_key = priority_to_feature[priority_pref]
                feature_value = features.get(feature_key, 0.5)
                # feature 값이 높을수록 보너스 (0.5 기준으로 ±0.15)
                score += (feature_value - 0.5) * 0.3

        # --- 가족 구성 매칭 (좌석 수) ---
        if 'family_size' in prefs:
            family_pref = prefs['family_size']
            seats = vehicle.get('seats', 5)

            # 가족 구성별 최소 필요 좌석 수
            min_seats_required = {0: 2, 1: 4, 2: 5, 3: 7}
            required = min_seats_required.get(family_pref, 5)

            if seats >= required:
                score += 0.1   # 좌석 수 충족
            else:
                score -= 0.1   # 좌석 수 부족

        # 점수 범위 제한 (0.0 ~ 1.0)
        return max(0.0, min(1.0, score))

    def _update_candidate_scores(self):
        """
        모든 차량의 매칭 점수 업데이트

        현재까지의 질문 응답을 기반으로 각 차량의 점수를 재계산합니다.
        이 점수는 observation에 포함되어 에이전트에게 전달됩니다.
        """
        for vehicle in self.vehicles:
            score = self._calculate_vehicle_match_score(vehicle, use_hidden=False)
            self.candidate_scores[vehicle['name']] = score

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        에이전트 액션 수행

        Args:
            action: 수행할 액션 (0 ~ 11)
                - 0-7: 질문하기 (해당 ID의 질문 수행)
                - 8: Top 1 차량 추천
                - 9: Top 2 차량 추천
                - 10: Top 3 차량 추천
                - 11: 추가 질문 요청

        Returns:
            observation: 새로운 상태 벡터
            reward: 이번 액션에 대한 보상
            terminated: 에피소드 정상 종료 여부 (추천 완료)
            truncated: 에피소드 강제 종료 여부 (최대 질문 초과)
            info: 디버깅/로깅용 추가 정보
        """
        reward = 0.0
        terminated = False
        truncated = False

        if action < self.n_questions:
            # === 질문 액션 ===
            reward = self._handle_question(action)

            # 최대 질문 횟수 초과 시 강제 종료
            if self.questions_count >= self.max_questions:
                truncated = True
        else:
            # === 추천 액션 ===
            recommend_type = action - self.n_questions  # 0, 1, 2, 또는 3
            reward, terminated = self._handle_recommendation(recommend_type)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_question(self, question_id: int) -> float:
        """
        질문 액션 처리

        Args:
            question_id: 질문 ID (0-7)

        Returns:
            이번 질문에 대한 보상

        처리 과정:
            1. 중복 질문 체크 → 페널티 후 종료
            2. 질문 수행 및 응답 저장
            3. 차량 점수 업데이트
            4. 보상 계산 (기본 페널티 + 정보 획득 보너스)
        """
        # 중복 질문 체크
        if question_id in self.asked_questions:
            return self.REWARD_DUPLICATE_PENALTY

        # 질문 수행
        self.asked_questions.add(question_id)
        self.questions_count += 1

        # 고객 응답 시뮬레이션
        # 숨겨진 선호도에서 해당 질문의 응답을 가져옴
        question = self.questions[question_id]
        attribute = question['attribute']

        if attribute in self.hidden_preferences:
            answer = self.hidden_preferences[attribute]
        else:
            # 선호도에 없는 속성은 랜덤 응답
            answer = self.np_random.integers(0, len(question['options']))

        self.question_answers[question_id] = answer

        # 차량 점수 업데이트 (새 정보 반영)
        self._update_candidate_scores()

        # === 보상 계산 ===
        reward = self.REWARD_QUESTION_PENALTY  # 기본 페널티

        # 정보 획득 보너스: 질문으로 인해 차량 점수 분산이 커지면 보너스
        # (좋은 질문 = 차량들을 잘 구분해주는 질문)
        scores = list(self.candidate_scores.values())
        if len(scores) > 1:
            variance = np.var(scores)
            reward += variance * 0.5

        return reward

    def _handle_recommendation(self, recommend_type: int) -> tuple[float, bool]:
        """
        추천 액션 처리

        Args:
            recommend_type: 추천 유형
                - 0: Top 1 추천 (1개)
                - 1: Top 2 추천 (2개)
                - 2: Top 3 추천 (3개)
                - 3: 추가 질문 요청 (에피소드 계속)

        Returns:
            (reward, terminated) 튜플
        """
        # 추가 질문 요청: 작은 페널티, 에피소드 계속
        if recommend_type == 3:
            return self.REWARD_EXTRA_QUESTION, False

        # 추천 차량 수 결정
        n_recommend = recommend_type + 1  # 1, 2, 또는 3개

        # 현재 점수 기준 상위 차량 선택
        sorted_vehicles = sorted(
            self.candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        recommended = [name for name, _ in sorted_vehicles[:n_recommend]]

        # 보상 계산
        reward = self._calculate_recommendation_reward(recommended)

        return reward, True  # 추천 후 에피소드 종료

    def _calculate_recommendation_reward(self, recommended: list) -> float:
        """
        추천 보상 계산

        Args:
            recommended: 추천된 차량 이름 목록

        Returns:
            총 보상

        보상 구성:
            1. 고객 만족도 (최대 10점): 추천 차량이 ideal_vehicles와 얼마나 일치하는지
            2. 효율성 보너스 (최대 2점): 적은 질문으로 추천했을 때
            3. 정확도 보너스 (3점): 1순위를 정확히 맞췄을 때
            4. 과다 추천 페널티 (-0.5점): 3개 이상 추천 시
        """
        ideal_vehicles = self.hidden_preferences.get('ideal_vehicles', [])
        ideal_scores = self.hidden_preferences.get('ideal_scores', {})

        # === 1. 고객 만족도 계산 ===
        # 추천 순위에 따른 가중치 적용 (첫 번째 추천이 가장 중요)
        match_score = 0.0
        for rank, rec_vehicle in enumerate(recommended):
            weight = 1.0 / (rank + 1)  # 1위: 1.0, 2위: 0.5, 3위: 0.33

            if rec_vehicle in ideal_vehicles:
                # ideal 차량과 일치: 순위에 따른 보너스
                ideal_rank = ideal_vehicles.index(rec_vehicle)
                match_score += weight * (1.0 - ideal_rank * 0.2)
            else:
                # 일치하지 않으면 해당 차량의 실제 점수로 부분 보상
                match_score += weight * ideal_scores.get(rec_vehicle, 0.3) * 0.5

        # 정규화 (추천 개수에 따른 최대값으로 나눔)
        max_possible = sum(1.0 / (i + 1) for i in range(len(recommended)))
        if max_possible > 0:
            match_score /= max_possible

        reward = match_score * self.REWARD_MAX_SATISFACTION

        # === 2. 효율성 보너스 ===
        if self.questions_count <= 2:
            reward += self.REWARD_EFFICIENCY_BONUS_HIGH
        elif self.questions_count <= 3:
            reward += self.REWARD_EFFICIENCY_BONUS_MID

        # === 3. 정확도 보너스 ===
        if ideal_vehicles and recommended[0] == ideal_vehicles[0]:
            reward += self.REWARD_EXACT_MATCH_BONUS

        # === 4. 과다 추천 페널티 ===
        if len(recommended) > 2:
            reward += self.REWARD_TOO_MANY_RECOMMEND

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        현재 상태를 observation 벡터로 변환

        Returns:
            69차원 float32 벡터 (모든 값은 -1 ~ 1 범위로 정규화)

        벡터 구성:
            [0-4]: 고객 기본 정보 (5차원)
            [5-44]: 질문 응답 상태 (8질문 x 5옵션 = 40차원)
            [45]: 질문 횟수 비율 (1차원)
            [46-68]: 차량별 매칭 점수 (23차원)
        """
        obs = []
        customer = self.current_customer

        # === 1. 고객 기본 정보 (5차원) ===
        # 나이: 20~70세를 -1~1로 정규화 (기준: 45세)
        obs.append((customer.get('age', 35) - 45) / 25)
        # 성별: male=1.0, female=-1.0
        obs.append(1.0 if customer.get('gender') == 'male' else -1.0)
        # 외국인 여부: True=1.0, False=-1.0
        obs.append(1.0 if customer.get('is_foreigner', False) else -1.0)
        # 직장 유무: True=1.0, False=-1.0
        obs.append(1.0 if customer.get('has_workplace', True) else -1.0)
        # 관심 차량 유무: 있으면 1.0, 없으면 -1.0
        has_interest = len(customer.get('interested_cars', [])) > 0
        obs.append(1.0 if has_interest else -1.0)

        # === 2. 질문 응답 상태 (40차원) ===
        # 각 질문에 대해 5개 옵션 one-hot 인코딩
        # 미응답 질문은 모두 0.0
        for q_id in range(self.n_questions):
            if q_id in self.question_answers:
                answer = self.question_answers[q_id]
                for opt_idx in range(5):
                    obs.append(1.0 if opt_idx == answer else -1.0)
            else:
                obs.extend([0.0] * 5)

        # === 3. 질문 횟수 (1차원) ===
        obs.append(self.questions_count / self.max_questions)

        # === 4. 차량별 매칭 점수 (23차원) ===
        for vehicle in self.vehicles:
            score = self.candidate_scores.get(vehicle['name'], 0.5)
            obs.append(score * 2 - 1)  # [0,1] → [-1,1] 변환

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict:
        """
        디버깅/로깅용 추가 정보 반환

        Returns:
            - questions_count: 현재까지 질문 횟수
            - asked_questions: 질문한 질문 ID 목록
            - customer_type: 고객 유형 (new/interested/experienced)
            - top_candidates: 현재 상위 3개 차량과 점수
        """
        return {
            'questions_count': self.questions_count,
            'asked_questions': list(self.asked_questions),
            'customer_type': self.current_customer.get('customer_type', 'new'),
            'top_candidates': sorted(
                self.candidate_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }

    def render(self):
        """
        환경 상태를 콘솔에 출력 (render_mode="human"일 때만 동작)

        출력 내용:
            - 고객 기본 정보
            - 현재 질문 횟수
            - 질문-응답 내역
            - 추천 후보 Top 3
        """
        if self.render_mode != "human":
            return

        print("\n" + "=" * 50)
        print(f"고객 정보: {self.current_customer.get('age')}세, "
              f"{self.current_customer.get('gender')}, "
              f"{self.current_customer.get('region')}")
        print(f"질문 횟수: {self.questions_count}/{self.max_questions}")
        print("-" * 50)

        # 질문-응답 내역 출력
        if self.question_answers:
            print("응답 내역:")
            for q_id, answer in self.question_answers.items():
                question = self.questions[q_id]
                options = question['options']
                answer_text = options[answer] if answer < len(options) else "기타"
                print(f"  Q: {question['text']}")
                print(f"  A: {answer_text}")

        print("-" * 50)

        # 추천 후보 출력
        print("추천 후보 (Top 3):")
        sorted_scores = sorted(
            self.candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for name, score in sorted_scores:
            print(f"  {name}: {score:.3f}")

        print("=" * 50)

    def close(self):
        """환경 리소스 정리 (현재는 특별한 정리 작업 없음)"""
        pass
