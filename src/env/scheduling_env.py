"""
시승 스케줄링 환경 (Phase 2)

강화학습 에이전트가 고객 요청과 센터 상태를 고려하여
최적의 시승 일정을 배정하는 환경입니다.
Gymnasium 인터페이스를 따릅니다.

핵심 개념:
    - 고객이 선호하는 일정이 가용하면 즉시 예약 확정
    - 불가능한 경우 대안을 제시하고, 고객의 수락/거절 시뮬레이션
    - 센터 자원(차량, 직원, 시간)을 효율적으로 배분하는 것이 목표

최적화:
    - 딕셔너리 대신 NumPy 배열 사용으로 O(n) → O(1) 접근
    - 차량 가용성 캐싱으로 매 스텝 재계산 방지
"""

import json
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SchedulingEnv(gym.Env):
    """
    시승 스케줄링 강화학습 환경

    에이전트는 고객 요청에 대해 예약을 확정하거나 대안을 제시합니다.
    목표: 높은 예약 성사율 + 고객 만족도 + 센터 자원 효율화

    Action Space (Discrete(6)):
        - 0: 예약 확정 (현재 요청 시간에 배정)
        - 1: 같은 날 다른 시간 제안
        - 2: 다음 날 같은 시간 제안
        - 3: 평일 대안 제안
        - 4: 다음 주 제안
        - 5: 인센티브 제공 (비선호 시간대 유도)

    Observation Space (Box, 다차원):
        - 고객 선호 정보: 선호 날짜 타입, 시간대, 유연성
        - 센터 상태: 슬롯별 가용성, 차량 가용성, 직원 상태
        - 현재 요청: 추천 차량, 요청 시도 횟수

    Reward 구조:
        - 예약 성사: +10
        - 선호 시간 매칭: +5
        - 대안 수락: +3
        - 대기시간 패널티: -2 * (시간/30분)
        - 부하 분산 보너스: +2
        - 고객 거절 (이탈): -5
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # =========================================================================
    # 보상 관련 상수
    # =========================================================================
    REWARD_BOOKING_CONFIRMED = 10.0     # 예약 성사
    REWARD_PREFERRED_TIME = 5.0         # 선호 시간 매칭
    REWARD_ALTERNATIVE_ACCEPTED = 3.0   # 대안 수락
    REWARD_LOAD_BALANCE = 2.0           # 부하 분산 보너스
    REWARD_WAITING_PENALTY = -0.5       # 대기 시간당 패널티
    REWARD_CUSTOMER_REJECTED = -5.0     # 고객 이탈
    REWARD_INVALID_ACTION = -1.0        # 무효 액션 (가용하지 않은 슬롯 확정 시도)

    # 시간 슬롯 상수
    SLOTS_PER_DAY = 9           # 09:00 ~ 17:00 (1시간 단위)
    DAYS_IN_SCHEDULE = 14       # 2주 스케줄

    def __init__(
        self,
        centers_path: str = None,
        vehicles_path: str = None,
        max_attempts: int = 3,
        render_mode: Optional[str] = None
    ):
        """
        환경 초기화

        Args:
            centers_path: 센터 데이터 JSON 경로
            vehicles_path: 차량 데이터 JSON 경로
            max_attempts: 최대 대안 제시 횟수 (초과 시 고객 이탈)
            render_mode: 렌더링 모드
        """
        super().__init__()

        # === 데이터 경로 설정 ===
        project_root = Path(__file__).parent.parent.parent
        centers_path = centers_path or str(project_root / "data" / "centers.json")
        vehicles_path = vehicles_path or str(project_root / "data" / "vehicles.json")

        # === 데이터 로드 ===
        self.centers_data = self._load_json(centers_path)
        self.centers = self.centers_data.get('centers', [])
        self.time_slots = self.centers_data.get('time_slots', [])
        self.flexibility_types = self.centers_data.get('customer_flexibility_types', [])
        self.peak_times = self.centers_data.get('peak_times', {})
        self.sim_config = self.centers_data.get('simulation_config', {})

        self.vehicles_data = self._load_json(vehicles_path)
        self.vehicles = self.vehicles_data.get('vehicles', [])

        # === 차량 ID → 인덱스 매핑 (O(1) 조회용) ===
        self.vehicle_id_to_idx = {v['id']: i for i, v in enumerate(self.vehicles)}

        # === 환경 설정 ===
        self.max_attempts = max_attempts
        self.render_mode = render_mode
        self.n_centers = len(self.centers)
        self.n_vehicles = len(self.vehicles)
        self.n_slots = self.SLOTS_PER_DAY * self.DAYS_IN_SCHEDULE

        # === Action Space 정의 ===
        # 0: 확정, 1-4: 대안 제시, 5: 인센티브
        self.action_space = spaces.Discrete(6)

        # === Observation Space 정의 ===
        # 고객 선호(6) + 센터 슬롯 가용성(126) + 차량 가용(n_vehicles) + 메타(4)
        obs_dim = 6 + self.n_slots + self.n_vehicles + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # === NumPy 배열 기반 상태 (성능 최적화) ===
        # schedule_state: (DAYS, SLOTS, n_vehicles) - 각 슬롯별 차량 가용 대수
        # staff_available: (DAYS, SLOTS) - 각 슬롯별 직원 가용 수
        # vehicle_availability: (n_vehicles,) - 차량별 총 가용 슬롯 (캐싱)
        self.schedule_state = None
        self.staff_available = None
        self._vehicle_availability_cache = None
        self._slot_availability_cache = None

        # === 에피소드 상태 변수 (reset에서 초기화) ===
        self.current_center = None          # 현재 센터
        self.current_customer = None        # 현재 고객 요청
        self.recommended_vehicle = None     # Phase 1에서 추천된 차량
        self.recommended_vehicle_idx = 0    # 추천 차량 인덱스
        self.attempt_count = 0              # 현재 대안 제시 횟수
        self.current_day = 0                # 현재 요청의 기준 날짜
        self.current_slot = 0               # 현재 요청의 기준 시간

    def _load_json(self, path: str) -> dict:
        """JSON 파일 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """
        환경 초기화 및 새 에피소드 시작

        Args:
            seed: 랜덤 시드
            options: 추가 옵션
                - center_idx: 특정 센터 지정
                - vehicle_id: Phase 1에서 추천된 차량 ID
                - prefill_ratio: 기존 예약 비율 (0.0 ~ 1.0)

        Returns:
            observation: 초기 상태
            info: 추가 정보
        """
        super().reset(seed=seed)

        # === 센터 선택 ===
        if options and 'center_idx' in options:
            center_idx = options['center_idx']
        else:
            center_idx = self.np_random.integers(0, self.n_centers)
        self.current_center = self.centers[center_idx]

        # === 추천 차량 설정 ===
        if options and 'vehicle_id' in options:
            self.recommended_vehicle = options['vehicle_id']
        else:
            # 센터에서 보유한 차량 중 랜덤 선택
            center_vehicles = [v['vehicle_id'] for v in self.current_center['vehicles']]
            self.recommended_vehicle = self.np_random.choice(center_vehicles)

        # 추천 차량 인덱스 캐싱
        self.recommended_vehicle_idx = self.vehicle_id_to_idx.get(self.recommended_vehicle, 0)

        # === 스케줄 상태 초기화 (NumPy 배열 사용) ===
        prefill_ratio = options.get('prefill_ratio', 0.3) if options else 0.3
        self._initialize_schedule(prefill_ratio)

        # === 고객 요청 생성 ===
        self._generate_customer_request()

        # === 상태 초기화 ===
        self.attempt_count = 0

        return self._get_observation(), self._get_info()

    def _initialize_schedule(self, prefill_ratio: float):
        """
        2주 스케줄 상태 초기화 (완전 벡터화 - 고성능)

        Args:
            prefill_ratio: 기존 예약으로 채울 비율
        """
        # === 센터 차량 정보를 배열로 변환 ===
        center_vehicle_counts = np.zeros(self.n_vehicles, dtype=np.int8)
        for v in self.current_center['vehicles']:
            idx = self.vehicle_id_to_idx.get(v['vehicle_id'], -1)
            if idx >= 0:
                center_vehicle_counts[idx] = v['count']

        # === fill_prob 매트릭스 생성 (완전 벡터화) ===
        # 주말 마스크: (DAYS,)
        days = np.arange(self.DAYS_IN_SCHEDULE)
        is_weekend = (days % 7) >= 5  # (DAYS,)

        # 피크 슬롯 마스크: (DAYS, SLOTS)
        weekday_peak_set = set(self.peak_times.get('weekday', []))
        weekend_peak_set = set(self.peak_times.get('weekend', []))

        is_peak = np.zeros((self.DAYS_IN_SCHEDULE, self.SLOTS_PER_DAY), dtype=bool)
        for slot in range(self.SLOTS_PER_DAY):
            for day in range(self.DAYS_IN_SCHEDULE):
                if is_weekend[day]:
                    is_peak[day, slot] = slot in weekend_peak_set
                else:
                    is_peak[day, slot] = slot in weekday_peak_set

        # fill_prob 계산: (DAYS, SLOTS)
        weekend_mult = np.where(is_weekend[:, np.newaxis], 1.2, 1.0)
        peak_mult = np.where(is_peak, 1.5, 1.0)
        fill_prob = np.minimum(0.9, prefill_ratio * weekend_mult * peak_mult)

        # === 스케줄 초기화 (이항분포 사용 - 완전 벡터화) ===
        # 각 차량의 최대 대수
        max_count = int(center_vehicle_counts.max()) if center_vehicle_counts.max() > 0 else 1

        # 한 번에 모든 랜덤값 생성: (DAYS, SLOTS, n_vehicles, max_count)
        random_vals = self.np_random.random((self.DAYS_IN_SCHEDULE, self.SLOTS_PER_DAY, self.n_vehicles, max_count))

        # fill_prob를 확장: (DAYS, SLOTS, 1, 1)
        fill_prob_expanded = fill_prob[:, :, np.newaxis, np.newaxis]

        # 예약 여부 판단
        booked_mask = random_vals < fill_prob_expanded  # (DAYS, SLOTS, n_vehicles, max_count)

        # 각 차량별 실제 대수만큼만 합산
        # count_mask: (n_vehicles, max_count) - 각 차량의 유효 슬롯
        count_range = np.arange(max_count)
        count_mask = count_range[np.newaxis, :] < center_vehicle_counts[:, np.newaxis]  # (n_vehicles, max_count)

        # 유효한 슬롯에서만 booked 카운트
        valid_booked = booked_mask & count_mask[np.newaxis, np.newaxis, :, :]  # (DAYS, SLOTS, n_vehicles, max_count)
        booked_count = valid_booked.sum(axis=3)  # (DAYS, SLOTS, n_vehicles)

        # 가용 대수 = 총 대수 - 예약된 수
        self.schedule_state = np.maximum(
            0,
            center_vehicle_counts[np.newaxis, np.newaxis, :] - booked_count
        ).astype(np.int8)

        # === 직원 가용성 초기화 (NumPy 배열) ===
        staff_count = self.current_center.get('staff_count', 3)
        variation = self.np_random.integers(-1, 2, size=(self.DAYS_IN_SCHEDULE, self.SLOTS_PER_DAY))
        self.staff_available = np.maximum(1, staff_count + variation).astype(np.int8)

        # === 캐시 초기화 ===
        self._update_availability_cache()

    def _generate_customer_request(self):
        """
        고객 시승 요청 생성

        고객의 선호 일정과 유연성을 시뮬레이션합니다.
        """
        # 선호 날짜 타입 (0: 이번주 평일, 1: 이번주 주말, 2: 다음주 평일, 3: 다음주 주말)
        date_type_weights = [0.2, 0.35, 0.2, 0.25]  # 주말 선호 높음
        preferred_date_type = self.np_random.choice(4, p=date_type_weights)

        # 날짜 타입에 따른 실제 날짜 계산
        if preferred_date_type == 0:  # 이번주 평일
            self.current_day = self.np_random.integers(0, 5)
        elif preferred_date_type == 1:  # 이번주 주말
            self.current_day = self.np_random.choice([5, 6])
        elif preferred_date_type == 2:  # 다음주 평일
            self.current_day = self.np_random.integers(7, 12)
        else:  # 다음주 주말
            self.current_day = self.np_random.choice([12, 13])

        # 선호 시간대 (0: 오전, 1: 오후, 2: 저녁)
        time_pref_weights = [0.3, 0.45, 0.25]
        preferred_time = self.np_random.choice(3, p=time_pref_weights)

        # 시간대에 따른 실제 슬롯 계산
        if preferred_time == 0:  # 오전 (09:00-12:00)
            self.current_slot = self.np_random.integers(0, 3)
        elif preferred_time == 1:  # 오후 (13:00-16:00)
            self.current_slot = self.np_random.integers(4, 7)
        else:  # 저녁 (16:00-18:00)
            self.current_slot = self.np_random.integers(7, 9)

        # 유연성 (high: 대안 수락률 80%, medium: 50%, low: 20%)
        flex_weights = [0.3, 0.5, 0.2]
        flexibility_idx = self.np_random.choice(3, p=flex_weights)
        flexibility = self.flexibility_types[flexibility_idx]

        self.current_customer = {
            'preferred_date_type': preferred_date_type,
            'preferred_day': self.current_day,
            'preferred_time': preferred_time,
            'preferred_slot': self.current_slot,
            'flexibility': flexibility['type'],
            'alt_accept_prob': flexibility['alt_accept_prob'],
            'vehicle_id': self.recommended_vehicle
        }

    def _update_availability_cache(self):
        """가용성 캐시 업데이트 (reset 및 예약 후 호출)"""
        # 슬롯별 가용성: 추천 차량과 직원 모두 가용한지 체크
        vid_idx = self.recommended_vehicle_idx
        vehicle_avail = self.schedule_state[:, :, vid_idx] > 0  # (DAYS, SLOTS)
        staff_avail = self.staff_available > 0  # (DAYS, SLOTS)

        # 둘 다 가용하면 1.0, 하나만 가용하면 0.0, 둘 다 불가하면 -1.0
        self._slot_availability_cache = np.where(
            vehicle_avail & staff_avail, 1.0,
            np.where(vehicle_avail | staff_avail, 0.0, -1.0)
        ).astype(np.float32).flatten()

        # 차량별 총 가용 슬롯 비율 (정규화)
        total_per_vehicle = self.schedule_state.sum(axis=(0, 1))  # (n_vehicles,)
        max_possible = self.n_slots * 2  # 최대값 추정
        self._vehicle_availability_cache = (
            np.minimum(1.0, total_per_vehicle / max_possible) * 2 - 1
        ).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        에이전트 액션 수행

        Args:
            action: 수행할 액션
                - 0: 예약 확정
                - 1: 같은 날 다른 시간 제안
                - 2: 다음 날 같은 시간 제안
                - 3: 평일 대안 제안
                - 4: 다음 주 제안
                - 5: 인센티브 제공

        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0
        terminated = False
        truncated = False

        if action == 0:
            # 예약 확정 시도
            reward, terminated = self._handle_confirm_booking()
        else:
            # 대안 제시
            reward, terminated = self._handle_suggest_alternative(action)

        # 최대 시도 횟수 초과
        if self.attempt_count >= self.max_attempts and not terminated:
            reward += self.REWARD_CUSTOMER_REJECTED
            truncated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_confirm_booking(self) -> tuple[float, bool]:
        """
        예약 확정 처리 (NumPy 배열 기반)

        Returns:
            (reward, terminated)
        """
        day, slot = self.current_day, self.current_slot
        vid_idx = self.recommended_vehicle_idx

        # 가용성 체크 (O(1) 배열 접근)
        vehicle_available = self.schedule_state[day, slot, vid_idx] > 0
        staff_available = self.staff_available[day, slot] > 0

        if not vehicle_available or not staff_available:
            # 가용하지 않은 슬롯에 확정 시도 → 무효 액션
            self.attempt_count += 1  # 실패해도 시도 횟수 증가 (무한루프 방지)
            return self.REWARD_INVALID_ACTION, False

        # 예약 확정
        self.schedule_state[day, slot, vid_idx] -= 1
        self.staff_available[day, slot] -= 1

        reward = self.REWARD_BOOKING_CONFIRMED

        # 선호 시간 매칭 보너스
        if (day == self.current_customer['preferred_day'] and
            slot == self.current_customer['preferred_slot']):
            reward += self.REWARD_PREFERRED_TIME

        # 부하 분산 보너스 (비피크 시간 예약 시)
        is_weekend = day % 7 >= 5
        peak_slots = self.peak_times.get('weekend' if is_weekend else 'weekday', [])
        if slot not in peak_slots:
            reward += self.REWARD_LOAD_BALANCE

        return reward, True

    def _handle_suggest_alternative(self, action: int) -> tuple[float, bool]:
        """
        대안 제시 처리 (NumPy 배열 기반)

        Args:
            action: 대안 유형 (1-5)

        Returns:
            (reward, terminated)
        """
        self.attempt_count += 1

        # 대안 슬롯 계산
        alt_day, alt_slot = self._calculate_alternative_slot(action)
        vid_idx = self.recommended_vehicle_idx

        # 대안 가용성 체크 (O(1) 배열 접근)
        vehicle_available = self.schedule_state[alt_day, alt_slot, vid_idx] > 0
        staff_available = self.staff_available[alt_day, alt_slot] > 0

        if not vehicle_available or not staff_available:
            # 대안도 불가능 → 작은 페널티
            return self.REWARD_WAITING_PENALTY, False

        # 고객 수락 여부 시뮬레이션
        accept_prob = self.current_customer['alt_accept_prob']

        # 인센티브 제공 시 수락률 증가
        if action == 5:
            accept_prob = min(1.0, accept_prob + 0.2)

        # 대안이 원래 요청과 가까울수록 수락률 증가
        day_diff = abs(alt_day - self.current_customer['preferred_day'])
        slot_diff = abs(alt_slot - self.current_customer['preferred_slot'])
        proximity_bonus = max(0, 0.2 - (day_diff * 0.05 + slot_diff * 0.02))
        accept_prob = min(1.0, accept_prob + proximity_bonus)

        if self.np_random.random() < accept_prob:
            # 고객 수락 → 예약 확정
            self.schedule_state[alt_day, alt_slot, vid_idx] -= 1
            self.staff_available[alt_day, alt_slot] -= 1

            reward = self.REWARD_BOOKING_CONFIRMED + self.REWARD_ALTERNATIVE_ACCEPTED

            # 대기 시간 패널티 (시도 횟수에 비례)
            reward += self.REWARD_WAITING_PENALTY * self.attempt_count

            return reward, True
        else:
            # 고객 거절 → 다시 시도 필요
            reward = self.REWARD_WAITING_PENALTY
            return reward, False

    def _calculate_alternative_slot(self, action: int) -> tuple[int, int]:
        """
        대안 슬롯 계산

        Args:
            action: 대안 유형

        Returns:
            (day, slot) 튜플
        """
        base_day = self.current_customer['preferred_day']
        base_slot = self.current_customer['preferred_slot']

        if action == 1:  # 같은 날 다른 시간
            alt_day = base_day
            # 다른 시간 선택 (현재와 ±2 슬롯 범위)
            offset = self.np_random.choice([-2, -1, 1, 2])
            alt_slot = max(0, min(self.SLOTS_PER_DAY - 1, base_slot + offset))

        elif action == 2:  # 다음 날 같은 시간
            alt_day = min(self.DAYS_IN_SCHEDULE - 1, base_day + 1)
            alt_slot = base_slot

        elif action == 3:  # 평일 대안
            # 가장 가까운 평일 찾기
            for offset in range(1, 8):
                candidate = base_day + offset
                if candidate < self.DAYS_IN_SCHEDULE and candidate % 7 < 5:
                    alt_day = candidate
                    break
            else:
                alt_day = base_day
            alt_slot = base_slot

        elif action == 4:  # 다음 주
            alt_day = min(self.DAYS_IN_SCHEDULE - 1, base_day + 7)
            alt_slot = base_slot

        else:  # action == 5: 인센티브 (비피크 시간 제안)
            alt_day = base_day
            # 비피크 슬롯 선택
            is_weekend = base_day % 7 >= 5
            peak_slots = set(self.peak_times.get('weekend' if is_weekend else 'weekday', []))
            non_peak = [s for s in range(self.SLOTS_PER_DAY) if s not in peak_slots]
            alt_slot = self.np_random.choice(non_peak) if non_peak else base_slot

        # 현재 요청 위치 업데이트 (다음 시도를 위해)
        self.current_day = alt_day
        self.current_slot = alt_slot

        return alt_day, alt_slot

    def _get_observation(self) -> np.ndarray:
        """
        현재 상태를 observation 벡터로 변환 (캐시 기반 최적화)

        Returns:
            정규화된 상태 벡터
        """
        customer = self.current_customer

        # === 1. 고객 선호 정보 (6차원) ===
        # 선호 날짜 타입 (one-hot, 4차원)
        date_type_onehot = np.full(4, -1.0, dtype=np.float32)
        date_type_onehot[customer['preferred_date_type']] = 1.0

        # 선호 시간대 (정규화): 0,1,2 → -1,0,1
        time_pref = np.float32((customer['preferred_time'] - 1) / 1.0)

        # 유연성 (high=1, medium=0, low=-1)
        flex_map = {'high': 1.0, 'medium': 0.0, 'low': -1.0}
        flexibility = np.float32(flex_map.get(customer['flexibility'], 0.0))

        customer_info = np.concatenate([
            date_type_onehot,
            [time_pref, flexibility]
        ])

        # === 2. 센터 슬롯 가용성 (캐시 사용) ===
        # _slot_availability_cache: (126,) 이미 계산됨

        # === 3. 차량 가용성 (캐시 사용) ===
        # _vehicle_availability_cache: (n_vehicles,) 이미 계산됨

        # === 4. 메타 정보 (4차원) ===
        meta_info = np.array([
            self.attempt_count / self.max_attempts,
            self.current_day / self.DAYS_IN_SCHEDULE * 2 - 1,
            self.current_slot / self.SLOTS_PER_DAY * 2 - 1,
            1.0 if self.current_day % 7 >= 5 else -1.0
        ], dtype=np.float32)

        # 전체 observation 조립
        return np.concatenate([
            customer_info,                      # 6
            self._slot_availability_cache,      # 126
            self._vehicle_availability_cache,   # n_vehicles
            meta_info                           # 4
        ])

    def _get_info(self) -> dict:
        """디버깅/로깅용 추가 정보"""
        day, slot = self.current_day, self.current_slot
        vid_idx = self.recommended_vehicle_idx

        return {
            'center': self.current_center['name'],
            'vehicle': self.recommended_vehicle,
            'requested_day': day,
            'requested_slot': slot,
            'slot_available': self.schedule_state[day, slot, vid_idx] > 0,
            'staff_available': self.staff_available[day, slot] > 0,
            'attempt_count': self.attempt_count,
            'customer_flexibility': self.current_customer['flexibility']
        }

    def render(self):
        """환경 상태를 콘솔에 출력"""
        if self.render_mode != "human":
            return

        day, slot = self.current_day, self.current_slot
        vid_idx = self.recommended_vehicle_idx

        print("\n" + "=" * 60)
        print(f"센터: {self.current_center['name']}")
        print(f"추천 차량: {self.recommended_vehicle}")
        print("-" * 60)
        print(f"고객 요청:")
        print(f"  선호 날짜: Day {self.current_customer['preferred_day']} "
              f"({'주말' if self.current_customer['preferred_day'] % 7 >= 5 else '평일'})")
        print(f"  선호 시간: Slot {self.current_customer['preferred_slot']} "
              f"({self.time_slots[self.current_customer['preferred_slot']]['label']})")
        print(f"  유연성: {self.current_customer['flexibility']}")
        print("-" * 60)
        print(f"현재 상태:")
        print(f"  요청 위치: Day {day}, Slot {slot}")
        print(f"  차량 가용: {self.schedule_state[day, slot, vid_idx]}대")
        print(f"  직원 가용: {self.staff_available[day, slot]}명")
        print(f"  시도 횟수: {self.attempt_count}/{self.max_attempts}")
        print("=" * 60)

    def close(self):
        """환경 리소스 정리"""
        pass


# === 환경 등록 (선택사항) ===
# gym.register(
#     id='SchedulingEnv-v0',
#     entry_point='src.env.scheduling_env:SchedulingEnv',
# )
