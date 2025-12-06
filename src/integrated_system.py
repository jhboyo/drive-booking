"""
Phase 3: 통합 시스템 (Integrated System)

Phase 1(차량 추천)과 Phase 2(스케줄링)를 연결하는 End-to-End 파이프라인임.
개별 Phase에서 학습된 Agent들을 통합하여 차량 추천부터 일정 예약까지
전체 프로세스를 하나의 시스템으로 처리함.

핵심 개념:
    - Phase 1 추천 결과가 Phase 2 입력으로 자동 전달
    - 통합 보상 = R1 + R2 + Synergy Bonus
    - End-to-End 최적화로 전체 시스템 성능 향상

파이프라인 흐름:
    고객 입력 → Phase 1 (차량 추천) → Phase 2 (스케줄링) → 예약 확정
"""

from pathlib import Path
from typing import Optional

import numpy as np

# Phase 1 환경 및 에이전트 (차량 추천)
from src.env.recommendation_env import VehicleRecommendationEnv
from src.agents.q_learning_agent import QLearningAgent

# Phase 2 환경 및 에이전트 (스케줄링)
from src.env.scheduling_env import SchedulingEnv
from src.agents.scheduling_agent import DQNAgent


class IntegratedSystem:
    """
    Phase 1 + Phase 2 통합 시스템

    차량 추천(Q-Learning)과 스케줄링(DQN)을 파이프라인으로 연결하여
    고객 입력부터 예약 확정까지 End-to-End 처리를 수행함.

    ========================================
    통합 보상 함수 (핵심 수식)
    ========================================

    Total_Reward = R1 + R2 + Synergy_Bonus

    R1 (Phase 1 보상):
        - 고객 만족도: 추천 차량과 고객 선호 매칭 점수 (최대 10점)
        - 효율성 보너스: 적은 질문으로 추천 시 추가 보상 (최대 2점)
        - 질문 페널티: 질문당 -0.2점

    R2 (Phase 2 보상):
        - 예약 성사: +10점
        - 선호 시간 매칭: +5점
        - 대기시간 페널티: 시도당 -0.5점

    Synergy_Bonus (통합 시너지):
        - 즉시 예약 성공: +5점 (Phase 2에서 첫 시도에 예약 성공)
        - 추천-스케줄 매칭: +3점 (추천 차량이 선호 시간에 가용)

    ========================================

    연구 기여점:
        1. Two-Phase RL 파이프라인 구조 제안
        2. Synergy Bonus를 통한 End-to-End 최적화
        3. 개별 학습 → 통합 미세조정 학습 전략
    """

    # =========================================================================
    # 시너지 보너스 상수 정의
    # =========================================================================
    # 즉시 예약 성공 보너스: Phase 2에서 대안 제시 없이 바로 예약 성공 시
    SYNERGY_IMMEDIATE_BOOKING = 5.0

    # 추천-스케줄 매칭 보너스: Phase 1 추천 차량이 고객 선호 시간에 가용할 때
    SYNERGY_SCHEDULE_MATCH = 3.0

    def __init__(
        self,
        phase1_agent: Optional[QLearningAgent] = None,
        phase2_agent: Optional[DQNAgent] = None,
        seed: int = 42
    ):
        """
        통합 시스템 초기화

        두 개의 독립적인 강화학습 환경과 에이전트를 생성하고,
        파이프라인으로 연결하기 위한 준비를 수행함.

        Args:
            phase1_agent: 사전 학습된 Phase 1 에이전트
                - None이면 새로운 Q-Learning 에이전트 생성
                - 기존 모델을 재사용하려면 학습된 에이전트 전달
            phase2_agent: 사전 학습된 Phase 2 에이전트
                - None이면 새로운 DQN 에이전트 생성
            seed: 랜덤 시드 (재현성 보장)
        """
        self.seed = seed

        # =====================================================================
        # Phase 1: 차량 추천 환경 및 에이전트 초기화
        # =====================================================================
        # 환경: 고객과 대화하며 최적 차량을 추천하는 Gymnasium 환경
        self.phase1_env = VehicleRecommendationEnv()

        # 에이전트: Q-Learning 기반 (테이블 기반 강화학습)
        # - 상태: 고객 프로필 + 질문 응답 (69차원)
        # - 액션: 질문(8개) + 추천(4개) = 12개
        self.phase1_agent = phase1_agent or QLearningAgent(
            n_actions=self.phase1_env.action_space.n,  # 12개 액션
            seed=seed
        )

        # =====================================================================
        # Phase 2: 스케줄링 환경 및 에이전트 초기화
        # =====================================================================
        # 환경: 시승 일정을 배정하는 Gymnasium 환경
        self.phase2_env = SchedulingEnv()

        # 에이전트: DQN 기반 (신경망 기반 강화학습)
        # - 상태: 고객 선호 + 센터 가용성 (159차원)
        # - 액션: 확정(1) + 대안 제시(5) = 6개
        self.phase2_agent = phase2_agent or DQNAgent(
            state_dim=self.phase2_env.observation_space.shape[0],  # 159차원
            action_dim=self.phase2_env.action_space.n,  # 6개 액션
            seed=seed
        )

        # =====================================================================
        # 에피소드 통계 추적 (학습 모니터링용)
        # =====================================================================
        self.episode_count = 0          # 완료된 에피소드 수
        self.total_rewards = []         # 에피소드별 총 보상 (R1+R2+Synergy)
        self.phase1_rewards = []        # 에피소드별 Phase 1 보상
        self.phase2_rewards = []        # 에피소드별 Phase 2 보상
        self.synergy_bonuses = []       # 에피소드별 시너지 보너스

    def run_episode(
        self,
        training: bool = False,
        customer_idx: Optional[int] = None,
        center_idx: Optional[int] = None
    ) -> dict:
        """
        End-to-End 에피소드 실행 (핵심 메서드)

        Phase 1 → Phase 2 순차 실행하여 전체 파이프라인 수행.
        하나의 고객이 차량 추천을 받고 예약까지 완료하는 전체 과정을 시뮬레이션함.

        실행 흐름:
            1. Phase 1 실행: 고객에게 질문하고 차량 추천
            2. Phase 1 결과 전달: 추천 차량 ID를 Phase 2에 전달
            3. Phase 2 실행: 추천 차량으로 시승 일정 배정
            4. 시너지 보너스 계산: 두 Phase 연계 효과 평가
            5. 통합 결과 생성: 모든 지표 취합

        Args:
            training: 학습 모드 여부
                - True: 에이전트 업데이트 수행 (탐험 포함)
                - False: 평가 모드 (greedy 정책)
            customer_idx: 특정 고객 지정 (테스트/디버깅용)
            center_idx: 특정 센터 지정 (테스트/디버깅용)

        Returns:
            에피소드 결과 딕셔너리:
                - total_reward: 총 보상 (R1 + R2 + Synergy)
                - end_to_end_success: 전체 성공 여부
                - questions_count: Phase 1에서 한 질문 수
                - attempt_count: Phase 2에서 시도한 횟수
                - 기타 상세 지표
        """
        # =====================================================================
        # Step 1: Phase 1 (차량 추천) 실행
        # =====================================================================
        phase1_result = self._run_phase1(
            training=training,
            customer_idx=customer_idx
        )

        # Phase 1 실패 시 조기 종료 (추천 완료 전에 truncated된 경우)
        if not phase1_result['success']:
            return self._create_result(
                phase1_result=phase1_result,
                phase2_result=None,
                synergy_bonus=0.0
            )

        # =====================================================================
        # Step 2: Phase 1 결과를 Phase 2에 전달
        # =====================================================================
        # Phase 1에서 추천된 차량 ID 추출
        recommended_vehicle = phase1_result['recommended_vehicle']

        # =====================================================================
        # Step 3: Phase 2 (스케줄링) 실행
        # =====================================================================
        phase2_result = self._run_phase2(
            vehicle_id=recommended_vehicle,  # 추천 차량 전달 (핵심 연결점)
            training=training,
            center_idx=center_idx
        )

        # =====================================================================
        # Step 4: 시너지 보너스 계산
        # =====================================================================
        # 두 Phase 간 연계 효과를 수치화
        synergy_bonus = self._calculate_synergy_bonus(
            phase1_result=phase1_result,
            phase2_result=phase2_result
        )

        # =====================================================================
        # Step 5: 통합 결과 생성
        # =====================================================================
        result = self._create_result(
            phase1_result=phase1_result,
            phase2_result=phase2_result,
            synergy_bonus=synergy_bonus
        )

        # =====================================================================
        # 통계 업데이트 (학습 모니터링용)
        # =====================================================================
        self.episode_count += 1
        self.total_rewards.append(result['total_reward'])
        self.phase1_rewards.append(result['phase1_reward'])
        self.phase2_rewards.append(result['phase2_reward'])
        self.synergy_bonuses.append(synergy_bonus)

        return result

    def _run_phase1(
        self,
        training: bool = False,
        customer_idx: Optional[int] = None
    ) -> dict:
        """
        Phase 1 (차량 추천) 실행

        고객 프로필 기반으로 대화형 추천을 수행함.
        에이전트가 질문을 선택하고, 적절한 시점에 차량을 추천함.

        강화학습 루프:
            while not done:
                1. 에이전트가 액션 선택 (질문 or 추천)
                2. 환경이 상태 전이 및 보상 계산
                3. (학습 모드) Q-table 업데이트

        Args:
            training: 학습 모드 여부
            customer_idx: 특정 고객 지정

        Returns:
            Phase 1 결과 딕셔너리:
                - success: 추천 완료 여부 (terminated=True)
                - reward: 에피소드 누적 보상
                - recommended_vehicle: 추천된 차량 ID
                - questions_count: 질문 횟수
        """
        # =====================================================================
        # 환경 초기화 (새 고객 생성)
        # =====================================================================
        options = {'customer_idx': customer_idx} if customer_idx is not None else None
        obs, info = self.phase1_env.reset(
            seed=self.seed + self.episode_count,  # 에피소드마다 다른 시드
            options=options
        )

        episode_reward = 0.0  # 에피소드 누적 보상
        steps = 0             # 스텝 수 (질문 + 추천 시도)
        done = False          # 에피소드 종료 플래그

        # =====================================================================
        # 강화학습 루프: 에이전트-환경 상호작용
        # =====================================================================
        while not done:
            # 1. 에이전트가 현재 상태에서 액션 선택
            #    - training=True: ε-greedy (탐험 포함)
            #    - training=False: greedy (최적 액션만)
            action = self.phase1_agent.select_action(obs, training=training)

            # 2. 환경에서 액션 실행 → 다음 상태, 보상, 종료 여부 반환
            next_obs, reward, terminated, truncated, info = self.phase1_env.step(action)

            # 3. 학습 모드일 때 Q-table 업데이트
            #    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
            if training:
                self.phase1_agent.update(obs, action, reward, next_obs, terminated, truncated)

            # 상태 전이
            obs = next_obs
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # =====================================================================
        # 에피소드 종료 처리
        # =====================================================================
        if training:
            # ε 감소 및 통계 업데이트
            self.phase1_agent.end_episode()

        # 추천된 차량 추출 (현재 점수 기준 상위 1개)
        top_candidates = info.get('top_candidates', [])
        recommended_vehicle = top_candidates[0][0] if top_candidates else None

        # 차량 이름을 ID 형식으로 변환 (Phase 2 환경과 호환)
        vehicle_id = self._get_vehicle_id(recommended_vehicle)

        return {
            'success': terminated,  # 정상 종료(추천 완료)면 성공
            'reward': episode_reward,
            'steps': steps,
            'questions_count': info.get('questions_count', 0),
            'recommended_vehicle': vehicle_id,  # Phase 2에 전달할 ID
            'recommended_vehicle_name': recommended_vehicle,  # 사람이 읽을 수 있는 이름
            'top_candidates': top_candidates  # 상위 3개 후보
        }

    def _run_phase2(
        self,
        vehicle_id: str,
        training: bool = False,
        center_idx: Optional[int] = None
    ) -> dict:
        """
        Phase 2 (스케줄링) 실행

        Phase 1에서 추천된 차량으로 시승 일정을 배정함.
        고객 선호 시간이 가용하면 즉시 확정, 불가하면 대안 제시.

        강화학습 루프:
            while not done:
                1. 에이전트가 액션 선택 (확정 or 대안 제시)
                2. 환경이 고객 수락/거절 시뮬레이션
                3. (학습 모드) DQN 업데이트

        Args:
            vehicle_id: Phase 1에서 추천된 차량 ID (핵심 입력)
            training: 학습 모드 여부
            center_idx: 특정 센터 지정

        Returns:
            Phase 2 결과 딕셔너리:
                - success: 예약 성사 여부
                - reward: 에피소드 누적 보상
                - attempt_count: 대안 제시 횟수
                - initial_available: 초기 선호 시간 가용 여부
        """
        # =====================================================================
        # 환경 초기화 (추천 차량 전달 - Phase 연결의 핵심)
        # =====================================================================
        options = {'vehicle_id': vehicle_id}  # Phase 1 결과를 Phase 2에 주입
        if center_idx is not None:
            options['center_idx'] = center_idx

        obs, info = self.phase2_env.reset(
            seed=self.seed + self.episode_count + 10000,  # Phase 1과 다른 시드
            options=options
        )

        # 초기 상태 저장 (시너지 보너스 계산용)
        initial_day = info['requested_day']      # 고객이 원하는 날짜
        initial_slot = info['requested_slot']    # 고객이 원하는 시간
        initial_available = info['slot_available']  # 해당 슬롯 가용 여부

        episode_reward = 0.0
        steps = 0
        done = False

        # =====================================================================
        # 강화학습 루프: 에이전트-환경 상호작용
        # =====================================================================
        while not done:
            # 1. 에이전트가 현재 상태에서 액션 선택
            #    액션: 0=확정, 1-4=대안제시, 5=인센티브
            action = self.phase2_agent.select_action(obs, training=training)

            # 2. 환경에서 액션 실행
            next_obs, reward, terminated, truncated, info = self.phase2_env.step(action)

            # 3. 학습 모드일 때 경험 저장 및 DQN 업데이트
            if training:
                # Experience Replay 버퍼에 경험 저장
                # 현재 스텝의 종료 여부를 전달해야 함 (done은 아직 이전 값)
                self.phase2_agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
                # 미니배치 학습 수행
                self.phase2_agent.update()

            obs = next_obs
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # =====================================================================
        # 에피소드 종료 처리
        # =====================================================================
        if training:
            # ε 감소 및 Target Network 동기화
            self.phase2_agent.end_episode()

        return {
            'success': terminated and episode_reward > 0,  # 정상 종료 + 긍정 보상 = 예약 성사
            'reward': episode_reward,
            'steps': steps,
            'attempt_count': info.get('attempt_count', 0),  # 대안 제시 횟수
            'initial_day': initial_day,          # 시너지 계산용
            'initial_slot': initial_slot,        # 시너지 계산용
            'initial_available': initial_available,  # 시너지 계산용 (핵심)
            'final_day': info.get('requested_day', initial_day),
            'final_slot': info.get('requested_slot', initial_slot),
            'center': info.get('center', ''),
            'vehicle': info.get('vehicle', vehicle_id)
        }

    def _calculate_synergy_bonus(
        self,
        phase1_result: dict,
        phase2_result: dict
    ) -> float:
        """
        시너지 보너스 계산 (Phase 3 핵심 기여)

        개별 Phase 독립 실행 대비 통합 실행의 이점을 수치화함.
        두 Phase 간 연계가 잘 될수록 높은 보너스 부여.

        시너지 보너스 구성:
            1. 즉시 예약 성공 (+5): 대안 제시 없이 바로 예약 성공
               - 의미: Phase 1이 가용한 차량을 잘 추천함
            2. 추천-스케줄 매칭 (+3): 추천 차량이 선호 시간에 가용
               - 의미: Phase 1이 스케줄 상황을 간접 반영함

        연구적 의의:
            - 개별 학습된 에이전트들의 협업 효과 측정
            - End-to-End 최적화의 필요성 입증

        Args:
            phase1_result: Phase 1 결과 딕셔너리
            phase2_result: Phase 2 결과 딕셔너리 (None 가능)

        Returns:
            시너지 보너스 값 (0.0 ~ 8.0)
        """
        # Phase 2 실행 안 됐으면 시너지 없음
        if phase2_result is None:
            return 0.0

        synergy = 0.0

        # =====================================================================
        # 시너지 1: 즉시 예약 성공 보너스 (+5점)
        # =====================================================================
        # 조건: Phase 2에서 첫 시도(attempt_count=0)에 예약 성공
        # 의미: Phase 1이 가용한 차량을 추천해서 바로 예약됨
        if phase2_result['success'] and phase2_result['attempt_count'] == 0:
            synergy += self.SYNERGY_IMMEDIATE_BOOKING

        # =====================================================================
        # 시너지 2: 추천-스케줄 매칭 보너스 (+3점)
        # =====================================================================
        # 조건: 추천된 차량이 고객 선호 시간에 가용했음
        # 의미: Phase 1 추천이 센터 가용성과 잘 맞음
        if phase2_result['initial_available']:
            synergy += self.SYNERGY_SCHEDULE_MATCH

        return synergy

    def _create_result(
        self,
        phase1_result: dict,
        phase2_result: Optional[dict],
        synergy_bonus: float
    ) -> dict:
        """
        통합 결과 딕셔너리 생성

        두 Phase의 결과와 시너지 보너스를 합쳐서
        End-to-End 성능을 종합적으로 평가할 수 있는 결과 생성.

        Args:
            phase1_result: Phase 1 결과
            phase2_result: Phase 2 결과 (None 가능)
            synergy_bonus: 시너지 보너스

        Returns:
            통합 결과 딕셔너리 (모든 지표 포함)
        """
        # 보상 계산: Total = R1 + R2 + Synergy
        phase1_reward = phase1_result['reward']
        phase2_reward = phase2_result['reward'] if phase2_result else 0.0
        total_reward = phase1_reward + phase2_reward + synergy_bonus

        return {
            # =====================================================================
            # 통합 지표 (핵심)
            # =====================================================================
            'total_reward': total_reward,           # 총 보상: R1 + R2 + Synergy
            'phase1_reward': phase1_reward,         # Phase 1 보상 (R1)
            'phase2_reward': phase2_reward,         # Phase 2 보상 (R2)
            'synergy_bonus': synergy_bonus,         # 시너지 보너스

            # =====================================================================
            # 성공 여부
            # =====================================================================
            'phase1_success': phase1_result['success'],  # Phase 1 성공 (추천 완료)
            'phase2_success': phase2_result['success'] if phase2_result else False,  # Phase 2 성공 (예약 성사)
            'end_to_end_success': (  # End-to-End 성공: 둘 다 성공해야 함
                phase1_result['success'] and
                phase2_result is not None and
                phase2_result['success']
            ),

            # =====================================================================
            # Phase 1 상세
            # =====================================================================
            'questions_count': phase1_result['questions_count'],  # 질문 횟수
            'recommended_vehicle': phase1_result.get('recommended_vehicle_name', ''),  # 추천 차량

            # =====================================================================
            # Phase 2 상세
            # =====================================================================
            'attempt_count': phase2_result['attempt_count'] if phase2_result else 0,  # 스케줄링 시도
            'preferred_time_match': (  # 선호 시간에 예약 성공 여부
                phase2_result is not None and
                phase2_result['success'] and
                phase2_result['final_day'] == phase2_result['initial_day'] and
                phase2_result['final_slot'] == phase2_result['initial_slot']
            ),

            # =====================================================================
            # 효율성 지표
            # =====================================================================
            'total_interactions': (  # 총 상호작용 횟수 (질문 + 스케줄링 시도)
                phase1_result['steps'] +
                (phase2_result['steps'] if phase2_result else 0)
            )
        }

    def _get_vehicle_id(self, vehicle_name: str) -> str:
        """
        차량 이름을 ID로 변환 (Phase 1 → Phase 2 연동)

        Phase 1 환경은 차량 이름 ("AVANTE Hybrid")을 사용하고,
        Phase 2 환경은 차량 ID ("avante_hybrid")를 사용함.
        이 메서드가 둘 사이의 변환을 담당함.

        Args:
            vehicle_name: 차량 이름 (예: "AVANTE Hybrid", "싼타페 HEV")

        Returns:
            차량 ID (예: "avante_hybrid", "싼타페_hev")
        """
        if vehicle_name is None:
            return "avante"  # 기본값

        # 이름을 ID 형식으로 변환
        # - 소문자로 변환
        # - 공백과 하이픈을 언더스코어로 대체
        vehicle_id = vehicle_name.lower().replace(" ", "_").replace("-", "_")
        return vehicle_id

    def get_stats(self) -> dict:
        """
        통합 시스템 통계 반환

        학습 진행 상황과 성능을 모니터링하기 위한 통계 제공.

        Returns:
            통계 딕셔너리:
                - episodes: 완료된 에피소드 수
                - mean_total_reward: 평균 총 보상
                - mean_synergy_bonus: 평균 시너지 보너스
                - phase1/2_agent_stats: 개별 에이전트 통계
        """
        if not self.total_rewards:
            return {
                'episodes': 0,
                'mean_total_reward': 0.0,
                'mean_phase1_reward': 0.0,
                'mean_phase2_reward': 0.0,
                'mean_synergy_bonus': 0.0
            }

        return {
            'episodes': self.episode_count,
            'mean_total_reward': float(np.mean(self.total_rewards)),
            'std_total_reward': float(np.std(self.total_rewards)),
            'mean_phase1_reward': float(np.mean(self.phase1_rewards)),
            'mean_phase2_reward': float(np.mean(self.phase2_rewards)),
            'mean_synergy_bonus': float(np.mean(self.synergy_bonuses)),
            'phase1_agent_stats': self.phase1_agent.get_stats(),
            'phase2_agent_stats': self.phase2_agent.get_stats()
        }

    def save(self, base_path: str):
        """
        모델 저장 (Phase 1, Phase 2 모두)

        학습된 모델을 파일로 저장하여 나중에 재사용 가능.
        - Phase 1: JSON 파일 (Q-table)
        - Phase 2: PyTorch 체크포인트 (DQN 가중치)

        Args:
            base_path: 저장 디렉토리 경로
        """
        save_dir = Path(base_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1 모델 저장 (Q-table을 JSON으로)
        self.phase1_agent.save(str(save_dir / "phase1_q_learning.json"))

        # Phase 2 모델 저장 (DQN 가중치를 .pth로)
        self.phase2_agent.save(str(save_dir / "phase2_dqn.pth"))

        print(f"모델 저장 완료: {save_dir}")

    def load(self, base_path: str):
        """
        저장된 모델 로드

        이전에 학습된 모델을 로드하여 평가 또는 추가 학습에 사용.

        Args:
            base_path: 로드할 디렉토리 경로
        """
        load_dir = Path(base_path)

        # Phase 1 모델 로드
        phase1_path = load_dir / "phase1_q_learning.json"
        if phase1_path.exists():
            self.phase1_agent.load(str(phase1_path))
            print(f"Phase 1 모델 로드: {phase1_path}")

        # Phase 2 모델 로드
        phase2_path = load_dir / "phase2_dqn.pth"
        if phase2_path.exists():
            self.phase2_agent.load(str(phase2_path))
            print(f"Phase 2 모델 로드: {phase2_path}")


# =============================================================================
# 학습 함수
# =============================================================================

def train_integrated(
    system: IntegratedSystem,
    n_episodes: int = 2000,
    phase1_pretrain: int = 500,
    phase2_pretrain: int = 500,
    log_interval: int = 100,
    verbose: bool = True
) -> dict:
    """
    통합 시스템 학습 (3단계 학습 전략)

    학습 전략:
        1단계. Phase 1 사전 학습: Q-Learning으로 차량 추천 정책 학습
        2단계. Phase 2 사전 학습: DQN으로 스케줄링 정책 학습
        3단계. 통합 학습: End-to-End 파이프라인으로 Synergy Bonus 반영

    이 전략의 장점:
        - 각 Phase가 독립적으로 안정적인 정책을 먼저 학습
        - 통합 학습에서 두 Phase 간 협업 최적화
        - Synergy Bonus를 통해 End-to-End 성능 향상

    Args:
        system: IntegratedSystem 인스턴스
        n_episodes: 통합 학습 에피소드 수
        phase1_pretrain: Phase 1 사전 학습 에피소드 수
        phase2_pretrain: Phase 2 사전 학습 에피소드 수
        log_interval: 로그 출력 간격
        verbose: 진행 상황 출력 여부

    Returns:
        학습 히스토리 딕셔너리:
            - phase1_pretrain_rewards: Phase 1 사전 학습 보상
            - phase2_pretrain_rewards: Phase 2 사전 학습 보상
            - integrated_rewards: 통합 학습 총 보상
            - synergy_bonuses: 시너지 보너스 히스토리
            - end_to_end_success: 성공 여부 히스토리
    """
    # 학습 히스토리 초기화
    history = {
        'phase1_pretrain_rewards': [],    # Phase 1 사전 학습 보상
        'phase2_pretrain_rewards': [],    # Phase 2 사전 학습 보상
        'integrated_rewards': [],          # 통합 학습 총 보상
        'phase1_rewards': [],              # 통합 학습 중 Phase 1 보상
        'phase2_rewards': [],              # 통합 학습 중 Phase 2 보상
        'synergy_bonuses': [],             # 시너지 보너스
        'end_to_end_success': []           # End-to-End 성공 여부
    }

    # =========================================================================
    # 1단계: Phase 1 사전 학습 (Q-Learning)
    # =========================================================================
    # 차량 추천 정책을 독립적으로 학습
    # 목표: 최소 질문으로 고객에게 적합한 차량 추천
    if verbose:
        print("=" * 60)
        print(f"[1단계] Phase 1 사전 학습 시작 ({phase1_pretrain} 에피소드)")
        print("=" * 60)

    for ep in range(phase1_pretrain):
        # 환경 초기화
        obs, _ = system.phase1_env.reset(seed=system.seed + ep)
        episode_reward = 0.0
        done = False

        # 에피소드 루프
        while not done:
            # 액션 선택 (ε-greedy)
            action = system.phase1_agent.select_action(obs, training=True)
            # 환경 스텝
            next_obs, reward, terminated, truncated, _ = system.phase1_env.step(action)
            # Q-table 업데이트
            system.phase1_agent.update(obs, action, reward, next_obs, terminated, truncated)
            obs = next_obs
            episode_reward += reward
            done = terminated or truncated

        # 에피소드 종료 처리 (ε 감소)
        system.phase1_agent.end_episode()
        history['phase1_pretrain_rewards'].append(episode_reward)

        # 주기적 로그 출력
        if verbose and (ep + 1) % log_interval == 0:
            recent = history['phase1_pretrain_rewards'][-log_interval:]
            print(f"  Phase 1 Episode {ep + 1:4d} | Avg Reward: {np.mean(recent):.2f}")

    # =========================================================================
    # 2단계: Phase 2 사전 학습 (DQN)
    # =========================================================================
    # 스케줄링 정책을 독립적으로 학습
    # 목표: 고객 선호 시간에 예약 성사, 대안 제시 전략 학습
    if verbose:
        print("\n" + "=" * 60)
        print(f"[2단계] Phase 2 사전 학습 시작 ({phase2_pretrain} 에피소드)")
        print("=" * 60)

    for ep in range(phase2_pretrain):
        # 환경 초기화
        obs, _ = system.phase2_env.reset(seed=system.seed + ep + 10000)
        episode_reward = 0.0
        done = False

        # 에피소드 루프
        while not done:
            # 액션 선택 (ε-greedy)
            action = system.phase2_agent.select_action(obs, training=True)
            # 환경 스텝
            next_obs, reward, terminated, truncated, _ = system.phase2_env.step(action)
            # Experience Replay에 저장 (현재 스텝의 종료 여부 전달)
            system.phase2_agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            # DQN 미니배치 업데이트
            system.phase2_agent.update()
            obs = next_obs
            episode_reward += reward
            done = terminated or truncated

        # 에피소드 종료 처리 (ε 감소, Target Network 동기화)
        system.phase2_agent.end_episode()
        history['phase2_pretrain_rewards'].append(episode_reward)

        # 주기적 로그 출력
        if verbose and (ep + 1) % log_interval == 0:
            recent = history['phase2_pretrain_rewards'][-log_interval:]
            print(f"  Phase 2 Episode {ep + 1:4d} | Avg Reward: {np.mean(recent):.2f}")

    # =========================================================================
    # 3단계: 통합 학습 (End-to-End)
    # =========================================================================
    # Phase 1 → Phase 2 파이프라인으로 실행
    # 목표: Synergy Bonus를 통해 두 Phase 간 협업 최적화
    if verbose:
        print("\n" + "=" * 60)
        print(f"[3단계] 통합 학습 시작 ({n_episodes} 에피소드)")
        print("=" * 60)

    for ep in range(n_episodes):
        # End-to-End 에피소드 실행 (Phase 1 → Phase 2)
        result = system.run_episode(training=True)

        # 히스토리 기록
        history['integrated_rewards'].append(result['total_reward'])
        history['phase1_rewards'].append(result['phase1_reward'])
        history['phase2_rewards'].append(result['phase2_reward'])
        history['synergy_bonuses'].append(result['synergy_bonus'])
        history['end_to_end_success'].append(1 if result['end_to_end_success'] else 0)

        # 주기적 로그 출력
        if verbose and (ep + 1) % log_interval == 0:
            recent_rewards = history['integrated_rewards'][-log_interval:]
            recent_success = history['end_to_end_success'][-log_interval:]
            recent_synergy = history['synergy_bonuses'][-log_interval:]

            print(f"  Episode {ep + 1:4d} | "
                  f"Total: {np.mean(recent_rewards):.2f} | "
                  f"Success: {np.mean(recent_success):.1%} | "
                  f"Synergy: {np.mean(recent_synergy):.2f}")

    if verbose:
        print("\n" + "=" * 60)
        print("학습 완료!")
        print("=" * 60)

    return history


# =============================================================================
# 평가 함수
# =============================================================================

def evaluate_integrated(
    system: IntegratedSystem,
    n_episodes: int = 100,
    verbose: bool = True
) -> dict:
    """
    통합 시스템 평가

    학습된 시스템의 성능을 다양한 지표로 측정함.
    평가 시에는 탐험 없이 greedy 정책만 사용.

    평가 지표:
        - 총 보상: R1 + R2 + Synergy
        - End-to-End 성공률: 추천 성공 AND 예약 성사
        - 선호 시간 매칭률: 고객이 원하는 시간에 예약
        - 효율성: 평균 질문 수, 평균 스케줄링 시도 횟수

    Args:
        system: IntegratedSystem 인스턴스
        n_episodes: 평가 에피소드 수
        verbose: 결과 출력 여부

    Returns:
        평가 결과 딕셔너리
    """
    # 결과 수집용 딕셔너리
    results = {
        'total_rewards': [],            # 총 보상
        'phase1_rewards': [],           # Phase 1 보상
        'phase2_rewards': [],           # Phase 2 보상
        'synergy_bonuses': [],          # 시너지 보너스
        'end_to_end_success': [],       # End-to-End 성공 여부
        'questions_counts': [],         # 질문 횟수
        'attempt_counts': [],           # 스케줄링 시도 횟수
        'preferred_time_matches': [],   # 선호 시간 매칭 여부
        'total_interactions': []        # 총 상호작용 횟수
    }

    # 평가 에피소드 실행
    for ep in range(n_episodes):
        # 평가 모드로 에피소드 실행 (training=False)
        result = system.run_episode(training=False)

        # 결과 수집
        results['total_rewards'].append(result['total_reward'])
        results['phase1_rewards'].append(result['phase1_reward'])
        results['phase2_rewards'].append(result['phase2_reward'])
        results['synergy_bonuses'].append(result['synergy_bonus'])
        results['end_to_end_success'].append(1 if result['end_to_end_success'] else 0)
        results['questions_counts'].append(result['questions_count'])
        results['attempt_counts'].append(result['attempt_count'])
        results['preferred_time_matches'].append(1 if result['preferred_time_match'] else 0)
        results['total_interactions'].append(result['total_interactions'])

    # =========================================================================
    # 통계 계산
    # =========================================================================
    summary = {
        'n_episodes': n_episodes,

        # 보상 통계
        'mean_total_reward': float(np.mean(results['total_rewards'])),
        'std_total_reward': float(np.std(results['total_rewards'])),
        'mean_phase1_reward': float(np.mean(results['phase1_rewards'])),
        'mean_phase2_reward': float(np.mean(results['phase2_rewards'])),
        'mean_synergy_bonus': float(np.mean(results['synergy_bonuses'])),

        # 성공률
        'end_to_end_success_rate': float(np.mean(results['end_to_end_success'])),
        'preferred_time_match_rate': float(np.mean(results['preferred_time_matches'])),

        # 효율성
        'mean_questions': float(np.mean(results['questions_counts'])),
        'mean_attempts': float(np.mean(results['attempt_counts'])),
        'mean_total_interactions': float(np.mean(results['total_interactions']))
    }

    # 결과 출력
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 3 통합 시스템 평가 결과")
        print("=" * 60)
        print(f"\n[보상]")
        print(f"  총 보상: {summary['mean_total_reward']:.2f} ± {summary['std_total_reward']:.2f}")
        print(f"  Phase 1 보상: {summary['mean_phase1_reward']:.2f}")
        print(f"  Phase 2 보상: {summary['mean_phase2_reward']:.2f}")
        print(f"  시너지 보너스: {summary['mean_synergy_bonus']:.2f}")
        print(f"\n[성공률]")
        print(f"  End-to-End 성공률: {summary['end_to_end_success_rate']:.1%}")
        print(f"  선호 시간 매칭률: {summary['preferred_time_match_rate']:.1%}")
        print(f"\n[효율성]")
        print(f"  평균 질문 수: {summary['mean_questions']:.2f}")
        print(f"  평균 스케줄링 시도: {summary['mean_attempts']:.2f}")
        print(f"  총 상호작용 횟수: {summary['mean_total_interactions']:.2f}")
        print("=" * 60)

    return summary


# =============================================================================
# 메인 실행 (테스트용)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: 통합 시스템 테스트")
    print("=" * 60)

    # 통합 시스템 생성
    system = IntegratedSystem(seed=42)

    # 간단한 학습 테스트 (에피소드 수 축소)
    history = train_integrated(
        system,
        n_episodes=100,       # 테스트용으로 축소
        phase1_pretrain=50,   # 테스트용으로 축소
        phase2_pretrain=50,   # 테스트용으로 축소
        log_interval=25,
        verbose=True
    )

    # 평가
    results = evaluate_integrated(
        system,
        n_episodes=50,
        verbose=True
    )

    # 모델 저장
    save_path = Path(__file__).parent.parent / "checkpoints" / "integrated"
    system.save(str(save_path))
