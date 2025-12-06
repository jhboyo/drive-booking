"""
DQN Agent for Scheduling Environment (Phase 2)

Deep Q-Network를 사용하여 시승 스케줄링 정책을 학습합니다.

핵심 개념:
    - Q(s, a; θ): 신경망으로 Q값을 근사
    - Experience Replay: 샘플 간 상관관계 제거
    - Target Network: 학습 안정성 확보
    - ε-greedy: 탐험과 활용의 균형
"""

import json
from collections import deque  # 양방향 큐 (버퍼용)
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn  # 신경망 모듈
import torch.optim as optim  # 옵티마이저


class QNetwork(nn.Module):
    """
    Q-Network: 상태 → Q값(액션별)

    3층 완전연결 신경망으로 Q값을 근사합니다.
    입력: 상태 벡터 (159차원)
    출력: 각 액션의 Q값 (6차원)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Q-Network 초기화

        Args:
            state_dim: 상태 벡터 차원 (159)
            action_dim: 액션 수 (6)
            hidden_dim: 은닉층 차원
        """
        super().__init__()

        # 3층 완전연결 신경망 구성
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # 입력층 → 은닉층1 (159 → 128)
            nn.ReLU(),                          # 활성화 함수
            nn.Linear(hidden_dim, hidden_dim),  # 은닉층1 → 은닉층2 (128 → 128)
            nn.ReLU(),                          # 활성화 함수
            nn.Linear(hidden_dim, action_dim)   # 은닉층2 → 출력층 (128 → 6)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파: Q값 계산

        Args:
            x: 상태 텐서 (batch_size, state_dim)

        Returns:
            Q값 텐서 (batch_size, action_dim)
        """
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay 버퍼

    과거 경험(상태, 액션, 보상, 다음상태, 종료여부)을 저장하고
    랜덤 샘플링하여 학습 샘플 간 상관관계를 제거합니다.
    이를 통해 학습 안정성이 향상됩니다.
    """

    def __init__(self, capacity: int = 10000):
        """
        버퍼 초기화

        Args:
            capacity: 버퍼 최대 크기 (초과 시 오래된 것부터 삭제)
        """
        self.buffer = deque(maxlen=capacity)  # 고정 크기 큐

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        경험 저장

        Args:
            state: 현재 상태
            action: 수행한 액션
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """
        랜덤 배치 샘플링

        Args:
            batch_size: 샘플링할 배치 크기

        Returns:
            (states, actions, rewards, next_states, dones) 튜플
        """
        # 버퍼에서 랜덤 인덱스 선택 (중복 없이)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # 각 요소별로 배열 생성
        states = np.array([t[0] for t in batch])       # 상태들
        actions = np.array([t[1] for t in batch])      # 액션들
        rewards = np.array([t[2] for t in batch])      # 보상들
        next_states = np.array([t[3] for t in batch])  # 다음 상태들
        dones = np.array([t[4] for t in batch])        # 종료 여부들

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """버퍼에 저장된 경험 수 반환"""
        return len(self.buffer)


class DQNAgent:
    """
    DQN 에이전트

    Deep Q-Network를 사용하여 스케줄링 정책을 학습합니다.

    ========================================
    핵심 수식
    ========================================

    1. Q-Learning 업데이트 (벨만 방정식):
       Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]

       여기서:
       - Q(s,a): 상태 s에서 액션 a의 가치
       - α: 학습률 (learning rate)
       - r: 즉각 보상
       - γ: 할인율 (discount factor, 0~1)
       - s': 다음 상태
       - max_a' Q(s',a'): 다음 상태에서 최대 Q값

    2. DQN 손실 함수 (MSE Loss):
       L(θ) = E[(r + γ · max_a' Q'(s',a'; θ⁻) - Q(s,a; θ))²]

       여기서:
       - θ: Q-Network 파라미터
       - θ⁻: Target Network 파라미터 (주기적 복사)
       - Q': Target Network

    3. ε-greedy 정책:
       π(a|s) = { argmax_a Q(s,a)  with probability (1-ε)
                { random action    with probability ε

    ========================================

    주요 구성요소:
        - Q-Network: 상태에서 Q값을 예측하는 신경망
        - Target Network: 안정적인 타겟 Q값 계산용 (주기적 업데이트)
        - Experience Replay: 과거 경험 저장 및 랜덤 샘플링
        - ε-greedy: 탐험(랜덤)과 활용(최적) 균형
    """

    def __init__(
        self,
        state_dim: int = 159,        # 상태 차원 (고객선호6 + 슬롯126 + 차량23 + 메타4)
        action_dim: int = 6,          # 액션 수 (확정1 + 대안5)
        hidden_dim: int = 128,        # 은닉층 뉴런 수
        learning_rate: float = 1e-3,  # 학습률 (Adam 옵티마이저용)
        discount_factor: float = 0.99, # 할인율 γ (미래 보상의 현재 가치)
        epsilon_start: float = 1.0,   # 초기 탐험률 (100% 랜덤)
        epsilon_end: float = 0.01,    # 최소 탐험률 (1% 랜덤)
        epsilon_decay: float = 0.995, # 탐험률 감소율 (에피소드당)
        buffer_size: int = 10000,     # Replay 버퍼 크기
        batch_size: int = 64,         # 미니배치 크기
        target_update_freq: int = 10, # Target Network 업데이트 주기
        seed: Optional[int] = None,   # 랜덤 시드
        device: str = None            # 연산 장치 ('cpu' 또는 'cuda')
    ):
        """DQN 에이전트 초기화"""

        # === 디바이스 설정 ===
        if device is None:
            # GPU 사용 가능하면 GPU, 아니면 CPU
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # === 랜덤 시드 설정 (재현성 보장) ===
        if seed is not None:
            torch.manual_seed(seed)      # PyTorch 시드
            np.random.seed(seed)         # NumPy 시드

        # === 하이퍼파라미터 저장 ===
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor  # γ
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # === ε-greedy 파라미터 ===
        self.epsilon = epsilon_start      # 현재 탐험률
        self.epsilon_end = epsilon_end    # 최소 탐험률
        self.epsilon_decay = epsilon_decay # 감소율

        # === Q-Network 생성 ===
        # 학습용 Q-Network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        # 타겟 Q-Network (안정적인 타겟값 계산용)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        # 초기에는 동일한 가중치로 시작
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 타겟 네트워크는 평가 모드 (gradient 계산 안함)

        # === 옵티마이저 (Adam) ===
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # === 손실 함수 (MSE) ===
        self.loss_fn = nn.MSELoss()

        # === Replay Buffer ===
        self.replay_buffer = ReplayBuffer(buffer_size)

        # === 랜덤 생성기 ===
        self.rng = np.random.default_rng(seed)

        # === 학습 통계 ===
        self.episode_count = 0   # 완료된 에피소드 수
        self.total_updates = 0   # 총 네트워크 업데이트 횟수
        self.losses = []         # 손실값 기록

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        ε-greedy 정책으로 액션 선택

        ========================================
        수식: ε-greedy 정책
        ========================================

        π(a|s) = { argmax_a Q(s,a; θ)   확률 (1-ε)  [활용]
                 { Uniform(A)           확률 ε      [탐험]

        여기서:
        - A: 가능한 액션 집합 {0, 1, 2, 3, 4, 5}
        - ε: 탐험률 (1.0 → 0.01로 점진적 감소)

        ========================================

        Args:
            state: 현재 상태 벡터 (159차원)
            training: 학습 모드 여부

        Returns:
            선택된 액션 (0: 확정, 1-5: 대안 제시)
        """
        # === 탐험 (Exploration) ===
        # 학습 중이고 ε 확률로 랜덤 액션 선택
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_dim)

        # === 활용 (Exploitation) ===
        # Q값이 가장 높은 액션 선택
        with torch.no_grad():  # gradient 계산 비활성화 (추론용)
            # 상태를 텐서로 변환 (배치 차원 추가)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Q값 계산
            q_values = self.q_network(state_tensor)
            # Q값이 최대인 액션 반환
            return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        경험을 Replay Buffer에 저장

        Args:
            state: 현재 상태
            action: 수행한 액션
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Q-Network 업데이트 (학습의 핵심!)

        ========================================
        수식
        ========================================

        1. 현재 Q값: Q(s, a; θ)

        2. 타겟 Q값 (벨만 방정식):
           y = r + γ · max_a' Q'(s', a'; θ⁻)

           종료 상태인 경우:
           y = r  (미래 보상 없음)

        3. 손실 함수 (MSE):
           L(θ) = (1/N) · Σ (y - Q(s, a; θ))²

        4. 그래디언트:
           ∇θ L(θ) = -(2/N) · Σ (y - Q(s,a;θ)) · ∇θ Q(s,a;θ)

        ========================================

        Returns:
            손실값 (버퍼 샘플 부족 시 None)
        """
        # === 버퍼 크기 확인 ===
        # 충분한 샘플이 없으면 학습 스킵
        if len(self.replay_buffer) < self.batch_size:
            return None

        # === 1. 미니배치 샘플링 ===
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # === 2. NumPy → PyTorch 텐서 변환 ===
        states = torch.FloatTensor(states).to(self.device)           # 상태 (64, 159)
        actions = torch.LongTensor(actions).to(self.device)          # 액션 (64,)
        rewards = torch.FloatTensor(rewards).to(self.device)         # 보상 (64,)
        next_states = torch.FloatTensor(next_states).to(self.device) # 다음상태 (64, 159)
        dones = torch.FloatTensor(dones).to(self.device)             # 종료여부 (64,)

        # === 3. 현재 Q값 계산 ===
        # 수식: Q(s, a; θ)
        # gather: 각 샘플에서 해당 액션의 Q값만 추출
        # 예: Q-Network 출력이 [Q0, Q1, Q2, Q3, Q4, Q5]이고 action=2면 Q2 추출
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # === 4. 타겟 Q값 계산 (Target Network 사용) ===
        with torch.no_grad():  # 타겟은 gradient 계산 안함 (역전파 X)
            # 수식: max_a' Q'(s', a'; θ⁻)
            # 다음 상태에서 가능한 모든 액션 중 최대 Q값
            next_q = self.target_network(next_states).max(dim=1)[0]

            # 수식: y = r + γ · max_a' Q'(s', a'; θ⁻) · (1 - done)
            # 벨만 방정식 적용
            # - 종료 상태(done=1): y = r (미래 보상 없음)
            # - 비종료 상태(done=0): y = r + γ · max Q'
            target_q = rewards + self.discount_factor * next_q * (1 - dones)

        # === 5. 손실 계산 (MSE) ===
        # 수식: L(θ) = (1/N) · Σ (y - Q(s,a;θ))²
        loss = self.loss_fn(current_q, target_q)

        # === 6. 역전파 및 가중치 업데이트 ===
        # 수식: θ ← θ - α · ∇θ L(θ)  (경사하강법)
        self.optimizer.zero_grad()  # gradient 초기화: ∇θ = 0
        loss.backward()             # 역전파: ∇θ L(θ) 계산
        # Gradient Clipping: ||∇θ|| ≤ 1.0으로 제한 (gradient 폭발 방지)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()       # 가중치 업데이트: θ ← θ - α · ∇θ L(θ)

        # === 통계 기록 ===
        self.total_updates += 1
        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def end_episode(self):
        """
        에피소드 종료 시 호출

        ========================================
        수식: ε 지수적 감소
        ========================================

        ε(t+1) = max(ε_min, ε(t) · decay)

        여기서:
        - ε(t): t번째 에피소드의 탐험률
        - decay: 감소율 (0.995)
        - ε_min: 최소 탐험률 (0.01)

        예시 (decay=0.995):
        - Episode 0:   ε = 1.000
        - Episode 100: ε = 0.606
        - Episode 500: ε = 0.082
        - Episode 1000: ε = 0.01 (최소값 도달)

        ========================================
        """
        self.episode_count += 1

        # === ε 지수적 감소 ===
        # 학습 초기: 많은 탐험 (ε ≈ 1)
        # 학습 후기: 적은 탐험 (ε → 0.01)
        # 수식: ε ← max(ε_min, ε × decay)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # === Target Network 동기화 ===
        # 주기적으로 Q-Network의 가중치를 Target Network에 복사
        # 이를 통해 타겟 Q값이 급격히 변하는 것을 방지
        if self.episode_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def get_stats(self) -> dict:
        """
        학습 통계 반환

        Returns:
            에피소드 수, 업데이트 수, 버퍼 크기, ε, 최근 손실 등
        """
        # 최근 100개 손실의 평균
        recent_loss = np.mean(self.losses[-100:]) if self.losses else 0.0
        return {
            'episodes': self.episode_count,
            'total_updates': self.total_updates,
            'buffer_size': len(self.replay_buffer),
            'epsilon': self.epsilon,
            'recent_loss': recent_loss,
            'device': str(self.device)
        }

    def save(self, path: str):
        """
        모델 저장 (체크포인트)

        저장 내용:
            - Q-Network 가중치
            - Target Network 가중치
            - 옵티마이저 상태
            - 학습 상태 (ε, 에피소드 수 등)
            - 하이퍼파라미터

        Args:
            path: 저장 경로 (.pth 파일)
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_updates': self.total_updates,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'discount_factor': self.discount_factor,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }, path)

    def load(self, path: str):
        """
        저장된 모델 로드

        Args:
            path: 로드할 파일 경로 (.pth 파일)
        """
        # 체크포인트 로드
        checkpoint = torch.load(path, map_location=self.device)

        # 네트워크 가중치 복원
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # 학습 상태 복원
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.total_updates = checkpoint['total_updates']


def train_dqn(
    env,
    n_episodes: int = 1000,
    seed: int = 42,
    verbose: bool = True,
    log_interval: int = 100,
    **agent_kwargs
) -> tuple:
    """
    DQN 에이전트 학습 함수

    학습 루프:
        1. 환경 초기화 (reset)
        2. 에피소드 진행 (액션 선택 → 환경 스텝 → 경험 저장 → 네트워크 업데이트)
        3. 에피소드 종료 처리 (ε 감소, Target 업데이트)
        4. 로그 출력

    Args:
        env: SchedulingEnv 환경
        n_episodes: 학습 에피소드 수
        seed: 랜덤 시드
        verbose: 학습 과정 출력 여부
        log_interval: 로그 출력 간격
        **agent_kwargs: DQNAgent 추가 인자

    Returns:
        (agent, training_history) 튜플
    """
    # === 환경 정보 추출 ===
    state_dim = env.observation_space.shape[0]  # 159
    action_dim = env.action_space.n              # 6

    # === 에이전트 생성 ===
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        **agent_kwargs
    )

    # === 학습 기록 초기화 ===
    training_history = {
        'episode_rewards': [],   # 에피소드별 총 보상
        'episode_lengths': [],   # 에피소드별 스텝 수
        'booking_success': [],   # 예약 성공 여부 (0 또는 1)
        'epsilons': [],          # ε 변화
        'losses': []             # 에피소드별 평균 손실
    }

    # === 학습 루프 ===
    for episode in range(n_episodes):
        # 1. 환경 초기화
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0       # 에피소드 누적 보상
        steps = 0                # 에피소드 스텝 수
        episode_losses = []      # 에피소드 내 손실값들

        done = False
        while not done:
            # 2. 액션 선택 (ε-greedy)
            action = agent.select_action(obs, training=True)

            # 3. 환경 스텝 실행
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 4. 경험 저장 (Replay Buffer)
            agent.store_transition(obs, action, reward, next_obs, done)

            # 5. Q-Network 업데이트
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            # 6. 상태 전이
            obs = next_obs
            episode_reward += reward
            steps += 1

        # === 에피소드 종료 처리 ===
        agent.end_episode()

        # === 학습 기록 저장 ===
        training_history['episode_rewards'].append(episode_reward)
        training_history['episode_lengths'].append(steps)
        # 예약 성공: 정상 종료(terminated) + 양수 보상
        training_history['booking_success'].append(1 if terminated and episode_reward > 0 else 0)
        training_history['epsilons'].append(agent.epsilon)
        training_history['losses'].append(np.mean(episode_losses) if episode_losses else 0)

        # === 로그 출력 ===
        if verbose and (episode + 1) % log_interval == 0:
            # 최근 log_interval 에피소드의 통계
            recent_rewards = training_history['episode_rewards'][-log_interval:]
            recent_success = training_history['booking_success'][-log_interval:]
            recent_losses = training_history['losses'][-log_interval:]
            avg_loss = np.mean([l for l in recent_losses if l > 0]) if any(l > 0 for l in recent_losses) else 0

            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {np.mean(recent_rewards):6.2f} | "
                  f"Success: {np.mean(recent_success):.1%} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f}")

    return agent, training_history


def evaluate_dqn(
    env,
    agent: DQNAgent,
    n_episodes: int = 100,
    seed: int = 42
) -> dict:
    """
    DQN 에이전트 평가 함수

    학습된 에이전트의 성능을 측정합니다.
    평가 시에는 탐험 없이 항상 최적 액션 선택 (greedy)

    Args:
        env: 평가용 환경
        agent: 학습된 DQNAgent
        n_episodes: 평가 에피소드 수
        seed: 랜덤 시드

    Returns:
        평가 결과 딕셔너리 (평균 보상, 성공률 등)
    """
    total_rewards = []       # 에피소드별 보상
    episode_lengths = []     # 에피소드별 길이
    booking_success = 0      # 예약 성공 횟수
    preferred_time_match = 0 # 선호 시간 매칭 횟수
    attempt_counts = []      # 시도 횟수

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)

        # 초기 선호 시간 저장 (나중에 매칭 여부 확인용)
        preferred_day = info['requested_day']
        preferred_slot = info['requested_slot']

        episode_reward = 0
        steps = 0

        done = False
        while not done:
            # 평가 시 greedy 정책 (training=False)
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # 결과 기록
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        attempt_counts.append(info.get('attempt_count', 0))

        # 예약 성공 판정
        if terminated and episode_reward > 0:
            booking_success += 1

            # 선호 시간에 예약되었는지 확인
            if (info['requested_day'] == preferred_day and
                info['requested_slot'] == preferred_slot):
                preferred_time_match += 1

    # === 결과 정리 ===
    results = {
        'agent': 'DQN',
        'n_episodes': n_episodes,
        'mean_reward': float(np.mean(total_rewards)),        # 평균 보상
        'std_reward': float(np.std(total_rewards)),          # 보상 표준편차
        'min_reward': float(np.min(total_rewards)),          # 최소 보상
        'max_reward': float(np.max(total_rewards)),          # 최대 보상
        'booking_success_rate': booking_success / n_episodes, # 예약 성사율
        'preferred_time_match_rate': preferred_time_match / n_episodes,  # 선호시간 매칭률
        'mean_attempts': float(np.mean(attempt_counts)),     # 평균 시도 횟수
        'mean_episode_length': float(np.mean(episode_lengths)),  # 평균 에피소드 길이
        'total_training_episodes': agent.episode_count       # 학습된 에피소드 수
    }

    return results


# === 메인 실행 ===
if __name__ == "__main__":
    from src.env.scheduling_env import SchedulingEnv

    # 환경 생성
    env = SchedulingEnv()

    print("=" * 60)
    print("DQN 에이전트 학습 시작 (Phase 2: Scheduling)")
    print(f"State dim: {env.observation_space.shape[0]}")  # 159
    print(f"Action dim: {env.action_space.n}")             # 6
    print("=" * 60)

    # 학습 수행 (500 에피소드)
    agent, history = train_dqn(
        env,
        n_episodes=500,
        seed=42,
        verbose=True,
        log_interval=50
    )

    print("\n" + "=" * 60)
    print("학습 완료! 평가 수행 중...")
    print("=" * 60)

    # 평가 수행 (100 에피소드)
    results = evaluate_dqn(env, agent, n_episodes=100)

    # 결과 출력
    print("\n=== DQN 평가 결과 ===")
    print(f"평균 보상: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"예약 성사율: {results['booking_success_rate']:.1%}")
    print(f"선호 시간 매칭률: {results['preferred_time_match_rate']:.1%}")
    print(f"평균 시도 횟수: {results['mean_attempts']:.2f}")

    # 모델 저장
    save_path = Path(__file__).parent.parent.parent / "checkpoints" / "dqn_scheduling.pth"
    agent.save(str(save_path))
    print(f"\n모델 저장: {save_path}")
