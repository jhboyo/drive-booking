"""
고객 프로필 생성기

학습용 시뮬레이션 고객 데이터를 생성합니다.
실제 현대자동차 사이트의 회원 정보 구조를 반영합니다.
"""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path


# 한국 지역 목록 (인구 비례 가중치 적용을 위해 사용)
REGIONS = [
    "서울", "경기", "인천", "부산", "대구", "대전", "광주", "울산", "세종",
    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]

# 현대자동차 차량 목록 (vehicles.json 기반, 폴백용)
VEHICLE_NAMES = [
    "아반떼", "쏘나타", "쏘나타 하이브리드", "그랜저", "그랜저 하이브리드",
    "캐스퍼", "베뉴", "코나", "코나 하이브리드", "코나 일렉트릭",
    "투싼", "투싼 하이브리드", "싼타페", "싼타페 하이브리드", "팰리세이드",
    "아이오닉 5", "아이오닉 6", "아이오닉 9", "아반떼 N", "아이오닉 5 N",
    "넥쏘", "스타리아"
]


@dataclass
class CustomerProfile:
    """
    고객 프로필 데이터 클래스

    실제 현대자동차 홈페이지의 회원 정보 구조를 반영합니다.
    - 기본 정보: 회원가입 시 수집되는 필수 정보
    - 활동 정보: 사이트 이용 과정에서 축적되는 정보
    """

    # === 기본 정보 (모든 고객 보유) ===
    age: int                # 나이 (생년월일에서 계산)
    gender: str             # 성별: "male" 또는 "female"
    is_foreigner: bool      # 외국인 여부
    region: str             # 거주 지역 (자택주소 기반)
    has_workplace: bool     # 직장 유무 (직장주소 등록 여부)

    # === 활동 정보 (기존 고객만 보유, 신규 고객은 빈 리스트) ===
    interested_cars: list = field(default_factory=list)      # 관심 차량 목록
    quote_history: list = field(default_factory=list)        # 견적 요청 내역
    test_drive_history: list = field(default_factory=list)   # 시승 신청 내역
    purchase_history: list = field(default_factory=list)     # 과거 구매 이력

    # === 고객 유형 ===
    # - new: 신규 고객 (기본 정보만 보유)
    # - interested: 관심 고객 (관심 차량 등록됨)
    # - experienced: 경험 고객 (시승/구매 이력 보유)
    customer_type: str = "new"

    def to_dict(self) -> dict:
        """딕셔너리로 변환 (JSON 저장용)"""
        return asdict(self)


class CustomerGenerator:
    """
    시뮬레이션 고객 프로필 생성기

    RL 에이전트 학습을 위한 가상 고객 데이터를 생성합니다.
    실제 고객 분포를 반영하여 현실적인 시뮬레이션 환경을 제공합니다.
    """

    def __init__(self, vehicles_path: str = None):
        """
        Args:
            vehicles_path: vehicles.json 파일 경로 (차량 정보 로드용)
        """
        self.vehicles = []
        if vehicles_path:
            self._load_vehicles(vehicles_path)

    def _load_vehicles(self, path: str):
        """차량 데이터 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vehicles = data.get('vehicles', [])

    def generate_one(self, customer_type: str = None) -> CustomerProfile:
        """
        단일 고객 프로필 생성

        Args:
            customer_type: 고객 유형 ("new", "interested", "experienced")
                          None이면 확률 기반 랜덤 선택
        Returns:
            CustomerProfile: 생성된 고객 프로필
        """
        # 고객 유형이 지정되지 않으면 확률적으로 선택
        # 실제 서비스에서 신규 고객 비율이 높음을 반영
        if customer_type is None:
            customer_type = random.choices(
                ["new", "interested", "experienced"],
                weights=[0.5, 0.3, 0.2]  # 신규 50%, 관심 30%, 경험 20%
            )[0]

        # === 기본 정보 생성 ===
        age = self._generate_age()
        gender = random.choice(["male", "female"])
        is_foreigner = random.random() < 0.05  # 외국인 비율 5%

        # 지역 선택 (인구 비례 가중치 적용: 수도권 > 광역시 > 기타)
        region = random.choices(
            REGIONS,
            weights=[0.25, 0.25, 0.08, 0.08, 0.05, 0.05, 0.04, 0.03, 0.02,
                     0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        )[0]
        has_workplace = random.random() < 0.7  # 직장인 비율 70%

        # === 활동 정보 생성 (고객 유형에 따라 다름) ===
        interested_cars = []
        quote_history = []
        test_drive_history = []
        purchase_history = []

        # 관심/경험 고객은 관심 차량 목록 보유
        if customer_type in ["interested", "experienced"]:
            interested_cars = self._generate_interested_cars(age)

        # 경험 고객은 견적/시승/구매 이력도 보유
        if customer_type == "experienced":
            quote_history = self._generate_quote_history(interested_cars)
            test_drive_history = self._generate_test_drive_history(interested_cars)
            purchase_history = self._generate_purchase_history(age)

        return CustomerProfile(
            age=age,
            gender=gender,
            is_foreigner=is_foreigner,
            region=region,
            has_workplace=has_workplace,
            interested_cars=interested_cars,
            quote_history=quote_history,
            test_drive_history=test_drive_history,
            purchase_history=purchase_history,
            customer_type=customer_type
        )

    def generate_batch(self, n: int, type_distribution: dict = None) -> list:
        """
        여러 고객 프로필 일괄 생성

        Args:
            n: 생성할 고객 수
            type_distribution: 고객 유형별 비율 (기본값: new 50%, interested 30%, experienced 20%)
        Returns:
            list[CustomerProfile]: 생성된 고객 프로필 목록
        """
        if type_distribution is None:
            type_distribution = {"new": 0.5, "interested": 0.3, "experienced": 0.2}

        profiles = []
        for _ in range(n):
            # 지정된 분포에 따라 고객 유형 선택
            customer_type = random.choices(
                list(type_distribution.keys()),
                weights=list(type_distribution.values())
            )[0]
            profiles.append(self.generate_one(customer_type))

        return profiles

    def _generate_age(self) -> int:
        """
        나이 생성 (정규분포 기반)

        평균 40세, 표준편차 12로 설정하여 20-70세 범위 내에서 생성
        자동차 구매 고객 연령 분포를 반영
        """
        age = int(random.gauss(40, 12))
        return max(20, min(70, age))  # 20-70세 범위로 클리핑

    def _generate_interested_cars(self, age: int) -> list:
        """
        관심 차량 목록 생성

        고객 나이에 따라 적합한 차량을 선택하여 1-3개 생성
        """
        n_cars = random.randint(1, 3)

        if self.vehicles:
            # 나이대별 적합 차량 후보군에서 선택
            candidates = self._get_age_appropriate_vehicles(age)
            selected = random.sample(candidates, min(n_cars, len(candidates)))
            return [v['name'] for v in selected]
        else:
            # vehicles.json 없으면 기본 목록에서 랜덤 선택
            return random.sample(VEHICLE_NAMES, n_cars)

    def _get_age_appropriate_vehicles(self, age: int) -> list:
        """
        나이대에 적합한 차량 필터링

        차량의 target_customers 속성과 고객 나이를 매칭하여
        적합도 점수 기반으로 차량 후보군 생성
        """
        appropriate = []

        for vehicle in self.vehicles:
            targets = vehicle.get('target_customers', [])
            score = 1.0  # 기본 점수

            # 나이대별 차량 선호도 가중치 적용
            if age < 30:
                # 20대: 소형차, 첫차, 도심형 선호
                if any(t in targets for t in ['young_single', 'first_car', 'city_driver']):
                    score *= 2.0
                if 'senior' in targets:
                    score *= 0.3  # 시니어 대상 차량은 낮은 확률
            elif age < 40:
                # 30대: 출퇴근용, 소가족, 테크 선호
                if any(t in targets for t in ['commuter', 'family_small', 'tech_lover']):
                    score *= 1.5
            elif age < 50:
                # 40대: 대가족, 비즈니스 선호
                if any(t in targets for t in ['family_large', 'business']):
                    score *= 1.5
            else:
                # 50대 이상: 프리미엄, 시니어, 비즈니스 선호
                if any(t in targets for t in ['luxury', 'senior', 'business']):
                    score *= 1.5
                if 'young_single' in targets:
                    score *= 0.5  # 청년 대상 차량은 낮은 확률

            # 점수 기반 확률적 선택
            if random.random() < score / 2:
                appropriate.append(vehicle)

        # 후보가 없으면 상위 5개 차량 반환 (폴백)
        return appropriate if appropriate else self.vehicles[:5]

    def _generate_quote_history(self, interested_cars: list) -> list:
        """
        견적 요청 내역 생성

        관심 차량 중 일부에 대해 견적을 요청했다고 가정
        """
        if not interested_cars:
            return []

        n_quotes = random.randint(0, len(interested_cars))
        return random.sample(interested_cars, n_quotes)

    def _generate_test_drive_history(self, interested_cars: list) -> list:
        """
        시승 신청 내역 생성

        관심 차량 중 최대 2개까지 시승 경험이 있다고 가정
        """
        if not interested_cars:
            return []

        n_drives = random.randint(0, min(2, len(interested_cars)))
        return random.sample(interested_cars, n_drives)

    def _generate_purchase_history(self, age: int) -> list:
        """
        과거 구매 이력 생성

        나이에 따라 과거 차량 구매 경험 생성 (25세 미만은 구매 이력 없음)
        10년당 최대 1대 구매했다고 가정
        """
        if age < 25:
            return []

        # 나이에 따른 최대 구매 횟수 계산
        n_purchases = min(random.randint(0, 2), (age - 20) // 10)
        if n_purchases == 0:
            return []

        return random.sample(VEHICLE_NAMES, n_purchases)

    def save_to_json(self, profiles: list, path: str):
        """
        프로필 목록을 JSON 파일로 저장

        메타 정보(총 개수, 유형별 분포)와 함께 저장
        """
        data = {
            "total_count": len(profiles),
            "type_distribution": {
                "new": sum(1 for p in profiles if p.customer_type == "new"),
                "interested": sum(1 for p in profiles if p.customer_type == "interested"),
                "experienced": sum(1 for p in profiles if p.customer_type == "experienced")
            },
            "profiles": [p.to_dict() for p in profiles]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """메인 함수: 샘플 고객 프로필 생성 및 저장"""
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).parent.parent.parent
    vehicles_path = project_root / "data" / "vehicles.json"
    output_path = project_root / "data" / "customer_profiles.json"

    # 생성기 초기화 (차량 데이터 로드)
    generator = CustomerGenerator(str(vehicles_path))

    # 100명의 고객 프로필 생성
    profiles = generator.generate_batch(100)

    # JSON 파일로 저장
    generator.save_to_json(profiles, str(output_path))

    print(f"Generated {len(profiles)} customer profiles")
    print(f"Saved to: {output_path}")

    # 샘플 출력 (처음 3명)
    print("\n=== Sample Profiles ===")
    for i, profile in enumerate(profiles[:3]):
        print(f"\n[{i+1}] {profile.customer_type.upper()} Customer")
        print(f"  Age: {profile.age}, Gender: {profile.gender}, Region: {profile.region}")
        print(f"  Workplace: {profile.has_workplace}")
        print(f"  Interested: {profile.interested_cars}")


if __name__ == "__main__":
    main()
