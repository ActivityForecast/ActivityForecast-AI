# AI 기반 실시간 날씨 예측 및 맞춤형 활동(운동) 추천 서비스

## 1. 프로젝트 개요

**프로젝트명**  
AI 기반 실시간 날씨 예측 및 맞춤형 활동 추천 서비스 플랫폼

**핵심 기능**

- 사용자가 **지역 + 운동 예정 시간 + 선호운동 2개**를 입력하면
- OpenWeatherMap API로 **실시간/미래 시간대의 날씨 + 대기질(PM2.5, PM10)** 을 조회하고
- 사전에 학습된 **XGBoost 모델**과 운동 메타데이터를 이용하여
- **실내/실외 / 카테고리 / 사용자 선호도**를 종합적으로 고려한 **최적 운동 1개를 추천**하는 FastAPI 기반 백엔드 서비스

---

## 2. 폴더 구조 예시

```text
project-root/
│
├─ app.py                       # FastAPI 서버 (운동 추천 API)
├─ category_model.pkl           # 학습된 XGBoost 카테고리 모델
├─ feature_scaler.pkl           # MinMaxScaler (모델 입력값 스케일러)
├─ category_label_encoder.pkl   # 카테고리 LabelEncoder
│
├─ activity_information.csv     # 운동 메타데이터 (활동명, 카테고리, 장소유형, 난이도, 추천계절, 시간대 등)
├─ user_preference.csv          # (선택) 모델 학습용 사용자 선호 데이터
│
├─ requirements.txt             # 파이썬 패키지 의존성 목록
└─ README.md                    # 프로젝트 설명 및 사용법

```
.pyc, __pycache__, 환경 변수 파일 등은 버전 관리에서 제외하는 것을 권장한다.

예시 .gitignore: 
```
*.pyc
__pycache__/
.env
```

## 3. 사전 준비
3.1. Python 환경

Python 3.9 ~ 3.11 권장

예시 (conda 사용 시): 
```
conda create -n weather-ai python=3.11
conda activate weather-ai 
```

3.2. 라이브러리 설치

pip install -r 아래
```
fastapi
uvicorn
pandas
scikit-learn
xgboost
joblib
requests
python-dotenv
```

4. OpenWeatherMap API 키 설정

OpenWeatherMap
 회원가입

상단 메뉴 → API keys 에서 API Key 발급 : 이건 이미 코드에 있습니다

프로젝트 루트 디렉터리에 .env 파일 생성 후 다음과 같이 작성:
```
OWM_API_KEY=9f8737894290204a7b2793f510443e46
```

5. 서버 실행 방법

1. 터미널에서 프로젝트 디렉터리로 이동:
```
cd /경로/프로젝트_폴더  # 예: /Users/.../졸업프로젝트/dataset
```
2. FastAPI 서버 실행:
```
uvicorn app:app --reload
```
접속 경로

API 문서 (Swagger UI):
http://127.0.0.1:8000/docs

기본 헬스 체크:
http://127.0.0.1:8000/

## 6. API 사용법
6.1. 엔드포인트

POST /recommend/by-location-and-user
→ 위치, 시간, 선호운동을 기반으로 운동 추천

6.2. 요청 바디(JSON 형식)
```
{
  "user_id": "demoUser",
  "location_name": "성남시 수정구",
  "target_datetime": "2025-11-07T09:00:00",
  "favorites": ["걷기", "배드민턴"]
}
```
* `user_id`
    * 임의의 사용자 식별자 (실제 추천 로직에서는 크게 사용하지 않음, 로그/확장용)
* `location_name`
    * 한글 주소 입력 가능 (예: `"서울특별시 종로구"`, `"부산광역시 해운대구"`, `"성남시 분당구"`)
    * 백엔드에서 OpenWeatherMap Geocoding API로 자동 위/경도 변환
* `target_datetime`
    * 운동 예정 시간 (현재 또는 미래 시간)
    * ISO 8601 형식: `"YYYY-MM-DDTHH:MM:SS"`
* `favorites`
    * 사용자가 UI에서 선택한 선호운동 2개 정도
    * `activity_information.csv`의 **활동명**과 최대한 일치하게 사용하는 것을 권장
    * 예: `["걷기"]`, `["조깅"]`, `["헬스", "요가"]`, `["등산", "자전거 타기"]`

6.3. 응답 예시
```
{
  "입력값": {
    "user_id": "demoUser",
    "location_name": "성남시 수정구",
    "target_datetime": "2025-11-07T09:00:00",
    "favorites_from_request": ["걷기", "배드민턴"]
  },
  "지오코딩": {
    "위도": 37.4503386,
    "경도": 127.1462933
  },
  "사용된_날씨": {
    "temperature": 14.27,
    "humidity": 56,
    "wind_speed": 1.3,
    "precipitation": 0,
    "pm25": 94.46,
    "pm10": 106.73,
    "pm_grade": "매우 나쁨",
    "season": "가을",
    "time_range": "오전",
    "owm_dt_txt": "2025-11-07 09:00:00",
    "is_rainy": false,
    "is_windy": false,
    "is_too_cold": false,
    "is_too_hot": false,
    "is_bad_air": true
  },
  "XGBoost_예측_카테고리": "유산소",
  "최종_선택_카테고리": "피트니스",
  "실내_우선여부": true,
  "추천_운동": "배드민턴",
  "추천_근거": "사용자 선호운동(실시간 입력): ['걷기', '배드민턴'] / 선호운동 중에서 현재 날씨/시간 조건에 맞는 운동을 우선 추천했습니다: ['배드민턴'] / 미세먼지·초미세먼지 수치가 높아 실내 운동(피트니스)을 우선 고려했습니다.",
  "디버그": {
    "favorites_raw": ["걷기", "배드민턴"],
    "favorites_clean": ["걷기", "배드민턴"],
    "favorite_categories": ["유산소", "구기스포츠"],
    "favorites_matched_in_category": ["배드민턴"]
  }
}

```
※ 실제 값은 조회 시점의 날씨 / 대기질 / 입력 위치 및 시간에 따라 달라진다.

## 7. 추천 로직 요약

1.  **위치 + 시간 입력**
    * 클라이언트에서 `location_name`, `target_datetime`, `favorites` 전달
    * 서버에서 OpenWeatherMap Geocoding API로 해당 위치의 위도/경도 조회

2.  **날씨 + 대기질 조회**
    * OpenWeatherMap Forecast API(5일/3시간 간격 예보)를 이용해 `target_datetime`과 가장 가까운 시각의 날씨를 선택
    * OpenWeatherMap Air Pollution API로 PM2.5, PM10 조회
    * 온도, 습도, 풍속, 강수량, PM2.5, PM10, 계절, 시간대 등을 **파생 특성으로 생성**

3.  **XGBoost로 활동 카테고리 예측**
    * 입력 특성:
        `(온도, 습도, 풍속, 강수량, pm25, season_code, location_type, difficulty, time_range 등 학습 시 사용한 피처)`
    * 출력: 카테고리 (예: `유산소`, `피트니스`, `구기스포츠`, `익스트림스포츠`)

4.  **실내/실외 판단**
    다음 조건 중 하나라도 해당하면 실내 운동 우선:
    * 비/눈 등 강수
    * 강풍
    * 너무 추운 날씨 (예: 0°C 이하)
    * 너무 더운 날씨 (예: 30°C 이상)
    * 미세먼지(PM10) 또는 초미세먼지(PM2.5)가 **나쁨** 또는 **매우 나쁨**
    **그렇지 않으면 실외 운동 우선**

5.  **사용자 선호 운동 반영**
    * `favorites`에 포함된 운동들의 카테고리(`activity_information.csv` 기준)를 계산
    * XGBoost 예측 카테고리와 선호 카테고리를 함께 고려하여 최종 카테고리 결정
    * 최종 카테고리 + 계절 + 시간대 + 실내/실외 조건에 맞는 운동 목록 **필터링**
    * 그 안에서:
        * **선호운동**이 포함되어 있으면 → **선호운동**을 최우선 추천
        * 포함되어 있지 않으면 → **조건**에 맞는 운동 중 1개를 랜덤 추천

## 8. 시연(데모) 플로우 예시

1.  사용자가 웹/앱에서 로그인
2.  선호운동 2개 선택 (예: 걷기, 헬스)
3.  지도/검색 UI에서 지역 선택 (예: `"성남시 수정구"`)
4.  운동 예정 시간 선택 (예: `"2025-11-07 09:00:00"`)
5.  프론트엔드에서 `/recommend/by-location-and-user` 로 위 JSON 형식으로 요청
6.  백엔드에서 날씨 + 대기질 + XGBoost 모델 + 선호운동을 종합하여 운동 1개 추천
7.  UI에 추천 결과와 함께:
    * 추천 운동명
    * 간단한 근거 (날씨/미세먼지/실내·실외/선호운동 반영 설명)
        를 표시

## 9. 주의 사항

* `activity_information.csv`와 `category_model.pkl`, `feature_scaler.pkl`, `category_label_encoder.pkl`은
    같은 피처 구조를 기준으로 학습/저장되어 있어야 한다.
* OpenWeatherMap 무료 플랜의 호출 제한을 고려하여 너무 잦은 호출은 피하는 것이 좋다.
* 실제 서비스 배포 시에는 `.env`와 API Key를 절대로 공개 저장소에 올리지 않도록 주의한다.
