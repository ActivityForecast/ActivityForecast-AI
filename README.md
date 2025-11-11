# AI 기반 실시간 날씨 예측 및 맞춤형 활동(운동) 추천 서비스

## 1. 프로젝트 개요

**프로젝트명**  
AI 기반 실시간 날씨 예측 및 맞춤형 활동 추천 서비스 플랫폼

**핵심 아이디어**

사용자가 **지역 + 운동 예정 시간 + 선호운동 2개**를 입력하면,

1. OpenWeatherMap API로 **해당 시점의 날씨 + 대기질(PM2.5, PM10)** 를 조회하고  
2. 사전에 학습된 **XGBoost 모델**로 날씨 기반 활동 **카테고리(유산소/피트니스/구기스포츠/익스트림)** 를 예측한 뒤  
3. **콘텐츠 기반 스코어링(선호운동/실내·실외/계절/시간대)** 과  
4. 사용자–운동 그래프에서 학습한 **GNN(GraphSAGE) 임베딩**을 이용해  
5. 최종적으로 **가장 점수가 높은 운동 1개를 추천**하는 FastAPI 기반 백엔드 서비스입니다.

---

## 2. 폴더 구조

```text
project-root/
│
├─ app.py                         # FastAPI 서버 (운동 추천 API)
│
├─ category_model.pkl             # XGBoost 카테고리 분류 모델
├─ feature_scaler.pkl             # MinMaxScaler (모델 입력값 스케일러)
├─ category_label_encoder.pkl     # 카테고리 LabelEncoder
│
├─ activity_information.csv       # 운동 메타데이터 (활동명, 카테고리, 장소유형, 난이도, 추천계절, 시간대 등)
├─ activity_gnn_embeddings.pt     # GNN(GraphSAGE)로 학습한 활동 임베딩 (activity_name → 벡터)
│
├─ Weather_Conditions.csv         # (선택) 날씨 데이터셋 – XGBoost 재학습/평가용
├─ user_preference.csv            # (선택) 사용자–운동 선호 데이터 – GNN 학습용
│
├─ requirements.txt               # 파이썬 패키지 의존성 목록
└─ README.md                      # 프로젝트 설명 및 사용법
```
__pycache__/, .env, .ipynb_checkpoints/ 등은 버전 관리에서 제외하는 것을 권장합니다.

예시 .gitignore:
```
*.pyc
__pycache__/
.env
.ipynb_checkpoints/
```
## 3. 사전 준비

### 3.1 Python 환경

* Python 3.9 ~ 3.11 권장

예시 (conda):

```
conda create -n weather-ai python=3.11
conda activate weather-ai
```
또는 venv:
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
### 3.2 라이브러리 설치

* requirements.txt 예시:
```
fastapi
uvicorn[standard]
pandas
scikit-learn
xgboost
joblib
requests
python-dotenv
torch     # GNN 임베딩 로드용 (CPU 버전으로 충분)
```
* 설치:
```
pip install -r requirements.txt
```
## 4. OpenWeatherMap API 키 설정

1.  OpenWeatherMap 회원가입
2.  상단 메뉴 → API keys에서 API Key 발급

### 4.1 간단 사용 (코드 상수 사용)

현재 `app.py`는 내부에 상수로 API 키를 넣도록 되어 있습니다.

```python
OWM_API_KEY = "9f8737894290204a7b2793f510443e46"
```
테스트용으로는 이 방식으로도 충분하지만, 실제 서비스 배포 시에는 .env 사용을 권장합니다.
## 4.2 추천 방식 - .env 사용 (선택)
프로젝트 루트에 .env 생성:
```
OWM_API_KEY = "9f8737894290204a7b2793f510443e46"
```
그리고 app.py 상단에서 python-dotenv로 로드하도록 수정할 수 있습니다. (현재 코드는 상수 방식 기준으로 작성되어 있으므로, 필요시 팀에서 선택적으로 변경)
## 5. 서버 실행 방법
### 5.1 프로젝트 디렉터리 이동
```
cd /경로/프로젝트_폴더
```
## 5.2 FastAPI 서버 실행

```
uvicorn app:app --host 0.0.0.0 --port 9000 --reload
새 터미널 창에서 ngrok http --domain=uncomely-alyse-undazed.ngrok-free.dev 9000
```
* 로컬에서 테스트:
    * Swagger Ui: http://127.0.0.1:9000/docs ↗
    * 헬스 체크: http://127.0.0.1:9000/ ↗
* 서버(예: 144.24.73.5)에서 돌릴 경우:
    * https://uncomely-alyse-undazed.ngrok-free.dev/docs ↗
## 6. API 사용법

### 6.1 엔드포인트

`POST` `/recommend/by-location-and-user`
→ 위치, 시간, 선호운동을 기반으로 운동 추천

### 6.2 요청 바디(JSON)

```json
{
  "user_id": "demoUser",
  "location_name": "서울특별시 강남구",
  "target_datetime": "2025-11-07T09:00:00",
  "favorites": ["걷기", "배드민턴"]
}
```
* `user_id`
    * 임의의 사용자 식별자 (로그/추후 확장용)
* `location_name`
    * 한글 주소 가능 (예: `"서울특별시 종로구"`, `"부산광역시 해운대구"`, `"성남시 수정구"`)
    * 백엔드에서 OpenWeatherMap Geocoding API로 위/경도 변환
* `target_datetime`
    * 운동 예정 시간 (현재 또는 미래 시각)
    * ISO 8601 형식: `"YYYY-MM-DDTHH:MM:SS"`
* `favorites`
    * 사용자가 선택한 선호운동 리스트 (1~2개 추천)
    * `activity_information.csv`의 활동명과 일치하도록 사용하는 것을 권장
        (예: `["걷기"]`, `["조깅"]`, `["헬스"]`, `["요가"]`, `["등산"]`, `["자전거 타기"]`)
### 6.3 응답 예시
```
{
  "입력값": {
    "user_id": "demoUser",
    "location_name": "서울특별시 강남구",
    "target_datetime": "2025-11-07T09:00:00",
    "favorites_from_request": ["걷기", "배드민턴"]
  },
  "지오코딩": {
    "위도": 37.5666791,
    "경도": 126.9782914
  },
  "사용된_날씨": {
    "temperature": 9.76,
    "humidity": 34,
    "wind_speed": 4.33,
    "precipitation": 0,
    "pm25": 3.91,
    "pm10": 6.66,
    "pm_grade": "좋음",
    "season": "가을",
    "time_range": "오전",
    "owm_dt_txt": "2025-11-10 09:00:00",
    "is_rainy": false,
    "is_windy": false,
    "is_too_cold": false,
    "is_too_hot": false,
    "is_bad_air": false
  },
  "XGBoost_예측_카테고리": "유산소",
  "최종_선택_카테고리": "유산소",
  "실내_우선여부": false,
  "추천_운동": "걷기",
  "추천_근거": "사용자 선호운동(실시간 입력): ['걷기', '배드민턴'] / 날씨와 공기 질이 비교적 양호하여, 실외 운동을 우선 고려했습니다. / 콘텐츠 기반 + GNN 임베딩 점수(score=5.00)가 가장 높은 운동을 추천했습니다.",
  "디버그": {
    "favorites_raw": ["걷기", "배드민턴"],
    "favorites_clean": ["걷기", "배드민턴"],
    "favorite_categories": ["유산소", "구기스포츠"],
    "favorites_matched_in_category": ["걷기"]
  }
}
```
* 실제 값은 조회 시점의 날씨 / 대기질 / 입력 위치 및 시간에 따라 달라집니다.
* GNN 임베딩 파일(activity_gnn_embeddings.pt)이 없으면, “콘텐츠 기반” 점수만 사용하여 추천하며, 관련 메시지가 로그에 찍힙니다.
## 7. 추천 로직 상세

### 7.1 위치 + 시간 입력
1.  클라이언트에서 `location_name`, `target_datetime`, `favorites` 전달
2.  서버에서 **OpenWeatherMap Geocoding API**로 해당 위치의 위도/경도를 조회

### 7.2 날씨 + 대기질 조회
* OpenWeatherMap **Forecast API(5일/3시간 간격)**로 예보 조회
    * `target_datetime`과 가장 가까운 시각의 예보를 선택 후,
    * 기온(temperature)
    * 습도(humidity)
    * 풍속(wind_speed)
    * 강수량(precipitation)
* OpenWeatherMap **Air Pollution API**로
    * PM2.5, PM10 → 공기질 등급 (좋음/보통/나쁨/매우 나쁨 ) 계산
* 날짜/시간으로부터
    * 계절( 봄/여름/겨...
    * 시간대( 오전/오후/저녁/야간 ) 파생

### 7.3 XGBoost로 활동 카테고리 예측 (환경 기반)
입력 특징:
* ( `온도`, `습도`, `풍속`, `강수량`, `pm25`, `season_code` )

출력:
* 활동 카테고리: `유산소`, `피트니스`, `구기스포츠`, `익스트림스포츠`
### 7.4 실내/실외 판단
다음 조건 중 하나라도 해당하면 실내 운동 우선:
* 강수량 > 0 (비/눈)
* 풍속 ≥ 8 m/s (강풍)
* 기온 ≤ 0°C (매우 추움)
* 기온 ≥ 30°C (매우 더움)
* 미세먼지(PM10) 또는 초미세먼지(PM2.5)가 나쁨 / 매우 나쁨

그 외에는 실외 운동 우선으로 판단합니다.

### 7.5 선호운동 + 카테고리 조정
* `favorites`에 포함된 운동명을 `activity_information.csv` 에서 찾아 해당 카테고리 목록을 얻습니다.
* XGBoost 예측 카테고리(`model_cat`)와 선호 카테고리들을 함께 고려하여 **최종 카테고리(`pred_category`)**를 선택:
    * 선호 카테고리가 하나뿐이면 → 그 카테고리 우선
    * 둘 이상이고 그 안에 `model_cat`이 포함되면 → `model_cat` 우선
    * 그렇지 않으면 - ...
    * 선호운동이 메타데이터와 매칭되지 않으면 → `model_cat` 사용

### 7.6 후보 운동 필터링 (1차: 규칙 기반)
`activity_information.csv`에서:
1.  `category == pred_category` 인 운동들 추출
2.  **계절 필터:**
    * `season` 컬럼이 비어 있거나, 현재 `season_str`를 포함하는 경우만 사용
3.  **시간대 필터:**
    * `time_range` 컬럼이 비어 있거나, 현재 `time_range_str`를 포함하는 경우만 사용
4.  **실내/실외 필터:**
    * 실내 우선인 경우 → `location_type`에 `"실내"` 포함
    * 실외 우선인 경우 → `location_type`에 `"실외"` 포함
        (없으면 카테고리 내 전체 후보 사용)
### 7.7 콘텐츠 기반 + GNN 스코어링 (2차: 학습/임베딩 기반 최적화)
각 후보 운동에 대해 점수를 계산합니다.

**(1) 콘텐츠 기반 점수**
* 선호운동 이름과 일치하면 **+3.0**
* 실내/실외 조건과 일치하면 **+2.0**
* 추천 계절과 일치하면 **+1.0**
* 추천 시간대와 일치하면 **+1.0**

**(2) GNN(GraphSAGE) 임베딩 점수**
1.  `user_preference.csv`를 기반으로 **user-activity 그래프**를 구성하고,
    **GraphSAGE**로 학습하여 `activity_gnn_embeddings.pt`를 생성합니다.
2.  서버에서는 이 파일을 로드해:
    * 각 `activity_name`에 대해 고정된 임베딩 벡터를 보유
    * 사용자의 선호운동들의 임베딩 평균 → **사용자 벡터(user_vec)**로 사용
3.  `user_vec`과 각 활동 벡터의 **cosine 유사도**를 계산하여,
    `0 ~ 2` 범위로 스케... .

**최종 점수 =**
**콘텐츠 기반 점수 (선호/실내-실외/계절/시간대)**
* **"GNN 유사도 점수"**

점수가 가장 높은 운동을 **최종 추천_운동**으로 반환합니다.
### 8. 시연(데모) 플로우 예시
1.  사용자가 앱/웹에서 로그인
2.  선호운동 2개 선택 (예: `걷기`, `배드민턴`)
3.  지역 입력 또는 지도에서 선택 (예: `"서울특별시 강남구"`)
4.  운동 예정 시간 선택 (예: `"2025-11-07 09:00:00"`)
5.  프론트엔드에서 `/recommend/by-location-and-user` 로 요청
6.  백엔드에서
    * Geocoding → 날씨/대기질 조회 → XGBoost 카테고리 예측
    * 실내/실외 판단 → 후보 운동 필터링
    * 콘텐츠 기반 + GNN 스코어링으로 최종 운동 1개 선택
7.  UI에 다음 정보 표시:
    * 추천 운동명
    * 간단한 추천 근거 (날씨/공기질/실내-실외/선호운동/GNN 반영 설명)

### 9. 주의 사항 및 팁
* `activity_information.csv` 와 `category_model.pkl`, `feature_scaler.pkl`, `category_label_encoder.pkl` 은
  동일한 피처 구조를 기준으로 학습/저장되어 있어야 합니다.
* `activity_gnn_embeddings.pt` 가 없을 경우:
    * 앱은 여전히 정상 동작하며,
    * GNN 점수는 0으로 처리되고 콘텐츠 기반 스코어링만 사용됩니다.
* OpenWeatherMap 무료 플랜의 호출 제한을 고려하여,
  너무 잦은 호출(짧은 주기 반복 요청)은 피하는 것이 좋습니다.
* 실제 서비스 배포 시 `.env` 와 API Key는 절대 공개 저장소에 커밋하지 않도록 주의하세요.
* 모델/임베딩을 업데이트할 때는:
    * **XGBoost:** `Weather_Conditions.csv` 기반 재학습 후 `.pkl` 파일 재생성
    * **GNN(GraphSAGE):** `user_preference.csv` + `activity_information.csv` 로 그래프 재구성 후
      `activity_gnn_embeddings.pt` 재생성
