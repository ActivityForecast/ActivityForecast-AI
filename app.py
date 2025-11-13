# -*- coding: utf-8 -*-
"""
app.py — 운동 추천 API (Kakao 지오코딩 + KMA/Open-Meteo + GNN)

- 지오코딩: Kakao Local 주소검색 + 역지오코딩(대한민국 대상)
    * 입력: 상세 도로명주소(예: "경기도 구리시 동구릉로30번길 33")
    * 출력: 카카오맵 기준 위도/경도 + 정규화된 도로명주소
    * Kakao 주소검색 결과 ↔ 역지오코딩 결과 주소가 일치하는 경우를 우선 사용

- 날씨: KMA 관측/초단기/단기(1h) 우선 + Open-Meteo(7일/1h)로 5일까지 보충
- 추천: 선호운동이 속한 카테고리들(합집합) 안에서 후보를 뽑고
        GNN 가중 / 실내·실외 정책 / 랜덤 샘플로 운동 3개 추천
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import os, re, random
import requests
import pandas as pd
import numpy as np
import joblib

# -------- optional: .env --------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- optional: torch (GNN) --------
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    print("torch import 실패. GNN 임베딩 비활성화:", e)

# =========================
# FastAPI
# =========================
app = FastAPI(
    title="운동 추천 API (Kakao 지오코딩 + KMA/Open-Meteo + GNN)",
    version="4.1.0-kakao-only-pref-based",
)

# =========================
# Kakao Local API (지오코딩 전용)
# =========================
# .env에 KAKAO_REST_API_KEY를 넣어두고 사용하는 걸 추천
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY", "e171de783420e9199d7edc58d475ffd2")
KAKAO_ADDRESS_URL  = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_REVERSE_URL  = "https://dapi.kakao.com/v2/local/geo/coord2address.json"

# =========================
# OWM (미세먼지용)
# =========================
OWM_API_KEY = os.getenv("OWM_API_KEY", "9f8737894290204a7b2793f510443e46")

# =========================
# KMA (관측/초단기/단기)
# =========================
KMA_API_KEY         = os.getenv("KMA_API_KEY", "e44a573223f2660f6169de6e9c31658a456d562ea2e1d84e4cc717baca25950a")
KMA_ULTRA_NCST_URL  = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
KMA_ULTRA_FCST_URL  = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
KMA_VILAGE_FCST_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

# Open-Meteo (무료, 7일 1시간)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# =========================
# 모델/스케일러/라벨인코더
# =========================
category_model = joblib.load("category_model.pkl")
scaler         = joblib.load("feature_scaler.pkl")  # temp, hum, ws, precip, pm25, pm10
le_category    = joblib.load("category_label_encoder.pkl")

# =========================
# 활동 메타데이터
# =========================
activity_df = pd.read_csv("activity_information.csv")
activity_df = activity_df.rename(columns={
    "활동명": "activity_name",
    "활동카테고리": "category",
    "장소유형": "indoor_outdoor",
    "난이도": "difficulty",
    "추천계절": "season",
    "적합시간대": "time_range",
})
for col in ["season", "time_range", "indoor_outdoor"]:
    if col not in activity_df.columns:
        activity_df[col] = np.nan

# =========================
# GNN 임베딩(옵션)
# =========================
if TORCH_AVAILABLE:
    try:
        gnn_obj = torch.load("activity_gnn_embeddings.pt", map_location="cpu")
        ACT_EMB = gnn_obj.get("activity_embeddings", None)
        ACT2IDX = gnn_obj.get("act2idx", {})
        if ACT_EMB is not None:
            ACT_EMB = torch.nn.functional.normalize(ACT_EMB, dim=1)
            print("GNN activity embeddings loaded.")
        else:
            print("GNN embeddings에 activity_embeddings 없음.")
    except Exception as e:
        ACT_EMB, ACT2IDX = None, {}
        print("GNN embeddings not loaded:", e)
else:
    ACT_EMB, ACT2IDX = None, {}
    print("torch 없음 → GNN 미사용.")

def build_user_embedding_from_favorites(favorite_list: List[str]):
    if ACT_EMB is None or not favorite_list:
        return None
    idx_list = [ACT2IDX.get(n) for n in favorite_list if n in ACT2IDX]
    idx_list = [i for i in idx_list if i is not None]
    if not idx_list:
        return None
    mat = ACT_EMB[idx_list]
    u = mat.mean(dim=0)
    return torch.nn.functional.normalize(u, dim=0)

def gnn_similarity_score(user_vec, activity_name: str) -> float:
    if user_vec is None:
        return 0.0
    idx = ACT2IDX.get(activity_name)
    if idx is None:
        return 0.0
    return float(torch.dot(user_vec, ACT_EMB[idx]).item())

# =========================
# 요청 스키마
# =========================
class LocationUserRequest(BaseModel):
    user_id: str
    location_name: str          # 상세 도로명주소 (대한민국)
    target_datetime: datetime
    favorites: Optional[List[str]] = None

# =========================
# 유틸: 계절/시간대/미세먼지 등급
# =========================
def get_season_from_month(m: int) -> str:
    if m in (3, 4, 5):  return "봄"
    if m in (6, 7, 8):  return "여름"
    if m in (9, 10, 11): return "가을"
    return "겨울"

def get_time_range_from_hour(h: int) -> str:
    if 6 <= h < 12:   return "오전"
    if 12 <= h < 18:  return "오후"
    if 18 <= h < 23:  return "저녁"
    return "야간"

def classify_pm_grade(pm25: float, pm10: float) -> str:
    if pm25 <= 15 and pm10 <= 30:   return "좋음"
    if pm25 <= 35 and pm10 <= 80:   return "보통"
    if pm25 <= 75 and pm10 <= 150:  return "나쁨"
    return "매우 나쁨"

def _same_no_space(a: str, b: str) -> bool:
    return re.sub(r"\s+", "", a or "") == re.sub(r"\s+", "", b or "")

# =========================
# Kakao 지오코딩 (대한민국 전용, 엄격 모드)
# =========================
def geocode_kakao_strict(query: str):
    """
    - 입력: 대한민국 상세 도로명주소(또는 일반 주소 문자열)
    - 동작:
        1) Kakao 주소검색(query) 호출
        2) 도로명 주소(ROAD_ADDR)를 우선 후보로 사용
        3) 각 후보에 대해:
            - 좌표(y, x)를 역지오코딩(coord2address)에 넣고
            - forward 주소와 reverse 주소의 address_name이
              공백 무시하고 완전히 일치하면 엄격 매칭 성공
        4) 그런 후보가 하나도 없으면:
            - 도로명 주소 후보들 중 첫 번째를 사용 (완화)
    - 반환: (lat, lon, normalized_address, addr_level)
        addr_level: "detailed" (ROAD_ADDR) or "admin" (그 외)
    """
    if (not KAKAO_REST_API_KEY) or ("여기에_너의_KAKAO" in KAKAO_REST_API_KEY):
        raise HTTPException(status_code=500, detail="Kakao REST API 키가 설정되지 않았습니다.")

    headers = {"Authorization": "KakaoAK " + KAKAO_REST_API_KEY}

    # 1) 주소검색
    r = requests.get(
        KAKAO_ADDRESS_URL,
        headers=headers,
        params={"query": query, "size": 10},
        timeout=7,
    )
    if r.status_code in (401, 403):
        raise HTTPException(status_code=500, detail=f"Kakao 인증 실패 (HTTP {r.status_code})")
    r.raise_for_status()
    data = r.json()
    docs = data.get("documents", [])
    if not docs:
        raise HTTPException(status_code=400, detail=f"Kakao 주소검색 결과 없음: {query}")

    # 도로명 주소 우선
    road_docs = [d for d in docs if d.get("address_type") == "ROAD_ADDR"]
    candidates = road_docs if road_docs else docs

    def _addr_name_from_doc(d):
        ra = d.get("road_address") or d.get("address") or {}
        addr_name = ra.get("address_name") or d.get("address_name") or ""
        return addr_name.strip()

    # 2) forward 주소 ↔ reverse 주소를 일치시키는 엄격 매칭
    for d in candidates:
        addr_name = _addr_name_from_doc(d)
        if not addr_name:
            continue

        lat = float(d["y"])
        lon = float(d["x"])

        r2 = requests.get(
            KAKAO_REVERSE_URL,
            headers=headers,
            params={"x": lon, "y": lat},
            timeout=7,
        )
        r2.raise_for_status()
        rev_docs = r2.json().get("documents", [])
        if not rev_docs:
            continue

        ra2 = rev_docs[0].get("road_address") or rev_docs[0].get("address") or {}
        rev_addr_name = (ra2.get("address_name") or "").strip()

        if _same_no_space(addr_name, rev_addr_name):
            addr_level = "detailed" if d.get("address_type") == "ROAD_ADDR" else "admin"
            return lat, lon, addr_name, addr_level

    # 3) 엄격 일치가 하나도 없으면 → 첫 번째 후보 사용 (도로명 주소 우선)
    first = candidates[0]
    addr_name = _addr_name_from_doc(first) or query
    lat = float(first["y"])
    lon = float(first["x"])
    addr_level = "detailed" if first.get("address_type") == "ROAD_ADDR" else "admin"
    return lat, lon, addr_name, addr_level

def geocode_location(location_name: str):
    """
    메인 지오코딩 함수
    - 지금은 Kakao만 사용 (대한민국 전용)
    - 반환: (lat, lon, geo_source, normalized_addr, addr_level)
    """
    raw = location_name.strip()
    lat, lon, addr_name, addr_level = geocode_kakao_strict(raw)
    return lat, lon, "kakao_strict", addr_name, addr_level

# =========================
# 시간/좌표 변환 (KMA용)
# =========================
def _dfs_xy_conv(lat, lon):
    RE = 6371.00877
    GRID = 5.0
    SLAT1 = 30.0
    SLAT2 = 60.0
    OLON = 126.0
    OLAT = 38.0
    XO = 43
    YO = 136
    DEGRAD = np.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD
    sn = np.tan(np.pi * 0.25 + slat2 * 0.5) / np.tan(np.pi * 0.25 + slat1 * 0.5)
    sn = np.log(np.cos(slat1) / np.cos(slat2)) / np.log(sn)
    sf = np.tan(np.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn) * (np.cos(slat1) / sn)
    ro = np.tan(np.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)
    ra = np.tan(np.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)
    theta = lon * DEGRAD - olon
    if theta > np.pi:
        theta -= 2.0 * np.pi
    if theta < -np.pi:
        theta += 2.0 * np.pi
    theta *= sn
    x = (ra * np.sin(theta)) + XO
    y = (ro - ra * np.cos(theta)) + YO
    return int(x + 0.5), int(y + 0.5)

def _kma_base_datetime(kst_dt: datetime):
    base = kst_dt.replace(minute=0, second=0, microsecond=0)
    if kst_dt.minute < 40:
        base -= timedelta(hours=1)
    return base.strftime("%Y%m%d"), base.strftime("%H%M")

def _kma_call(url, params):
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    return data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

def _build_kst(dt: datetime) -> datetime:
    KST = timezone(timedelta(hours=9))
    return dt.astimezone(KST) if dt.tzinfo else dt.replace(tzinfo=KST)

# =========================
# KMA: 관측/예보 수집
# =========================
def get_kma_nowcast(lat: float, lon: float, ref_dt: datetime):
    if not KMA_API_KEY:
        return []
    KST_dt = _build_kst(ref_dt)
    base_date, base_time = _kma_base_datetime(KST_dt)
    nx, ny = _dfs_xy_conv(lat, lon)
    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": 60,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    items = _kma_call(KMA_ULTRA_NCST_URL, params)
    row = {}
    for it in items:
        cat = it.get("category")
        val = it.get("obsrValue")
        row[cat] = val
    out = []
    try:
        t = int(round(float(row.get("T1H"))))
        h = int(round(float(row.get("REH"))))
        w = float(row.get("WSD", 0.0))
        p_raw = row.get("RN1", "0")
        try:
            p = 0.0 if "강수없음" in str(p_raw) else float(str(p_raw).replace("mm", "").strip())
        except Exception:
            p = 0.0
        ts = KST_dt.replace(minute=0, second=0, microsecond=0)
        out.append(
            {
                "dt": ts,
                "temperature": t,
                "humidity": h,
                "wind_speed": w,
                "precipitation": p,
                "src": "kma_nowcast",
            }
        )
    except Exception:
        pass
    return out

def get_kma_ultra_fcst(lat: float, lon: float, ref_dt: datetime):
    if not KMA_API_KEY:
        return []
    KST_dt = _build_kst(ref_dt)
    base_date, base_time = _kma_base_datetime(KST_dt)
    nx, ny = _dfs_xy_conv(lat, lon)
    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": 200,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    items = _kma_call(KMA_ULTRA_FCST_URL, params)
    slots = {}
    for it in items:
        fd, ft = it.get("fcstDate"), it.get("fcstTime")
        key = f"{fd}{ft}"
        if key not in slots:
            slots[key] = {}
        slots[key][it.get("category")] = it.get("fcstValue")
    out = []
    for key, vals in slots.items():
        y, m, d = int(key[:4]), int(key[4:6]), int(key[6:8])
        hh, mm = int(key[8:10]), int(key[10:12])
        ts = datetime(y, m, d, hh, mm, tzinfo=timezone(timedelta(hours=9)))
        try:
            t = int(round(float(vals["T1H"])))
            h = int(round(float(vals["REH"])))
            w = float(vals.get("WSD", 0.0))
        except Exception:
            continue
        p_raw = vals.get("RN1", "0")
        try:
            p = 0.0 if "강수없음" in str(p_raw) else float(str(p_raw).replace("mm", "").strip())
        except Exception:
            p = 0.0
        out.append(
            {
                "dt": ts,
                "temperature": t,
                "humidity": h,
                "wind_speed": w,
                "precipitation": p,
                "src": "kma_ultra_fcst",
            }
        )
    out.sort(key=lambda x: x["dt"])
    return out

def get_kma_vilage_fcst(lat: float, lon: float, ref_dt: datetime, hours: int = 72):
    if not KMA_API_KEY:
        return []
    KST_dt = _build_kst(ref_dt)
    base_date, base_time = _kma_base_datetime(KST_dt)
    nx, ny = _dfs_xy_conv(lat, lon)
    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": 1000,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    items = _kma_call(KMA_VILAGE_FCST_URL, params)
    keep = {"TMP", "REH", "WSD", "PCP"}
    bucket = {}
    for it in items:
        cat = it.get("category")
        if cat not in keep:
            continue
        key = f"{it.get('fcstDate')}{it.get('fcstTime')}"
        if key not in bucket:
            bucket[key] = {}
        bucket[key][cat] = it.get("fcstValue")
    out = []
    for key, vals in bucket.items():
        y, m, d = int(key[:4]), int(key[4:6]), int(key[6:8])
        hh, mm = int(key[8:10]), int(key[10:12])
        ts = datetime(y, m, d, hh, mm, tzinfo=timezone(timedelta(hours=9)))
        if ts < KST_dt:
            continue
        try:
            t = int(round(float(vals["TMP"])))
            h = int(round(float(vals["REH"])))
            w = float(vals.get("WSD", 0.0))
        except Exception:
            continue
        p_raw = vals.get("PCP", "0")
        try:
            p = 0.0 if "강수없음" in str(p_raw) else float(str(p_raw).replace("mm", "").strip())
        except Exception:
            p = 0.0
        out.append(
            {
                "dt": ts,
                "temperature": t,
                "humidity": h,
                "wind_speed": w,
                "precipitation": p,
                "src": "kma_vilage_fcst",
            }
        )
    out.sort(key=lambda x: x["dt"])
    end_cut = _build_kst(ref_dt) + timedelta(hours=hours)
    out = [x for x in out if x["dt"] <= end_cut]
    return out

# =========================
# Open-Meteo 7일/1시간
# =========================
def get_open_meteo_hourly(lat: float, lon: float, start_dt: datetime, hours: int = 120):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "Asia/Seoul",
        "forecast_days": 7,
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    hr = data.get("hourly", {})
    times = hr.get("time", [])
    temp = hr.get("temperature_2m", [])
    hum  = hr.get("relative_humidity_2m", [])
    wind = hr.get("wind_speed_10m", [])
    pcp  = hr.get("precipitation", [])
    series = []
    for i in range(min(len(times), len(temp), len(hum), len(wind), len(pcp))):
        ts = pd.to_datetime(times[i]).to_pydatetime().replace(tzinfo=timezone(timedelta(hours=9)))
        series.append(
            {
                "dt": ts,
                "temperature": int(round(float(temp[i]))),
                "humidity": int(round(float(hum[i]))),
                "wind_speed": float(wind[i]),
                "precipitation": float(pcp[i]),
                "src": "open_meteo",
            }
        )
    series.sort(key=lambda x: x["dt"])
    KST_start = _build_kst(start_dt).replace(minute=0, second=0, microsecond=0)
    end = KST_start + timedelta(hours=hours)
    series = [x for x in series if KST_start <= x["dt"] <= end]
    return series

# =========================
# 5일(120h) 1시간 타임라인 빌드
# =========================
def build_5day_hourly_series(lat: float, lon: float, start_dt: datetime):
    KST_start = _build_kst(start_dt).replace(minute=0, second=0, microsecond=0)
    KST_end   = KST_start + timedelta(hours=120)
    timeline = []
    timeline += get_kma_nowcast(lat, lon, KST_start)
    timeline += get_kma_ultra_fcst(lat, lon, KST_start)
    timeline += get_kma_vilage_fcst(lat, lon, KST_start, hours=72)
    om = get_open_meteo_hourly(lat, lon, KST_start, hours=120)

    if om:
        pri = {"kma_nowcast": 4, "kma_ultra_fcst": 3, "kma_vilage_fcst": 2, "open_meteo": 1}
        bucket = {}
        for x in timeline + om:
            k = x["dt"]
            if KST_start <= k <= KST_end:
                if (k not in bucket) or (pri.get(x["src"], 0) > pri.get(bucket[k]["src"], 0)):
                    bucket[k] = x
        merged = [bucket[k] for k in sorted(bucket.keys())]
    else:
        merged = [x for x in timeline if KST_start <= x["dt"] <= KST_end]

    for x in merged:
        x["season"] = get_season_from_month(_build_kst(x["dt"]).month)
        x["time_range"] = get_time_range_from_hour(_build_kst(x["dt"]).hour)
    return merged

def get_weather_at_datetime(lat: float, lon: float, target_dt: datetime):
    series = build_5day_hourly_series(lat, lon, target_dt)
    if not series:
        raise RuntimeError("시계열 생성 실패")
    tgt = _build_kst(target_dt).replace(minute=0, second=0, microsecond=0)
    for x in series:
        if x["dt"] == tgt:
            return x
    best = min(series, key=lambda x: abs(x["dt"] - tgt))
    return best

# =========================
# 미세먼지(간단: OWM 공기질)
# =========================
def get_air_quality(lat: float, lon: float):
    try:
        r = requests.get(
            "http://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": OWM_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        comp = (data.get("list") or [{}])[0].get("components", {})
        pm25 = float(comp.get("pm2_5", 0.0))
        pm10 = float(comp.get("pm10", 0.0))
    except Exception:
        pm25, pm10 = 0.0, 0.0
    return pm25, pm10, classify_pm_grade(pm25, pm10)

# =========================
# 스케일러 입력 벡터(6개)
# =========================
def build_feature_vector(weather: dict) -> np.ndarray:
    x = np.array(
        [[
            weather["temperature"],
            weather["humidity"],
            weather["wind_speed"],
            weather["precipitation"],
            weather["pm25"],
            weather["pm10"],
        ]],
        dtype=float,
    )
    return scaler.transform(x)

# =========================
# 추천 엔드포인트 (선호 카테고리 합집합 기반)
# =========================
@app.post("/recommend/by-location-and-user")
def recommend_by_location_and_user(req: LocationUserRequest):
    # 1) 지오코딩 (대한민국 Kakao)
    lat, lon, geo_source, normalized_addr, addr_level = geocode_location(req.location_name)

    # 2) 타겟 시각 날씨(정시)
    w = get_weather_at_datetime(lat, lon, req.target_datetime)
    pm25, pm10, pm_grade = get_air_quality(lat, lon)
    w["pm25"], w["pm10"], w["pm_grade"] = pm25, pm10, pm_grade

    is_rainy    = float(w["precipitation"]) > 0.1
    is_windy    = float(w["wind_speed"])   >= 7.0
    is_too_cold = int(w["temperature"])    <= 0
    is_too_hot  = int(w["temperature"])    >= 30
    is_bad_air  = pm_grade in ["나쁨", "매우 나쁨"]
    indoor_required = is_rainy or is_windy or is_too_cold or is_too_hot or is_bad_air

    # 3) 카테고리 예측 (XGBoost)
    weather_for_model = {
        "temperature": int(w["temperature"]),
        "humidity": int(w["humidity"]),
        "wind_speed": float(w["wind_speed"]),
        "precipitation": float(w["precipitation"]),
        "pm25": float(pm25),
        "pm10": float(pm10),
    }
    X_scaled = build_feature_vector(weather_for_model)
    cat_idx  = category_model.predict(X_scaled)[0]
    pred_category = le_category.inverse_transform([cat_idx])[0]

    # 4) 선호 카테고리 기반으로 추천 범위 결정
    fav_list_raw = req.favorites or []
    fav_list = [f.strip().replace(",", "") for f in fav_list_raw if f and f.strip()]
    fav_meta = activity_df[activity_df["activity_name"].isin(fav_list)]
    fav_cats = list(fav_meta["category"].dropna().unique())

    reason_parts = []
    if fav_list:
        reason_parts.append(f"사용자 선호운동(입력): {fav_list}")

    # base_categories = 실제 추천에 사용할 카테고리 집합
    if not fav_cats:
        base_categories = [pred_category]
        reason_parts.append(f"선호 카테고리 없음 → 모델 예측 '{pred_category}' 사용")
    elif len(fav_cats) == 1:
        base_categories = fav_cats
        reason_parts.append(
            f"선호 카테고리 '{fav_cats[0]}' 우선 사용 (모델 예측 '{pred_category}'는 참고)"
        )
    else:
        base_categories = fav_cats
        reason_parts.append(
            f"여러 선호 카테고리 {fav_cats}에서 추천 (모델 예측 '{pred_category}'는 참고)"
        )

    if indoor_required:
        reason_parts.append(
            f"날씨/대기 불리(비:{is_rainy}, 강풍:{is_windy}, 추움:{is_too_cold}, 더움:{is_too_hot}, 미세먼지:{pm_grade}) → 실내 우선"
        )
    else:
        reason_parts.append("날씨/공기질 양호 → 실외 우선")

    # 5) 후보 필터 (선호 카테고리 합집합 기반)
    candidates = activity_df[activity_df["category"].isin(base_categories)].copy()

    season_str = get_season_from_month(_build_kst(w["dt"]).month)
    time_range_str = get_time_range_from_hour(_build_kst(w["dt"]).hour)

    tmp = candidates[
        (candidates["season"].isna())
        | (candidates["season"].astype(str).str.contains(season_str))
    ]
    if not tmp.empty:
        candidates = tmp

    tmp = candidates[
        (candidates["time_range"].isna())
        | (candidates["time_range"].astype(str).str.contains(time_range_str))
    ]
    if not tmp.empty:
        candidates = tmp

    if indoor_required:
        tmp = candidates[candidates["indoor_outdoor"] == "실내"]
        if not tmp.empty:
            candidates = tmp

    if candidates.empty:
        candidates = activity_df[activity_df["category"].isin(base_categories)].copy()

    # 6) 점수 + 랜덤 샘플(상위 10개 내에서 3개 무작위)
    favorite_set = set(fav_list)
    user_vec = build_user_embedding_from_favorites(fav_list)

    scored = []
    for _, row in candidates.iterrows():
        act = row["activity_name"]
        s = 3.0
        if act in favorite_set:
            s += 2.0
        s += gnn_similarity_score(user_vec, act) if user_vec is not None else 0.0
        if indoor_required and row["indoor_outdoor"] == "실외":
            s -= 2.0
        scored.append((act, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    pool = scored[: min(10, len(scored))]
    rng = random.Random()
    rng.seed(os.urandom(16))
    rng.shuffle(pool)
    top = pool[: min(3, len(pool))]

    recommended_activity   = top[0][0] if top else None
    recommended_activities = [a for a, _ in top]

    reason = " / ".join(reason_parts)
    if top:
        reason += f" / 상위 후보 10개 중 랜덤 샘플 → {recommended_activities}"

    weather_info = {
        "temperature": int(w["temperature"]),
        "humidity": int(w["humidity"]),
        "wind_speed": float(w["wind_speed"]),
        "precipitation": float(w["precipitation"]),
        "pm25": float(pm25),
        "pm10": float(pm10),
        "pm_grade": pm_grade,
        "season": season_str,
        "time_range": time_range_str,
        "dt_kst": _build_kst(w["dt"]).isoformat(),
        "source": w["src"],
        "is_rainy": is_rainy,
        "is_windy": is_windy,
        "is_too_cold": is_too_cold,
        "is_too_hot": is_too_hot,
        "is_bad_air": is_bad_air,
    }

    return {
        "입력값": {
            "user_id": req.user_id,
            "location_name": req.location_name,
            "정규화_된_주소": normalized_addr,
            "주소_레벨": addr_level,  # detailed/admin
            "target_datetime": req.target_datetime.isoformat(),
            "favorites_from_request": fav_list_raw,
        },
        "지오코딩": {"위도": lat, "경도": lon},
        "지오코딩_출처": geo_source,
        "사용된_날씨": weather_info,
        "XGBoost_예측_카테고리": pred_category,
        "최종_선택_카테고리": base_categories,
        "실내_우선여부": indoor_required,
        "추천_운동": recommended_activity,
        "추천_운동_목록": recommended_activities,
        "추천_근거": reason,
    }

# =========================
# 5일(120h) 예보만 반환(검증용)
# =========================
@app.post("/forecast/5days")
def forecast_5days(req: LocationUserRequest):
    lat, lon, src, norm, lvl = geocode_location(req.location_name)
    series = build_5day_hourly_series(lat, lon, req.target_datetime)
    if not series:
        raise HTTPException(status_code=500, detail="5일 예보 시계열 생성 실패")
    return {
        "입력값": {
            "location_name": req.location_name,
            "target_datetime": req.target_datetime.isoformat(),
        },
        "위치": {"lat": lat, "lon": lon},
        "지오코딩_출처": src,
        "정규화_된_주소": norm,
        "주소_레벨": lvl,
        "해상도": "1시간",
        "길이(시간)": len(series),
        "시계열": [
            {
                "dt": _build_kst(x["dt"]).isoformat(),
                "temperature": int(x["temperature"]),
                "humidity": int(x["humidity"]),
                "wind_speed": float(x["wind_speed"]),
                "precipitation": float(x["precipitation"]),
                "season": x["season"],
                "time_range": x["time_range"],
                "source": x["src"],
            }
            for x in series
        ],
    }
