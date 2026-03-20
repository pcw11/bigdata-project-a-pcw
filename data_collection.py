import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv, find_dotenv

# ============================================================
# 1. API 설정
# ============================================================
# .env 파일에서 API 키 로드
load_dotenv(find_dotenv())
API_KEY = os.getenv("API_KEY")

# 에어코리아 대기오염정보 - 시도별 실시간 측정정보 조회
url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"

# ============================================================
# 2. 요청 파라미터 설정
# ============================================================
params = {
    "serviceKey": API_KEY,        # 인증키
    "returnType": "json",         # 응답 형식
    "numOfRows": "100",           # 한 페이지에 가져올 데이터 수
    "pageNo": "1",                # 페이지 번호
    "sidoName": "서울",           # 시도 이름
    "ver": "1.0"                  # API 버전
}

# ============================================================
# 3. API 호출
# ============================================================
print("▶ API 호출 중...")
response = requests.get(url, params=params)

# 응답 상태 확인
print(f"▶ 응답 상태 코드: {response.status_code}")
if response.status_code != 200:
    print(f"▶ API 호출 실패! 상태 코드: {response.status_code}")
    print(f"▶ 응답 내용: {response.text[:200]}")
    exit()
else:
    print("▶ API 호출 성공!")

# ============================================================
# 4. JSON 응답 파싱
# ============================================================
data = response.json()

# 응답 구조 확인 (디버깅용)
print("\n▶ 응답 구조 확인 (최상위 키):")
print(json.dumps(list(data.keys()), ensure_ascii=False))

try:
    header = data["response"]["header"]
    print(f"\n▶ 결과 코드: {header['resultCode']}")
    print(f"▶ 결과 메시지: {header['resultMsg']}")

    body = data["response"]["body"]
    total_count = body["totalCount"]
    print(f"▶ 전체 데이터 수: {total_count}")

    items = body["items"]
    print(f"▶ items의 타입: {type(items)}")

    if isinstance(items, dict):
        # 형태 B: items가 dict이면 "item" 키 안에 실제 리스트 있음
        items = items.get("item", items)
        print("▶ items['item']에서 데이터 추출 (형태 B)")

    if isinstance(items, list):
        print(f"▶ 가져온 데이터 수: {len(items)}")
    else:
        print(f"▶ 예상치 못한 items 타입: {type(items)}")
        print(f"▶ items 내용 미리보기: {str(items)[:300]}")
        exit()

except KeyError as e:
    print(f"▶ 응답 구조가 예상과 다릅니다. 키 오류: {e}")
    print(f"▶ 전체 응답 일부 미리보기:\n{json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
    exit()

# ============================================================
# 5. DataFrame 변환
# ============================================================
df = pd.DataFrame(items)
print(f"\n▶ DataFrame 크기: {df.shape}")
print(f"▶ 컬럼 목록: {df.columns.tolist()}")
print("\n▶ 처음 5행:")
print(df.head())

# ============================================================
# 6. 필요한 컬럼만 선택하고 이름 변경
# ============================================================
columns_map = {
    "stationName": "측정소",
    "dataTime": "측정일시",
    "pm10Value": "미세먼지(PM10)",
    "pm25Value": "초미세먼지(PM2.5)",
    "o3Value": "오존(O3)",
    "no2Value": "이산화질소(NO2)",
    "coValue": "일산화탄소(CO)",
    "so2Value": "아황산가스(SO2)",
    "pm10Grade": "PM10등급",
    "pm25Grade": "PM25등급"
}

# 존재하는 컬럼만 선택
available_cols = {k: v for k, v in columns_map.items() if k in df.columns}
df_selected = df[list(available_cols.keys())].rename(columns=available_cols)

print(f"\n▶ 선택된 컬럼: {df_selected.columns.tolist()}")
print(df_selected.head(10))

# ============================================================
# 7. CSV 파일로 저장
# ============================================================
df_selected.to_csv("air_quality_seoul.csv", index=False, encoding="utf-8-sig")
print("\n▶ 'air_quality_seoul.csv' 파일로 저장 완료!")