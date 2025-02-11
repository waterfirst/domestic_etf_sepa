# 국내 ETF 분석 및 백테스트 시스템

이 프로젝트는 국내 ETF에 대한 SEPA 전략 기반 분석과 백테스트 기능을 제공하는 두 개의 독립적인 웹 애플리케이션으로 구성되어 있습니다.

## 1. ETF SEPA 전략 분석기 (app1.py)

SEPA(Secondary Evaluation Price Action) 전략을 기반으로 국내 ETF를 분석하고 추천하는 시스템입니다.

### 주요 기능
- 국내 상위 ETF 실시간 데이터 분석
- SEPA 전략 기반 점수 산출
- 기술적 지표 분석 (RSI, MACD, 이동평균 등)
- 상위 추천 ETF 시각화
- 추천 ETF 목록 JSON 파일 저장

### 분석 지표
- SEPA 점수
- 기술적 지표 (RSI, MACD, 볼린저 밴드)
- 거래량 분석
- 추세 강도
- 이중 바닥 패턴 분석

## 2. ETF 백테스트 시스템 (app2.py)

ETF의 과거 데이터를 기반으로 다양한 매매 전략을 테스트하고 성과를 분석하는 시스템입니다.

### 주요 기능
- JSON 파일 기반 ETF 데이터 로드
- 개별 ETF 백테스트
- 전체 ETF 일괄 분석
- 매매 시그널 시각화
- 성과 분석 리포트 생성

### 백테스트 지표
- 총 수익률
- 승률
- 최대 낙폭
- 샤프 비율
- 평균 보유 기간
- 수익 요인

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/yourusername/your-repo-name.git

# 필요 패키지 설치
pip install -r requirements.txt
```

## 실행 방법

### ETF SEPA 분석기 실행
```bash
streamlit run app1.py
```

### ETF 백테스트 시스템 실행
```bash
streamlit run app2.py
```

## 사용된 기술

- Python 3.9+
- Streamlit
- Pandas
- yfinance
- Plotly
- NumPy

## 데이터 소스

- Yahoo Finance Korea (yfinance)
- 한국거래소 (KRX)

## 주의사항

1. 백테스트 결과는 과거 데이터를 기반으로 하며, 미래의 수익을 보장하지 않습니다.
2. 실제 투자 시에는 거래 수수료와 슬리피지를 고려해야 합니다.
3. 레버리지 ETF의 경우 변동성이 크므로 특별한 주의가 필요합니다.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 기여 방법

1. 이 저장소를 Fork 합니다.
2. 새로운 Branch를 생성합니다 (`git checkout -b feature/AmazingFeature`).
3. 변경사항을 Commit 합니다 (`git commit -m 'Add some AmazingFeature'`).
4. Branch에 Push 합니다 (`git push origin feature/AmazingFeature`).
5. Pull Request를 생성합니다.

## 서비스 주소

국내 ETF 분석_추천 - (https://domesticetfsepa-2l5qz9n84k5iu2betf4frb.streamlit.app/)

ETF 백테스팅 서비스- [https://domestic-etfsepa-backtest.streamlit.app/]
