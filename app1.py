import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

# 페이지 기본 설정

# 페이지 기본 설정
st.set_page_config(page_title="국내 ETF 분석 대시보드", page_icon="📈", layout="wide")


def get_top_kr_etfs():
    """
    수익률 상위 국내 상장 ETF 티커 목록을 반환합니다.
    """
    try:
        # 수익률 상위 ETF 티커 (각 티커 뒤에 .KS 추가)
        etf_tickers = [
            "473460.KS",  # KODEX 미국서학개미
            "473590.KS",  # ACE 미국주식베스트셀러
            "426030.KS",  # TIMEFOLIO 미국나스닥100액티브
            "461340.KS",  # HANARO 글로벌생성형AI액티브
            "456600.KS",  # TIMEFOLIO 글로벌AI인공지능액티브
            "407830.KS",  # 에셋플러스 글로벌플랫폼액티브
            "465580.KS",  # ACE 미국빅테크TOP7 Plus
            "411420.KS",  # KODEX 미국메타버스나스닥액티브
            "461900.KS",  # PLUS 미국테크TOP10
            "381170.KS",  # TIGER 미국테크TOP10 INDXX
            "498270.KS",  # KIWOOM 미국양자컴퓨팅
            "457990.KS",  # PLUS 태양광&ESS
            "457480.KS",  # ACE 테슬라밸류체인액티브
            "461910.KS",  # PLUS 미국테크TOP10레버리지
            "465610.KS",  # ACE 미국빅테크TOP7 Plus레버리지
            "488080.KS",  # TIGER 반도체TOP10레버리지
            "483340.KS",  # ACE 구글밸류체인액티브
            "494310.KS",  # KODEX 반도체레버리지
            "377990.KS",  # TIGER Fn신재생에너지
            "228790.KS",  # TIGER 화장품
            "409820.KS",  # KODEX 미국나스닥100레버리지
            "379800.KS",  # KODEX 미국SNP500
            "394350.KS",  # KIWOOM 글로벌퓨처모빌리티
            "306530.KS",  # HANARO 코스닥150선물레버리지
            "233740.KS",  # KODEX 코스닥150레버리지
            "233160.KS",  # TIGER 코스닥150 레버리지
            "278240.KS",  # RISE 코스닥150선물레버리지
            "144600.KS",  # KODEX 은선물(H)
            "491820.KS",  # HANARO 전력설비투자
            "418660.KS",  # TIGER 미국나스닥100레버리지
            "466950.KS",  # TIGER 글로벌AI액티브
            "487240.KS",  # KODEX AI전력핵심설비
            "423920.KS",  # TIGER 미국필라델피아반도체레버리지
            "225040.KS",  # TIGER 미국S&P500레버리지
            "479850.KS",  # HANARO K-뷰티
            "438320.KS",  # TIGER 차이나항셍테크레버리지
            "483320.KS",  # ACE 엔비디아밸류체인액티브
            "463050.KS",  # TIMEFOLIO K바이오액티브
            "486450.KS",  # SOL 미국AI전력인프라
            "471040.KS",  # KoAct 글로벌AI&로봇액티브
            "414270.KS",  # ACE 글로벌자율주행액티브
            "261070.KS",  # TIGER 코스닥150바이오테크
            "473500.KS",  # KIWOOM 글로벌전력반도체
            "465660.KS",  # TIGER 일본반도체FACTSET
            "243890.KS",  # TIGER 200에너지화학레버리지
            "476000.KS",  # UNICORN 포스트IPO액티브
            "495940.KS",  # RISE 미국AI테크액티브
            "480310.KS",  # TIGER 글로벌온디바이스AI
            "472160.KS",  # TIGER 미국테크TOP10 INDXX(H)
            "462900.KS",  # KoAct 바이오헬스케어액티브
            "304780.KS",  # HANARO 200선물레버리지
            "314250.KS",  # KODEX 미국빅테크10(H)
            "122630.KS",  # KODEX 레버리지
            "253150.KS",  # PLUS 200선물레버리지
            "123320.KS",  # TIGER 레버리지
            "252400.KS",  # RISE 200선물레버리지
            "453950.KS",  # TIGER TSMC파운드리밸류체인
            "267770.KS",  # TIGER 200선물레버리지
            "364970.KS",  # TIGER 바이오TOP10
            "253250.KS",  # KIWOOM 200선물레버리지
            "462330.KS",  # KODEX 2차전지산업레버리지
            "261920.KS",  # ACE 필리핀MSCI(합성)
            "498050.KS",  # HANARO 바이오코리아액티브
            "453650.KS",  # KODEX 미국S&P500금융
            "497570.KS",  # TIGER 미국필라델피아AI반도체나스닥
            "452250.KS",  # ACE 미국30년국채선물레버리지(합성 H)
            "486240.KS",  # DAISHIN343 AI반도체&인프라액티브
            "426410.KS",  # PLUS 미국대체투자Top10
            "446770.KS",  # ACE 글로벌반도체TOP4 Plus SOLACTIVE
            "152500.KS",  # ACE 레버리지
            "490090.KS",  # TIGER 미국AI빅테크10
            "306950.KS",  # KODEX KRX300레버리지
            "261220.KS",  # KODEX WTI원유선물(H)
            "464310.KS",  # TIGER 글로벌AI&로보틱스 INDXX
            "442580.KS",  # PLUS 글로벌HBM반도체
            "494340.KS",  # (Incomplete ETF name, please verify)
            "459580.KS",  # KODEX CD금리액티브(합성)
            "360750.KS",  # TIGER 미국S&P500
            "069500.KS",  # KODEX 200
            "357870.KS",  # TIGER CD금리투자KIS(합성)
            "488770.KS",  # KODEX 머니마켓액티브
            "133690.KS",  # TIGER 미국나스닥100
            "423160.KS",  # KODEX KOFR금리액티브(합성)
            "379800.KS",  # KODEX 미국S&P500
            "449170.KS",  # TIGER KOFR금리액티브(합성)
            "381170.KS",  # TIGER 미국테크TOP10 INDXX
            "305720.KS",  # KODEX 2차전지산업
            "305540.KS",  # TIGER 2차전지테마
            "091160.KS",  # KODEX 반도체
            "364980.KS",  # TIGER 2차전지TOP10
            "465580.KS",  # ACE 미국빅테크TOP7 Plus
            "483340.KS",  # ACE 구글밸류체인액티브
            "483320.KS",  # ACE 엔비디아밸류체인액티브
            "490090.KS",  # TIGER 미국AI빅테크10
            "495940.KS",  # RISE 미국AI테크액티브
            "456600.KS",  # TIMEFOLIO 글로벌AI인공지능액티브
        ]

        # 중복 제거
        etf_tickers = list(set(etf_tickers))

        # 정렬 (선택사항)
        etf_tickers.sort()

        return etf_tickers

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# 예시 사용
if __name__ == "__main__":
    etfs = get_top_kr_etfs()
    print("총 ETF 수:", len(etfs))
    for ticker in etfs:
        print(ticker)


def display_top_etf_recommendations(df_results):
    """상위 3개 ETF에 대한 상세 추천 정보를 표시합니다."""
    st.subheader("🌟 상위 3개 ETF 상세 분석")

    top_3_etfs = df_results.head(3)

    for idx, etf in top_3_etfs.iterrows():
        with st.expander(
            f"#{idx+1} {etf['ETF명']} (SEPA 점수: {etf['SEPA_점수']:.1f})",
            expanded=True,
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                # 기본 정보
                st.markdown("#### 📊 투자 포인트")
                points = [
                    f"• SEPA 전략 적합도: {etf['SEPA_점수']:.1f}점",
                    f"• 이중 바닥 패턴: {'확인됨 ✅' if etf['SEPA_조건'].get('이중 바닥 패턴', False) else '미확인 ❌'}",
                    f"• 패턴 강도: {etf['SEPA_조건'].get('패턴 강도', '0.00')}",
                    f"• 최근 모멘텀: {'강세' if etf['1개월수익률'] > 0 else '약세'}",
                ]
                st.markdown("\n".join(points))

                # 기술적 지표 분석
                st.markdown("#### 📈 기술적 지표")
                tech_analysis = [
                    f"• RSI: {etf['기술적_지표']['추세강도']:.1f} ({'과매수' if etf['기술적_지표']['추세강도'] > 70 else '과매도' if etf['기술적_지표']['추세강도'] < 30 else '중립'})",
                    f"• MACD: {'상승 추세' if etf['기술적_지표']['MACD_Signal'] > 0 else '하락 추세'}",
                    f"• 볼린저 밴드 위치: {etf['기술적_지표']['볼린저위치']:.1f}%",
                ]
                st.markdown("\n".join(tech_analysis))

            with col2:
                # 투자위험도 및 추천 전략
                st.markdown("#### ⚠️ 투자위험도")
                risk_level = (
                    "높음"
                    if etf["변동성"] > df_results["변동성"].quantile(0.75)
                    else (
                        "중간"
                        if etf["변동성"] > df_results["변동성"].quantile(0.25)
                        else "낮음"
                    )
                )
                st.warning(f"위험도: {risk_level}")

                st.markdown("#### 💡 투자 전략")
                strategy = [
                    "• 진입 전략: "
                    + (
                        "현재가 매수 가능"
                        if etf["기술적_지표"]["볼린저위치"] < 80
                        else "조정 시 매수 추천"
                    ),
                    (
                        "• 이중 바닥 형성 이후 추세 전환 예상"
                        if etf["SEPA_조건"].get("이중 바닥 패턴", False)
                        else "• 일반적 추세 추종 전략 권장"
                    ),
                    f"• 목표수익률: {etf['1개월수익률']*1.5:.1f}%",
                ]
                st.markdown("\n".join(strategy))


@st.cache_data(ttl=3600)  # 1시간 캐시
def calculate_technical_indicators(df):
    """ETF의 기술적 지표를 계산합니다."""
    if len(df) < 60:  # 최소 60일치 데이터 필요
        return None

    try:
        # 이동평균선
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        df["MA150"] = df["Close"].rolling(window=150).mean()
        df["MA200"] = df["Close"].rolling(window=200).mean()

        # RSI 계산
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # 볼린저 밴드
        df["MA20_std"] = df["Close"].rolling(window=20).std()
        df["Upper_band"] = df["MA20"] + (df["MA20_std"] * 2)
        df["Lower_band"] = df["MA20"] - (df["MA20_std"] * 2)

        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        return df
    except Exception as e:
        st.error(f"지표 계산 중 오류 발생: {str(e)}")
        return None


def check_sepa_conditions(df):
    """SEPA 전략 조건을 확인합니다."""
    if df is None or len(df) < 200:
        return False, {}

    try:
        latest = df.iloc[-1]
        month_ago = df.iloc[-30]

        # SEPA 조건 체크
        criteria = {
            "현재가가 200일선 위": latest["Close"] > latest["MA200"],
            "150일선이 200일선 위": latest["MA150"] > latest["MA200"],
            "50일선이 150/200일선 위": (latest["MA50"] > latest["MA150"])
            and (latest["MA50"] > latest["MA200"]),
            "현재가가 5일선 위": latest["Close"] > latest["MA5"],
            "200일선 상승 추세": latest["MA200"] > month_ago["MA200"],
        }

        # 52주 최저가 대비 상승률 계산
        year_low = df["Low"].tail(252).min()
        price_above_low = (latest["Close"] / year_low - 1) > 0.3
        criteria["52주 최저가 대비 30% 이상"] = price_above_low

        all_conditions_met = all(criteria.values())

        return all_conditions_met, criteria

    except Exception as e:
        st.error(f"SEPA 조건 체크 중 오류 발생: {str(e)}")
        return False, {}


def analyze_etf(ticker):
    """ETF 분석 함수 업데이트"""
    try:
        etf = yf.Ticker(ticker)
        df = etf.history(period="1y")

        if df.empty:
            return None

        df = calculate_technical_indicators(df)
        if df is None:
            return None

        # SEPA 점수 및 조건 확인
        sepa_score, sepa_conditions = check_sepa_conditions_etf(df)

        info = etf.info
        latest = df.iloc[-1]

        # 수익률 계산
        returns = calculate_returns(df)

        # 추가 기술적 지표
        technical_indicators = calculate_additional_indicators(df)

        result = {
            "티커": ticker.replace(".KS", ""),
            "ETF명": info.get("longName", "N/A"),
            "현재가": latest["Close"],
            "SEPA_점수": sepa_score,
            "SEPA_조건": sepa_conditions,
            "기술적_지표": technical_indicators,
            **returns,
            "변동성": df["Close"].std(),
            "거래량": latest["Volume"],
            "차트데이터": df,
        }

        return result

    except Exception as e:
        st.error(f"{ticker} 분석 중 오류 발생: {str(e)}")
        return None


def calculate_returns(df):
    """수익률 계산 함수"""
    latest = df.iloc[-1]
    returns = {}
    periods = {
        "1주일수익률": 5,
        "1개월수익률": 20,
        "3개월수익률": 60,
        "6개월수익률": 120,
        "1년수익률": 240,
    }

    for period_name, days in periods.items():
        if len(df) >= days:
            returns[period_name] = (
                (latest["Close"] / df.iloc[-days]["Close"]) - 1
            ) * 100
        else:
            returns[period_name] = 0

    return returns


def calculate_additional_indicators(df):
    """추가 기술적 지표 계산"""
    latest = df.iloc[-1]

    return {
        "추세강도": latest["RSI"],
        "MACD_Signal": latest["MACD"] - latest["Signal"],
        "볼린저위치": (latest["Close"] - latest["Lower_band"])
        / (latest["Upper_band"] - latest["Lower_band"])
        * 100,
        "거래량증감": (df["Volume"].tail(5).mean() / df["Volume"].tail(20).mean() - 1)
        * 100,
    }


def create_etf_chart(ticker, df):
    """ETF 차트를 Plotly로 생성"""
    fig = go.Figure()

    # 캔들스틱 차트
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="가격"
        )
    )

    # 이동평균선
    ma_periods = {
        "MA5": (5, "purple"),
        "MA20": (20, "blue"),
        "MA50": (50, "green"),
        "MA200": (200, "red")
    }

    for ma_name, (period, color) in ma_periods.items():
        ma = df["Close"].rolling(window=period).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma,
                name=ma_name,
                line=dict(color=color)
            )
        )

    # 거래량 차트
    colors = ['red' if row['Close'] < row['Open'] else 'green' 
              for i, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="거래량",
            marker_color=colors,
            opacity=0.5,
            yaxis="y2"
        )
    )

    # 차트 레이아웃
    fig.update_layout(
        title=f"{ticker} 가격 차트",
        yaxis_title="가격",
        xaxis_title="날짜",
        height=600,
        yaxis2=dict(
            title="거래량",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def check_sepa_conditions_etf(df):
    """향상된 SEPA 전략 조건 확인 함수"""
    if df is None or len(df) < 60:
        return 0, {}

    try:
        latest = df.iloc[-1]
        month_ago = df.iloc[-20] if len(df) >= 20 else df.iloc[0]

        # 기본 SEPA 조건
        base_criteria = {
            "현재가 > MA200": latest["Close"] > latest["MA200"],
            "MA50 > MA200": latest["MA50"] > latest["MA200"],
            "현재가 > MA20": latest["Close"] > latest["MA20"],
            "거래량 증가": df["Volume"].tail(20).mean() > df["Volume"].tail(60).mean(),
        }

        # 모멘텀 조건
        momentum_criteria = {
            "RSI > 50": latest["RSI"] > 50,
            "MACD 상승": latest["MACD"] > latest["Signal"],
            "단기 상승추세": latest["MA5"] > latest["MA20"],
        }

        # 이중 바닥 패턴 분석 - 가중치 상향 조정
        double_bottom = detect_double_bottom(df)
        pattern_score = (
            double_bottom["pattern_strength"] if double_bottom["has_pattern"] else 0
        )

        # 점수 계산 - 이중 바닥 패턴 가중치 40%로 상향
        base_score = sum(base_criteria.values()) * 15  # 기본 조건 (30%)
        momentum_score = sum(momentum_criteria.values()) * 10  # 모멘텀 (30%)
        pattern_score = pattern_score * 100  # 이중 바닥 패턴 (40%)

        total_score = base_score * 0.3 + momentum_score * 0.3 + pattern_score * 0.4

        # 조건 통합
        all_criteria = {
            **base_criteria,
            **momentum_criteria,
            "이중 바닥 패턴": double_bottom["has_pattern"],
            "패턴 강도": f"{pattern_score:.2f}",
        }

        return total_score, all_criteria

    except Exception as e:
        st.error(f"SEPA 조건 체크 중 오류 발생: {str(e)}")
        return 0, {}

def display_sepa_etfs(df_results):
    """SEPA ETF 표시 함수 업데이트"""
    st.subheader("🎯 SEPA 전략 기반 ETF 분석")

    # SEPA 점수 기준 정렬
    df_results = df_results.sort_values("SEPA_점수", ascending=False)
    top_etfs = df_results.head(15)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📊 상위 추천 ETF")
        display_df = top_etfs[
            ["ETF명", "현재가", "SEPA_점수", "1개월수익률", "3개월수익률", "1년수익률"]
        ].copy()

        # Plotly table로 변경
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_df.columns),
                fill_color='royalblue',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                fill_color=['white'],
                align='left',
                font=dict(color='darkslategray', size=11),
                format=[None, ",.0f", ".1f", ".2f", ".2f", ".2f"]
            )
        )])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 💡 ETF 유형 분석")
        etf_types = pd.Series(
            [etf_name.split()[0] for etf_name in top_etfs["ETF명"]]
        ).value_counts()

        fig = px.pie(
            values=etf_types.values,
            names=etf_types.index,
            title="상위 ETF 운용사 분포"
        )
        st.plotly_chart(fig)



def detect_double_bottom(df, threshold=0.05):  # threshold를 5%로 완화
    """
    이중 바닥 패턴을 감지하는 함수 - 기준 완화 및 개선
    """
    try:
        # 최근 200일 데이터로 제한
        df = df.tail(200).copy()

        # 저점 찾기 (window size를 더 작게 조정)
        df["Low_min"] = (
            df["Low"].rolling(window=15, center=True).min()
        )  # 20일에서 15일로 완화
        potential_bottoms = []

        for i in range(15, len(df) - 15):
            if abs(df["Low"].iloc[i] - df["Low_min"].iloc[i]) < 0.001:  # 근사값도 허용
                potential_bottoms.append((i, df["Low"].iloc[i]))

        pattern_metrics = {
            "has_pattern": False,
            "pattern_strength": 0,
            "bottom_depth": 0,
            "recovery_strength": 0,
            "volume_confirmation": 0,
        }

        if len(potential_bottoms) >= 2:
            # 마지막 두 저점 분석
            bottom1_idx, bottom1_price = potential_bottoms[-2]
            bottom2_idx, bottom2_price = potential_bottoms[-1]

            # 저점 간 가격 차이 확인 (기준 완화)
            price_diff_pct = abs(bottom2_price - bottom1_price) / bottom1_price

            # 저점 간 기간 확인 (15일로 완화)
            time_between_bottoms = bottom2_idx - bottom1_idx

            if price_diff_pct <= threshold and time_between_bottoms >= 15:
                pattern_metrics["has_pattern"] = True

                # 저점의 깊이
                prev_high = (
                    df["High"].iloc[max(0, bottom1_idx - 15) : bottom1_idx].max()
                )
                depth = (prev_high - min(bottom1_price, bottom2_price)) / prev_high
                pattern_metrics["bottom_depth"] = depth

                # 회복 강도 - 현재가와 두 번째 저점과의 차이
                current_price = df["Close"].iloc[-1]
                recovery = (current_price - bottom2_price) / bottom2_price
                pattern_metrics["recovery_strength"] = recovery

                # 거래량 확인 - 기준 완화
                avg_volume = df["Volume"].tail(30).mean()  # 최근 30일 평균으로 변경
                bottom2_volume = (
                    df["Volume"].iloc[bottom2_idx : bottom2_idx + 5].mean()
                )  # 5일 평균으로 변경
                volume_increase = bottom2_volume / avg_volume
                pattern_metrics["volume_confirmation"] = volume_increase

                # 종합 패턴 강도 계산 - 가중치 조정
                pattern_metrics["pattern_strength"] = (
                    (1 - price_diff_pct) * 0.25  # 저점 간 유사성
                    + min(1.0, depth) * 0.25  # 저점 깊이
                    + min(1.5, recovery) * 0.3  # 회복 강도 (가중치 증가)
                    + min(1.0, volume_increase) * 0.2  # 거래량 확인
                )

        return pattern_metrics

    except Exception as e:
        print(f"이중 바닥 패턴 감지 중 오류: {str(e)}")
        return {"has_pattern": False, "pattern_strength": 0}


def save_top_etfs_to_json(df_results):
    """상위 10개 ETF 정보를 JSON 파일로 저장"""
    top_10_etfs = df_results.head(10)

    # JSON으로 저장할 데이터 준비
    etf_data = []
    for _, row in top_10_etfs.iterrows():
        etf_info = {
            "ticker": row["티커"] + ".KS",
            "name": row["ETF명"],
            "current_price": float(row["현재가"]),
            "sepa_score": float(row["SEPA_점수"]),
            "monthly_return": float(row["1개월수익률"]),
            "quarterly_return": float(row["3개월수익률"]),
        }
        etf_data.append(etf_info)

    # 현재 날짜로 파일명 생성
    current_date = datetime.now().strftime("%Y%m%d")
    filename = f"top_etfs_{current_date}.json"

    try:
        # JSON 파일 저장
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(etf_data, f, ensure_ascii=False, indent=2)
        return filename
    except Exception as e:
        raise Exception(f"파일 저장 중 오류: {str(e)}")



def main():
    st.title("국내 ETF SEPA 전략 분석 대시보드 📈")
    st.markdown("---")

    # 세션 스테이트 초기화
    if "analyzed_results" not in st.session_state:
        st.session_state["analyzed_results"] = None

    # 분석 시작 버튼
    if st.session_state["analyzed_results"] is None:
        if st.button("ETF 분석 시작"):
            with st.spinner("ETF 분석 중..."):
                start_time = time.time()

                # ETF 리스트 가져오기
                tickers = get_top_kr_etfs()
                if not tickers:
                    st.error("ETF 리스트를 가져오는데 실패했습니다.")
                    return

                # 멀티스레딩으로 병렬 처리
                analyzed_etfs = []
                progress_bar = st.progress(0)

                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_etf = {
                        executor.submit(analyze_etf, ticker): ticker
                        for ticker in tickers
                    }

                    completed = 0
                    for future in future_to_etf:
                        result = future.result()
                        if result is not None:
                            analyzed_etfs.append(result)
                        completed += 1
                        progress_bar.progress(completed / len(tickers))

                if analyzed_etfs:
                    df_results = pd.DataFrame(analyzed_etfs)
                    df_results = df_results.sort_values("SEPA_점수", ascending=False)
                    st.session_state["analyzed_results"] = df_results
                    end_time = time.time()
                    st.success(f"분석 완료! 실행 시간: {end_time - start_time:.2f}초")
                else:
                    st.error("분석 가능한 ETF가 없습니다.")
                    return

    # 분석 결과가 있는 경우 표시
    if st.session_state["analyzed_results"] is not None:
        df_results = st.session_state["analyzed_results"]
        top_10_etfs = df_results.head(10)

        # JSON 다운로드 버튼
        if st.button("상위 10개 ETF 저장"):
            json_str = top_10_etfs.to_json(orient='records', force_ascii=False)
            st.download_button(
                label="JSON 파일 다운로드",
                data=json_str,
                file_name="top_10_etfs.json",
                mime="application/json"
            )

        # 상위 10개 ETF 표시
        st.subheader("🏆 SEPA 전략 상위 10개 ETF")

        # 선택 가능한 ETF 목록 생성
        etf_options = [
            f"{row['티커']} - {row['ETF명']}" for _, row in top_10_etfs.iterrows()
        ]

        selected_etf = st.selectbox("분석할 ETF 선택", etf_options)

        if selected_etf:
            # 선택된 ETF의 티커 추출
            selected_ticker = selected_etf.split(" - ")[0]
            etf_data = df_results[df_results["티커"] == selected_ticker].iloc[0]

            col1, col2 = st.columns([3, 1])

            with col1:
                # 차트 표시
                chart = create_etf_chart(etf_data["ETF명"], etf_data["차트데이터"])
                st.plotly_chart(chart, use_container_width=True)

            with col2:
                # ETF 정보 표시
                st.subheader("📊 ETF 정보")
                metrics = {
                    "현재가": f"₩{etf_data['현재가']:,.0f}",
                    "SEPA 점수": f"{etf_data['SEPA_점수']:.1f}점",
                    "1개월수익률": f"{etf_data['1개월수익률']:.2f}%",
                    "3개월수익률": f"{etf_data['3개월수익률']:.2f}%",
                    "6개월수익률": f"{etf_data['6개월수익률']:.2f}%",
                }

                for key, value in metrics.items():
                    st.metric(key, value)

                # SEPA 조건 테이블을 Plotly로 표시
                st.markdown("#### 💡 SEPA 전략 조건")
                if isinstance(etf_data["SEPA_조건"], dict):
                    condition_data = pd.DataFrame({
                        "조건": list(etf_data["SEPA_조건"].keys()),
                        "충족여부": ["✅" if v else "❌" 
                                   for v in etf_data["SEPA_조건"].values()]
                    })
                    
                    fig = go.Figure(data=[go.Table(
                        header=dict(
                            values=["<b>조건</b>", "<b>충족여부</b>"],  # 볼드체 적용
                            fill_color='royalblue',
                            align='center',  # 중앙 정렬
                            font=dict(color='white', size=14)
                        ),
                        cells=dict(
                            values=[condition_data["조건"], condition_data["충족여부"]],
                            fill_color=['white', 'white'],
                            align=['left', 'center'],  # 조건은 왼쪽, 충족여부는 중앙 정렬
                            font=dict(color=['black', 'black'], size=13),  # 글자색 검정으로 명시
                            height=30  # 셀 높이 조정
                        )
                    )])
                    
                    # 테이블 레이아웃 설정
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=len(condition_data) * 35 + 40  # 데이터 행 수에 따라 높이 자동 조정
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                if isinstance(etf_data["SEPA_조건"], dict):
                    condition_data = pd.DataFrame({
                        "조건": list(etf_data["SEPA_조건"].keys()),
                        "충족여부": ["✅" if v else "❌" 
                                   for v in etf_data["SEPA_조건"].values()]
                    })
                    
                    fig = go.Figure(data=[go.Table(
                        header=dict(
                            values=list(condition_data.columns),
                            fill_color='royalblue',
                            align='left',
                            font=dict(color='white')
                        ),
                        cells=dict(
                            values=[condition_data[col] for col in condition_data.columns],
                            fill_color=['white'],
                            align='left'
                        )
                    )])
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

        # 상위 10개 ETF 테이블
        st.markdown("---")
        st.subheader("📋 SEPA 전략 상위 10개 ETF 목록")

        display_cols = [
            "티커",
            "ETF명",
            "현재가",
            "SEPA_점수",
            "1개월수익률",
            "3개월수익률",
            "6개월수익률",
        ]
        
        # Plotly table로 변경
        display_df = top_10_etfs[display_cols].copy()
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_df.columns),
                fill_color='royalblue',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                fill_color=['white'],
                align='left',
                font=dict(color='darkslategray', size=11),
                format=[None, None, ",.0f", ".1f", ".2f", ".2f", ".2f"]  # 각 컬럼의 포맷 지정
            )
        )])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
