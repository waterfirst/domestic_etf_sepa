import json
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time

# ETF 목록 정의
TOP_ETFS = [
    {"ticker": "418660.KS", "name": "Mirae Asset Tiger Synth-Nasdaq 100 Leverage Etf"},
    {"ticker": "471040.KS", "name": "KoAct Global AI and Robotics Active ETF"},
    {"ticker": "409820.KS", "name": "Samsung Kodex US Nasdaq100 Leverage ETF"},
    {"ticker": "144600.KS", "name": "Kodex Silver Futures Special Asset ETF"},
    {"ticker": "483340.KS", "name": "Kim Ace Google Value Chain Active ETF"},
    {"ticker": "133690.KS", "name": "Mirae Asset TIGER USA NASDAQ 100 ETF"},
    {"ticker": "091160.KS", "name": "Kodex Semicon"},
    {"ticker": "394350.KS", "name": "Kiwoom Global Future Mobility ETF"},
    {"ticker": "472160.KS", "name": "Tiger Us Tech top10 Indxx ETF"},
    {"ticker": "487240.KS", "name": "Kodex AI Electric Power Core Facilities Etf"},
]


def create_etf_chart_with_signals(df, trades_df=None):
    """ETF 차트에 매수/매도 시그널을 강조하여 표시"""
    fig = go.Figure()

    # 캔들스틱 차트
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="가격",
        )
    )

    # 이동평균선 추가
    ma_colors = {"MA5": "purple", "MA20": "blue", "MA50": "green", "MA200": "red"}
    for ma, color in ma_colors.items():
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ma],
                    name=ma,
                    line=dict(color=color, width=1),
                    opacity=0.7,
                )
            )

    # 매수/매도 시그널 표시
    if trades_df is not None and not trades_df.empty:
        # 매수 시그널
        fig.add_trace(
            go.Scatter(
                x=trades_df["entry_date"],
                y=trades_df["entry_price"],
                mode="markers+text",
                name="매수",
                marker=dict(
                    symbol="triangle-up",
                    size=15,
                    color="green",
                    line=dict(width=2, color="darkgreen"),
                ),
                text="매수",
                textposition="top center",
                textfont=dict(size=12, color="green"),
                hovertemplate="매수가: %{y:,.0f}원<br>날짜: %{x}<extra></extra>",
            )
        )

        # 매도 시그널에 대한 텍스트 데이터 준비
        exit_texts = [f"매도 ({returns:.1f}%)" for returns in trades_df["returns"]]

        # 매도 시그널
        fig.add_trace(
            go.Scatter(
                x=trades_df["exit_date"],
                y=trades_df["exit_price"],
                mode="markers+text",
                name="매도",
                marker=dict(
                    symbol="triangle-down",
                    size=15,
                    color="red",
                    line=dict(width=2, color="darkred"),
                ),
                text="매도",
                textposition="bottom center",
                textfont=dict(size=12, color="red"),
                hovertemplate="매도가: %{y:,.0f}원<br>날짜: %{x}<br>수익률: {%returnrate}%<extra></extra>".format(
                    returnrate=trades_df["returns"].round(2).values
                ),
            )
        )

    # 차트 레이아웃 설정
    fig.update_layout(
        title="ETF 가격 차트 및 매매 시그널",
        yaxis_title="가격",
        xaxis_title="날짜",
        height=800,  # 차트 크기 증가
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
    )

    return fig


class OptimizedETFBacktest:
    """최적화된 ETF 백테스팅 클래스"""

    def __init__(self, df, initial_capital=10000000):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.trades = []
        self.portfolio_value = initial_capital

        # 매매 전략 파라미터
        self.params = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_threshold": 1.5,  # 평균 거래량 대비
            "trailing_stop_pct": 0.05,  # 5%
            "profit_target_pct": 0.15,  # 15%
            "max_loss_pct": 0.07,  # 7%
        }

    def optimize_parameters(self):
        """그리드 서치를 통한 매매 파라미터 최적화"""
        best_params = self.params.copy()
        best_return = 0

        # 파라미터 그리드 정의
        param_grid = {
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "volume_threshold": [1.3, 1.5, 1.7],
            "trailing_stop_pct": [0.03, 0.05, 0.07],
            "profit_target_pct": [0.1, 0.15, 0.2],
            "max_loss_pct": [0.05, 0.07, 0.09],
        }

        # 그리드 서치 수행
        for params in self._generate_param_combinations(param_grid):
            self.params = params
            returns = self.run_backtest()
            if returns > best_return:
                best_return = returns
                best_params = params.copy()

        self.params = best_params
        return best_params

    def _generate_param_combinations(self, param_grid):
        """파라미터 조합 생성"""
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))

    def check_entry_conditions(self, current_idx):
        """매수 조건 확인"""
        if current_idx < 20:  # 충분한 데이터 필요
            return False

        current = self.df.iloc[current_idx]
        prev = self.df.iloc[current_idx - 1]

        # RSI 조건
        rsi_condition = current["RSI"] < self.params["rsi_oversold"]

        # 거래량 조건
        volume_ma = self.df["Volume"].rolling(20).mean().iloc[current_idx]
        volume_condition = (
            current["Volume"] > volume_ma * self.params["volume_threshold"]
        )

        # 이동평균선 조건
        ma_condition = (
            current["Close"] > current["MA20"]
            and current["MA20"] > current["MA50"]
            and current["MA50"] > current["MA200"]
        )

        # MACD 조건
        macd_condition = (
            current["MACD"] > current["Signal"] and prev["MACD"] <= prev["Signal"]
        )

        return rsi_condition and volume_condition and ma_condition and macd_condition

    def check_exit_conditions(self, entry_price, current_price, highest_price):
        """매도 조건 확인"""
        profit_pct = (current_price - entry_price) / entry_price
        drawdown_pct = (highest_price - current_price) / highest_price

        # 익절 조건
        if profit_pct >= self.params["profit_target_pct"]:
            return True, "익절"

        # 손절 조건
        if profit_pct <= -self.params["max_loss_pct"]:
            return True, "손절"

        # 트레일링 스탑
        if drawdown_pct >= self.params["trailing_stop_pct"]:
            return True, "트레일링 스탑"

        return False, None

    def run_backtest(self):
        """백테스트 실행"""
        trades = []
        in_position = False
        entry_price = 0
        highest_price = 0

        for i in range(len(self.df)):
            current_price = self.df["Close"].iloc[i]

            if not in_position:
                if self.check_entry_conditions(i):
                    entry_price = current_price
                    highest_price = current_price
                    entry_date = self.df.index[i]
                    in_position = True
            else:
                highest_price = max(highest_price, current_price)
                should_exit, exit_reason = self.check_exit_conditions(
                    entry_price, current_price, highest_price
                )

                if should_exit:
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "entry_price": entry_price,
                            "exit_date": self.df.index[i],
                            "exit_price": current_price,
                            "returns": (current_price - entry_price)
                            / entry_price
                            * 100,
                            "exit_reason": exit_reason,
                        }
                    )
                    in_position = False

        self.trades = pd.DataFrame(trades)
        return self.calculate_performance()

    def calculate_performance(self):
        """성과 지표 계산"""
        if len(self.trades) == 0:
            return 0

        total_return = self.trades["returns"].sum()
        win_rate = len(self.trades[self.trades["returns"] > 0]) / len(self.trades) * 100
        max_drawdown = self.calculate_max_drawdown()
        sharpe_ratio = self.calculate_sharpe_ratio()

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }

    def calculate_max_drawdown(self):
        """최대 낙폭 계산"""
        cumulative_returns = (1 + self.trades["returns"] / 100).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min() * 100

    def calculate_sharpe_ratio(self):
        """샤프 비율 계산"""
        returns = self.trades["returns"]
        if len(returns) < 2:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)  # 연율화


def analyze_and_optimize_etf(df):
    """ETF 분석 및 전략 최적화"""
    backtest = OptimizedETFBacktest(df)

    # 파라미터 최적화
    optimal_params = backtest.optimize_parameters()

    # 최적화된 파라미터로 백테스트 실행
    performance = backtest.run_backtest()

    return backtest.trades, performance, optimal_params


def display_optimization_results(trades_df, performance, optimal_params):
    """최적화 결과 표시"""
    st.subheader("📊 백테스트 결과")

    # 성과 지표 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 수익률", f"{performance['total_return']:.2f}%")
    with col2:
        st.metric("승률", f"{performance['win_rate']:.2f}%")
    with col3:
        st.metric("최대 낙폭", f"{performance['max_drawdown']:.2f}%")
    with col4:
        st.metric("샤프 비율", f"{performance['sharpe_ratio']:.2f}")

    # 최적화된 파라미터 표시
    st.subheader("⚙️ 최적화된 매매 파라미터")
    param_df = pd.DataFrame([optimal_params]).T
    param_df.columns = ["값"]
    st.dataframe(param_df)

    # 매매 기록 표시
    st.subheader("📝 매매 기록")
    st.dataframe(trades_df)


class EnhancedETFBacktest:
    def __init__(self, df, etf_info, initial_capital=10000000):
        """백테스팅 클래스 초기화"""
        self.df = df.copy()
        self.etf_info = etf_info
        self.initial_capital = initial_capital
        self.trades = []

        # ETF 유형에 따른 매매 파라미터 설정
        self.target_profit = 0.15  # 기본 목표 수익률
        self.stop_loss = -0.07  # 기본 손절률
        self.trailing_stop = 0.05  # 기본 트레일링 스탑

        # 레버리지 ETF인 경우 파라미터 조정
        if "leverage" in etf_info["name"].lower():
            self.target_profit = 0.12
            self.stop_loss = -0.05
            self.trailing_stop = 0.03

    def find_double_bottom(self, window_size=20):
        """이중 바닥 패턴 찾기"""
        signals = []

        for i in range(window_size, len(self.df) - window_size):
            window = self.df.iloc[i - window_size : i + 1]

            # 저점 찾기
            if len(window) >= 2:
                low1 = window["Low"].iloc[:-10].min()
                low2 = window["Low"].iloc[-10:].min()

                # 저점 간 가격 차이 확인 (5% 이내)
                if abs(low2 - low1) / low1 < 0.05:
                    # 거래량 확인
                    avg_volume = window["Volume"].mean()
                    recent_volume = window["Volume"].iloc[-5:].mean()

                    if recent_volume > avg_volume * 1.2:  # 거래량 20% 이상 증가
                        signals.append(
                            {
                                "date": self.df.index[i],
                                "price": self.df["Close"].iloc[i],
                                "pattern": "double_bottom",
                            }
                        )

        return signals

    def check_sepa_conditions(self, current):
        """SEPA 전략 조건 확인"""
        return (
            current["Close"] > current["MA200"]
            and current["MA50"] > current["MA200"]
            and current["Close"] > current["MA20"]
            and current["RSI"] > 50
            and current["MACD"] > current["Signal"]
        )

    def execute_trade(self, entry_point):
        """매매 실행"""
        entry_date = entry_point["date"]
        entry_price = entry_point["price"]
        position = {
            "entry_date": entry_date,
            "entry_price": entry_price,
            "highest_price": entry_price,
        }

        # 진입 시점 이후 데이터
        entry_idx = self.df.index.get_loc(entry_date)
        for i in range(entry_idx + 1, len(self.df)):
            current = self.df.iloc[i]
            current_price = current["Close"]

            # 최고가 갱신
            if current_price > position["highest_price"]:
                position["highest_price"] = current_price

            # 매도 조건 체크
            if current_price >= entry_price * (1 + self.target_profit):
                return self.record_trade(position, current, "target_profit")
            elif current_price <= entry_price * (1 + self.stop_loss):
                return self.record_trade(position, current, "stop_loss")
            elif current_price <= position["highest_price"] * (1 - self.trailing_stop):
                return self.record_trade(position, current, "trailing_stop")
            elif current_price < current["MA20"]:
                return self.record_trade(position, current, "ma_cross")

        # 마지막까지 매도되지 않은 경우
        return self.record_trade(position, self.df.iloc[-1], "end_of_period")

    def record_trade(self, position, exit_data, exit_reason):
        """거래 기록"""
        trade = {
            "entry_date": position["entry_date"],
            "entry_price": position["entry_price"],
            "exit_date": exit_data.name,
            "exit_price": exit_data["Close"],
            "exit_reason": exit_reason,
            "hold_days": (exit_data.name - position["entry_date"]).days,
            "returns": (exit_data["Close"] - position["entry_price"])
            / position["entry_price"]
            * 100,
        }
        self.trades.append(trade)
        return trade

    def run_backtest(self):
        """백테스팅 실행"""
        # 매수 시그널 찾기
        entry_signals = self.find_double_bottom()

        # 각 시그널에 대해 SEPA 조건 확인 및 매매 실행
        for signal in entry_signals:
            current = self.df.loc[signal["date"]]
            if self.check_sepa_conditions(current):
                self.execute_trade(signal)

        # 결과 분석
        trades_df = pd.DataFrame(self.trades)

        if len(trades_df) > 0:
            performance = {
                "total_trades": len(trades_df),
                "winning_trades": len(trades_df[trades_df["returns"] > 0]),
                "avg_return": trades_df["returns"].mean(),
                "max_return": trades_df["returns"].max(),
                "min_return": trades_df["returns"].min(),
                "avg_hold_days": trades_df["hold_days"].mean(),
                "win_rate": len(trades_df[trades_df["returns"] > 0])
                / len(trades_df)
                * 100,
                "profit_factor": abs(
                    trades_df[trades_df["returns"] > 0]["returns"].sum()
                    / trades_df[trades_df["returns"] < 0]["returns"].sum()
                    if len(trades_df[trades_df["returns"] < 0]) > 0
                    else float("inf")
                ),
            }
        else:
            performance = {
                "total_trades": 0,
                "winning_trades": 0,
                "avg_return": 0,
                "max_return": 0,
                "min_return": 0,
                "avg_hold_days": 0,
                "win_rate": 0,
                "profit_factor": 0,
            }

        return trades_df, performance


def display_trade_analysis(trades_df, performance):
    """거래 분석 결과 표시"""
    # 성과 지표 표시
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "총 거래 수",
            f"{performance['total_trades']}회",
            delta=f"승률 {performance['win_rate']:.1f}%",
        )

    with col2:
        st.metric(
            "평균 수익률",
            f"{performance['avg_return']:.2f}%",
            delta=f"최대 {performance['max_return']:.2f}%",
        )

    with col3:
        st.metric(
            "최대 손실",
            f"{performance['min_return']:.2f}%",
            delta=f"평균 보유 {performance['avg_hold_days']:.1f}일",
        )

    with col4:
        st.metric(
            "수익 요인",
            f"{performance['profit_factor']:.2f}",
            delta="양호" if performance["profit_factor"] > 2 else "주의",
        )

    if not trades_df.empty:
        st.subheader("📊 수익률 분포")

        # 수익률 히스토그램
        fig_hist = px.histogram(
            trades_df,
            x="returns",
            nbins=20,
            title="거래별 수익률 분포",
            labels={"returns": "수익률 (%)", "count": "거래 횟수"},
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        # 매도 사유 분석
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 매도 사유 분석")
            exit_reasons = trades_df["exit_reason"].value_counts()
            fig_pie = px.pie(
                values=exit_reasons.values,
                names=exit_reasons.index,
                title="매도 사유 분포",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("📅 보유 기간 분석")
            fig_hold = px.box(
                trades_df, y="hold_days", points="all", title="거래별 보유 기간"
            )
            st.plotly_chart(fig_hold, use_container_width=True)

        # 상세 거래 기록
        st.subheader("📝 상세 거래 기록")
        display_df = trades_df.copy()
        display_df["entry_date"] = display_df["entry_date"].dt.strftime("%Y-%m-%d")
        display_df["exit_date"] = display_df["exit_date"].dt.strftime("%Y-%m-%d")
        display_df["returns"] = display_df["returns"].round(2).astype(str) + "%"
        display_df.columns = [
            "진입일",
            "진입가격",
            "청산일",
            "청산가격",
            "청산사유",
            "보유기간",
            "수익률",
        ]

        st.dataframe(
            display_df.sort_values("진입일", ascending=False), use_container_width=True
        )


def display_backtest_summary(results):
    """백테스트 결과 요약 표시"""
    st.subheader("📊 전체 ETF 백테스팅 결과")

    summary_data = []
    for result in results:
        etf_info = result["etf_info"]
        perf = result["performance"]

        summary_data.append(
            {
                "ETF": f"{etf_info['name']} ({etf_info['ticker']})",
                "총 거래수": perf["total_trades"],
                "승률(%)": f"{perf['win_rate']:.1f}%",
                "평균수익률(%)": f"{perf['avg_return']:.2f}%",
                "최대수익(%)": f"{perf['max_return']:.2f}%",
                "최대손실(%)": f"{perf['min_return']:.2f}%",
                "평균보유기간": f"{perf['avg_hold_days']:.1f}일",
                "수익요인": f"{perf['profit_factor']:.2f}",
            }
        )

    df = pd.DataFrame(summary_data)

    # 결과를 평균 수익률 기준으로 정렬
    df["평균수익률_정렬용"] = df["평균수익률(%)"].str.rstrip("%").astype(float)
    df = df.sort_values("평균수익률_정렬용", ascending=False)
    df = df.drop("평균수익률_정렬용", axis=1)

    st.dataframe(
        df.style.highlight_max(
            subset=["평균수익률(%)", "승률(%)", "수익요인"], axis=0
        ).highlight_min(subset=["최대손실(%)"], axis=0),
        use_container_width=True,
    )


def calculate_technical_indicators(df):
    """
    기술적 지표 계산 함수

    Parameters:
        df (pd.DataFrame): OHLCV 데이터를 포함한 DataFrame

    Returns:
        pd.DataFrame: 기술적 지표가 추가된 DataFrame
    """
    try:
        # 이동평균선 계산
        for window in [5, 20, 50, 200]:
            df[f"MA{window}"] = df["Close"].rolling(window=window).mean()

        # RSI 계산 (14일)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD 계산
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["Signal"]

        # 볼린저 밴드 (20일)
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

        # 거래량 이동평균
        df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()

        # 일간 변동성 (ATR 기반)
        high_low = df["High"] - df["Low"]
        high_close = abs(df["High"] - df["Close"].shift())
        low_close = abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["ATR"] = true_range.rolling(window=14).mean()

        # 추세 강도 (ADX)
        plus_dm = df["High"].diff()
        minus_dm = df["Low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr14 = true_range.rolling(window=14).sum()
        plus_di14 = 100 * (plus_dm.rolling(window=14).sum() / tr14)
        minus_di14 = 100 * (minus_dm.rolling(window=14).sum() / tr14)
        df["ADX"] = (
            (abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100)
            .rolling(window=14)
            .mean()
        )

        return df

    except Exception as e:
        st.error(f"기술적 지표 계산 중 오류 발생: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def fetch_etf_data(ticker):
    """단일 ETF 데이터 로드"""
    try:
        etf = yf.Ticker(ticker)
        df = etf.history(period="1y")
        if not df.empty:
            return calculate_technical_indicators(df)
        return None
    except Exception as e:
        st.error(f"ETF 데이터 로드 중 오류 ({ticker}): {str(e)}")
        return None


def load_all_etf_data():
    """모든 ETF 데이터 로드"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    etf_data = []
    total_etfs = len(TOP_ETFS)

    for idx, etf in enumerate(TOP_ETFS):
        status_text.text(f"Loading {etf['ticker']} ({idx+1}/{total_etfs})")
        df = fetch_etf_data(etf["ticker"])

        if df is not None:
            etf_data.append({"ticker": etf["ticker"], "name": etf["name"], "data": df})

        progress_bar.progress((idx + 1) / total_etfs)
        time.sleep(0.1)

    status_text.text("데이터 로딩 완료!")
    time.sleep(1)
    status_text.empty()

    return etf_data


def create_analysis_charts(df, trades_df=None):
    """차트 생성"""
    fig = go.Figure()

    # 캔들스틱
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="가격",
        )
    )

    # 이동평균선
    colors = {"MA20": "blue", "MA50": "green", "MA200": "red"}
    for ma, color in colors.items():
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=color)))

    # 거래 지점 표시
    if trades_df is not None and not trades_df.empty:
        fig.add_trace(
            go.Scatter(
                x=trades_df["entry_date"],
                y=trades_df["entry_price"],
                mode="markers",
                name="매수",
                marker=dict(color="green", size=10, symbol="triangle-up"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trades_df["exit_date"],
                y=trades_df["exit_price"],
                mode="markers",
                name="매도",
                marker=dict(color="red", size=10, symbol="triangle-down"),
            )
        )

    fig.update_layout(
        title="가격 차트 및 거래 지점",
        yaxis_title="가격",
        height=600,
        template="plotly_dark",
    )

    return fig


def run_backtest(selected_data):
    """백테스팅 실행"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 백테스팅 단계 정의
    steps = ["초기화", "매수 시그널 분석", "매도 시그널 분석", "성과 계산"]

    for idx, step in enumerate(steps):
        status_text.text(f"백테스팅 진행 중: {step}")
        progress_bar.progress((idx + 1) / len(steps))
        time.sleep(0.5)

    backtest = EnhancedETFBacktest(
        selected_data["data"],
        {"ticker": selected_data["ticker"], "name": selected_data["name"]},
    )
    trades_df, performance = backtest.run_backtest()

    status_text.text("백테스팅 완료!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

    return trades_df, performance


def load_etf_json():
    """JSON 파일에서 ETF 정보 로드"""
    uploaded_file = st.file_uploader("ETF 목록 JSON 파일 선택", type=["json"])
    if uploaded_file is not None:
        try:
            json_data = json.load(uploaded_file)
            # JSON 데이터 검증
            required_fields = ["ticker", "name"]
            if all(
                all(field in item for field in required_fields) for item in json_data
            ):
                return json_data
            else:
                st.error("잘못된 JSON 형식입니다. ticker와 name 필드가 필요합니다.")
                return None
        except Exception as e:
            st.error(f"JSON 파일 로드 중 오류 발생: {str(e)}")
            return None
    return None


def main():
    st.set_page_config(
        page_title="ETF 백테스팅 대시보드", page_icon="📈", layout="wide"
    )
    st.title("📊 ETF 백테스팅 분석 대시보드")
    st.markdown("---")

    # 데이터 소스 선택
    data_source = st.radio(
        "데이터 소스 선택", ["JSON 파일 업로드", "기본 ETF 목록 사용"], horizontal=True
    )

    # 세션 스테이트 초기화
    if "etf_data" not in st.session_state:
        st.session_state.etf_data = None

    if data_source == "JSON 파일 업로드":
        # JSON 파일 로드
        etf_list = load_etf_json()

        if etf_list:
            st.success(f"{len(etf_list)}개의 ETF 정보를 불러왔습니다.")

            # ETF 데이터 로드 버튼
            if st.button("ETF 데이터 로드"):
                with st.spinner("ETF 데이터 로딩 중..."):
                    st.session_state.etf_data = []
                    progress_bar = st.progress(0)

                    for idx, etf in enumerate(etf_list):
                        try:
                            df = fetch_etf_data(etf["ticker"])
                            if df is not None:
                                st.session_state.etf_data.append(
                                    {
                                        "ticker": etf["ticker"],
                                        "name": etf["name"],
                                        "data": df,
                                    }
                                )
                        except Exception as e:
                            st.warning(f"{etf['ticker']} 데이터 로드 실패: {str(e)}")

                        progress_bar.progress((idx + 1) / len(etf_list))

                    st.success(
                        f"{len(st.session_state.etf_data)}개 ETF 데이터 로드 완료!"
                    )
        else:
            st.info("JSON 파일을 업로드해주세요.")
            return

    else:  # 기본 ETF 목록 사용
        if st.session_state.etf_data is None:
            with st.spinner("ETF 데이터 로딩 중..."):
                st.session_state.etf_data = load_all_etf_data()

    # 데이터 로드 확인
    if not st.session_state.etf_data:
        st.error("ETF 데이터를 로드할 수 없습니다.")
        return

    # ETF 선택 및 백테스팅 섹션
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_etf = st.selectbox(
            "분석할 ETF 선택",
            options=[etf["name"] for etf in st.session_state.etf_data],
            format_func=lambda x: f"{x} ({[e['ticker'] for e in st.session_state.etf_data if e['name'] == x][0]})",
        )

    with col2:
        start_backtest = st.button("백테스팅 시작")
        run_all = st.button("전체 ETF 분석")

    # 선택된 ETF 데이터 가져오기
    selected_data = next(
        etf for etf in st.session_state.etf_data if etf["name"] == selected_etf
    )

    # 백테스팅 실행 (단일 ETF)
    if start_backtest:
        trades_df, performance = run_backtest(selected_data)

        # 차트 표시
        st.subheader("📈 거래 분석 차트")
        chart = create_analysis_charts(selected_data["data"], trades_df)
        st.plotly_chart(chart, use_container_width=True)

        # 백테스팅 결과 표시
        display_trade_analysis(trades_df, performance)

    # 전체 ETF 분석
    if run_all:
        with st.spinner("전체 ETF 분석 중..."):
            progress_bar = st.progress(0)
            results = []

            for idx, etf_data in enumerate(st.session_state.etf_data):
                backtest = EnhancedETFBacktest(
                    etf_data["data"],
                    {"ticker": etf_data["ticker"], "name": etf_data["name"]},
                )
                trades_df, performance = backtest.run_backtest()
                results.append(
                    {
                        "etf_info": {
                            "ticker": etf_data["ticker"],
                            "name": etf_data["name"],
                        },
                        "trades": trades_df,
                        "performance": performance,
                    }
                )
                progress_bar.progress((idx + 1) / len(st.session_state.etf_data))

            display_backtest_summary(results)
            progress_bar.empty()


if __name__ == "__main__":
    main()
