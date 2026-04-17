from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🔹 Fetch Data
def get_stock_data(symbol: str):
    df = yf.download(symbol, period="1y")
    df.reset_index(inplace=True)
    return df


# 🔹 Process Data
def process_data(df):
    # 🔥 Fix multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'])

    # Handle missing values
    df = df.ffill()

    # Metrics
    df['Daily Return'] = (df['Close'] - df['Open']) / df['Open']
    df['MA7'] = df['Close'].rolling(7).mean()
    df['52W High'] = df['Close'].rolling(252).max()
    df['52W Low'] = df['Close'].rolling(252).min()
    df['Volatility'] = df['Daily Return'].rolling(7).std()
    # Mock Sentiment based on daily return
    df['Sentiment'] = df['Daily Return'].apply(
        lambda x: "Positive 😊" if x > 0 else "Negative 😐"
    )

    return df


# 🔹 API 1: Get last 30 days data
@app.get("/data/{symbol}")
def get_data(symbol: str):
    df = get_stock_data(symbol)

    if df.empty:
        return {"error": "No data found"}

    df = process_data(df)
    df = df.tail(30)

# 🔥 Fix NaN / Infinity issue
    df = df.replace([float('inf'), float('-inf')], None)
    df = df.fillna(0)

    return df.to_dict(orient="records")
    
@app.get("/summary/{symbol}")
def get_summary(symbol: str):
    df = get_stock_data(symbol)

    if df is None or df.empty:
        return {"error": "No data found"}

    df = process_data(df)

    # Clean values (important)
    df = df.replace([float('inf'), float('-inf')], None)
    df = df.fillna(0)

    summary = {
        "52_week_high": float(df['Close'].max()),
        "52_week_low": float(df['Close'].min()),
        "average_close": float(df['Close'].mean())
    }

    return summary

@app.get("/companies")
def get_companies():
    companies = [
        {"name": "Infosys", "symbol": "INFY.NS"},
        {"name": "TCS", "symbol": "TCS.NS"},
        {"name": "Reliance", "symbol": "RELIANCE.NS"},
        {"name": "HDFC Bank", "symbol": "HDFCBANK.NS"}
    ]
    return companies

@app.get("/compare")
def compare_stocks(symbol1: str, symbol2: str):
    df1 = process_data(get_stock_data(symbol1))
    df2 = process_data(get_stock_data(symbol2))

    df1 = df1.replace([float('inf'), float('-inf')], None).fillna(0)
    df2 = df2.replace([float('inf'), float('-inf')], None).fillna(0)

    return {
        symbol1: {
            "last_30_days_change": float(df1['Close'].iloc[-1] - df1['Close'].iloc[-30])
        },
        symbol2: {
            "last_30_days_change": float(df2['Close'].iloc[-1] - df2['Close'].iloc[-30])
        }
    }

@app.get("/predict/{symbol}")
def predict(symbol: str):
    df = process_data(get_stock_data(symbol))

    last = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]

    prediction = last + (last - prev)

    return {"predicted_price": float(prediction)}