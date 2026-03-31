# 🚀 TradeMind: AI Trading Assistant

🔗 Live Demo: https://trademind-rtngbc9crw2yfupn6zqfps.streamlit.app/

## 📌 Overview
TradeMind is an AI-powered trading assistant built during a hackathon.  
It analyzes stock data and generates buy/sell signals with confidence scores, helping users make more informed trading decisions.

## 💡 Motivation
Many retail investors struggle to interpret market signals and often rely on guesswork.  
This project aims to bridge that gap by combining data analysis with AI-generated insights.

## ⚙️ Features
- 📈 Real-time stock data analysis (via yfinance)
- 🤖 AI-powered signal generation (Buy/Sell + Confidence)
- 🧠 Natural language explanations of trading signals
- 🖥️ Interactive web app built with Streamlit

## 🧠 How It Works
1. Fetch stock market data
2. Analyze price trends and patterns
3. Use AI to generate trading insights
4. Output actionable signals with explanations

## 🛠 Tech Stack
- Python
- Streamlit
- yfinance
- Antropic API
- Tavily API
- Pandas / NumPy

## 🚀 How to Run Locally
Install: pip install streamlit yfinance pandas anthropic requests
Export: ANTHROPIC_API_KEY and TAVILY_API_KEY
Run: streamlit run app.py
