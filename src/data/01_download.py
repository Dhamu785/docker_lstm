# %%
import yfinance as yf

ticker = yf.Ticker("HDFCBANK.NS")
df = ticker.history(period="max")

df.to_csv("HDFC_max.csv")
# %%
