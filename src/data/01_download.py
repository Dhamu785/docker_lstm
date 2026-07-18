# %%
import yfinance as yf

ticker = yf.Ticker("AMZN")
df = ticker.history(period="max")

df.to_csv("AMZN_max.csv")
# %%
