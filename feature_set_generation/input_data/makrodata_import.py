import yfinance as yf
import pandas as pd
from datetime import datetime


# importing and transforming VIX

df = yf.Ticker("^VIX").history(start="1999-06-01", end = "2026-08-01" ,interval = "1d", auto_adjust = False)
df["date"] = df.index
df = df.sort_values("date").reset_index(drop = True)
df["date"] = pd.to_datetime(df["date"]).dt.date 
df.columns = [x.lower() for x in df.columns]
df = df[["date","close"]]
df["metric"] = "^VIX"


gpr = pd.read_csv(r"C:\Users\Nutzer\Desktop\scientific project\makrodata\gpr_index.csv",sep = ";")
gpr["date"] = pd.to_datetime(gpr["date"], dayfirst= True).dt.date 
gpr = gpr.sort_values("date").reset_index(drop = True)
gpr["close"] = gpr["GPRD"]
gpr["close"] = gpr["close"].astype(str).str.replace(".","")
gpr["close"] = gpr["close"].astype(str).str.replace(",",".").astype(float)
gpr["metric"] = "GPR"
gpr = gpr[["date","close","metric"]]

df = pd.concat([df,gpr], ignore_index= True)

baa10y = pd.read_csv(r"C:\Users\Nutzer\Desktop\scientific project\makrodata\BAA10Y.csv", sep = ",")
baa10y["date"] = pd.to_datetime(baa10y["observation_date"]).dt.date 
baa10y = baa10y.sort_values("date").reset_index(drop = True)
baa10y["close"] = baa10y["BAA10Y"]
baa10y["metric"] = "BAA10Y"
baa10y = baa10y[["date","close","metric"]]

df = pd.concat([df,baa10y], ignore_index= True)


t10y2y = pd.read_csv(r"C:\Users\Nutzer\Desktop\scientific project\makrodata\T10Y2Y.csv", sep = ",")
t10y2y["date"] = pd.to_datetime(t10y2y["observation_date"]).dt.date 
t10y2y = t10y2y.sort_values("date").reset_index(drop = True)
t10y2y["close"] = t10y2y["T10Y2Y"]
t10y2y["metric"] = "T10Y2Y"
t10y2y = t10y2y[["date","close","metric"]]
print(t10y2y)

df = pd.concat([df,t10y2y], ignore_index= True)

print(df.isna().sum())
print((df["close"] == 0).sum())


#print(df.loc[df["close"] == 0,["date","metric"]])    # as long only T10Y2Y shows that its fine, slope = 0 it so interesting in this case

df["close"] = df.groupby("metric")["close"].transform(lambda x :x.interpolate(method = "linear"))


#df.to_csv(r"macro_data.csv", sep = ",")












