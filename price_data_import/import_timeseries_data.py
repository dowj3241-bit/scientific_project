import yfinance as yf
import pandas as pd
from datetime import datetime


tickers = { "GC=F" : ("2000-08-30","2026-08-01","metal"),        #gold-futures
            "HG=F" : ("2000-08-30","2026-08-01","metal"),        #copper-futures
            "NG=F" : ("2000-08-30","2026-08-01","energy"),       #natural gas futures
            "CL=F" : ("2000-08-23","2026-08-01","energy"),       #crude oil futures
            "ZM=F" : ("2000-05-15","2026-08-01","agricultural"), #soy meal futures
            "ZW=F" : ("2000-07-17","2026-08-01","agricultural"), #wheat futures             
            "^GSPC" : ("2000-01-01","2026-08-01","equity"),      # s&p-500
           }



storage = []


for key, content in tickers.items():
    placeholder = yf.Ticker(key).history(start = content[0], end = content[1], interval = "1d", auto_adjust = False)
    placeholder["date"] = placeholder.index
    placeholder = placeholder.sort_values("date").reset_index(drop = True)  #to order them in ascending order for the comming transformation 
    placeholder.columns = [column_name.lower() for column_name in placeholder.columns] 
    placeholder = placeholder[["date","close","low","high"]]
    placeholder["tkr"] = key
    placeholder["category"] = content[2]
    print(f"The ticker:{key} was successfully loaded and added {len(placeholder)} rows to the storagelist.")
    storage.append(placeholder)


df = pd.concat(storage, ignore_index = True)

#-----------------------------------------------------------------------------------------------------------------------------------------------------transformation pipeline

timestamp = datetime.now()
report_name = f"data_report_{timestamp:%Y%m%d_%H%M%S}.txt" #think about how windows allows some signs and some not.....pain in the ass man

report_data = []
warnings = []

def log(text):
    report_data.append(str(text))

def warnings_assembler(text):
    warnings.append(str(text))


# datumspalte transformieren
try:
    df["date"] = pd.to_datetime(df["date"]).dt.date 
except Exception:
    warnings_assembler(f"- Transforming the data column to format YY-mm-dd fail")

# auf reihe - duplikate prüfen

try:
    n_duplicates = df.duplicated(subset = ["date","tkr"]).sum()     #important for report 
    df = df.drop_duplicates(subset =["date", "tkr"]).reset_index(drop = True)
except Exception as i:
    warnings_assembler(f"checking for duplicates was not performed -> {i}")

# auf nan prüfen und auffüllen
interpolated = 0
try:
    n_nan_before = df[["close","low","high"]].isna().sum().sum()  # important for report

    if n_nan_before > 0:
        for x in ["close","low","high"]:
            df[x] = df.groupby("tkr")[x].transform(lambda value: value.interpolate(method="linear"))

    n_nan_after = df[["close","low","high"]].isna().sum().sum()    
    interpolated = n_nan_before - n_nan_after                                                     #imporant for report
except Exception as i:
    warnings_assembler(f"couldnt check for NaN or interpolate -> {i}")

# auf nullzeilen und ngeativzeilen in den preisen prüfen und ersetzen
zeros_replaced = 0
try:
    n_zero = (df[["close","low","high"]] == 0).sum().sum()                                                  #important for report
    n_negative = (df[["close","low","high"]] < 0).sum().sum()                                               #important for report        
    if n_zero > 0:
    
        for x in ["close","low","high"]:
            df[x] = df[x].replace(0, pd.NA) 

        for x in ["close","low","high"]:
            df[x] = df.groupby("tkr")[x].transform(lambda value: value.interpolate(method="linear"))

    n_zero_after = (df[["close","low","high"]] == 0).sum().sum()
    zeros_replaced = n_zero - n_zero_after                                                                          #important for report
        
    if n_negative > 0: 
         warnings_assembler(f"{n_negative} negative prices found(e.g. WTI 2020-04-20, not replaced by choice)")                                                                                       # important for report

except Exception as i:
    warnings_assembler(f"couldnt handle 0 or negative replacements -> {i}")

#df.to_csv(rf"C:\Users\Nutzer\Desktop\scientific project\data\commodities_price_series_data_{timestamp:%Y%m%d_%H%M%S}.csv", index = False, sep=",")


#------------------------------------------------------------------------------------------------------------------------------- report body descriptive

log("Descriptive Statistics of the Dataframe")
log("-" * 70)
log(f"shape(row/columns): {df.shape[0]} rows x {df.shape[1]} columns")
log(f"timespan       : {df['date'].min()} - {df['date'].max()}")
log(f"Assets (ticker): {df['tkr'].nunique()}")
log(" ")

log("Number of rows per asset:")
for x in df['tkr'].unique():
    log(f"{x}: {(df['tkr'] == x).sum()}")
log(" ")

log("Number of rows per category:")
for x in df['category'].unique():
    log(f"{x}: {(df['category'] == x).sum()}")
log(" ")

log("postion dimensions per asset:")
for key in df["tkr"].unique():
    filterd = df[df["tkr"] == key]                              #should work , filterd once and operations afterwards
    log(f"{key}:"
        f"Max-Closing-Price: {filterd['close'].max():.2f} , " 
        f"Min-Closing-Price: {filterd['close'].min():.2f} , "
        f"Mean-Closing-Price: {filterd['close'].mean():.2f} , "
        f"date coverd: {filterd['date'].min()} - {filterd['date'].max()}")  # maybe a oneline with grouby could work here better? maybe future
log(" ")
log("datatype per column")
for x, y in df.dtypes.items():
    log(f"{x}:  {y}")
#------------------------------------------------------------------------------------------------------------------------------------- # report body transformation process
log(" ")
log("Transformation Process Report")
log("-" * 70)
log(f"duplicates found and dropped: {n_duplicates}")
log(f"NaN found: {n_nan_before}")
log(f"NaN filled through Interpolation: {interpolated}")
log(f"0-Values found: {n_zero}")
log(f"0-Values replaced with Interpolation: {zeros_replaced}")
log(f"Negative-Values found: {n_negative}")
log(" ")
#----------------------------------------------------------------------------------------------------------------------------------------- #report head

with open(report_name, "w", encoding = "utf-8") as report:
    report.write("=" * 70 + "\n\n")
    report.write("Data Report \n")
    report.write(f"timestamp: {timestamp:%Y%m%d_%H%M%S}\n")
    report.write("=" * 70 + "\n\n")

    report.write("Warnings Block \n")
    report.write("-" * 70 + "\n")
    if warnings:
        for w in warnings:
            report.write(f"   ! {w}\n")
    else:
        report.write(" no warnings during the process\n")
    report.write("\n")

    report.write("\n".join(report_data) + "\n") #plus here to ensure report ends with a empty row

print(f"Report was written: {report_name} with {(len(warnings))} Warnings")





