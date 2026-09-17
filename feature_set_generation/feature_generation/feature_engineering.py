import pandas as pd
import numpy as np

#------------------------------------------loading

BASE = rf"C:\Users\Nutzer\Desktop\scientific project\feature_set_generation\input_data"

macro_data = pd.read_csv(BASE + r"\macro_data.csv", sep = ",") #provides the macro data at onset

main_data = pd.read_csv(BASE + r"\df_features_input.csv", sep = ",") # provides the pricedata,panic probabilities and returns 

main_data = main_data.sort_values(by = ["tkr","date"]).reset_index(drop = True)   


target_values = pd.read_csv(BASE + r"\target_values.csv", sep = ",") #provides the episodes and target variables

target_values =  target_values[["tkr","onset","duration"]] #add Cluster_ID if data is clusterd,also activate down in the dictionary otherwise not saved

#------------------------------------------- transformation of target to log_duration to fight impact of right-skewed target distribution a bit

target_values["class"] = (target_values["duration"] >= 3).astype(int)  #True = 1 , shorter = 0

print(target_values["class"].value_counts())

#------------------------------------------- market based features

def volatility_before_t(date: str, tkr: str, t : int):
    placeholder = main_data[main_data["tkr"] == tkr].reset_index(drop = True)
    i = placeholder.index[placeholder["date"]== date][0]   #index returns a index-object so i need to get the int from the "list"
    if i - t < 0 :   # stops negative slicing
        return np.nan
    window = placeholder["log_returns"].iloc[i-t : i]   #iloc sind die enden exklusiv, daher window from onset[ day1 ,day2,day..t
    mean_return =  window.mean()
    final_sum = ((window - mean_return)**2).sum() #summiert das ergebnis
    return np.sqrt(final_sum/ (t-1)) #gibt die vola aus 

def volatility_ratio(date: str, tkr:str):
    days_5_before = volatility_before_t(date = date, tkr = tkr, t = 5)
    days_20_before = volatility_before_t(date = date, tkr = tkr, t = 20)

    if np.isnan(days_5_before) or np.isnan(days_20_before) or days_20_before == 0:
        return np.nan

    return days_5_before/days_20_before

def momentum(date: str, tkr: str, t : int):
    placeholder = main_data[main_data["tkr"] == tkr].reset_index(drop = True)
    i = placeholder.index[placeholder["date"]== date][0]   #index returns a index-object so i need to get the int from the "list"
    if i - t < 0 :   # stops negative slicing
        return np.nan
    window = placeholder["log_returns"].iloc[i-t:i]
    momentum = window.sum()
    return momentum

def max_drawdown_t(date: str, tkr: str, t : int):   #biggest negative 
    placeholder = main_data[main_data["tkr"] == tkr].reset_index(drop = True)
    i = placeholder.index[placeholder["date"]== date][0]   #index returns a index-object so i need to get the int from the "list"
    if i - t < 0 :   # stops negative slicing
        return np.nan
    window = placeholder["close"].iloc[i-t:i]
    window_max = window.cummax() #cummax takes a series and returns a series on which at each index point is the current highest max and by deviding both in the next line we come up with the draw down
    #print("test", window_max)
    draw_down = window/window_max - 1.0  #-1 to see the drawdown in percent 
    return draw_down.min()

def onset_return(date:str, tkr:str):
    placeholder = main_data[main_data["tkr"] == tkr].reset_index(drop = True)
    i = placeholder.index[placeholder["date"]== date][0]
    log_return = placeholder["log_returns"].iloc[i]
    return log_return*100

def intra_day_range(date:str, tkr:str):
    placeholder = main_data[main_data["tkr"] == tkr].reset_index(drop = True)
    i = placeholder.index[placeholder["date"]== date][0]
    close = placeholder["close"].iloc[i]
    if close <= 0:
     return np.nan
    high = placeholder["high"].iloc[i]
    low = placeholder["low"].iloc[i]
    intra_range = (high-low)/close      #divide by close makes it compareable
    return intra_range

#------------------------------------------------------------------------------------------------------------------------- regime-based features

def panic_onset(date:str, tkr:str):
    placeholder = main_data[main_data["tkr"] == tkr].reset_index(drop = True)
    i = placeholder.index[placeholder["date"]== date][0]
    panic = placeholder["panic_prob"].iloc[i]
    return panic 

def max_panic_before_onset(date: str, tkr: str, t : int):
    placeholder = main_data[main_data["tkr"] == tkr].reset_index(drop = True)
    i = placeholder.index[placeholder["date"]== date][0]   #index returns a index-object so i need to get the int from the "list"
    if i - t < 0 :   # stops negative slicing
        return np.nan
    window = placeholder["panic_prob"].iloc[i-t:i]
    return window.max()


#---------------------------------------------------------------------------------------------------------------------- cross-asset features


def cross_asset_panic(date: str):        #mean of the panic at onset
    placeholder = main_data[main_data["date"] == date].reset_index(drop = True)
    mean = placeholder["panic_prob"].mean()
    return mean 

def cross_asset_panic_before_t(date: str, t:int):    #mean panic before onset    , dataformat needs to be sorted by tkr and then by date otehrwise throws nan
    i = main_data.index[main_data["date"]== date]  #gives us a index_list exactly once per tkr (dates per tkr are unique, index also in longform)
    liste = []
    #print(i)
    
    for element in i:
        #print(main_data.iloc[element-t:element]["tkr"])
        if element - t < 0 or main_data.iloc[element-t:element]["tkr"].nunique() != 1:  #avoids taking data from dfr tickers, input data needs to be sorted!    
            continue
        else:
            window = main_data["panic_prob"].iloc[element-t:element]
            mean = window.mean()        
            liste.append(mean)

    if len(liste) == 0 :             
        print("empty list")
        return np.nan

    if len(liste) ==1:
        print("only one asset")
    
    mean = np.mean(liste)
    return mean 

def asset_return_correlation(date:str, t:int):
    wide = main_data.pivot(index = "date",columns="tkr", values= "log_returns")   #every column is a tkr with returns
    i = wide.index.get_loc(date)                                                #each date is unique
    #if value == 17:
    #        print(i)
    if i - t < 0:
        print(f"is true for entity {value}")
        return np.nan
    window = wide.iloc[i-t:i]

    #90% from t as miniman criteria for variance 
    min_obs = np.ceil(0.9 * t)
    valid_cols = window.columns[window.notna().sum() >= min_obs]
    window = window[valid_cols]

    #no selfcorrelation

    if window.shape[1] < 2:
        return np.nan

    corr = window.corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1) #corr.shapes is (7,7), k=1 is a diagonal offset meaning the line above the diagonal (everythings is 1 here) 1 = True, 0 = False (boolean indexing)
    return corr.values[mask].mean()


#---------------------------------------------------------------------------------------------------------------------- macro-context

def return_makro(date:str):
    dictionary = {}
    for i in macro_data["metric"].unique():
        metric_value  = macro_data[(macro_data["date"]== date) & (macro_data["metric"] == i)]["close"]  #wir ziehen mit iloc den wert aus der gefilterten serie
        if not metric_value.empty:
            dictionary[i] = metric_value.iloc[0] 
        else:
            dictionary[i] = np.nan
    return dictionary

#------------------------------------------------------------------------------------------------------------------------creation of feature table 



feature_dictionary = {}

for value in range(0, len(target_values)):
    row = target_values.iloc[value]

    macro_dictionary = {}
    macro_dictionary = return_makro(row["onset"])  #reference for later in the dictionary

    feature_dictionary[value] = {

        #basic description characteristics

        "id" : value,
        #"cluster_id" : row["Cluster_ID"],  # activate if clusterd
        "onset" : row["onset"],
        "tkr" : row["tkr"],
        "duration": row["duration"],
        "class": row["class"],

    #market features
        "vola_t_5" : volatility_before_t(date = row["onset"],tkr = row["tkr"],t = 5),
        "vola_t_10": volatility_before_t(date = row["onset"],tkr = row["tkr"],t = 10),
        "vola_t_15": volatility_before_t(date = row["onset"],tkr = row["tkr"],t = 15),
        "vola_t_20": volatility_before_t(date = row["onset"],tkr = row["tkr"],t = 20),
        "vola_t_30": volatility_before_t(date = row["onset"],tkr = row["tkr"],t = 30),

        "vola_ratio": volatility_ratio(date = row["onset"],tkr = row["tkr"]),

        "momentum_t-5" : momentum(date = row["onset"],tkr = row["tkr"], t=5),
        "momentum_t-10": momentum(date = row["onset"],tkr = row["tkr"], t=10),
        "momentum_t-15": momentum(date = row["onset"],tkr = row["tkr"], t=15),
        "momentum_t-20": momentum(date = row["onset"],tkr = row["tkr"], t=20),
        "momentum_t-30": momentum(date = row["onset"],tkr = row["tkr"], t=30),

        "max_drawdown_t-5": max_drawdown_t(date = row["onset"],tkr = row["tkr"], t=5),
        "max_drawdown_t-10": max_drawdown_t(date = row["onset"],tkr = row["tkr"], t=10),
        "max_drawdown_t-15": max_drawdown_t(date = row["onset"],tkr = row["tkr"], t=15),
        "max_drawdown_t-20": max_drawdown_t(date = row["onset"],tkr = row["tkr"], t=20),
        "max_drawdown_t-30": max_drawdown_t(date = row["onset"],tkr = row["tkr"], t=30),

        "onset_return": onset_return(date = row["onset"],tkr = row["tkr"]),

        "intra_day_range": intra_day_range(date = row["onset"],tkr = row["tkr"]),
   

    # regime features

        "p-onset" : panic_onset(date = row["onset"],tkr = row["tkr"]),

        "max_panic_t-5": max_panic_before_onset(date = row["onset"],tkr = row["tkr"], t=5),
        "max_panic_t-10": max_panic_before_onset(date = row["onset"],tkr = row["tkr"], t=10),
        "max_panic_t-15": max_panic_before_onset(date = row["onset"],tkr = row["tkr"], t=15),
        "max_panic_t-20": max_panic_before_onset(date = row["onset"],tkr = row["tkr"], t=20),
        "max_panic_t-30": max_panic_before_onset(date = row["onset"],tkr = row["tkr"], t=30),

    # crossasset

        "mean_panic_onset" : cross_asset_panic(date = row["onset"]),

        "asset_panic_t-5":  cross_asset_panic_before_t(date = row["onset"], t=5),
        "asset_panic_t-10": cross_asset_panic_before_t(date = row["onset"], t=10),
        "asset_panic_t-15": cross_asset_panic_before_t(date = row["onset"], t=15),
        "asset_panic_t-20": cross_asset_panic_before_t(date = row["onset"], t=20),
        "asset_panic_t-30": cross_asset_panic_before_t(date = row["onset"], t=30),

        "asset_correl_t-10": asset_return_correlation(date = row["onset"], t=10),
        "asset_correl_t-15": asset_return_correlation(date = row["onset"], t=15),
        "asset_correl_t-20": asset_return_correlation(date = row["onset"], t=20),
        "asset_correl_t-30": asset_return_correlation(date = row["onset"], t=30),


    # makro features

        "^VIX" : macro_dictionary["^VIX"], 
        "GPR" : macro_dictionary["GPR"],
        "BAA10Y" : macro_dictionary["BAA10Y"],
        "T10Y2Y" : macro_dictionary["T10Y2Y"]



    }
    print(f"{value} done")
    
df = pd.DataFrame.from_dict(feature_dictionary, orient = "index")


#print(df.isna().sum())

#sub_1 = df[df["asset_correl_t-5"].isna() | df["asset_correl_t-10"].isna() | df["asset_correl_t-15"].isna() | df["asset_correl_t-20"].isna() | df["asset_correl_t-30"].isna()]

#print(sub_1)


df.to_csv( rf"C:\Users\Nutzer\Desktop\scientific project\feature_set_generation\output_data\feature_table.csv", index = False)







