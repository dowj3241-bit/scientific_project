import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle 

# dont forget to set save in the end for output

#------------------------------------------------------------------------------------------------------------------------ input is the csv-data-set produced by import_timesries_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

df = pd.read_csv(r"C:\Users\Nutzer\Desktop\scientific project\data\price_series_data.csv", delimiter = ",")                  #set correct path

df["date"] = pd.to_datetime(df["date"])                     # for the almost non-existent hance, that the format of the date is not yy-mm-dd

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#------------------------------------------------------------------------------------------------------------------------ calculating log_Returns & cleaning of NaN and negative prices

df = df[["date","tkr","close"]].sort_values(["tkr","date"])

df = df.drop(df[df["close"] <= 0].index)

df = df.reset_index(drop=True)

df["log_returns"] = df.groupby("tkr")["close"].transform(lambda x : np.log(x / x.shift(1)))

df = df.drop(df[df["log_returns"].isna()].index)

df = df.reset_index(drop=True)

#------------------------------------------------------------------------------------------------------------------------- markov modell fitting 2- regimes

models_2_regime = {}

storage = []

llf_tracking_2_regime = {}

for tkr, elements in df.groupby("tkr"):

    elements = elements.sort_values("date")   # just for the fear that smth isnt sorted

    return_series = pd.Series(elements["log_returns"].values, index = elements["date"].values)

    return_series = return_series * 100  #scaling for msm

    llf_tracker = []
    best = None

    for i in range(0,12):

        np.random.seed(i) # statsmodells uses this as anchor for randomstarts for reproducibility
        try:
            modelcard = sm.tsa.MarkovRegression(return_series, k_regimes = 2, switching_variance= True).fit(search_reps = 10)

            llf_tracker.append(modelcard.llf)

            if best is None or best.llf < modelcard.llf:
                best = modelcard

        except Exception as e:
            llf_tracker.append(None)

    llf_tracking_2_regime[tkr] = llf_tracker

    models_2_regime[tkr] = best

    if best.params["sigma2[1]"] > best.params["sigma2[0]"]:
        prob_series = best.smoothed_marginal_probabilities[1]

    else:
        prob_series = best.smoothed_marginal_probabilities[0]           #series-object,which is easier to be turned back in df and modfied then

    out = prob_series.rename("panic_prob").reset_index()
    out = out.rename(columns={"index": "date"})
    out["tkr"] = tkr
    storage.append(out)

    print(f"model {tkr} fitted, lets go!")

panic_data = pd.concat(storage, ignore_index=True)

df = df.merge(panic_data, on = ["date", "tkr"], how = "left")



#----------------------------------------------------------------------------------------------------------------------------------------- fitting 3-regime-model

models_3_regime = {}

storage = []

llf_tracking_3_regime = {}

for tkr, elements in df.groupby("tkr"):

    elements = elements.sort_values("date")   # just for the fear that smth isnt sorted

    return_series = pd.Series(elements["log_returns"].values, index = elements["date"].values)

    return_series = return_series * 100  #scaling for msm

    llf_tracker = []
    best = None

    for i in range(0,12):
  
          np.random.seed(i) # statsmodells uses this as anchor for randomstarts for reproducibility
          try:
              modelcard = sm.tsa.MarkovRegression(return_series, k_regimes = 3, switching_variance= True).fit(maxiter = 1000, search_reps = 10)

              llf_tracker.append(modelcard.llf)
            
              if best is None or best.llf < modelcard.llf:
                  best = modelcard
  
          except Exception as e:
              llf_tracker.append(None)  

    llf_tracking_3_regime[tkr] = llf_tracker

    models_3_regime[tkr] = best 

    sigma2 = best.params.filter(like="sigma2").values
    order = np.argsort(sigma2)

    probabilities = best.smoothed_marginal_probabilities
      
    out = pd.DataFrame({
        "date" : return_series.index,
        "p_3r_calm" : probabilities[order[0]].values,
        "p_3r_middle": probabilities[order[1]].values,
        "p_3r_panic" :   probabilities[order[2]].values})

    
    out["tkr"] = tkr
    storage.append(out)

    print(f"model {tkr} fitted, lets go!")

bundled_panic_data = pd.concat(storage, ignore_index=True)

df = df.merge(bundled_panic_data, on = ["date", "tkr"], how = "left")


#---------------------------------------------------------------------------------------------------------------------------------------------------------


tracking = pd.DataFrame.from_dict(llf_tracking_2_regime)
tracking.to_csv(rf"2_regime_model_training.csv", index = False, sep=",")

tracking = pd.DataFrame.from_dict(llf_tracking_3_regime)
tracking.to_csv(rf"3_regime_model_training.csv", index = False, sep=",")


#with open("markov_2_regime_models.pickle", "wb") as f:
#    pickle.dump(models_2_regime, f)

#with open("markov_3_regime_models.pickle", "wb") as f:
#    pickle.dump(models_3_regime, f)


#df.to_csv(rf"log_returns_p_panic_data.csv", index = False, sep=",") # dont forget to set path 

print("done")













