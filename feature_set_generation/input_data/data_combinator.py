import pandas as pd 
import pickle
import numpy as np
import statsmodels.api as sm

BASE = r"C:\Users\Nutzer\Desktop\scientific project"

markov_model = BASE + r"\markov_model_documentation\markov_3_regime_models.pickle" #provides unsmoothed proboialitly of high volatility regime per day

price_time_series = pd.read_csv(BASE + r"\data\price_series_data.csv", sep = ",")  #provides closing price, daily high, daily low per day

log_returns = pd.read_csv(BASE + r"\data\log_returns_p_panic_data.csv", sep = ",")   #provides log returns per date

with open(markov_model,"rb") as f:
    regime_model = pickle.load(f)


#-------------------------------------------------------extracting raw probability for high volatility state per day per asset from current model
liste = []

for tkr, content in regime_model.items():

    # identification of which part is highest sigma in all regimes

    filtered_probs = regime_model[tkr].filtered_marginal_probabilities
    sigmas = np.array([regime_model[tkr].params[f"sigma2[{i}]"] for i in range(0,len(filtered_probs.columns))])   #we have as many probabilities columns as indexes so k is always implicilty given
    max_sigma_pos = np.argmax(sigmas)

    #print(max_sigma_pos)

    # assignment of probabiliteis of the highest variance regime and transformation of the serie to df with index of date
    serie = filtered_probs[max_sigma_pos]
    serie = serie.to_frame(name="panic_prob")

    #addtional transformations
    serie["date"] = serie.index
    serie = serie.sort_values("date").reset_index(drop = True) 
    serie["tkr"] = tkr
    liste.append(serie)
    print(f"{tkr} done, next round!")

prob_df = pd.concat(liste)
prob_df = prob_df.sort_values(["tkr","date"]).reset_index(drop = True) #erst nach tkr, dann nach date, index angleichen
prob_df["date"] = prob_df["date"].astype(str) #vorbereitung zum mergen, da nur gleiche typen mergen 

#----------------------------------------------------------------------------------mergen wie bei sql

sub1 = price_time_series[["date","tkr","close","low","high"]]
sub2 = log_returns[["date","tkr","log_returns"]]

df_combined = pd.merge(sub1,sub2, on = ["date","tkr"], how = "outer")
df_combined = pd.merge(df_combined,prob_df, on =["date","tkr"], how = "outer" )
df_combined = df_combined.dropna().reset_index(drop = True) #cleaning it since some day have no log returns and no prob because they are the first days

#--------------------------------------------------------------------------------saving

#df_combined.to_csv(r"df_features_input.csv",index=False, sep =",")











