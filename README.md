# Does the Storm Last? A Machine Learning Approach to Predict the Duration of Panic Volatility after Market Shocks
A leakage-safe machine learning framework to predict the duration of shock-induced panic volatility across energy, metals, agriculture, and equities (2000–2026). Utilizes 3-state Markov-switching models for regime detection and tree-based ensembles (XGBoost/Random Forest) to forecast whether market instabilities will resolve quickly or persist.


This project asks whether machine learning can predict how long market instability lasts after a major shock. The basic question is if at the day a crisis starts, we are to say anything useful about how long it will run?

The project compares several model families to see which ones work, whether they beat random-chance baselines, and which kinds of information carry signal. The features come from price movements, cross-asset information, and macroeconomic conditions. Where different models agree that a feature matters, that feature is treated as a more promising signal.

The aim is evidence on three points: whether crisis duration is predictable at all, which methods look most promising, and which signals drive predictions. The results are meant to support later research and to give policymakers, investors, and other stakeholders something to work with in periods of extreme uncertainty.

<div align="center">

  <img src="data/Alter_(Raste)_1881_84_Arthur_Fitger.jpg" alt="Nach dem vollendeten Werk ausruhn" width="600px">

  > *„NACH DEM VOLLENDETEN WERK AUSRUHN, O SCHOENSTER GEDANKE,*
  > *JEDER BEGEHRT ES, DOCH WIE SELTEN GEWAEHRT ES DAS GLUECK.“*
  > <br>— Arthur Fitger (Hamburger Kunsthalle)

</div>

---
## Pipeline

The scripts **needs to be run in the order** below. Each one writes a CSV or a pickle file that the next one reads.

### 1. Import

`import_timeseries_data.py` downloads daily prices for seven assets from Yahoo Finance: gold, copper, natural gas, crude oil, soy meal, wheat, and the S&P 500. It keeps date, close, low, high, ticker, and category, then cleans the data. It drops duplicate rows, fills gaps by linear interpolation, and replaces zero prices the same way. Negative prices stay in on purpose, since crude oil really did trade below zero in April 2020. The script writes a text report that records what it found and what it changed.

`makrodata_import.py` builds the macro table. It pulls the VIX from Yahoo Finance and reads three files from disk: the Geopolitical Risk index, the BAA-to-10-year credit spread, and the 10-year-minus-2-year yield curve slope. All four end up in one long table with the columns date, close, and metric. Missing days are interpolated per metric.

### 2. Markov model fitting

`markov_model_fitting.py` turns closing prices into log returns, scales them by 100, and fits a Markov switching model per asset. It fits both a two-regime and a three-regime version, each with switching variance. Each fit runs twelve times under different random seeds, and the run with the highest log-likelihood wins. The script logs every log-likelihood so the spread across seeds can be checked later.

The regime with the highest variance is read as the panic state. The model gives a daily probability of being in that state, which is the core signal for everything downstream.

### 3. Transformation

`data_combinator.py` joins the three inputs into one table for feature building: prices, log returns, and the daily panic probability from the Markov model. It merges on date and ticker and drops rows with missing values, which removes the first day of each series.

### 4. Target value generation

`target_variable_generator.py` turns the daily panic probability into episodes. It marks every day with a probability of 0.5 or higher as a panic day. Single calm days that sit between two panic days are filled in, so that one quiet day does not split one crisis into two. The script then reads off each run of panic days and records its ticker, start date, end date, and length in trading days.

`filler_analysis.py` checks how much the gap width matters. It compares episode counts, mean and median length, and total days across gap widths from 0 to 10, and lists which episodes merge as the gap grows.

`target_value_cluster_&_visuals_pipeline.py` filters out episodes shorter than three trading days and groups the rest into crisis clusters. Two episodes belong to the same cluster if their start dates lie within 30 trading days of each other. A cluster that touches two or more assets is called multi-asset, one that touches a single asset is called idiosyncratic. The script also draws the charts: clusters over time, the duration distribution, the effect of each candidate length filter, and the raw against the log-transformed duration.

Clusters serve two purposes. They describe the data, and they act as groups in the cross-validation later, so that two episodes from the same crisis never sit on both sides of a train-test split.

### 5. Feature generation

`feature_engineering.py` builds one row per episode. Every window ends the day before the onset, so no feature can see the crisis it is meant to predict.

**Market features:** volatility over the 5, 10, 15, 20, and 30 days before onset; the ratio of 5-day to 20-day volatility; cumulative return over the same five windows; the largest drawdown in each window; the return on the onset day itself; and the high-low range on the onset day, divided by the close so that assets can be compared.

**Regime features:** the highest panic probability in each of the five windows before it

**Cross-asset features:** the mean panic probability across all assets on the onset day and across each window before it, plus the mean pairwise correlation of returns across assets. A column enters the correlation only if it has data on at least 90 percent of the days in the window.

**Macro features:** the VIX, the Geopolitical Risk index, the credit spread, and the yield curve slope on the onset day.

**Target:** episodes lasting three trading days or longer are class 1(persistent crisis), shorter ones class 0(short crisis).

### 6. Model training

`modell_training_classification.py` runs the comparison. Every model sits in the same pipeline: median imputation with a missing-value indicator, standard scaling, then the model itself. An optional L1 feature selection step is in the code but switched off.

Five models compete: random forest, XGBoost, a support vector machine with an RBF kernel, logistic regression with Elastic Net penelization, Gaussian naive Bayes

Data is sorted chronologically and split such that 80% are training and 20% testing. since AUC and F1 are not defined there. The script reports ROC-AUC, macro F1, and balanced accuracy with and 0.95 confidence intervall over a boostrapped resampling of 10 aswell as a stasitistical significance. Feature importance comes from permutation importance


## Design parameters

| Parameter | Value | Where |
|---|---|---|
| Assets | 7 (2 metal, 2 energy, 2 agricultural, 1 equity) | import |
| Period | 2000 to 2026, daily | import |
| Regimes | 2 and 3, switching variance | Markov fitting |
| Random starts per fit | 12 seeds, best log-likelihood kept | Markov fitting |
| Panic state | regime with the highest variance | Markov fitting |
| Probability type for labels | smoothed | target generation |
| Probability type for features | filtered | transformation |
| Panic threshold | 0.8 | target generation |
| Gap fill | 1 trading day | target generation |
| Class cut | 3 trading days | feature generation |
| Feature windows | 5, 10, 15, 20, 30 days before onset | feature generation |
| Minimum coverage for correlation | 90 percent of window days | feature generation |

Labels use smoothed probabilities, which look at the whole series and so give the cleanest reading of where a crisis really began and ended. Features use filtered probabilities, which only look backwards and so match what an observer would have known on the day. This keeps the target honest and the features free of hindsight.

## File order

```
import_timeseries_data.py     ->  price_series_data.csv
makrodata_import.py           ->  macro_data.csv
markov_model_fitting.py       ->  markov_*_regime_models.pickle
                                  log_returns_p_panic_data.csv
target_variable_generator.py  ->  durationepisodes_unprocessed.csv
filler_analysis.py            ->  gap sensitivity tables
target_value_cluster_&_visuals_pipeline.py -> episodes_clustered_filter=0.csv
data_combinator.py            ->  df_features_input.csv
feature_engineering.py        ->  feature_table.csv
modell_training_classification.py -> model_results.csv & feature importance tables
```

**!!!Paths are hard-coded at the top of each script and need to be changed to the local directory. Most write steps are commented out, so uncomment the line at the end of a script to save its output.!!!**

## Requirements

pandas, numpy, statsmodels, scikit-learn, xgboost, yfinance, matplotli

#Links to data

# Assset-Data-Links:

Gold-Future (GC=F): https://de.finance.yahoo.com/quote/GC=F/

Copper-Future (HG=F): https://de.finance.yahoo.com/quote/HG=F/

Natural-Gas-Future (NG=F): https://de.finance.yahoo.com/quote/NG=F/

Crude-Oil-Future (CL=F): https://de.finance.yahoo.com/quote/CL=F/

Soybean-Meal-Future (ZM=F): https://de.finance.yahoo.com/quote/ZM=F/

Chicago-SRW-Wheat-Futures (ZW=F): https://de.finance.yahoo.com/quote/ZW=F/ 

S&P-500 (^GSPC): https://de.finance.yahoo.com/quote/%5EGSPC/ 

# Macro-Data-Links:

GPR-Index: 

https://www.matteoiacoviello.com/gpr.htm


10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity (T10Y2Y):

https://fred.stlouisfed.org/series/T10Y2y


Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity (BAA10Y):

https://fred.stlouisfed.org/series/baa10y


CBOE Volatility Index (^VIX):

https://de.finance.yahoo.com/quote/%5EVIX/

---

## Used Python-Packages:

This project runs on open-source packages. Please cite them if you build on this work.

**Matplotlib** — Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)

**NumPy** — Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)

**pandas** — McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 445, 56–61.
The pandas development team (2020). *pandas-dev/pandas: Pandas*. Zenodo. [doi:10.5281/zenodo.3509134](https://doi.org/10.5281/zenodo.3509134)

**scikit-learn** — Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

**statsmodels** — Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. *9th Python in Science Conference*.

**XGBoost** — Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

**yfinance** — Aroussi, R. (2026). *yfinance: Download market data from Yahoo! Finance API* (Version 0.2.x) [Python package]. GitHub. https://github.com/ranaroussi/yfinance

<details>
<summary>BibTeX</summary>

```bibtex
@article{Hunter2007,
  author  = {Hunter, J. D.},
  title   = {Matplotlib: A 2D graphics environment},
  journal = {Computing in Science \& Engineering},
  volume  = {9},
  number  = {3},
  pages   = {90--95},
  year    = {2007},
  doi     = {10.1109/MCSE.2007.55}
}

@article{Harris2020,
  author  = {Harris, Charles R. and Millman, K. Jarrod and van der Walt, St{\'e}fan J. and others},
  title   = {Array programming with {NumPy}},
  journal = {Nature},
  volume  = {585},
  pages   = {357--362},
  year    = {2020},
  doi     = {10.1038/s41586-020-2649-2}
}

@inproceedings{McKinney2010,
  author    = {McKinney, Wes},
  title     = {Data structures for statistical computing in {Python}},
  booktitle = {Proceedings of the 9th Python in Science Conference},
  volume    = {445},
  pages     = {56--61},
  year      = {2010}
}

@software{reback2020pandas,
  author    = {{The pandas development team}},
  title     = {pandas-dev/pandas: Pandas},
  month     = feb,
  year      = {2020},
  publisher = {Zenodo},
  version   = {latest},
  doi       = {10.5281/zenodo.3509134},
  url       = {https://doi.org/10.5281/zenodo.3509134}
}

@article{scikit-learn,
  author  = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
             and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
             and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A.
             and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  title   = {Scikit-learn: Machine Learning in {Python}},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011}
}

@inproceedings{seabold2010statsmodels,
  author    = {Seabold, Skipper and Perktold, Josef},
  title     = {statsmodels: Econometric and statistical modeling with {Python}},
  booktitle = {9th Python in Science Conference},
  year      = {2010}
}

@inproceedings{Chen2016,
  author    = {Chen, Tianqi and Guestrin, Carlos},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on
               Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2016},
  doi       = {10.1145/2939672.2939785}
}

@software{aroussi_yfinance,
  author  = {Aroussi, Ran},
  title   = {yfinance: Download market data from {Yahoo!} {Finance} {API}},
  version = {0.2.x},
  year    = {2026},
  url     = {https://github.com/ranaroussi/yfinance}
}
```

</details>
































