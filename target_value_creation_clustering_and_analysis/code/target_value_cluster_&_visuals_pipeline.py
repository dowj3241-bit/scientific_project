import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

episodes_raw = pd.read_csv(r"C:\Users\Nutzer\Desktop\scientific_project\data\durationepisodes_unprocessed.csv", sep = ",")

print(len(episodes_raw))

days = pd.read_csv(r"C:\Users\Nutzer\Desktop\scientific_project\data\log_returns_p_panic_data.csv", sep=",")  #has a list of all trading days coverd

trading_days = pd.Index(sorted(pd.to_datetime(days["date"]).unique()))

CRISIS_GAP = 30   # businessdays, if the gape betweens onsets is bigger then the integer its a news "panic_regime-cluster"



mighty_filter = 0


episodes_raw["onset"] = pd.to_datetime(episodes_raw["onset"])
episodes = episodes_raw[episodes_raw["duration"] >= mighty_filter].sort_values("onset").reset_index(drop=True)   #the mighty filter 

episodes["onset_pos"] = trading_days.searchsorted(episodes["onset"].to_numpy())
episodes["Cluster_ID"] = (episodes["onset_pos"].diff() > CRISIS_GAP).cumsum()


#----------------------------------------------------------------------------------------------------------------------------------------------------------descriptive analysis of the formed clusters

summary = episodes.groupby("Cluster_ID").agg(
    Cluster_Start = ("onset", "min"),
    Cluster_End = ("onset", "max"),
    Episodes = ("tkr", "size"),
    Assets_involved = ("tkr", "nunique"),
    Assets = ("tkr", lambda s: ",".join(sorted(s.unique()))),
    Median_Duration = ("duration", "median"),
    Min_Duration = ("duration", "min"),
    Max_Duration = ("duration", "max"),)

summary["type"] = np.where(summary["Assets_involved"] >= 2, "multi-asset", "ideosyncratic")
#summary.sort_values("Cluster_Start").to_csv(f"cluster_summary_filter={mighty_filter}.csv", sep=",")           #csv saving here
print(summary.to_string())

# --- basic input for featureengineering
episodes["n_assets_in_cluster"]  = episodes["Cluster_ID"].map(summary["Assets_involved"])
episodes["n_epis_in_cluster"] = episodes["Cluster_ID"].map(summary["Episodes"])
episodes["Cluster_Type"] = np.where(episodes["n_assets_in_cluster"] >= 2, "multi-asset", "ideosyncratic")
#episodes.to_csv(f"episodes_clustered_filter={mighty_filter}.csv", index=False, sep=",")                        #csv saving here
print(episodes)


# could be activated if needed , lets see what my supervisor says about the rest
overview = summary.groupby("type").agg(
    n_clusters = ("Episodes", "size"),
    n_episodes = ("Episodes", "sum"),
    Median_Duration = ("Median_Duration", "median"),)

overview["ep_share_%"] = (overview["n_episodes"] / overview["n_episodes"].sum() * 100).round(1)

#---------------------------------------------------------------------------------------------------------------------------------------------------------- visualisation of clusters found over time

s = summary.reset_index().copy()
s["start"] = pd.to_datetime(s["Cluster_Start"])

broad  = s[s["Assets_involved"] >= 2]   # multi-asset 
narrow = s[s["Assets_involved"] <  2]   

fig, ax = plt.subplots(figsize=(14, 4.5))

# idiosynkcratic clusters
ax.vlines(narrow["Cluster_Start"], 0, narrow["Assets_involved"], color="#b0b0b0", linewidth=1.0, zorder=1)
ax.scatter(narrow["Cluster_Start"], narrow["Assets_involved"], s=18, color="#b0b0b0", zorder=2,
           label="idiosyncratic")

# multi-Asset-Cluster 
ax.vlines(broad["Cluster_Start"], 0, broad["Assets_involved"], color="#c0392b", linewidth=2.0, zorder=3)
ax.scatter(broad["Cluster_Start"], broad["Assets_involved"], s=45, color="#c0392b", zorder=4,
           label="multi-asset cluster (>=2)")


ymax = int(s["Assets_involved"].max())
ax.set_ylim(0, ymax + 5.8)          # headroom for three label rows
ax.set_yticks(range(0, ymax + 1))
ax.set_ylabel("number of involved assets")
#ax.set_title(f"Crisis clusters over time ({len(s)} clusters)", pad=28)

EVENTS = {
    4:  ("9/11 Attacks",                 0,    0),
    8:  ("Iraq Invasion",                1,  160),
    13: ("Chinese Credit Crunch", 2, 260),
    20: ("Amaranth Default\n Australian Millennium Drought",        0, 100),
    24: ("Early GFC", 1, 180),
    25: ("Global Financial\nCrisis",     2,   260),
    27: ("Greek Debt Crisis,\nFlash Crash",   0,  210),
    29: ("USDA Grain Shock",             1,  430),
    32: ("US Downgrade,\nEuro Crisis",   2,  560),
    46: ("Covid-19 Crash",               0, -260),
    52: ("Russian Invasion \n of Ukraine",1,   60),
    59: ("US Tariffs Shock",                 0,  100),
    62: ("Middle East Tension",      2,  210),
}
ROW_Y = {0: ymax + 4.5, 1: ymax + 3.1, 2: ymax + 1.7}

for cid, (text, row, shift) in EVENTS.items():
    r = s.loc[s["Cluster_ID"] == cid]
    if r.empty:
        continue
    r = r.iloc[0]
    ax.annotate(
        text,
        xy=(r["start"], r["Assets_involved"] + 0.25),
        xytext=(r["start"] + pd.Timedelta(days=shift), ROW_Y[row]),
        ha="center", va="bottom", fontsize=8, linespacing=1.15, zorder=5,
        arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.7, alpha=0.6,
                        shrinkA=2, shrinkB=0,
                        connectionstyle="angle,angleA=0,angleB=90,rad=0"),
    )

ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", which="minor", length=3, color="#999999")
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
          frameon=False, borderaxespad=0)

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------histogram duration distribution 

dur = episodes_raw.loc[episodes_raw["duration"] >= mighty_filter, "duration"]

mean, med, skew = dur.mean(), dur.median(), dur.skew()


fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(dur, bins=range(0, int(dur.max()) + 6, 2), color="steelblue", edgecolor="white")

ax.axvline(med,  color="crimson", ls="-",  lw=2, label=f"median = {med:.0f}")
ax.axvline(mean, color="black",   ls="--", lw=2, label=f"mean = {mean:.0f}")

ax.set_yscale("log")

ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)


ax.set_xlabel("Episode duration (trading days)")
ax.set_ylabel("Count (log scale)")
ax.set_xticks(range(0, int(dur.max()) + 25, 10)) 
ax.set_title(f" Distribution of episode durations "
             f"(n = {len(dur)}).", fontsize=10, loc="left")
ax.legend()

plt.tight_layout()
plt.show()
#-----------------------------------------------------------------------------------filter analysis

filters = [0,1,2,3,4,5,6,7,8,9,10]
tickers = sorted(episodes_raw["tkr"].unique())

rows = {}
for f in filters:
    sub = episodes_raw[episodes_raw["duration"] >= f]
    n   = len(sub)
    counts = sub["tkr"].value_counts()
    # "12 (18%)" per ticker
    rows[f] = {t: f"{counts.get(t, 0)} ({counts.get(t, 0)/n:.0%})" for t in tickers}

table = pd.DataFrame(rows).T                     # index = filter, cols = tickers
table.index.name = "min_duration"
table["TOTAL"] = [len(episodes_raw[episodes_raw["duration"] >= f]) for f in filters]

#print(table)
#table.to_csv(r"asset_composition_by_filter.csv")