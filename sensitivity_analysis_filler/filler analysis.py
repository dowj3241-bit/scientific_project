import pandas as pd
import numpy as np

#---------------------------------------------------------------------------------------------------------------------------------- gap analysis 

base = r"C:\Users\Nutzer\Desktop\scientific project\filler_width_testing"  #change to current directory
datasets = {
    gap: pd.read_csv(rf"{base}\durationepisodes_unprocessed_gap_{gap}.csv")
    for gap in range(11)
}

print(datasets[0].columns.tolist())
print(datasets[0].head())


def summarize(datasets, dur_col="duration"):
    rows = []
    for gap in sorted(datasets):
        d = datasets[gap][dur_col]
        rows.append({
            "gap":        gap,
            "n_episodes": len(d),
            "dur_mean":   round(d.mean(), 1),
            "dur_median": d.median(),
            "dur_max":    int(d.max()),
            "dur_q90":    round(d.quantile(0.90), 1),
            "total_days": int(d.sum()),
        })
    out = pd.DataFrame(rows)
    out["merges_vs_g0"]    = out.loc[0, "n_episodes"] - out["n_episodes"]
    out["fill_days_vs_g0"] = out["total_days"] - out.loc[0, "total_days"]
    return out

tab = summarize(datasets)

tab.to_csv(rf"{base}\filler_summary_for_raw_data.csv", index=False)

#-------------------------------------------------------------------------------------------------tracking of changes

def merged_away(df_lo, df_hi):
    hi_onsets = set(zip(df_hi["tkr"], df_hi["onset"]))
    mask = [(t, o) not in hi_onsets
            for t, o in zip(df_lo["tkr"], df_lo["onset"])]
    return df_lo[mask]

changes = []
for g in range(10):
    lost = merged_away(datasets[g], datasets[g+1]).copy()
    lost["step"] = f"{g}->{g+1}"
    changes.append(lost)

changes = pd.concat(changes, ignore_index=True)
print(changes.to_string(index=False))
changes.to_csv(r"C:\Users\Nutzer\Desktop\scientific project\filler_width_testing\list_of_changes.csv", index=False)