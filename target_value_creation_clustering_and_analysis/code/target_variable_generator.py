import numpy as np
import pandas as pd

#----------------------------------------------------------------------------------------------------------------------------------------------------- #core functions


def sequence_filler(df : pd.DataFrame, reference_column : str, index_column_title: str, gap : int):

    df = df.sort_values(["tkr", "date"]).reset_index(drop = True).copy() #einmal nicht immer wieder

    column_name = f"{reference_column}_gap{gap}"

    df[column_name] = (df[reference_column] >= 0.5).astype(int) #ceil turns everything above 0 to 1 and to integer, so thresholding before is required 

    for tkr in df[index_column_title].unique():

        positions = df.index[df[index_column_title] == tkr]

        start, stop = positions[0], positions[-1] + 1
        i = start

        while i < stop:
            
            if df.at[i , column_name] == 0:
                j = i

                while j < stop and df.at[j, column_name] == 0 :
                    j += 1
                surrounded = i > start and j < stop and df.at[i-1, column_name] == 1 and df.at[j, column_name] == 1 # isn bool value, wird true wenn alles erfüllt ist

                if surrounded and (j-i) <= gap:
                    df.loc[i:j-1, column_name] = 1 # j-1 weil loc ende mit einschlie0t
                i = j # macht hier einfach weiter am ende des intervalls

            else:
                 i += 1

    return df



def duration_generator(df : pd.DataFrame, reference_column: str , index_column : str):

    df = df.sort_values([index_column, "date"]).reset_index(drop = True)

    output_list = []

    for tkr in df[index_column].unique():

        positions = df.index[df[index_column] == tkr]

        start, stop = positions[0], positions[-1] + 1 

        i = start

        while i < stop:

            if df.at[i, reference_column] == 1:
                j = i

                while j < stop and df.at[j, reference_column] >= 1:
                    j += 1

                sequence_endings = i > start and j < stop and df.at[i-1,reference_column] == 0 and df.at[j, reference_column] == 0 #i-1 could be redundant since if -block only is triggerd when we enter a 1 (but keep its my touch to it)

                if sequence_endings:
                    ticker = tkr #from the loop 
                    onset_date = df.at[i,"date"]
                    year = pd.to_datetime(onset_date).year
                    end_date = df.at[j-1,"date"]
                    duration = j-i
                    output_list.append({"tkr": ticker, "year": year, "onset" :onset_date,"end" : end_date, "duration": duration})  #this one is for sure able to be shortent in regards of the the lines above till the if-block,this way just more readable

                i = j 

            else:
                i += 1

    return pd.DataFrame(output_list)

#-------------------------------------------------------------------------------------------------------------------------------------------------------- main loop


df_ = pd.read_csv(r"C:\Users\Nutzer\Desktop\scientific project\data\log_returns_p_panic_data.csv", delimiter=",")


# only keep probabilities above threshold

columns_to_work_on = ["panic_prob", "p_3r_panic"]


threshold = 0.8

for i in columns_to_work_on:
    threshold = threshold
    df_[f"{i}_th{threshold}"]  = df_[i].where(df_[i] >= threshold, other = 0)  #assign new column we cleaned up thresholds of smoothed probabilities above 5 of both states (we never work with the frist column)


#--------------------------------------------------------------------------------------------------------------------------------- target generation


#create target variable set and diffrent sets for sensitivity analysis for gap parameter g 

#for i in range(0,11):
    #gap = i

    #new_df = sequence_filler(df= df_, reference_column= f"p_3r_panic_th{threshold}", index_column_title= "tkr", gap=gap)

    #duration_df = duration_generator(df = new_df, reference_column= f"p_3r_panic_th{threshold}_gap{gap}", index_column= "tkr")

    #duration_df.to_csv(f"durationepisodes_unprocessed_gap_{gap}.csv",index=False, sep = ",")


#-------------------------------------------------------------------------------------------------------------------------- # saving of the finale set

gap = 2

new_df = sequence_filler(df= df_, reference_column= f"p_3r_panic_th{threshold}", index_column_title= "tkr", gap=gap)

# create the durationset

duration_df = duration_generator(df = new_df, reference_column= f"p_3r_panic_th{threshold}_gap{gap}", index_column= "tkr")

duration_df.to_csv("durationepisodes_unprocessed.csv",index=False, sep = ",")

print(len(duration_df))


