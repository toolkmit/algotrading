import pandas as pd
import numpy as np

def calc_pivots(data):
    df = data.copy()
    df['ph'] = np.nan #pivot highs
    df['pl'] = np.nan #pivot lows
    df['swing'] = np.nan #swings
    
    #Initialize internal vars
    lower_low = higher_high = False
    pivot1_found = pivot1_was_low = False
    pivot1_index = pivot1_value = 0
    pivot2_found = pivot2_was_low = False
    pivot2_index = pivot2_value = 0

    for i in df.index[1:]:
        prev_low = df.at[i-1, 'low']
        curr_low = df.at[i, 'low']
        prev_high = df.at[i-1, 'high']
        curr_high = df.at[i, 'high']
        
        if lower_low:
            if curr_low > prev_low:
                df.at[i-1, 'pl'] = prev_low
                lower_low = False
        else:
            if curr_low < prev_low:
                lower_low = True
                
        if higher_high:
            if curr_high < prev_high:
                df.at[i-1, 'ph'] = prev_high
                higher_high = False
        else:
            if curr_high > prev_high:
                higher_high = True
                
        
        #This is the first pivot that we find
        if not pivot1_found:
            #pivot is a high
            if not np.isnan(df.at[i-1, 'ph']) and np.isnan(df.at[i-1, 'pl']):
                pivot1_was_low = False
                pivot1_index = i-1
                pivot1_found = True
                pivot1_value = df.at[i-1, 'ph']
                df.at[i-1, 'swing'] = df.at[i-1, 'ph']
            #pivot is a low
            elif not np.isnan(df.at[i-1, 'pl']) and np.isnan(df.at[i-1, 'ph']):
                pivot1_was_low = True
                pivot1_index = i-1
                pivot1_found = True
                pivot1_value = df.at[i-1, 'pl']
                df.at[i-1, 'swing'] = df.at[i-1, 'pl']
            elif not np.isnan(df.at[i-1, 'pl']) and not np.isnan(df.at[i-1, 'ph']):
                pass #unique case where bar is both pivot high and low
            
        elif pivot1_found and not pivot2_found:
            if not np.isnan(df.at[i-1, 'ph']) and np.isnan(df.at[i-1, 'pl']):
                if pivot1_was_low:
                    pivot2_was_low = False
                    pivot2_index = i-1
                    pivot2_found = True
                    pivot2_value = df.at[i-1, 'ph']
                    df.at[i-1, 'swing'] = df.at[i-1, 'ph']
                elif df.at[i-1, 'ph'] >= pivot1_value:
                    #update pivot1 if this is a higher high
                    df.at[pivot1_index, 'swing'] = np.nan
                    pivot1_index = i-1
                    pivot1_value = df.at[i-1, 'ph']
                    df.at[i-1, 'swing'] = df.at[i-1, 'ph']
            elif not np.isnan(df.at[i-1, 'pl']) and np.isnan(df.at[i-1, 'ph']):
                if not pivot1_was_low:
                    pivot2_was_low = True
                    pivot2_index = i-1
                    pivot2_found = True
                    pivot2_value = df.at[i-1, 'pl']
                    df.at[i-1, 'swing'] = df.at[i-1, 'pl']
                elif df.at[i-1, 'pl'] <= self.pivot1_value:
                    #update pivot1 if this is a lower low
                    df.at[pivot1_index, 'swing'] = np.nan
                    pivot1_index = i-1
                    pivot1_value = df.at[i-1, 'pl']
                    df.at[i-1, 'swing'] = df.at[i-1, 'pl']
            elif not np.isnan(df.at[i-1, 'pl']) and not np.isnan(df.at[i-1, 'ph']):
                #unique case where bar is both pivot high and low  
                pivot2_index = i-1
                pivot2_found = True
                if not pivot1_was_low: #Pivot 1 was high, so we make Pivot2 a low
                    pivot2_was_low = True
                    pivot2_value = df.at[i-1, 'pl']
                    df.at[i-1, 'swing'] = df.at[i-1, 'pl']
                else: #Pivot 1 was low, so we make Pivot2 a high
                    pivot2_was_low = False
                    pivot2_value = df.at[i-1, 'ph']
                    df.at[i-1, 'swing'] = df.at[i-1, 'ph']
                    
        elif pivot1_found and pivot2_found:
            if not np.isnan(df.at[i-1, 'ph']) and np.isnan(df.at[i-1, 'pl']):
                if pivot2_was_low: # 3 pivot sequence complete -- update p1 & p2 fields
                    pivot1_was_low = pivot2_was_low
                    pivot1_index = pivot2_index
                    pivot1_value = pivot2_value
                    pivot2_was_low = False
                    pivot2_index = i-1
                    pivot2_value = df.at[i-1, 'ph']
                    df.at[i-1, 'swing'] = df.at[i-1, 'ph']
                elif df.at[i-1, 'ph'] >= pivot2_value:
                    #update pivot2 if this is a higher high
                    df.at[pivot2_index, 'swing'] = np.nan
                    pivot2_index = i-1
                    pivot2_value = df.at[i-1, 'ph']
                    df.at[i-1, 'swing'] = df.at[i-1, 'ph']
            elif not np.isnan(df.at[i-1, 'pl']) and np.isnan(df.at[i-1, 'ph']):
                if not pivot2_was_low: #3 pivot seq complete
                    pivot1_was_low = pivot2_was_low
                    pivot1_index = pivot2_index
                    pivot1_value = pivot2_value
                    pivot2_was_low = True
                    pivot2_index = i-1
                    pivot2_value = df.at[i-1, 'pl']
                    df.at[i-1, 'swing'] = df.at[i-1, 'pl']
                elif df.at[i-1, 'pl'] <= pivot2_value:
                    #update pivot2 if this is a lower low
                    df.at[pivot2_index, 'swing'] = np.nan
                    pivot2_index = i-1
                    pivot2_value = df.at[i-1, 'pl']
                    df.at[i-1, 'swing'] = df.at[i-1, 'pl']
            elif not np.isnan(df.at[i-1, 'pl']) and not np.isnan(df.at[i-1, 'pl']):
                #unique case where bar is both pivot high and low      
                pivot1_was_low = pivot2_was_low
                pivot1_index = pivot2_index
                pivot1_value = pivot2_value
                pivot2_index = i-1
                
                if not pivot2_was_low: #2nd pivot was a high so 3rd pivot is a low
                    pivot2_was_low = True
                    pivot2_value = df.at[i-1, 'pl']
                    df.at[i-1, 'swing'] = df.at[i-1, 'pl']
                else:  # 2nd pivot was low so 3rd pivot is a high
                    pivot2_was_low = False
                    pivot2_value = df.at[i-1, 'ph']
                    df.at[i-1, 'swing'] = df.at[i-1, 'ph']
    
    return df