import pandas as pd
import numpy as np
from plume_learn.plume_utils.manage_plume import remove_all_0_plume

def modify_df(df_in, growth_id, growth_name):
    """
    Modify the dataframe with real unit (distance, time) from pixel.
    
    Args:
    df (pd.DataFrame): dataframe to be modified
    """
    df = df_in.copy()
    
    df.rename(columns={'plume_index': 'Plume Index'}, inplace=True)
    
    df.rename(columns={'time_index': 'Time (µs)'}, inplace=True)
    df['Time (µs)'] = df['Time (µs)'] * 500e-3 # convert ns to ms
    
    df.rename(columns={'Distance': 'Distance (m)'}, inplace=True)
     # convert pixel to meter: target to substrate distance is 54mm or 347 pixels.
    df['Distance (m)'] = df['Distance (m)'] * 54/1000/347 
    
    df.rename(columns={'Velocity': 'Velocity (m/s)'}, inplace=True)
    # convert pixel to meter: target to substrate distance is 54mm or 347 pixels; convert time to second: time interval is 500ns.
    df['Velocity (m/s)'] = df['Velocity (m/s)'] * 54/1000/347 / 500e-9

    df.rename(columns={'Area': 'Area (a.u.)'}, inplace=True)
    
    df = remove_all_0_plume(df.reset_index(), index_label='Plume Index', metric='Area (a.u.)', viz=False)
    df.rename(columns={'Growth': 'Growth Name'}, inplace=True)
    
    # df['Growth ID'] = growth_id
    # df['Growth Name'] = growth_name
    
    df['Threshold'] = df['Threshold'].astype(str)
    df = df[df['Threshold']=='200']
    return df