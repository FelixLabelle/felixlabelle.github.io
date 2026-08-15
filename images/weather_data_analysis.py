import pandas as pd

weather_data_df = pd.read_csv("berlin_daily_weather_1960_2025.csv")
weather_data_df['YEAR'] = weather_data_df['DATE'].str.split('/').apply(lambda x: x[0])
weather_data_df['MONTH'] = weather_data_df['DATE'].str.split('/').apply(lambda x: x[1])
weather_data_df['DAY'] = weather_data_df['DATE'].str.split('/').apply(lambda x: x[2])
weather_data_df['SORTABLE_DAY'] = weather_data_df['MONTH'] * 100 + weather_data_df['DAY'] 
year_based_means = weather_data_df.groupby('YEAR').aggregate({"TMIN":"mean", "TMAX": "mean"})
last_frost_range = (weather_data_df['TMIN'] > 0) & (weather_data_df['MONTH'])
last_frost_range_df = weather_data_df[last_frost_range]
import pdb;pdb.set_trace()
last_frost_range_df.groupby('YEAR').sortby('SORTABLE_DAY')
# sort by sortable sortable day, only keep last