# CA-housing-price-prediction
Predicting the California housing price using significant environmental features such as air quality, water quality, oil wells and visualizing the current state of environmental factors and housing prices in California

<br />

### Dataset Overview

|Dataset Name|Description|Key Columns|Source URL|
|:----------:|:---------:|:---------:|:--------:|
|housing_price.csv|This is the California average housing price by month-year and county.|Mon-Yr, County_name, County_name2, ...|https://car.sharefile.com/share/view/s0c02663a5c54e23a|
|lab_results.csv|This is the California water quality station dataset by month-year and county.|STATION_TYPE, LATITUDE,	LONGITUDE,	STATUS, COUNTY_NAME, PARAMETER, RESULTS, …|https://data.cnra.ca.gov/dataset/water-quality-data/resource/a9e7ef50-54c3-4031-8e44-aa46f3c660fe|
|air_quality.csv|This is the California air quality dataset by year and county. Each air quality column indicates the number of days in a year that was categorized in corresponding air quality.|State, County, Year, Days with AQI, Good Days, Moderate Days, …|https://aqs.epa.gov/aqsweb/airdata/download_files.html#Annual|
|wells_data.csv|This is the California oil & gas wells dataset by month-year and county.|WellNumber, WellStatus, WellType, WellTypeLa, AreaName, District,	CountyName,...|https://gis.conservation.ca.gov/portal/home/item.html?id=0d30c4d9ac8f4f84a53a145e7d68eb6b|

<br />

### Data Preprocessing
Comprehensive data preprocessing is conducted to 3 final datasets mentioned above. The detailed steps are listed below.\
1. Drop missing values: we define missing values as the values in each dataset that are either np.nan or blank strings. There are about three hundreds of missing values in total in three datasets, which takes a very small proportion of the data volume. Therefore, we drop them because it does not affect the analysis results and model performance.
2. Drop duplicates: checking duplicate values is necessary. We drop all duplicate values in three datasets.
3. Align timestamps: three environment datasets cover different time periods. We have to align them so that the ML models can learn from the same time range. To not lose much information, we decide to choose data from the year 1990 to 2022. 
4. One-hot Encoding: categorical values cannot be consumed by most of the ML models and will cause errors. Hence, we decide to use the one-hot encoding technique to convert categorical data into 0/1 binary value. This technique may create a very sparse dataset if there are many categorical features. However, this does not affect the performance of our ML models.
5. Train / Test data split: To have a good model training and achieve a good training error. After considering the train / test trade off, we choose to use 70% of final data as our training data and the rest 30% as testing data. Within training data, we use 20% as validation data for k fold cross validation.
