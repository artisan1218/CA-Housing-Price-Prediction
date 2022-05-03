# CA-housing-price-prediction
Predicting the California housing price using significant environmental features such as air quality, water quality, oil wells and visualizing the current state of environmental factors and housing prices in California

<br />

### Dataset Overview
In the folder `data`, there are four dataset file in `.csv` format. The detailed information of these files are shown below:

|Dataset Name|Description|Key Columns|Source URL|
|:----------:|:---------:|:---------:|:--------:|
|housing_price.csv|This is the California average housing price by month-year and county.|Mon-Yr, County_name, County_name2, ...|https://car.sharefile.com/share/view/s0c02663a5c54e23a|
|lab_results.csv|This is the California water quality station dataset by month-year and county.|STATION_TYPE, LATITUDE,	LONGITUDE,	STATUS, COUNTY_NAME, PARAMETER, RESULTS, …|https://data.cnra.ca.gov/dataset/water-quality-data/resource/a9e7ef50-54c3-4031-8e44-aa46f3c660fe|
|air_quality.csv|This is the California air quality dataset by year and county. Each air quality column indicates the number of days in a year that was categorized in corresponding air quality.|State, County, Year, Days with AQI, Good Days, Moderate Days, …|https://aqs.epa.gov/aqsweb/airdata/download_files.html#Annual|
|wells_data.csv|This is the California oil & gas wells dataset by month-year and county.|WellNumber, WellStatus, WellType, WellTypeLa, AreaName, District,	CountyName,...|https://gis.conservation.ca.gov/portal/home/item.html?id=0d30c4d9ac8f4f84a53a145e7d68eb6b|

<br />

### Data Preprocessing
Comprehensive data preprocessing is conducted to 3 final datasets mentioned above. The detailed steps are listed below.
1. Drop missing values: we define missing values as the values in each dataset that are either np.nan or blank strings. There are about three hundreds of missing values in total in three datasets, which takes a very small proportion of the data volume. Therefore, we drop them because it does not affect the analysis results and model performance.
2. Drop duplicates: checking duplicate values is necessary. We drop all duplicate values in three datasets.
3. Align timestamps: three environment datasets cover different time periods. We have to align them so that the ML models can learn from the same time range. To not lose much information, we decide to choose data from the year 1990 to 2022. 
4. One-hot Encoding: categorical values cannot be consumed by most of the ML models and will cause errors. Hence, we decide to use the one-hot encoding technique to convert categorical data into 0/1 binary value. This technique may create a very sparse dataset if there are many categorical features. However, this does not affect the performance of our ML models.
5. Train / Test data split: To have a good model training and achieve a good training error. After considering the train / test trade off, we choose to use 70% of final data as our training data and the rest 30% as testing data. Within training data, we use 20% as validation data for k fold cross validation.


We came up with a solution that joins the housing price dataset with all three environment dataset separately. In this way, we get three final datasets eventually and each dataset is manageable in size. The visualization is shown as the following. 

![image](https://user-images.githubusercontent.com/25105806/166407384-164703e1-7236-49fa-ad60-3fe5adeeb404.png)

<br />

### Model Training
The three files in the root directory `LSTM.ipynb`, `Neural Network.ipynb`, `XGBoost.ipynb` are our model training code for corresponding models respectively. It captures our implementations of the models we used, how we trained the models and how we exported the models.

<br />

### Dashboard
Dashboard is the final step of this project implementation and is the product we need to deliver to our customer. The core idea behind it is to have a user friendly interface and can show enough information for users to explore. The top four sections of this dashboard can help users to explore housing prices and the environment in California. And the last section is used to forecast the housing price. The demo video has been uploaded to https://www.youtube.com/watch?v=7M9wgudshW4

We put all model-inferencing and visualization source code in the folder `dashboard`. 
* The sub-folder `model_file` stores the core LSTM model we've trained and used to make prediction on the housing price based on environmental issues. The three models are for air pollution, water pollution and oil&wells pollution respectively. The result of these models will be aggregated together later.
* The sub-folder `scaler_file` stores the trained `StandardScaler` from `from sklearn.preprocessing import StandardScaler`. We will use these pre-trained scaler to scale the user input to the range(0, 1) so that our models can predict.
* The stand-alone file `vis.py` is the core code for our dashboard. The dashboard is built by using Python's streamlit package. We have to convert the file to `.py` file otherwise streamlit will not start. To start the dashboard, simply use the command `streamlit run path/to/streamlit/dashboard/file`


Here are some screenshots of the dashbaord:
![image](https://user-images.githubusercontent.com/25105806/166408337-5ab5ebc1-8a6c-493e-b343-3d4dac474cad.png)
![image](https://user-images.githubusercontent.com/25105806/166408354-ccf32fa2-9363-4baf-8ae4-6ea1dfd06204.png)
![image](https://user-images.githubusercontent.com/25105806/166408365-42f48b8c-7141-4690-b179-33b9c10e1825.png)

