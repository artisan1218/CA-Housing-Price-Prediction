import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import plotly.express as px
from plotly.graph_objs import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from pickle import dump
from pickle import load

#===========================Helper Functions#===========================
def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)

# rmse metric for tensorflow nn model
def rmse(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))

def scale_data(data, type):
    if type=='air':
      scaler = load(open('/Users/yuqixiao/Desktop/DSCI-560/dashboard/scaler_file/air_scaler.pkl', 'rb'))
      np_data = np.array(data).reshape(-1, 1)
      scaled_columns = scaler.transform(np_data[:15].reshape(1, -1)) 
      pred = np.concatenate([scaled_columns, np_data[15:].reshape(1, -1)], axis=1)
    elif type=='oil':
      scaler = load(open('/Users/yuqixiao/Desktop/DSCI-560/dashboard/scaler_file/oil_scaler.pkl', 'rb'))
      np_data = np.array(data).reshape(-1, 1)
      scaled_columns = scaler.transform(np_data[:3].reshape(1, -1)) 
      pred = np.concatenate([scaled_columns, np_data[3:].reshape(1, -1)], axis=1)
    elif type=='water':
      scaler = load(open('/Users/yuqixiao/Desktop/DSCI-560/dashboard/scaler_file/water_scaler.pkl', 'rb'))
      np_data = np.array(data).reshape(-1, 1)
      scaled_columns = scaler.transform(np_data[:3].reshape(1, -1)) 
      pred = np.concatenate([scaled_columns, np_data[3:].reshape(1, -1)], axis=1)

    return pred
  

def predict(county_name, raw_year, air_pollution, oil_pollution, water_pollution, air_county, oil_county, water_county):
    final_result = []
    if county_name in air_county:
      # air model
      air_index = air_county.index(county_name)
      air_county_onehot = [0] * air_index + [1] + [0] * (len(air_county) - air_index - 1)
      year = 2022 - int(raw_year)
      air_input = scale_data([year] + list(air_pollution.values()) + air_county_onehot, 'air')
      air_result = air_model.predict(air_input)
      air_result = np.ravel(air_result)
      final_result.append(air_result[0])

    if county_name in oil_county:
      # oil model
      oil_index = oil_county.index(county_name)
      oil_county_onehot = [0] * oil_index + [1] + [0] * (len(oil_county) - oil_index - 1)
      oil_input = scale_data(list(oil_pollution.values()) + oil_county_onehot, 'oil') 
      oil_result = oil_model.predict(oil_input)
      oil_result = np.ravel(oil_result)
      final_result.append(oil_result[0])

    if county_name in water_county:
      # water model
      water_index = water_county.index(county_name)
      water_county_onehot = [0] * water_index + [1] + [0] * (len(water_county) - water_index - 1)
      water_input = scale_data(list(water_pollution.values()) + water_county_onehot + water_pollution_type, 'water') 
      water_result = water_model.predict(water_input)
      water_result = np.ravel(water_result)
      final_result.append(water_result[0])

    return sum(final_result) / len(final_result) 



st.set_page_config(layout="wide")

#load datasets
hp_df = load_dataset('/Users/yuqixiao/Desktop/DSCI-560/dashboard/data/housing_price_raw.csv')
hp_df['year'] = hp_df['Mon-Yr'].apply(lambda x: int(x[-4:]))

wq_df = load_dataset('/Users/yuqixiao/Desktop/DSCI-560/dashboard/data/water_quality.csv')
wq_df['Mon-Yr'] = wq_df['Month'] + '-' + wq_df['Year'].astype(str)
wq_df['Mon-Yr'] = pd.to_datetime(wq_df['Mon-Yr'], infer_datetime_format=True)

aq_df = load_dataset('/Users/yuqixiao/Desktop/DSCI-560/dashboard/data/air_quality.csv')

ow_df = load_dataset('/Users/yuqixiao/Desktop/DSCI-560/dashboard/data/oil_wells.csv')

st.markdown("<h1 style='text-align: center; color: black;'>California Housing Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: blue;'>Team Random</h4>", unsafe_allow_html=True)


Col1, Col2, Col3 = st.columns((2,1,1))
show_fig = False
Col1.subheader('Explore California Housing Price')
with Col2:
    county_selection = st.multiselect("Select county", list(hp_df['county'].unique()), key=1)
    show_fig = True if county_selection else False
with Col3:
    year_selection = st.select_slider('Select a time range', range(1990, 2023), value=(2000, 2020), key=1)

nCol1, nCol2 = st.columns(2)
with nCol1:
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    top_10_hp_county = hp_df[hp_df['year'] == 2021][['county', 'year', 'price']].groupby(['county', 'year']).mean().sort_values('price', ascending=False).reset_index()['county'][:10]
    fig1 = px.line(hp_df[hp_df['county'].isin(top_10_hp_county)], x='Mon-Yr', y='price', color='county')
    fig1.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis_showgrid': False, 
            'yaxis_showgrid': False,
            'title': {
                'text': "Top 10 Highest Housing Price in CA",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
            })
    fig1.update_xaxes(zeroline=False)
    st.plotly_chart(fig1, use_container_width=True)

with nCol2:
    if show_fig:
        hp_df_selection = hp_df[
            (hp_df['county'].isin(county_selection)) & 
            (hp_df['year'] >= year_selection[0]) & 
            (hp_df['year'] <= year_selection[1])
        ]
        st.markdown('')
        see_data = st.expander('You can click here to see the raw data 👉')
        with see_data:
            st.dataframe(data=hp_df_selection[['Mon-Yr', 'county', 'price']].reset_index(drop=True))

        fig2 = px.line(hp_df_selection, x="Mon-Yr", y="price", color='county')
        fig2.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis_showgrid': False, 
            'yaxis_showgrid': False
            })
        st.plotly_chart(fig2, use_container_width=True)


show_fig = False
Col1, Col2, Col3 = st.columns((2,1,1))
Col1.subheader('Explore California Water Quality')
with Col2:
    county_selection2 = st.selectbox("Select a county", list(wq_df['County'].unique()))
    show_fig = True
with Col3:
    if show_fig:
        wq_df_selection = wq_df[(wq_df['County'] == county_selection2) & (wq_df['Year'] >= year_selection[0]) & (wq_df['Year'] <= year_selection[1])]
        water_pollution_selection = st.selectbox("Select a water contamination", ['Select a water contamination'] + list(wq_df_selection['Parameter'].unique()))

nCol1, nCol2 = st.columns(2)   
with nCol1:
    if show_fig:
        top_10_water_pollution = wq_df_selection[['County', 'Parameter', 'Ratio']].groupby(['County', 'Parameter']).mean().reset_index().sort_values('Ratio', ascending=False).reset_index(drop=True)[:10]
        fig4 = px.bar(top_10_water_pollution, x="Ratio", y="Parameter", orientation='h')
        fig4.update_layout({
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'xaxis_showgrid': False, 
                'yaxis_showgrid': False,
                'title': {
                    'text': "Top 10 Excessive Pollutant",
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}
                })
        fig4.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)

with nCol2:
    if water_pollution_selection != 'Select a water contamination':
        wq_df_selection = wq_df_selection[wq_df_selection['Parameter'] == water_pollution_selection][['County', 'Year', 'Result', 'Limit']]
        wq_df_selection = wq_df_selection.groupby(['County', 'Year']).mean().reset_index()
        fig3 = px.line(wq_df_selection, x="Year", y="Result", color='County')
        fig3.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis_showgrid': False, 
            'yaxis_showgrid': False
            })
        fig3.add_hline(
            y=wq_df_selection['Limit'].mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text="threshold")
        st.plotly_chart(fig3, use_container_width=True)


Col1, Col2 = st.columns(2)
Col1.subheader('Explore California Air Quality')

nCol1, nCol2, nCol3, nCol4 = st.columns(4)
aq_df_selection = aq_df[(aq_df['County'].isin(county_selection)) & (aq_df['Year'] <= year_selection[1]) & (aq_df['Year'] >= year_selection[0])].reset_index(drop=True)
with nCol1:
    if show_fig:
        air_quality_selection1 = st.selectbox("Select an air quality measurement", list(aq_df.columns)[3:], key=1, index=9)
        fig5 = px.line(aq_df_selection[['County', 'Year', air_quality_selection1]], x="Year", y=air_quality_selection1, color='County', height=200)
        fig5.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis_showgrid': False, 
            'yaxis_showgrid': False,
            'yaxis_title': None,
            'showlegend': False,
            'margin': dict(
            l = 5,        # left
            r = 5,        # right
            t = 10,        # top
            b = 10,        # bottom
            )
            })
        # fig5.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig5, use_container_width=True)

with nCol2:
    if show_fig:
        air_quality_selection2 = st.selectbox("", list(aq_df.columns)[3:], key=2, index=10)
        fig6 = px.line(aq_df_selection[['County', 'Year', air_quality_selection2]], x="Year", y=air_quality_selection2, color='County', height=200)
        fig6.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis_showgrid': False, 
            'yaxis_showgrid': False,
            'yaxis_title': None,
            'showlegend': False,
            'margin': dict(
            l = 5,        # left
            r = 5,        # right
            t = 10,        # top
            b = 10,        # bottom
            )
            })
        # fig6.update_traces(line_color='#ff7f0e')
        st.plotly_chart(fig6, use_container_width=True)

with nCol3:
    if show_fig:
        air_quality_selection3 = st.selectbox("", list(aq_df.columns)[3:], key=3, index=12)
        fig7 = px.line(aq_df_selection[['County', 'Year', air_quality_selection3]], x="Year", y=air_quality_selection3, color='County', height=200)
        fig7.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis_showgrid': False, 
            'yaxis_showgrid': False,
            'yaxis_title': None,
            'showlegend': False,
            'margin': dict(
            l = 5,        # left
            r = 5,        # right
            t = 10,        # top
            b = 10,        # bottom
            )
            })
        # fig7.update_traces(line_color='#2ca02c')
        st.plotly_chart(fig7, use_container_width=True)

with nCol4:
    if show_fig:
        air_quality_selection4 = st.selectbox("", list(aq_df.columns)[3:], key=4, index=14)
        fig8 = px.line(aq_df_selection[['County', 'Year', air_quality_selection4]], x="Year", y=air_quality_selection4, color='County', height=200)
        fig8.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis_showgrid': False, 
            'yaxis_showgrid': False,
            'yaxis_title': None,
            'showlegend': True,
            'margin': dict(
            l = 5,        # left
            r = 5,        # right
            t = 10,        # top
            b = 10,        # bottom
            )
            })
        # fig8.update_traces(line_color='#9467bd')
        st.plotly_chart(fig8, use_container_width=True)


Col1, Col2, Col3 = st.columns((2,1,1))
Col1.subheader('Explore California Oil & Gas Wells')
with Col2:
    county_selection3 = st.multiselect('Select county', ow_df['CountyName'].unique())
    ow_df_select = ow_df[ow_df['CountyName'].isin(county_selection3)]
with Col3:
    well_year = st.slider('Select a year', 1990, 2022, 2000)
    ow_df_select = ow_df_select[ow_df_select['SpudYear'] <= well_year]

nCol1, nCol2 = st.columns(2)
with nCol1:
    top_10_wells = ow_df.groupby('CountyName').count()['WellStatus'].reset_index().sort_values('WellStatus', ascending=False).reset_index(drop=True)[:10]
    fig9 = px.bar(top_10_wells, x="WellStatus", y="CountyName", orientation='h')
    fig9.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis_showgrid': False, 
            'yaxis_showgrid': False,
            'title': {
                'text': "Top 10 County with Most Gas Wells",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
            })
    fig9.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig9, use_container_width=True)

with nCol2:
    st.map(ow_df_select)



#=======================Forecasting#=======================

Col1, Col2 = st.columns(2)
Col1.subheader('Housing Price Forecast')


air_model = tf.keras.models.load_model('/Users/yuqixiao/Desktop/DSCI-560/dashboard/model_file/LSTM_air.h5', custom_objects={"rmse": rmse})
oil_model = tf.keras.models.load_model('/Users/yuqixiao/Desktop/DSCI-560/dashboard/model_file/LSTM_oil.h5', custom_objects={"rmse": rmse})
water_model = tf.keras.models.load_model('/Users/yuqixiao/Desktop/DSCI-560/dashboard/model_file/LSTM_water.h5', custom_objects={"rmse": rmse})

air_county = ['Alameda', 'Amador', 'Butte', 'Calaveras', 'Del Norte', 'El Dorado',
              'Fresno', 'Glenn', 'Humboldt', 'Kern', 'Kings', 'Lake', 'Lassen',
              'Los Angeles', 'Madera', 'Marin', 'Mariposa', 'Mendocino', 'Merced',
              'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Plumas',
              'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 'San Diego',
              'San Francisco', 'San Joaquin', 'San Luis Obispo', 'San Mateo',
              'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta', 'Siskiyou',
              'Solano', 'Sonoma', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare',
              'Tuolumne', 'Ventura', 'Yolo']

oil_county = ['Alameda','Amador', 'Butte', 'Del Norte', 'Fresno', 'Glenn', 'Humboldt', 'Kern',
              'Kings', 'Lake', 'Lassen', 'Los Angeles', 'Madera', 'Marin',
              'Mendocino', 'Merced', 'Mono', 'Monterey', 'Napa', 'Orange', 'Placer',
              'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 'San Diego',
              'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara',
              'Santa Clara', 'Santa Cruz', 'Shasta', 'Siskiyou', 'Solano', 'Sonoma',
              'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Ventura', 'Yolo', 'Yuba']

water_county = ['Alameda', 'Amador', 'Butte', 'Calaveras','Del Norte',
                'El Dorado','Fresno','Glenn','Humboldt','Kern','Kings',
                'Lake','Lassen','Los Angeles','Marin','Mendocino','Merced',
                'Mono','Monterey','Napa','Nevada','Placer','Plumas',
                'Riverside','Sacramento','San Benito','San Bernardino',
                'San Diego','San Joaquin','San Luis Obispo','Santa Barbara',
                'Santa Clara','Santa Cruz','Shasta','Siskiyou','Solano','Sonoma',
                'Stanislaus','Sutter','Tehama','Tulare','Tuolumne','Ventura',
                'Yolo','Yuba']

water_pollution_list = ['(Aminomethyl)phosphonic acid',
                        '1,1,1,2-Tetrachloroethane',
                        '1,1,1-Trichloroethane',
                        '1,1,2,2-Tetrachloroethane',
                        '1,1,2-Trichloroethane',
                        '1,1,2-Trichlorotrifluoroethane',
                        '1,1-Dichloroethane',
                        '1,1-Dichloroethene',
                        '1,1-Dichloropropene',
                        '1,2,3-Trichlorobenzene',
                        '1,2,3-Trichloropropane',
                        '1,2,4-Trichlorobenzene',
                        '1,2,4-Trimethylbenzene',
                        '1,2-Dibromo-3-chloropropane (DBCP)',
                        '1,2-Dibromoethane (EDB)',
                        '1,2-Dichlorobenzene',
                        '1,2-Dichloroethane',
                        '1,2-Dichloropropane',
                        '1,3,5-Trimethylbenzene',
                        '1,3-Dichlorobenzene',
                        '1,3-Dichloropropane',
                        '1,4-Dichlorobenzene',
                        '1-Naphthol',
                        '2,2-Dichloropropane',
                        '2,3,7,8-Tetrachlorodibenzo-p-dioxin',
                        '2,3-Dibromopropionic acid',
                        '2,4,5-T',
                        '2,4,5-TP (Silvex)',
                        '2,4-D',
                        '2,4-DB',
                        '2-Chlorotoluene',
                        '3,5-Dichlorobenzoic Acid',
                        '3-Hydroxycarbofuran',
                        "4,4'-Dibromoctafluorobiphenyl",
                        '4-Bromofluorobenzene',
                        '4-Chlorotoluene',
                        '4-Isopropyltoluene',
                        '4-Nitrophenol',
                        '5-Day Biochemical Oxygen Demand',
                        'Acenaphthene',
                        'Acenaphthylene',
                        'Acephate',
                        'Acifluorfen',
                        'Alachlor',
                        'Aldicarb',
                        'Aldicarb sulfone',
                        'Aldicarb sulfoxide',
                        'Aldrin',
                        'Alkalinity',
                        'Allethrin',
                        'Aminomethylphosphonic Acid (AMPA)',
                        'Anthracene',
                        'Ash Free Dry Mass',
                        'Atrazine',
                        'Azinphos methyl (Guthion)',
                        'BHC-alpha',
                        'BHC-beta',
                        'BHC-delta',
                        'BHC-gamma (Lindane)',
                        'Benfluralin',
                        'Bentazon',
                        'Benz(a)anthracene',
                        'Benzene',
                        'Benzo(a)pyrene',
                        'Benzo(b)fluoranthene',
                        'Benzo(ghi)perylene',
                        'Benzo(k)fluoranthene',
                        'Bicarbonate (HCO3-)',
                        'Bifenthrin',
                        'Biochemical Oxygen Demand',
                        'Bis(2-ethylhexyl) adipate',
                        'Bolstar',
                        'Bromacil',
                        'Bromide',
                        'Bromobenzene',
                        'Bromochloroacetic Acid (BCAA)',
                        'Bromochloroacetonitrile',
                        'Bromochloromethane',
                        'Bromodichloromethane',
                        'Bromoform',
                        'Bromomethane',
                        'Butachlor',
                        'Captan',
                        'Carbaryl',
                        'Carbofuran',
                        'Carbon tetrachloride',
                        'Carbonaceous Biochemical Oxygen Demand',
                        'Carbophenothion (Trithion)',
                        'Chlordane',
                        'Chlorobenzene',
                        'Chloroethane',
                        'Chloroform',
                        'Chloromethane',
                        'Chlorophyll a',
                        'Chloropicrin',
                        'Chloropropham',
                        'Chlorothalonil',
                        'Chlorpropham',
                        'Chlorpyrifos',
                        'Chorpropham',
                        'Chrysene',
                        'Clostridium perfringens',
                        'Color',
                        'Conductance',
                        'Coumaphos',
                        'Cryptosporidium',
                        'Cyanazine',
                        'Cyfluthrin',
                        'Cypermethrin',
                        'DDD (all isomers)',
                        'DDE (all isomers)',
                        'DDT (all isomers)',
                        'Dacthal (DCPA)',
                        'Dalapon',
                        'Deltamethrin/Tralomethrin',
                        'Demeton (Demeton O + Demeton S)',
                        'Demeton-O',
                        'Demeton-S',
                        'Diazinon',
                        'Dibenz(a,h)anthracene',
                        'Dibromoacetic Acid (DBAA)',
                        'Dibromoacetonitrile',
                        'Dibromochloromethane',
                        'Dibromomethane',
                        'Dicamba',
                        'Dichloran',
                        'Dichloroacetic Acid (DCAA)',
                        'Dichloroacetonitrile',
                        'Dichlorodifluoromethane',
                        'Dichloroprop',
                        'Dichlorprop',
                        'Dichlorvos',
                        'Dicofol',
                        'Dieldrin',
                        'Dimethoate',
                        'Dinoseb (DNPB)',
                        'Diquat',
                        'Dissolved Acid Hydrolyzable Phosphorus',
                        'Dissolved Aluminum',
                        'Dissolved Ammonia',
                        'Dissolved Antimony',
                        'Dissolved Arsenic',
                        'Dissolved Barium',
                        'Dissolved Beryllium',
                        'Dissolved Bicarbonate (HCO3-)',
                        'Dissolved Boron',
                        'Dissolved Bromide',
                        'Dissolved Cadmium',
                        'Dissolved Calcium',
                        'Dissolved Carbonate (CO3--)',
                        'Dissolved Chloride',
                        'Dissolved Chromium',
                        'Dissolved Chromium, hexavalent (Cr6+)',
                        'Dissolved Cobalt',
                        'Dissolved Copper',
                        'Dissolved Fluoride',
                        'Dissolved Hardness',
                        'Dissolved Hydroxide (OH-)',
                        'Dissolved Iron',
                        'Dissolved Lead',
                        'Dissolved Lithium',
                        'Dissolved Magnesium',
                        'Dissolved Manganese',
                        'Dissolved Mercury',
                        'Dissolved Methylmercury',
                        'Dissolved Molybdenum',
                        'Dissolved Nickel',
                        'Dissolved Nitrate',
                        'Dissolved Nitrate + Nitrite',
                        'Dissolved Nitrite',
                        'Dissolved Organic Carbon',
                        'Dissolved Organic Nitrogen',
                        'Dissolved Phosphorus',
                        'Dissolved Potassium',
                        'Dissolved Selenium',
                        'Dissolved Silica (SiO2)',
                        'Dissolved Silver',
                        'Dissolved Sodium',
                        'Dissolved Strontium',
                        'Dissolved Sulfate',
                        'Dissolved Thallium',
                        'Dissolved Total Kjeldahl Nitrogen',
                        'Dissolved Vanadium',
                        'Dissolved Zinc',
                        'Dissolved ortho-Phosphate',
                        'Disulfoton',
                        'Diuron',
                        'Endosulfan sulfate',
                        'Endosulfan-I',
                        'Endosulfan-II',
                        'Endothal',
                        'Endrin',
                        'Endrin aldehyde',
                        'Escherichia coli',
                        'Esfenvalerate',
                        'Ethion',
                        'Ethoprop',
                        'Ethyl benzene',
                        'Ethylbenzene',
                        'Ethylene Dibromide',
                        'Ethylenethiourea',
                        'Fecal Coliform',
                        'Fensulfothion',
                        'Fenthion',
                        'Fenvalerate',
                        'Fluoranthene',
                        'Fluorene',
                        'Formetanate hydrochloride',
                        'Giardia lamblia',
                        'Glyphosate',
                        'Haloacetic Acid Formation Potential (HAAFP)',
                        'Heptachlor',
                        'Heptachlor epoxide',
                        'Hexachlorobenzene',
                        'Hexachlorobutadiene',
                        'Hexachlorocyclopentadiene',
                        'Indeno(1,2,3-cd)pyrene',
                        'Inositol 1,4,5-Trisphosphate',
                        'Isopropylbenzene',
                        'MCPA',
                        'MCPP',
                        'Malathion',
                        'Merphos',
                        'Methidathion',
                        'Methiocarb',
                        'Methomyl',
                        'Methoxychlor',
                        'Methyl tert-butyl ether (MTBE)',
                        'Methylene Blue Active Substances (MBAS)',
                        'Methylene chloride',
                        'Metolachlor',
                        'Metribuzin',
                        'Mevinphos',
                        'Molinate',
                        'Monobromoacetic Acid (MBAA)',
                        'Monochloroacetic Acid (MCAA)',
                        'Naled',
                        'Naphthalene',
                        'Napropamide',
                        'Norflurazon',
                        'Oil and Grease',
                        'Oxamyl',
                        'Oxyfluorfen',
                        "PCB's",
                        'PCB-1016',
                        'PCB-1221',
                        'PCB-1232',
                        'PCB-1242',
                        'PCB-1248',
                        'PCB-1254',
                        'PCB-1260',
                        'Parathion, Ethyl',
                        'Parathion, Methyl',
                        'Pendimethalin',
                        'Pentachloronitrobenzene (PCNB)',
                        'Pentachlorophenol (PCP)',
                        'Permethrin',
                        'Phenanthrene',
                        'Phenol',
                        'Pheophytin a',
                        'Phorate',
                        'Phosalone',
                        'Phosmet',
                        'Picloram',
                        'Prallethrin',
                        'Profenofos',
                        'Prometryn',
                        'Propachlor',
                        'Propargite',
                        'Propetamphos',
                        'Propoxur',
                        'Pyrene',
                        'Ronnel',
                        'Settleable Solids',
                        'Simazine',
                        'Soil Solids',
                        'Specific Conductance',
                        'Styrene',
                        'Sumithrin',
                        'Suspended + Volatile Suspended Solids',
                        'Suspended Solids',
                        'Tefluthrin',
                        'Tetrachloroethene',
                        'Tetrachloroethylene',
                        'Tetrachlorofluoromethane',
                        'Tetrachlorvinphos',
                        'Thiobencarb',
                        'Thionazine (Zinophos)',
                        'Tokuthion',
                        'Toluene',
                        'Total Alkalinity',
                        'Total Aluminum',
                        'Total Ammonia',
                        'Total Antimony',
                        'Total Arsenic',
                        'Total Asbestos, Chrysotile',
                        'Total Barium',
                        'Total Beryllium',
                        'Total Boron',
                        'Total Cadmium',
                        'Total Calcium',
                        'Total Chromium',
                        'Total Cobalt',
                        'Total Coliform',
                        'Total Copper',
                        'Total Cyanide',
                        'Total Dissolved Solids',
                        'Total Hardness',
                        'Total Iron',
                        'Total Kjeldahl Nitrogen',
                        'Total Lead',
                        'Total Lithium',
                        'Total Magnesium',
                        'Total Manganese',
                        'Total Mercury',
                        'Total Methylmercury',
                        'Total Molybdenum',
                        'Total Nickel',
                        'Total Oil and Grease',
                        'Total Organic Carbon',
                        'Total Organic Nitrogen',
                        'Total Parathion, ethyl & methyl',
                        'Total Phosphorus',
                        'Total Potassium',
                        'Total Selenium',
                        'Total Silica (SiO2)',
                        'Total Silver',
                        'Total Sodium',
                        'Total Strontium',
                        'Total Suspended Solids',
                        'Total Thallium',
                        'Total Vanadium',
                        'Total Xylene, (total)',
                        'Total Zinc',
                        'Total ortho-Phosphate',
                        'Toxaphene',
                        'Trichloroacetic Acid (TCAA)',
                        'Trichloroacetonitrile',
                        'Trichloroethene',
                        'Trichlorofluoromethane',
                        'Trichloronate',
                        'Triclopyr',
                        'Trifluralin',
                        'Trihalomethane Formation Potential (THMFP)',
                        'Turbidity',
                        'UV Absorbance @254nm',
                        'Vinyl chloride',
                        'Volatile Solids',
                        'Volatile Suspended Solids',
                        'Xylene, (total)',
                        'bis(2-Ethylhexyl) phthalate',
                        'cis-1,2-Dichloroethene',
                        'cis-1,3-Dichloropropene',
                        'l-Cyhalothrin',
                        'm + p Xylene',
                        'm-Xylene',
                        'n-Butylbenzene',
                        'n-Propylbenzene',
                        "o,p'-DDE",
                        'o-Xylene',
                        "p,p'-DDD",
                        "p,p'-DDE",
                        "p,p'-DDT",
                        'p-Xylene',
                        'pH',
                        's,s,s-Tributyl Phosphorotrithioate (DEF)',
                        'sec-Butylbenzene',
                        'tert-Butylbenzene',
                        'trans-1,2-Dichloroethene',
                        'trans-1,3-Dichloropropene']

with st.form('Form'):
    year = st.number_input('Year', value=2022)
    county = st.selectbox('Select a county', ['Select a county'] + list(set(air_county + oil_county + water_county)))
    aq = st.selectbox('Select air quality', ['Select air quality', 'Good', 'Moderate', 'Bad'])
    wq = st.multiselect('Select water quality', ['All'] + water_pollution_list)
    if wq == 'All':
        wq = water_pollution_list

    submitted = st.form_submit_button('Forecast')


if aq == 'Bad':
    good_days = 365 
    moderate_days = 0
    unhealthy_days_sens = 0
    unhealthy_days = 0
    very_unhealthy_days = 0
    hazard_days = 0
    co_days = 2
    median_aqi = 15
    no2_days = 10
    ozone_days = 10
    so2_days = 10
    pm2_days = 10
    pm10_days = 10
elif aq == 'Moderate':
    good_days = 0
    moderate_days = 365
    unhealthy_days_sens = 0
    unhealthy_days = 0
    very_unhealthy_days = 0
    hazard_days = 0
    co_days = 100
    median_aqi = 100
    no2_days = 100
    ozone_days = 100
    so2_days = 100
    pm2_days = 100
    pm10_days = 100
else:
    good_days = 0
    moderate_days = 0
    unhealthy_days_sens = 0
    unhealthy_days = 365
    very_unhealthy_days = 0
    hazard_days = 0
    co_days = 300
    median_aqi = 300
    no2_days = 300
    ozone_days = 300
    so2_days = 300
    pm2_days = 300
    pm10_days = 300


# above should sum up 

price_age = 2022 - year

air_pollution = {
    'Good Days': good_days,
    'Moderate Days': moderate_days,
    'Unhealthy for Sensitive Groups Days': unhealthy_days_sens,
    'Unhealthy Days': unhealthy_days,
    'Very Unhealthy Days': very_unhealthy_days,
    'Hazardous Days': hazard_days,
    'Median AQI': median_aqi,
    'Days CO': co_days,
    'Days NO2': no2_days,
    'Days Ozone': ozone_days,
    'Days SO2': so2_days,
    'Days PM2.5': pm2_days, 
    'Days PM10': pm10_days,
    'house_price_age': price_age 
}

# -----------------------------------------------------------------------

spudage = 50 # 29 to 102
well_ratio = 0.1 # 0 to 0.3

oil_pollution = {
    'SpudAge': spudage,
    'ActiveWellRatio': well_ratio,
    'house_price_age': price_age
}

# ------------------------------------------------------------------------------------

result = 300 # -1 to 5000000.0
polluted = 1 # 0 or 1
selected_pollution = wq # select from water_pollution_list

water_pollution = {
    'Result': result,
    'Polluted': polluted,
    'house_price_age': price_age
}

water_pollution_index = dict()
for idx, name in enumerate(water_pollution_list):
    water_pollution_index[name] = idx

water_pollution_type = [0] * len(water_pollution_list)
for pollution in selected_pollution:
    water_pollution_type[water_pollution_index[pollution]] = 1


if submitted:
    pred_price = round(predict(county, year, air_pollution, oil_pollution, water_pollution, air_county, oil_county, water_county))
    st.success('Predicted price of {} in {} is $ {}'.format(county, year, str(pred_price)))
