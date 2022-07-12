from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.parsers import JSONParser

import json
import os
import pandas as pd
import datetime
import numpy as np
from pycaret.regression import *
import plotly.express as px



# Create your views here.

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_path = os.path.join(directory,'prediction_src/')

def index(request):
    result  = { }
    return JsonResponse(result,safe=False)

@api_view(["GET"])
def get_time_series_data(request):
    
    if settings.MODEL_LOAD_FLAG == "False":        
        graph_dic = {}
        #Time Series Analysis
        #load contact data
        contact_df = pd.read_csv(source_path+'TransactionDataMay24Contact.csv')
        #load transaction data
        trans_df = pd.read_csv(source_path+'TransactionDataMay24Trans.csv')

        #load location based clv model
        clv_model = load_model(source_path+'restaurant_loc_clv_model')

        #load time series model
        ts_clv_model = load_model(source_path+'restaurant_time_series_model_v_1')

        #actual dataset start & end time
        ts_act_start = '2021-05-01 00:00:00'
        ts_act_end = '2022-04-30 00:00:00'

        #future forecast start & end time
        ts_prediction_start = '2021-05-01'
        ts_prediction_end = '2022-05-30'


        #executive summary filter from date & to date
        act_start = ''#'2022-01-01 00:00:00'
        act_end = ''#'2022-04-30 00:00:00'
        year = 2021
        month = 0
        custom_filter = True
        is_filter = False
        print("*****************************************************************************************")
        print(act_start)
        print(act_end)
        print("*****************************************************************************************")
        prev_act_start = ''#'2021-01-01 00:00:00'
        prev_act_end = ''#'2021-04-30 00:00:00'
        prev_year = 0
        prev_month = 0
        prev_clv_not_found = False
        

        settings.TRANSACTION_DATAFRAME = trans_df.copy()
        #thersold date
        thersold_date_time = trans_df['OrderDate'].max()
        tmp_date = thersold_date_time.split(" ")
        thersold_date = tmp_date[0]

        settings.THERSOLD_MAX_DATE = thersold_date

        #future forecast time frequency
        ts_prediction_freq = 'D' #As Days

        contact_df.drop(['PostCode'],axis=1,inplace=True)
        contact_df.drop(['Age'],axis=1,inplace=True)
        contact_df.drop(['BirthDate'],axis=1,inplace=True)

        #remove null & duplicate values
        contact_df = contact_df.dropna()
        contact_df = contact_df.drop_duplicates()


        #convert date column into datetime formate
        trans_df['OrderDate'] = trans_df['OrderDate'].astype(str)
        trans_df['OrderDate'] = pd.to_datetime(trans_df.OrderDate, format='%Y-%m-%d')
        trans_df['OrderDate'] = trans_df['OrderDate'].dt.date


        #create future forecast request data
        future_dates = pd.date_range(start = ts_prediction_start, end = ts_prediction_end, freq = ts_prediction_freq)
        future_df = pd.DataFrame()

        # extract date, month and year from dates
        future_df['Month'] = [i.month for i in future_dates]
        future_df['Year'] = [i.year for i in future_dates]
        future_df['day_of_week'] = [i.dayofweek for i in future_dates]
        future_df['day_of_year'] = [i.dayofyear for i in future_dates]

        #predict the data using timeseries model
        predictions_future = predict_model(ts_clv_model, data=future_df)

        #Pre-Process the actual data
        ts_trans_df = trans_df.copy()
        ts_trans_df.drop(['Source'],axis=1,inplace=True)
        ts_trans_df.drop(['Location'],axis=1,inplace=True)
        ts_trans_df = ts_trans_df.dropna()
        ts_trans_df = ts_trans_df.drop_duplicates()


        startDate = pd.to_datetime(ts_act_start).date()
        ts_trans_df.drop(ts_trans_df[ts_trans_df['OrderDate'] < startDate].index, inplace = True)

        endDate = pd.to_datetime(ts_act_end).date()
        ts_trans_df.drop(ts_trans_df[ts_trans_df['OrderDate'] > endDate].index, inplace = True)

        #remove negative transaction value
        ts_trans_df.drop(ts_trans_df[ts_trans_df['Transactional Value'] <= 0].index, inplace = True)

        #group dataset by date
        ts_trans_df = ts_trans_df.groupby(['OrderDate']).agg({"Transactional Value" : np.sum}).reset_index()

        #convert date column into datetime formate
        ts_trans_df['OrderDate'] = pd.to_datetime(ts_trans_df['OrderDate'])

        # extract date, month and year from dates
        ts_trans_df['Month'] = [i.month for i in ts_trans_df['OrderDate']]
        ts_trans_df['Year'] = [i.year for i in ts_trans_df['OrderDate']]
        ts_trans_df['day_of_week'] = [i.dayofweek for i in ts_trans_df['OrderDate']]
        ts_trans_df['day_of_year'] = [i.dayofyear for i in ts_trans_df['OrderDate']]

        #remove date column
        ts_trans_df.drop("OrderDate", axis=1, inplace=True)

        #concatenate actual and future data
        concat_df = pd.concat([ts_trans_df,predictions_future], axis=0)

        ts_trans_df_tmp = ts_trans_df.groupby(['Year','day_of_year']).agg({"Transactional Value" : np.sum}).cumsum().reset_index()
        ts_trans_df_tmp['Transactional Value'] = round(ts_trans_df_tmp['Transactional Value'],2)
        ts_trans_df_tmp.insert(0, 'days', range(1, 1 + len(ts_trans_df_tmp)))
        
        ts_trans_df_tmp = pd.merge(ts_trans_df_tmp, ts_trans_df_tmp, on=['Year','day_of_year'])
        ts_trans_df_tmp.insert(0, 'days', range(1, 1 + len(ts_trans_df_tmp)))
        

        predictions_future_tmp = predictions_future.groupby(['Year','day_of_year']).agg({"Label" : np.sum}).cumsum().reset_index()
        predictions_future_tmp['Label'] = round(predictions_future_tmp['Label'],2)
        predictions_future_tmp.insert(0, 'days', range(1, 1 + len(predictions_future_tmp)))

        predictions_future_tmp = pd.merge(predictions_future_tmp, predictions_future_tmp, on=['Year','day_of_year'])
        predictions_future_tmp.insert(0, 'days', range(1, 1 + len(predictions_future_tmp)))
        


        day_of_year_list = json.loads(predictions_future_tmp["days_x"].to_json(orient = 'records'))        
        prediction_val_list = json.loads(predictions_future_tmp["Label_y"].to_json(orient = 'records'))
        transaction_val_list = json.loads(ts_trans_df_tmp["Transactional Value_y"].to_json(orient = 'records'))
                      

        graph_dic["transaction_val"] = transaction_val_list
        graph_dic["prediction_val"] = prediction_val_list        
        graph_dic["day_of_year"] = day_of_year_list        
        
        #Prediction Analysis        
                
        loc_trans_df = trans_df.copy()

        #pre-process the actual dataset
        loc_trans_df.drop(['Source'],axis=1,inplace=True)
        #drop nan values
        loc_trans_df = loc_trans_df.dropna()
        loc_trans_df = loc_trans_df.drop_duplicates()
        loc_trans_df['no_visit'] = 1
        loc_trans_df['customer_age'] = pd.to_datetime(loc_trans_df['OrderDate']).dt.date
        loc_trans_df['days_since_last_visit'] = pd.to_datetime(loc_trans_df['OrderDate']).dt.date
        loc_trans_df['recency'] = pd.to_datetime(loc_trans_df['OrderDate']).dt.date

        #add thersold date
        loc_trans_df['thersold_date'] = thersold_date
        loc_trans_df['thersold_date'] = pd.to_datetime(loc_trans_df['thersold_date']).dt.date
        thersoldDate = pd.to_datetime(thersold_date_time).date()

        #group data by ID & Location
        loc_trans_df = loc_trans_df.groupby(['CtcID','Location']).agg({"no_visit" : np.sum, "Transactional Value" : np.mean, 'customer_age' : lambda x: (thersoldDate - x.min()),'recency' : lambda x: (x.max() - x.min()), 'days_since_last_visit' : lambda x: (thersoldDate - x.max())}).reset_index()

        #convert date column to int
        loc_trans_df['customer_age'] = loc_trans_df['customer_age'].dt.days.astype('int64')
        loc_trans_df['recency'] = loc_trans_df['recency'].dt.days.astype('int64')
        loc_trans_df['days_since_last_visit'] = loc_trans_df['days_since_last_visit'].dt.days.astype('int64')

        #predict the future 3 month target by using saved model
        prediction_clv = predict_model(clv_model,data = loc_trans_df)

        #add, rename and remove the column
        prediction_clv['Total_Transaction'] = prediction_clv['Transactional Value']
        prediction_clv['frequency'] = round(prediction_clv['recency'] / prediction_clv['no_visit'] , 3)
        prediction_clv = prediction_clv.rename(columns={"Label" : "CLV_3M"})

        #location based avg CLV of All customer
        prediction_cus_clv = prediction_clv.groupby(['Location']).agg({"CLV_3M" : np.mean,"Transactional Value" : np.mean,"Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()

        #add, rename and remove the column
        prediction_cus_clv['Total_Transaction'] = round(prediction_cus_clv['Total_Transaction'],3)
        prediction_cus_clv = prediction_cus_clv.rename(columns={"Transactional Value" : "AVG_Transaction"})
        prediction_cus_clv = prediction_cus_clv.rename(columns={"CtcID" : "Total_Customer"})
        prediction_cus_clv = prediction_cus_clv.rename(columns={"CLV_3M" : "AVG_CLV_3M"})


        #CLV of loyal customer
        prediction_loyal_clv = prediction_clv.loc[(prediction_clv.no_visit >= 3)]

        prediction_loyal_clv = prediction_loyal_clv.groupby(['Location']).agg({"CLV_3M" : np.mean,"Transactional Value" : np.mean,"Total_Transaction" : np.sum,"CtcID" : np.size}).reset_index()

        #add, rename and remove the column
        prediction_loyal_clv = prediction_loyal_clv.rename(columns={"CtcID" : "No_Loyal_Cus"})
        prediction_loyal_clv = prediction_loyal_clv.rename(columns={"CLV_3M" : "AVG_Loyal_CLV_3M"})
        prediction_loyal_clv = prediction_loyal_clv.rename(columns={"Transactional Value" : "AVG_Loyal_Transaction"})
        prediction_loyal_clv = prediction_loyal_clv.rename(columns={"Total_Transaction" : "Total_Loyal_Transaction"})

        #merge the results (All + Loyal)
        prediction_cus_clv = pd.merge(prediction_cus_clv, prediction_loyal_clv, on=["Location"])

        #loyal customer %
        prediction_cus_clv['Loyal_Cus_Percent'] = round(prediction_cus_clv['No_Loyal_Cus'] / prediction_cus_clv['Total_Customer'],3)*100

        #CLV of new customer
        prediction_new_clv = prediction_clv.loc[(prediction_clv.no_visit == 1)]

        prediction_new_clv = prediction_new_clv.groupby(['Location']).agg({"CLV_3M" : np.mean,"Transactional Value" : np.mean,"Total_Transaction" : np.sum,"CtcID" : np.size}).reset_index()

        #add, rename and remove the column
        prediction_new_clv = prediction_new_clv.rename(columns={"CtcID" : "No_New_Cus"})
        prediction_new_clv = prediction_new_clv.rename(columns={"CLV_3M" : "AVG_New_CLV_3M"})
        prediction_new_clv = prediction_new_clv.rename(columns={"Transactional Value" : "AVG_New_Transaction"})
        prediction_new_clv = prediction_new_clv.rename(columns={"Total_Transaction" : "Total_New_Transaction"})

        #merge the results (All + Loyal + New)
        prediction_cus_clv = pd.merge(prediction_cus_clv, prediction_new_clv, on=["Location"])

        #CLV of repeat customer
        prediction_repeat_clv = prediction_clv.loc[(prediction_clv.no_visit > 1)]

        prediction_repeat_clv = prediction_repeat_clv.groupby(['Location']).agg({"CLV_3M" : np.mean,"Transactional Value" : np.mean,"Total_Transaction" : np.sum,"CtcID" : np.size}).reset_index()

        #add, rename and remove the column
        prediction_repeat_clv = prediction_repeat_clv.rename(columns={"CtcID" : "No_Repeat_Cus"})
        prediction_repeat_clv = prediction_repeat_clv.rename(columns={"CLV_3M" : "AVG_Repeat_CLV_3M"})
        prediction_repeat_clv = prediction_repeat_clv.rename(columns={"Transactional Value" : "AVG_Repeat_Transaction"})
        prediction_repeat_clv = prediction_repeat_clv.rename(columns={"Total_Transaction" : "Total_Repeat_Transaction"})

        #merge the results (All + Loyal + New + Repeat)
        prediction_cus_clv = pd.merge(prediction_cus_clv, prediction_repeat_clv, on=["Location"])

        #GPS Location
        gps_data = [['Brasserie Blanc Bath', round(51.470632100409205,3), round(-2.2437504810404594,3)], 
                    ['Brasserie Blanc Beaconsfield',round(51.60555049378384,3), round(-0.6232441772891227,3)], 
                    ['Brasserie Blanc Bournemouth',round(50.71542477599034,3), round(-1.8823695557218925,3)],
                    ['Brasserie Blanc Bristol',round(51.45735146391432,3), round(-2.587050672941342,3)],
                ['Brasserie Blanc Chancery Lane',round(51.514551046856994,3), round(-0.11096812243497467,3)],
                ['Brasserie Blanc Cheltenham',round(51.896524401044026,3), round(-2.0801534391442793,3)],
                ['Brasserie Blanc Chichester',round(50.836870662861244,3), round(-0.7732422764306373,3)],
                ['Brasserie Blanc Hale Barns',round(53.36401268187584,3), round(-2.297013832745419,3)],
                ['Brasserie Blanc Knutsford',round(53.30483997510202,3), round(-2.3732921112251844,3)],
                ['Brasserie Blanc Leeds',round(53.79339573894752,3), round(-1.5438611828324094,3)],
                ['Brasserie Blanc Milton Keynes',round(52.037017847390516,3), round(-0.7623994657914837,3)],
                ['Brasserie Blanc Oxford',round(51.761050422119354,3), round(-1.2671100539495332,3)],
                ['Brasserie Blanc Portsmouth',round(50.79550523243025,3), round(-1.1029575324825527,3)],
                ['Brasserie Blanc Southbank',round(51.50578329682557,3), round(-0.11441317955228329,3)],
                ['Brasserie Blanc Threadneedle Street',round(51.51437604018978,3), round(-0.08581769785899936,3)],
                ['Brasserie Blanc Tower of London',round(51.51246352311413,3), round(-0.07109553035904186,3)],
                ['Brasserie Blanc Winchester',round(51.064856996055184,3), round(-1.315253933507015,3)],
                ['Fulham Reach',round(51.48752469290043,3), round(-0.22612417944571334,3)],
                ['Reigate',round(51.23736456284525,3), round(-0.20570696737151656,3)],
                ['The Barley Mow',round(51.244089991503735,3), round(-0.005469128619243311,3)],
                ['The Black Horse - Reigate',round(51.240461484451906,3), round(-0.22025829727803417,3)],
                ['The Black Horse - Thame',round(51.74684094634539,3), round(-0.9768882083396363,3)],
                ['The Boot',round(51.970287947998806,3), round(-0.8552062617633214,3)],
                ['The British Queen',round(52.13076814763469,3), round(0.24560552947044778,3)],
                ['The Cricketers',round(51.02965610965665,3), round(-0.9584448650060563,3)],
                ['The Hare',round(51.778125180549445,3), round(0.3823966403305349,3)],
                ['The Highwayman',round(52.23782882511647,3), round(-0.1557309467433792,3)],
                ["The Jobber's Rest",round(51.55787567264048,3), round(0.2736065515794826,3)],
                ['The Jolly Farmer',round(51.65487088641414,3), round(-0.10703228880398843,3)],
                ["The King's Arms",round(51.80696540817324,3), round(-0.4918656387015799,3)],
                ['The Kings Head',round(51.46988818132948,3), round(-0.5213254310719632,3)],
                ['The March Hare',round(51.277723926185146,3), round(-0.5114192553269904,3)],
                ['The Oaks',round(51.670135037891534,3), round(-0.49660080081038094,3)],
                ['The Oakwood',round(51.42894087992989,3), round(-0.9351928814517441,3)],
                ['The Queens Head',round(51.3846917726277,3), round(-0.5603681779193056,3)],
                ['The Red Deer',round(51.062929690860294,3), round(-0.3261965821938817,3)],
                ['The Sun Inn',round(51.37183581259531,3), round(-0.4684112326011627,3)],
                ['The Victoria',round(51.506568292823005,3), round(-0.9446416737091597,3)],
                ['The White Bear',round(51.54609734269894,3), round(-0.025242128037043843,3)]]

        gps_df = pd.DataFrame(gps_data, columns=['Location','Latitude', 'Longitude'])

        #merge the results (All + Loyal + New + Repeat)
        prediction_cus_clv = pd.merge(prediction_cus_clv, gps_df, on=["Location"])

        graph_dic["loc_total_transaction"] = json.loads(prediction_cus_clv["Total_Transaction"].to_json(orient = 'records'))
        graph_dic["loc_avg_clv"] = json.loads(prediction_cus_clv["AVG_CLV_3M"].to_json(orient = 'records'))
        graph_dic["loc_avg_loyal_clv"] = json.loads(prediction_cus_clv["AVG_Loyal_CLV_3M"].to_json(orient = 'records'))
        graph_dic["loc_loyal_cus_percentage"] = json.loads(prediction_cus_clv["Loyal_Cus_Percent"].to_json(orient = 'records'))
        graph_dic["no_new_cus"] = json.loads(prediction_cus_clv["No_New_Cus"].to_json(orient = 'records'))
        graph_dic["no_repeat_cus"] = json.loads(prediction_cus_clv["No_Repeat_Cus"].to_json(orient = 'records'))
        graph_dic["tot_repeat_transaction"] = json.loads(prediction_cus_clv["Total_Repeat_Transaction"].to_json(orient = 'records'))
        graph_dic["latitude"] = json.loads(prediction_cus_clv["Latitude"].to_json(orient = 'records'))
        graph_dic["longitude"] = json.loads(prediction_cus_clv["Longitude"].to_json(orient = 'records'))
        graph_dic["map_location"] = json.loads(prediction_cus_clv["Location"].to_json(orient = 'records'))
        
        settings.PREDICTION_GRAPH_DATA = graph_dic

        #Age-Category
        #drop nan values
        trans_df = trans_df.dropna()
        trans_df = trans_df.drop_duplicates()

        trans_df_new = trans_df.copy()

        #group the data
        trans_df_new = trans_df_new.groupby(['CtcID','Location','OrderDate','Source']).agg({"Transactional Value" : np.sum}).reset_index()

        #merge transaction and contact data
        final_df = pd.merge(trans_df_new, contact_df, how="left", on=["CtcID"])
        final_df['Age Category'] = final_df['Age Category'].fillna('Unknown')

        #group actual data by age-category
        Age_Category_DF = final_df.groupby(['Age Category','Location']).agg({"CtcID": np.size,"Transactional Value" : np.sum}).reset_index()
        #rename the column
        Age_Category_DF = Age_Category_DF.rename(columns={"CtcID" : "Total_Cus"})
        Age_Category_DF = Age_Category_DF.rename(columns={"Transactional Value" : "AGE_Cat_Total_Transaction"})
        Age_Category_DF['AGE_Cat_Total_Transaction'] = round(Age_Category_DF['AGE_Cat_Total_Transaction'],3)

        #calculate total transaction value
        Total_Trans_DF = Age_Category_DF.groupby(['Location']).agg({"AGE_Cat_Total_Transaction" : np.sum}).reset_index()
        Total_Trans_DF = Total_Trans_DF.rename(columns={"AGE_Cat_Total_Transaction" : "Location_Total_Transaction"})

        #merge transaction and total transaction data
        Age_Category_DF = pd.merge(Age_Category_DF, Total_Trans_DF, how="left", on=["Location"])

        #age-category revenue percentage
        Age_Category_DF['AGE_Cat_Revenue_Per'] = round(Age_Category_DF['AGE_Cat_Total_Transaction'] / Age_Category_DF['Location_Total_Transaction'],3) * 100

        Age_Category_DF.drop(['Location_Total_Transaction'],axis=1,inplace=True)

        #merge CLV and contact data
        final_clv_df = pd.merge(prediction_clv, contact_df, how="left", on=["CtcID"])
        final_clv_df['Age Category'] = final_clv_df['Age Category'].fillna('Unknown')

        final_clv_df['AVG_CLV_3M'] = final_clv_df['CLV_3M']
        final_clv_df['MEDIAN_CLV_3M'] = final_clv_df['CLV_3M']
        final_clv_df['MIN_CLV_3M'] = final_clv_df['CLV_3M']
        final_clv_df['MAX_CLV_3M'] = final_clv_df['CLV_3M']
        final_clv_df['STD_CLV_3M'] = final_clv_df['CLV_3M']

        #group clv data by age-category
        Age_Category_CLV_DF = final_clv_df.groupby(['Age Category','Location']).agg({"CtcID": np.size,"CLV_3M" : np.sum,"AVG_CLV_3M" : np.mean,"MEDIAN_CLV_3M" : np.median,"MIN_CLV_3M" : np.min,"MAX_CLV_3M" : np.max,"STD_CLV_3M" : np.std}).reset_index()

        Age_Category_CLV_DF = Age_Category_CLV_DF.rename(columns={"CtcID" : "Total_CLV_Cus"})

        #merge actual and clv data of grouped by age-category
        Age_Category_DF = pd.merge(Age_Category_DF, Age_Category_CLV_DF, on=["Age Category","Location"])
        Age_Category_DF['CLV_3M'] = round(Age_Category_DF['CLV_3M'],2)
            

        settings.ANALYTICS_GRAPH_DATAFRAME = Age_Category_DF.copy()

        

        #Revenue Distribution over time
        trans_df_revenue = trans_df.copy()

        #group by orderdate
        trans_df_revenue = trans_df_revenue.groupby(['OrderDate','Location']).agg({"CtcID": np.size,"Transactional Value" : np.mean}).reset_index()

        #convert date column to datetime
        trans_df_revenue['OrderDate'] = pd.to_datetime(trans_df_revenue['OrderDate'])

        # extract date, month and year from dates
        trans_df_revenue['Month'] = [i.month for i in trans_df_revenue['OrderDate']]
        trans_df_revenue['Year'] = [i.year for i in trans_df_revenue['OrderDate']]
        trans_df_revenue['day_of_week'] = [i.dayofweek for i in trans_df_revenue['OrderDate']]
        trans_df_revenue['day_of_year'] = [i.dayofyear for i in trans_df_revenue['OrderDate']]

        #rename & remove column
        trans_df_revenue['Transactional Value'] = round(trans_df_revenue['Transactional Value'],3)
        trans_df_revenue = trans_df_revenue.rename(columns={"CtcID" : "Total_Cus"})
        trans_df_revenue.drop(['OrderDate'],axis=1,inplace=True)
        #group by day_of_week
        trans_df_revenue = trans_df_revenue.groupby(['day_of_week','Location']).agg({"Total_Cus": np.sum,"Transactional Value" : np.mean}).reset_index()

        settings.REVENUE_GRAPH_DATAFRAME = trans_df_revenue.copy()
        

        #prediction segment
        #All Customer
        prediction_cus_clv = prediction_clv.copy()
        prediction_cus_clv['Total_Transaction'] = prediction_cus_clv['Transactional Value']
        prediction_cus_clv = prediction_cus_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size, "frequency" : np.mean}).reset_index()
        prediction_cus_clv['Total_Transaction'] = round(prediction_cus_clv['Total_Transaction'],3)
        prediction_cus_clv['frequency'] = round(prediction_cus_clv['frequency'],3)
        prediction_cus_clv = prediction_cus_clv.rename(columns={"Total_Transaction" : "CUS_REVENUE"})
        prediction_cus_clv = prediction_cus_clv.rename(columns={"Transactional Value" : "CUS_AOV"})
        prediction_cus_clv = prediction_cus_clv.rename(columns={"CtcID" : "CUS_Users"})
        prediction_cus_clv = prediction_cus_clv.rename(columns={"CLV_3M" : "CUS_CLV"})
        prediction_cus_clv = prediction_cus_clv.rename(columns={"frequency" : "CUS_Frequency"})

        #Loyal Customer
        prediction_loyal_clv = prediction_clv.copy()
        prediction_loyal_clv = prediction_clv.loc[(prediction_clv.no_visit >= 3)]
        prediction_loyal_clv['Total_Transaction'] = prediction_loyal_clv['Transactional Value']
        prediction_loyal_clv = prediction_loyal_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size, "frequency" : np.mean}).reset_index()
        prediction_loyal_clv['Total_Transaction'] = round(prediction_loyal_clv['Total_Transaction'],3)
        prediction_loyal_clv['frequency'] = round(prediction_loyal_clv['frequency'],3)
        prediction_loyal_clv = prediction_loyal_clv.rename(columns={"Total_Transaction" : "LOYAL_REVENUE"})
        prediction_loyal_clv = prediction_loyal_clv.rename(columns={"Transactional Value" : "LOYAL_AOV"})
        prediction_loyal_clv = prediction_loyal_clv.rename(columns={"CtcID" : "LOYAL_Users"})
        prediction_loyal_clv = prediction_loyal_clv.rename(columns={"CLV_3M" : "LOYAL_CLV"})
        prediction_loyal_clv = prediction_loyal_clv.rename(columns={"frequency" : "LOYAL_Frequency"})

        #Repeat Customer
        prediction_repeat_clv = prediction_clv.copy()
        prediction_repeat_clv = prediction_repeat_clv.loc[(prediction_repeat_clv.no_visit >= 2)]
        prediction_repeat_clv['Total_Transaction'] = prediction_repeat_clv['Transactional Value']
        prediction_repeat_clv = prediction_repeat_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size, "frequency" : np.mean}).reset_index()
        prediction_repeat_clv['Total_Transaction'] = round(prediction_repeat_clv['Total_Transaction'],3)
        prediction_repeat_clv['frequency'] = round(prediction_repeat_clv['frequency'],3)
        prediction_repeat_clv = prediction_repeat_clv.rename(columns={"Total_Transaction" : "REPEAT_REVENUE"})
        prediction_repeat_clv = prediction_repeat_clv.rename(columns={"Transactional Value" : "REPEAT_AOV"})
        prediction_repeat_clv = prediction_repeat_clv.rename(columns={"CtcID" : "REPEAT_Users"})
        prediction_repeat_clv = prediction_repeat_clv.rename(columns={"CLV_3M" : "REPEAT_CLV"})
        prediction_repeat_clv = prediction_repeat_clv.rename(columns={"frequency" : "REPEAT_Frequency"})

        #Top Spender
        prediction_top_clv = prediction_clv.copy()
        prediction_top_clv['Total_Transaction'] = prediction_top_clv['Transactional Value']
        total_cus = prediction_cus_clv['CUS_Users'].sum()
        total_cus = total_cus * 0.25
        total_cus = round(total_cus)
        prediction_top_clv = prediction_top_clv.groupby(['CtcID']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "frequency" : np.mean}).reset_index()
        prediction_top_clv.sort_values(by='Total_Transaction', ascending=False,inplace=True)
        prediction_top_clv_ID = prediction_top_clv['CtcID'].iloc[:total_cus]
        prediction_top_spend_clv = prediction_clv[prediction_clv.CtcID.isin(prediction_top_clv_ID)]
        prediction_top_spend_clv['Total_Transaction'] = prediction_top_spend_clv['Transactional Value']
        prediction_top_clv = prediction_top_spend_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size, "frequency" : np.mean}).reset_index()
        prediction_top_clv['Total_Transaction'] = round(prediction_top_clv['Total_Transaction'],3)
        prediction_top_clv['frequency'] = round(prediction_top_clv['frequency'],3)
        prediction_top_clv = prediction_top_clv.rename(columns={"Total_Transaction" : "TOP_REVENUE"})
        prediction_top_clv = prediction_top_clv.rename(columns={"Transactional Value" : "TOP_AOV"})
        prediction_top_clv = prediction_top_clv.rename(columns={"CtcID" : "TOP_Users"})
        prediction_top_clv = prediction_top_clv.rename(columns={"CLV_3M" : "TOP_CLV"})
        prediction_top_clv = prediction_top_clv.rename(columns={"frequency" : "TOP_Frequency"})

        #Low Spender
        prediction_low_clv = prediction_clv.copy()
        prediction_low_clv['Total_Transaction'] = prediction_low_clv['Transactional Value']
        prediction_low_clv = prediction_low_clv.groupby(['CtcID']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "frequency" : np.mean}).reset_index()
        prediction_low_clv.sort_values(by='Total_Transaction', ascending=True,inplace=True)
        prediction_low_clv_ID = prediction_low_clv['CtcID'].iloc[:total_cus]
        prediction_low_spend_clv = prediction_clv[prediction_clv.CtcID.isin(prediction_low_clv_ID)]
        prediction_low_spend_clv['Total_Transaction'] = prediction_low_spend_clv['Transactional Value']
        prediction_low_clv = prediction_low_spend_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size, "frequency" : np.mean}).reset_index()
        prediction_low_clv['Total_Transaction'] = round(prediction_low_clv['Total_Transaction'],3)
        prediction_low_clv['frequency'] = round(prediction_low_clv['frequency'],3)
        prediction_low_clv = prediction_low_clv.rename(columns={"Total_Transaction" : "LOW_REVENUE"})
        prediction_low_clv = prediction_low_clv.rename(columns={"Transactional Value" : "LOW_AOV"})
        prediction_low_clv = prediction_low_clv.rename(columns={"CtcID" : "LOW_Users"})
        prediction_low_clv = prediction_low_clv.rename(columns={"CLV_3M" : "LOW_CLV"})
        prediction_low_clv = prediction_low_clv.rename(columns={"frequency" : "LOW_Frequency"})

        #At Risk Customer
        prediction_risk_clv = prediction_clv.copy()
        prediction_risk_clv = prediction_risk_clv.loc[(prediction_risk_clv.days_since_last_visit > (2 * prediction_risk_clv.frequency))]
        prediction_risk_clv['Total_Transaction'] = prediction_risk_clv['Transactional Value']
        prediction_risk_clv = prediction_risk_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size, "frequency" : np.mean}).reset_index()
        prediction_risk_clv['Total_Transaction'] = round(prediction_risk_clv['Total_Transaction'],3)
        prediction_risk_clv['frequency'] = round(prediction_risk_clv['frequency'],3)
        prediction_risk_clv = prediction_risk_clv.rename(columns={"Total_Transaction" : "RISK_REVENUE"})
        prediction_risk_clv = prediction_risk_clv.rename(columns={"Transactional Value" : "RISK_AOV"})
        prediction_risk_clv = prediction_risk_clv.rename(columns={"CtcID" : "RISK_Users"})
        prediction_risk_clv = prediction_risk_clv.rename(columns={"CLV_3M" : "RISK_CLV"})
        prediction_risk_clv = prediction_risk_clv.rename(columns={"frequency" : "RISK_Frequency"})

        # #Churn Customer
        prediction_churn_clv = prediction_clv.copy()
        prediction_churn_clv = prediction_churn_clv.loc[(prediction_churn_clv.no_visit > 1)]
        cus_total = prediction_churn_clv['CtcID'].size
        cus_avg_freq = round(prediction_churn_clv['frequency'].sum()/cus_total,3)
        prediction_churn_clv = prediction_churn_clv.loc[(prediction_churn_clv.days_since_last_visit > (3 * cus_avg_freq))]
        prediction_churn_clv['Total_Transaction'] = prediction_churn_clv['Transactional Value']
        prediction_churn_clv = prediction_churn_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size, "frequency" : np.mean}).reset_index()
        prediction_churn_clv['Total_Transaction'] = round(prediction_churn_clv['Total_Transaction'],3)
        prediction_churn_clv['frequency'] = round(prediction_churn_clv['frequency'],3)
        prediction_churn_clv = prediction_churn_clv.rename(columns={"Total_Transaction" : "CHURN_REVENUE"})
        prediction_churn_clv = prediction_churn_clv.rename(columns={"Transactional Value" : "CHURN_AOV"})
        prediction_churn_clv = prediction_churn_clv.rename(columns={"CtcID" : "CHURN_Users"})
        prediction_churn_clv = prediction_churn_clv.rename(columns={"CLV_3M" : "CHURN_CLV"})
        prediction_churn_clv = prediction_churn_clv.rename(columns={"frequency" : "CHURN_Frequency"})

        settings.ALL_CUSTOMER_DATAFRAME = prediction_cus_clv.copy()
        settings.LOYAL_CUSTOMER_DATAFRAME = prediction_loyal_clv.copy()
        settings.REPEAT_CUSTOMER_DATAFRAME = prediction_repeat_clv.copy()
        settings.TOP_SPENDER_DATAFRAME = prediction_top_clv.copy()
        settings.LOW_SPENDER_DATAFRAME = prediction_low_clv.copy()
        settings.RISK_CUSTOMER_DATAFRAME = prediction_risk_clv.copy()
        settings.CHURN_CUSTOMER_DATAFRAME = prediction_churn_clv.copy()
        settings.PREDICTION_CLV_DATAFRAME = prediction_clv.copy()

        #Executive Summary Report

        #one time execution
        prediction_summary_clv = prediction_clv.copy()
        thersold = thersold_date.split("-")
        thersold_yr = int(thersold[0])
        thersold_mnth = int(thersold[1])
        thersold_day = int(thersold[2])
        a_date = datetime.date(thersold_yr, thersold_mnth, thersold_day)
        prediction_summary_clv['thersold_day'] = a_date
        prediction_summary_clv['customer_age_days'] = pd.to_timedelta(prediction_summary_clv['customer_age'], unit='D')
        prediction_summary_clv['OrderDate'] = prediction_summary_clv['thersold_day'] - prediction_summary_clv['customer_age_days']
        prediction_summary_clv.drop(['thersold_day'],axis=1,inplace=True)
        prediction_summary_clv.drop(['customer_age_days'],axis=1,inplace=True)
        prediction_summary_clv_prev = prediction_summary_clv.copy()

        if(len(act_start.strip())):
            startDate = pd.to_datetime(act_start).date()
            startDate_tmp = pd.to_datetime(act_start).date()
            prev_start = str(startDate_tmp - pd.DateOffset(years=1))
            prev_start = pd.to_datetime(prev_start).date()
            prediction_summary_clv.drop(prediction_summary_clv[prediction_summary_clv['OrderDate'] < startDate].index, inplace = True)
            prediction_summary_clv_prev.drop(prediction_summary_clv_prev[prediction_summary_clv_prev['OrderDate'] < prev_start].index, inplace = True)
            
            if(len(act_end.strip())):
                endDate = pd.to_datetime(act_end).date()
                endDate_tmp = pd.to_datetime(act_end).date()
                prev_end = str(endDate_tmp - pd.DateOffset(years=1))
                prev_end = pd.to_datetime(prev_end).date()
                prediction_summary_clv.drop(prediction_summary_clv[prediction_summary_clv['OrderDate'] > endDate].index, inplace = True)
                prediction_summary_clv_prev.drop(prediction_summary_clv_prev[prediction_summary_clv_prev['OrderDate'] > prev_end].index, inplace = True)
                is_filter = True
            else:
                custom_filter = False
        else:
            custom_filter = False
            

        prediction_summary_clv['OrderDate'] = pd.to_datetime(prediction_summary_clv['OrderDate'])

        # extract date, month and year from dates
        prediction_summary_clv['Month'] = [i.month for i in prediction_summary_clv['OrderDate']]
        prediction_summary_clv['Year'] = [i.year for i in prediction_summary_clv['OrderDate']]
        prediction_summary_clv['day_of_week'] = [i.dayofweek for i in prediction_summary_clv['OrderDate']]
        prediction_summary_clv['day_of_year'] = [i.dayofyear for i in prediction_summary_clv['OrderDate']]

        r, c = prediction_summary_clv_prev.shape
        if r > 0 :
            prediction_summary_clv_prev['OrderDate'] = pd.to_datetime(prediction_summary_clv_prev['OrderDate'])

            # extract date, month and year from dates
            prediction_summary_clv_prev['Month'] = [i.month for i in prediction_summary_clv_prev['OrderDate']]
            prediction_summary_clv_prev['Year'] = [i.year for i in prediction_summary_clv_prev['OrderDate']]
            prediction_summary_clv_prev['day_of_week'] = [i.dayofweek for i in prediction_summary_clv_prev['OrderDate']]
            prediction_summary_clv_prev['day_of_year'] = [i.dayofyear for i in prediction_summary_clv_prev['OrderDate']]
        else:
            prev_clv_not_found = True  
            
        if not custom_filter:
            if(year > 0):
                prev_year = year - 1
                prediction_summary_clv.drop(prediction_summary_clv[prediction_summary_clv['Year'] != year].index, inplace = True)  
                prediction_summary_clv_prev.drop(prediction_summary_clv_prev[prediction_summary_clv_prev['Year'] != prev_year].index, inplace = True)        
                is_filter = True
                
                if(month > 0):
                    prev_month = month - 1
                    prediction_summary_clv.drop(prediction_summary_clv[prediction_summary_clv['Month'] != month].index, inplace = True)
                    prediction_summary_clv_prev.drop(prediction_summary_clv_prev[prediction_summary_clv_prev['Month'] != prev_month].index, inplace = True)        
                    is_filter = True
                    
        r, c = prediction_summary_clv_prev.shape
        if r == 0 :
            prev_clv_not_found = True

        #location based avg CLV of All customer
        prediction_summary_all_clv = prediction_summary_clv.groupby(['Location']).agg({"no_visit" : np.sum, "CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()

        #add, rename and remove the column
        prediction_summary_all_clv['Total_Transaction'] = round(prediction_summary_all_clv['Total_Transaction'],3)
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_ALL"})
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"Transactional Value" : "AVG_REVENUE_ALL"})
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"CtcID" : "TOTAL_USERS"})
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"CLV_3M" : "AVG_CLV_ALL"})
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"no_visit" : "Total_NO_ORDERS"})

        executive_summary_df = prediction_summary_all_clv

        #Repeat Customer
        prediction_summary_repeat_clv = prediction_summary_clv.copy()
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.loc[(prediction_summary_repeat_clv.no_visit >= 2)]
        prediction_summary_repeat_clv['Total_Transaction'] = prediction_summary_repeat_clv['Transactional Value']
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
        prediction_summary_repeat_clv['Total_Transaction'] = round(prediction_summary_repeat_clv['Total_Transaction'],3)
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_REPEAT"})
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.rename(columns={"Transactional Value" : "AVG_REVENUE_REPEAT"})
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.rename(columns={"CtcID" : "REPEAT_USERS"})
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.rename(columns={"CLV_3M" : "AVG_CLV_REPEAT"})

        executive_summary_df = pd.merge(executive_summary_df, prediction_summary_repeat_clv, on=["Location"])

        #New Customer
        prediction_summary_new_clv = prediction_summary_clv.copy()
        prediction_summary_new_clv = prediction_summary_new_clv.loc[(prediction_summary_new_clv.no_visit == 1)]
        prediction_summary_new_clv['Total_Transaction'] = prediction_summary_new_clv['Transactional Value']
        prediction_summary_new_clv = prediction_summary_new_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
        prediction_summary_new_clv['Total_Transaction'] = round(prediction_summary_new_clv['Total_Transaction'],3)
        prediction_summary_new_clv = prediction_summary_new_clv.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_NEW"})
        prediction_summary_new_clv = prediction_summary_new_clv.rename(columns={"Transactional Value" : "AVG_REVENUE_NEW"})
        prediction_summary_new_clv = prediction_summary_new_clv.rename(columns={"CtcID" : "NEW_USERS"})
        prediction_summary_new_clv = prediction_summary_new_clv.rename(columns={"CLV_3M" : "AVG_CLV_NEW"})

        executive_summary_df = pd.merge(executive_summary_df, prediction_summary_new_clv, on=["Location"])

        #Churn Customer
        prediction_summary_churn_clv = prediction_summary_clv.copy()
        prediction_summary_churn_clv = prediction_summary_churn_clv.loc[(prediction_summary_churn_clv.no_visit > 1)]
        cus_total = prediction_summary_churn_clv['CtcID'].size
        cus_avg_freq = round(prediction_summary_churn_clv['frequency'].sum()/cus_total,3)
        prediction_summary_churn_clv = prediction_summary_churn_clv.loc[(prediction_summary_churn_clv.days_since_last_visit > (3 * cus_avg_freq))]
        prediction_summary_churn_clv['Total_Transaction'] = prediction_summary_churn_clv['Transactional Value']
        prediction_summary_churn_clv = prediction_summary_churn_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
        prediction_summary_churn_clv['Total_Transaction'] = round(prediction_summary_churn_clv['Total_Transaction'],3)
        prediction_summary_churn_clv = prediction_summary_churn_clv.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_CHURN"})
        prediction_summary_churn_clv = prediction_summary_churn_clv.rename(columns={"Transactional Value" : "AVG_REVENUE_CHURN"})
        prediction_summary_churn_clv = prediction_summary_churn_clv.rename(columns={"CtcID" : "CHURN_USERS"})
        prediction_summary_churn_clv = prediction_summary_churn_clv.rename(columns={"CLV_3M" : "AVG_CLV_CHURN"})

        executive_summary_df = pd.merge(executive_summary_df, prediction_summary_churn_clv, on=["Location"])

        executive_summary_df['TOTAL_USERS'] = executive_summary_df['TOTAL_USERS'] - executive_summary_df['CHURN_USERS']

        executive_summary_df_final = executive_summary_df[['Location','Total_NO_ORDERS','AVG_REVENUE_ALL','AVG_CLV_ALL','TOTAL_REVENUE_ALL','TOTAL_USERS','REPEAT_USERS','NEW_USERS','CHURN_USERS']]


        if not prev_clv_not_found:
            #location based avg CLV of All customer
            prediction_all_clv_prev = prediction_summary_clv_prev.groupby(['Location']).agg({"no_visit" : np.sum, "CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()

            #add, rename and remove the column
            prediction_all_clv_prev['Total_Transaction'] = round(prediction_all_clv_prev['Total_Transaction'],3)
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_ALL"})
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"Transactional Value" : "AVG_REVENUE_ALL"})
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"CtcID" : "TOTAL_USERS"})
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"CLV_3M" : "AVG_CLV_ALL"})
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"no_visit" : "Total_NO_ORDERS"})

            executive_summary_df_prev = prediction_all_clv_prev

            #Repeat Customer
            prediction_repeat_clv_prev = prediction_summary_clv_prev.copy()
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.loc[(prediction_repeat_clv_prev.no_visit >= 2)]
            prediction_repeat_clv_prev['Total_Transaction'] = prediction_repeat_clv_prev['Transactional Value']
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
            prediction_repeat_clv_prev['Total_Transaction'] = round(prediction_repeat_clv_prev['Total_Transaction'],3)
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_REPEAT"})
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.rename(columns={"Transactional Value" : "AVG_REVENUE_REPEAT"})
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.rename(columns={"CtcID" : "REPEAT_USERS"})
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.rename(columns={"CLV_3M" : "AVG_CLV_REPEAT"})

            executive_summary_df_prev = pd.merge(executive_summary_df_prev, prediction_repeat_clv_prev, on=["Location"])

            #New Customer
            prediction_new_clv_prev = prediction_summary_clv_prev.copy()
            prediction_new_clv_prev = prediction_new_clv_prev.loc[(prediction_new_clv_prev.no_visit == 1)]
            prediction_new_clv_prev['Total_Transaction'] = prediction_new_clv_prev['Transactional Value']
            prediction_new_clv_prev = prediction_new_clv_prev.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
            prediction_new_clv_prev['Total_Transaction'] = round(prediction_new_clv_prev['Total_Transaction'],3)
            prediction_new_clv_prev = prediction_new_clv_prev.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_NEW"})
            prediction_new_clv_prev = prediction_new_clv_prev.rename(columns={"Transactional Value" : "AVG_REVENUE_NEW"})
            prediction_new_clv_prev = prediction_new_clv_prev.rename(columns={"CtcID" : "NEW_USERS"})
            prediction_new_clv_prev = prediction_new_clv_prev.rename(columns={"CLV_3M" : "AVG_CLV_NEW"})

            executive_summary_df_prev = pd.merge(executive_summary_df_prev, prediction_new_clv_prev, on=["Location"])

            #Churn Customer
            prediction_churn_clv_prev = prediction_summary_clv_prev.copy()
            prediction_churn_clv_prev = prediction_churn_clv_prev.loc[(prediction_churn_clv_prev.no_visit > 1)]
            cus_total = prediction_churn_clv_prev['CtcID'].size
            cus_avg_freq = round(prediction_churn_clv_prev['frequency'].sum()/cus_total,3)
            prediction_churn_clv_prev = prediction_churn_clv_prev.loc[(prediction_churn_clv_prev.days_since_last_visit > (3 * cus_avg_freq))]
            prediction_churn_clv_prev['Total_Transaction'] = prediction_churn_clv_prev['Transactional Value']
            prediction_churn_clv_prev = prediction_churn_clv_prev.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
            prediction_churn_clv_prev['Total_Transaction'] = round(prediction_churn_clv_prev['Total_Transaction'],3)
            prediction_churn_clv_prev = prediction_churn_clv_prev.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_CHURN"})
            prediction_churn_clv_prev = prediction_churn_clv_prev.rename(columns={"Transactional Value" : "AVG_REVENUE_CHURN"})
            prediction_churn_clv_prev = prediction_churn_clv_prev.rename(columns={"CtcID" : "CHURN_USERS"})
            prediction_churn_clv_prev = prediction_churn_clv_prev.rename(columns={"CLV_3M" : "AVG_CLV_CHURN"})

            executive_summary_df_prev = pd.merge(executive_summary_df_prev, prediction_churn_clv_prev, on=["Location"])

            executive_summary_df_prev['TOTAL_USERS'] = executive_summary_df_prev['TOTAL_USERS'] - executive_summary_df_prev['CHURN_USERS']

            executive_summary_df_final_prev = executive_summary_df_prev[['Location','Total_NO_ORDERS','AVG_REVENUE_ALL','AVG_CLV_ALL','TOTAL_REVENUE_ALL','TOTAL_USERS','REPEAT_USERS','NEW_USERS','CHURN_USERS']]
            
            
            r1, c1 = executive_summary_df_final_prev.shape
            if r1>0 :
                executive_summary_df_final['Total_NO_ORDERS_Changes'] = round((executive_summary_df_final['Total_NO_ORDERS'] - executive_summary_df_final_prev['Total_NO_ORDERS'])/executive_summary_df_final_prev['Total_NO_ORDERS'],2)*100
                executive_summary_df_final['AVG_REVENUE_ALL_Changes'] = round((executive_summary_df_final['AVG_REVENUE_ALL'] - executive_summary_df_final_prev['AVG_REVENUE_ALL'])/executive_summary_df_final_prev['AVG_REVENUE_ALL'],2)*100
                executive_summary_df_final['AVG_CLV_ALL_Changes'] = round((executive_summary_df_final['AVG_CLV_ALL'] - executive_summary_df_final_prev['AVG_CLV_ALL'])/executive_summary_df_final_prev['AVG_CLV_ALL'],2)*100
                executive_summary_df_final['TOTAL_REVENUE_ALL_Changes'] = round((executive_summary_df_final['TOTAL_REVENUE_ALL'] - executive_summary_df_final_prev['TOTAL_REVENUE_ALL'])/executive_summary_df_final_prev['TOTAL_REVENUE_ALL'],2)*100
                executive_summary_df_final['TOTAL_USERS_Changes'] = round((executive_summary_df_final['TOTAL_USERS'] - executive_summary_df_final_prev['TOTAL_USERS'])/executive_summary_df_final_prev['TOTAL_USERS'],2)*100
                executive_summary_df_final['REPEAT_USERS_Changes'] = round((executive_summary_df_final['REPEAT_USERS'] - executive_summary_df_final_prev['REPEAT_USERS'])/executive_summary_df_final_prev['REPEAT_USERS'],2)*100
                executive_summary_df_final['NEW_USERS_Changes'] = round((executive_summary_df_final['NEW_USERS'] - executive_summary_df_final_prev['NEW_USERS'])/executive_summary_df_final_prev['AVG_REVENUE_ALL'],2)*100
                executive_summary_df_final['CHURN_USERS_Changes'] = round((executive_summary_df_final['CHURN_USERS'] - executive_summary_df_final_prev['CHURN_USERS'])/executive_summary_df_final_prev['CHURN_USERS'],2)*100
        else:
            executive_summary_df_final['Total_NO_ORDERS_Changes'] = '-'
            executive_summary_df_final['AVG_REVENUE_ALL_Changes'] = '-'
            executive_summary_df_final['AVG_CLV_ALL_Changes'] = '-'
            executive_summary_df_final['TOTAL_REVENUE_ALL_Changes'] = '-'
            executive_summary_df_final['TOTAL_USERS_Changes'] = '-'
            executive_summary_df_final['REPEAT_USERS_Changes'] = '-'
            executive_summary_df_final['NEW_USERS_Changes'] = '-'
            executive_summary_df_final['CHURN_USERS_Changes'] = '-'
            executive_summary_df_final_prev = pd.DataFrame()
            executive_summary_df_final_prev["TOTAL_USERS"] = '-'
            executive_summary_df_final_prev["Total_NO_ORDERS"] = '-'
            executive_summary_df_final_prev["TOTAL_REVENUE_ALL"] = '-'
            executive_summary_df_final_prev["AVG_REVENUE_ALL"] = '-'
            executive_summary_df_final_prev["NEW_USERS"] = '-'
            executive_summary_df_final_prev["REPEAT_USERS"] = '-'
            executive_summary_df_final_prev["CHURN_USERS"] = '-'
            executive_summary_df_final_prev["AVG_CLV_ALL"] = '-'

        executive_summary_df_final.fillna('-', inplace=True)
        summary_location = "The Sun Inn"
        executive_summary_dict = {}
        print("*********************************************************************************")
        print(executive_summary_df_final[executive_summary_df_final["Location"] == summary_location].size)
        print("*********************************************************************************")

        if executive_summary_df_final[executive_summary_df_final["Location"] == summary_location].size != 0:

            executive_summary_dict["customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["TOTAL_USERS"].tolist()[0]
            executive_summary_dict["customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["TOTAL_USERS_Changes"].tolist()[0] 
            executive_summary_dict["customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["TOTAL_USERS"].tolist()[0]                 

            executive_summary_dict["orders_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["Total_NO_ORDERS"].tolist()[0]  
            executive_summary_dict["orders_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["Total_NO_ORDERS_Changes"].tolist()[0]
            executive_summary_dict["orders_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["Total_NO_ORDERS"].tolist()[0] 
            executive_summary_dict["revenue_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["TOTAL_REVENUE_ALL"].tolist()[0] 
            executive_summary_dict["revenue_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["TOTAL_REVENUE_ALL_Changes"].tolist()[0] 
            executive_summary_dict["revenue_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["TOTAL_REVENUE_ALL"].tolist()[0] 
            executive_summary_dict["aov_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["AVG_REVENUE_ALL"].tolist()[0]  
            executive_summary_dict["aov_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["AVG_REVENUE_ALL_Changes"].tolist()[0]  
            executive_summary_dict["aov_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["AVG_REVENUE_ALL"].tolist()[0] 
            executive_summary_dict["new_customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["NEW_USERS"].tolist()[0]  
            executive_summary_dict["new_customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["NEW_USERS_Changes"].tolist()[0] 
            executive_summary_dict["new_customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["NEW_USERS"].tolist()[0] 
            executive_summary_dict["repeat_customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["REPEAT_USERS"].tolist()[0]
            executive_summary_dict["repeat_customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["REPEAT_USERS_Changes"].tolist()[0]  
            executive_summary_dict["repeat_customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["REPEAT_USERS"].tolist()[0]  
            executive_summary_dict["churn_customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["CHURN_USERS"].tolist()[0]  
            executive_summary_dict["churn_customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["CHURN_USERS_Changes"].tolist()[0]
            executive_summary_dict["churn_customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["CHURN_USERS"].tolist()[0] 
            executive_summary_dict["aclv_customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["AVG_CLV_ALL"].tolist()[0]  
            executive_summary_dict["aclv_customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["AVG_CLV_ALL_Changes"].tolist()[0] 
            executive_summary_dict["aclv_customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["AVG_CLV_ALL"].tolist()[0] 
                
        print("Dataframe initialization process completed... ")
        settings.EXECUTIVE_SUM_DICT = executive_summary_dict
        settings.MODEL_LOAD_FLAG = "True"
                        
            
    
    result = {}
    result["prediction_graph"] = settings.PREDICTION_GRAPH_DATA
    age_analytics_df = settings.ANALYTICS_GRAPH_DATAFRAME.copy()
    revenue_df = settings.REVENUE_GRAPH_DATAFRAME.copy()

    all_customer_df = settings.ALL_CUSTOMER_DATAFRAME.copy()
    loyal_customer_df = settings.LOYAL_CUSTOMER_DATAFRAME.copy()
    repeat_customer_df = settings.REPEAT_CUSTOMER_DATAFRAME.copy()
    top_customer_df = settings.TOP_SPENDER_DATAFRAME.copy()
    low_customer_df = settings.LOW_SPENDER_DATAFRAME.copy()
    risk_customer_df = settings.RISK_CUSTOMER_DATAFRAME.copy()
    chrun_customer_df = settings.CHURN_CUSTOMER_DATAFRAME.copy()
    executive_summary = settings.EXECUTIVE_SUM_DICT
    location_name = "The Sun Inn"
    
    
    
    top_customer_df[top_customer_df["Location"] == "The Sun Inn"]
    low_customer_df[low_customer_df["Location"] == "The Sun Inn"]
    risk_customer_df[risk_customer_df["Location"] == "The Sun Inn"]
    chrun_customer_df[chrun_customer_df["Location"] == "The Sun Inn"]
    

    age_analytics_df = age_analytics_df[age_analytics_df['Location'] == 'The Sun Inn']
    revenue_df = revenue_df[revenue_df['Location'] == location_name]

    age_list,age_rev_percentage,age_clv,age_cus_group,age_rev = [],[],[],[],[]    
    unknown_rev_precentage = 0
    unknown_avg_clv = 0
    unknown_total_cus = 0
    unknown_val_reve = 0
    for x in range(8):        
        if x == 7 :
            unknown_rev_precentage = json.loads(age_analytics_df["AGE_Cat_Revenue_Per"].to_json(orient = 'records'))[x]
            unknown_avg_clv = json.loads(age_analytics_df["AVG_CLV_3M"].to_json(orient = 'records'))[x]
            unknown_total_cus = json.loads(age_analytics_df["Total_Cus"].to_json(orient = 'records'))[x]
            unknown_val_reve = json.loads(age_analytics_df["AGE_Cat_Total_Transaction"].to_json(orient = 'records'))[x]
        else:
            age_list.append(json.loads(age_analytics_df["Age Category"].to_json(orient = 'records'))[x])
            age_rev_percentage.append(json.loads(age_analytics_df["AGE_Cat_Revenue_Per"].to_json(orient = 'records'))[x])
            age_clv.append(json.loads(age_analytics_df["AVG_CLV_3M"].to_json(orient = 'records'))[x])
            age_cus_group.append(json.loads(age_analytics_df["Total_Cus"].to_json(orient = 'records'))[x])
            age_rev.append(json.loads(age_analytics_df["AGE_Cat_Total_Transaction"].to_json(orient = 'records'))[x])
    age_analytics_dic = {}
    age_analytics_dic["age_category"] = age_list 
    age_analytics_dic["age_category_rev_percentage"] = age_rev_percentage
    age_analytics_dic["avg_clv"] = age_clv
    age_analytics_dic["no_cus_age_group"] = age_cus_group
    age_analytics_dic["val_of_revenue"] = age_rev
    

    revenue_dic = {"day_of_day_of_week" : json.loads(revenue_df["day_of_week"].to_json(orient = 'records'))} 
    revenue_dic["transaction_value"] = json.loads(revenue_df["Transactional Value"].to_json(orient = 'records'))

    

    all_customer_dic = {"users" : all_customer_df[all_customer_df["Location"] == location_name]["CUS_Users"].tolist()[0]}
    all_customer_dic["frequency"] = all_customer_df[all_customer_df["Location"] == location_name]["CUS_Frequency"].tolist()[0]
    all_customer_dic["aov"] = all_customer_df[all_customer_df["Location"] == location_name]["CUS_AOV"].tolist()[0]
    all_customer_dic["clv"] = all_customer_df[all_customer_df["Location"] == location_name]["CUS_CLV"].tolist()[0]
    all_customer_dic["revenue"] = all_customer_df[all_customer_df["Location"] == location_name]["CUS_REVENUE"].tolist()[0]

    loyal_customer_dic = {"users" : loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_Users"].tolist()[0]}
    loyal_customer_dic["frequency"] = loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_Frequency"].tolist()[0]
    loyal_customer_dic["aov"] = loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_AOV"].tolist()[0]
    loyal_customer_dic["clv"] = loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_CLV"].tolist()[0]
    loyal_customer_dic["revenue"] = loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_REVENUE"].tolist()[0]

    repeat_customer_dic = {"users" : repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_Users"].tolist()[0]}
    repeat_customer_dic["frequency"] = repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_Frequency"].tolist()[0]
    repeat_customer_dic["aov"] = repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_AOV"].tolist()[0]
    repeat_customer_dic["clv"] = repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_CLV"].tolist()[0]
    repeat_customer_dic["revenue"] = repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_REVENUE"].tolist()[0]

    top_customer_dic = {"users" : top_customer_df[top_customer_df["Location"] == location_name]["TOP_Users"].tolist()[0]}
    top_customer_dic["frequency"] = top_customer_df[top_customer_df["Location"] == location_name]["TOP_Frequency"].tolist()[0]
    top_customer_dic["aov"] = top_customer_df[top_customer_df["Location"] == location_name]["TOP_AOV"].tolist()[0]
    top_customer_dic["clv"] = top_customer_df[top_customer_df["Location"] == location_name]["TOP_CLV"].tolist()[0]
    top_customer_dic["revenue"] = top_customer_df[top_customer_df["Location"] == location_name]["TOP_REVENUE"].tolist()[0]

    low_customer_dic = {"users" : low_customer_df[low_customer_df["Location"] == location_name]["LOW_Users"].tolist()[0]}
    low_customer_dic["frequency"] = low_customer_df[low_customer_df["Location"] == location_name]["LOW_Frequency"].tolist()[0]
    low_customer_dic["aov"] = low_customer_df[low_customer_df["Location"] == location_name]["LOW_AOV"].tolist()[0]
    low_customer_dic["clv"] = low_customer_df[low_customer_df["Location"] == location_name]["LOW_CLV"].tolist()[0]
    low_customer_dic["revenue"] = low_customer_df[low_customer_df["Location"] == location_name]["LOW_REVENUE"].tolist()[0]

    risk_customer_dic = {"users" : risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_Users"].tolist()[0]}
    risk_customer_dic["frequency"] = risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_Frequency"].tolist()[0]
    risk_customer_dic["aov"] = risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_AOV"].tolist()[0]
    risk_customer_dic["clv"] = risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_CLV"].tolist()[0]
    risk_customer_dic["revenue"] = risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_REVENUE"].tolist()[0]

    churn_customer_dic = {"users" : chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_Users"].tolist()[0]}
    churn_customer_dic["frequency"] = chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_Frequency"].tolist()[0]
    churn_customer_dic["aov"] = chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_AOV"].tolist()[0]
    churn_customer_dic["clv"] = chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_CLV"].tolist()[0]
    churn_customer_dic["revenue"] = chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_REVENUE"].tolist()[0]
    

    result["all_cus_clv"] = all_customer_dic
    result["loyal_cus_clv"] = loyal_customer_dic
    result["repeat_cus_clv"] = repeat_customer_dic
    result["top_cus_clv"] = top_customer_dic
    result["low_cus_clv"] = low_customer_dic
    result["risk_cus_clv"] = risk_customer_dic
    result["churn_cus_clv"] = churn_customer_dic
    
    result["analytics_graph"] = age_analytics_dic
    result["analytics_graph_unknown"] = unknown_rev_precentage
    result["unknown_avg_clv"] = unknown_avg_clv
    result["unknown_tot_cus"] = unknown_total_cus
    result["unknown_val_of_rev"] = unknown_val_reve
    result["revenue_graph"] = revenue_dic

    result["executive_summary"] = executive_summary    
    
    
    return JsonResponse(result,safe=False)

@api_view(["GET"])
def age_based_location_data(request):
    result = {}
    location = "The Sun Inn"
    if 'location' in request.GET:
        location = request.query_params["location"]
    age_analytics_df = settings.ANALYTICS_GRAPH_DATAFRAME.copy()
    age_analytics_df = age_analytics_df[age_analytics_df['Location'] == location]

    age_list,age_rev_percentage,age_clv,age_cus_group,age_rev = [],[],[],[],[]
    unknown_rev_precentage = 0
    unknown_avg_clv = 0
    unknown_total_cus = 0
    unknown_val_reve = 0
    for x in range(8):        
        if x == 7 :
            unknown_rev_precentage = json.loads(age_analytics_df["AGE_Cat_Revenue_Per"].to_json(orient = 'records'))[x]
            unknown_avg_clv = json.loads(age_analytics_df["AVG_CLV_3M"].to_json(orient = 'records'))[x]
            unknown_total_cus = json.loads(age_analytics_df["Total_Cus"].to_json(orient = 'records'))[x]
            unknown_val_reve = json.loads(age_analytics_df["AGE_Cat_Total_Transaction"].to_json(orient = 'records'))[x]
        else:
            age_list.append(json.loads(age_analytics_df["Age Category"].to_json(orient = 'records'))[x])
            age_rev_percentage.append(json.loads(age_analytics_df["AGE_Cat_Revenue_Per"].to_json(orient = 'records'))[x])
            age_clv.append(json.loads(age_analytics_df["AVG_CLV_3M"].to_json(orient = 'records'))[x])
            age_cus_group.append(json.loads(age_analytics_df["Total_Cus"].to_json(orient = 'records'))[x])
            age_rev.append(json.loads(age_analytics_df["AGE_Cat_Total_Transaction"].to_json(orient = 'records'))[x])
    age_analytics_dic = {}
    age_analytics_dic["age_category"] = age_list 
    age_analytics_dic["age_category_rev_percentage"] = age_rev_percentage
    age_analytics_dic["avg_clv"] = age_clv
    age_analytics_dic["no_cus_age_group"] = age_cus_group
    age_analytics_dic["val_of_revenue"] = age_rev
    
    
    result["analytics_graph"] = age_analytics_dic
    result["analytics_graph_unknown"] = unknown_rev_precentage
    result["unknown_avg_clv"] = unknown_avg_clv
    result["unknown_tot_cus"] = unknown_total_cus
    result["unknown_val_of_rev"] = unknown_val_reve
    return JsonResponse(result,safe=False)

@api_view(["GET"])
def revenue_based_location_data(request):
    result = {}    
    location = "The Sun Inn"
    if 'location' in request.GET:
        location = request.query_params["location"]        
    
    
    revenue_df = settings.REVENUE_GRAPH_DATAFRAME.copy()
    revenue_df = revenue_df[revenue_df['Location'] == location]    
    revenue_dic = {"day_of_day_of_week" : json.loads(revenue_df["day_of_week"].to_json(orient = 'records'))} 
    revenue_dic["transaction_value"] = json.loads(revenue_df["Transactional Value"].to_json(orient = 'records'))    
    result["revenue_graph"] = revenue_dic
    return JsonResponse(result,safe=False)

@api_view(["GET"])
def location_based_segment_data(request):    
    result = {}
    location_name = "The Sun Inn"
    if 'location' in request.GET:
        location_name = request.query_params["location"]
    all_customer_df = settings.ALL_CUSTOMER_DATAFRAME.copy()
    loyal_customer_df = settings.LOYAL_CUSTOMER_DATAFRAME.copy()
    repeat_customer_df = settings.REPEAT_CUSTOMER_DATAFRAME.copy()
    top_customer_df = settings.TOP_SPENDER_DATAFRAME.copy()
    low_customer_df = settings.LOW_SPENDER_DATAFRAME.copy()
    risk_customer_df = settings.RISK_CUSTOMER_DATAFRAME.copy()
    chrun_customer_df = settings.CHURN_CUSTOMER_DATAFRAME.copy()
    
    print(location_name)
    all_customer_dic = {"users" : all_customer_df[all_customer_df["Location"] == location_name]["CUS_Users"].tolist()}
    all_customer_dic["frequency"] = all_customer_df[all_customer_df["Location"] == location_name]["CUS_Frequency"].tolist() 
    all_customer_dic["aov"] = all_customer_df[all_customer_df["Location"] == location_name]["CUS_AOV"].tolist()
    all_customer_dic["clv"] = all_customer_df[all_customer_df["Location"] == location_name]["CUS_CLV"].tolist()
    all_customer_dic["revenue"] = all_customer_df[all_customer_df["Location"] == location_name]["CUS_REVENUE"].tolist()

    loyal_customer_dic = {"users" : loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_Users"].tolist()}
    loyal_customer_dic["frequency"] = loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_Frequency"].tolist()
    loyal_customer_dic["aov"] = loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_AOV"].tolist()
    loyal_customer_dic["clv"] = loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_CLV"].tolist()
    loyal_customer_dic["revenue"] = loyal_customer_df[loyal_customer_df["Location"] == location_name]["LOYAL_REVENUE"].tolist()

    repeat_customer_dic = {"users" : repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_Users"].tolist()}
    repeat_customer_dic["frequency"] = repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_Frequency"].tolist()
    repeat_customer_dic["aov"] = repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_AOV"].tolist()
    repeat_customer_dic["clv"] = repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_CLV"].tolist()
    repeat_customer_dic["revenue"] = repeat_customer_df[repeat_customer_df["Location"] == location_name]["REPEAT_REVENUE"].tolist()

    top_customer_dic = {"users" : top_customer_df[top_customer_df["Location"] == location_name]["TOP_Users"].tolist()}
    top_customer_dic["frequency"] = top_customer_df[top_customer_df["Location"] == location_name]["TOP_Frequency"].tolist()
    top_customer_dic["aov"] = top_customer_df[top_customer_df["Location"] == location_name]["TOP_AOV"].tolist()
    top_customer_dic["clv"] = top_customer_df[top_customer_df["Location"] == location_name]["TOP_CLV"].tolist()
    top_customer_dic["revenue"] = top_customer_df[top_customer_df["Location"] == location_name]["TOP_REVENUE"].tolist()

    low_customer_dic = {"users" : low_customer_df[low_customer_df["Location"] == location_name]["LOW_Users"].tolist()}
    low_customer_dic["frequency"] = low_customer_df[low_customer_df["Location"] == location_name]["LOW_Frequency"].tolist()
    low_customer_dic["aov"] = low_customer_df[low_customer_df["Location"] == location_name]["LOW_AOV"].tolist()
    low_customer_dic["clv"] = low_customer_df[low_customer_df["Location"] == location_name]["LOW_CLV"].tolist()
    low_customer_dic["revenue"] = low_customer_df[low_customer_df["Location"] == location_name]["LOW_REVENUE"].tolist()

    risk_customer_dic = {"users" : risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_Users"].tolist()}
    risk_customer_dic["frequency"] = risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_Frequency"].tolist()
    risk_customer_dic["aov"] = risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_AOV"].tolist()
    risk_customer_dic["clv"] = risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_CLV"].tolist()
    risk_customer_dic["revenue"] = risk_customer_df[risk_customer_df["Location"] == location_name]["RISK_REVENUE"].tolist()

    churn_customer_dic = {"users" : chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_Users"].tolist()}
    churn_customer_dic["frequency"] = chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_Frequency"].tolist()
    churn_customer_dic["aov"] = chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_AOV"].tolist()
    churn_customer_dic["clv"] = chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_CLV"].tolist()
    churn_customer_dic["revenue"] = chrun_customer_df[chrun_customer_df["Location"] == location_name]["CHURN_REVENUE"].tolist()
    

    result["all_cus_clv"] = all_customer_dic
    result["loyal_cus_clv"] = loyal_customer_dic
    result["repeat_cus_clv"] = repeat_customer_dic
    result["top_cus_clv"] = top_customer_dic
    result["low_cus_clv"] = low_customer_dic
    result["risk_cus_clv"] = risk_customer_dic
    result["churn_cus_clv"] = churn_customer_dic

    return JsonResponse(result)

@api_view(["GET"])
def location_based_excutive_summary(request):

        act_start = ''#'2022-01-01 00:00:00'
        act_end = ''#'2022-04-30 00:00:00'
        year = 2020
        month = 0
        custom_filter = True
        is_filter = False
        
        summary_location = "The Sun Inn"
        if 'location' in request.GET:
            summary_location = request.query_params["location"]                                                

        if 'year_val' in request.GET:
            year = int(request.query_params["year_val"])
            act_start = ''
            act_end = ''
        if 'month_val' in request.GET:
            month = int(request.query_params["month_val"]) 

        if 'start_dt_val' in request.GET:
            print(type(request.query_params["start_dt_val"]))
            act_start = str(request.query_params["start_dt_val"]) 
            print(type(act_start))
            year = 0
            month = 0
        if 'end_dt_val' in request.GET:
            act_end = str(request.query_params["end_dt_val"])                     

        #executive summary filter from date & to date
        
        
        prev_act_start = ''#'2021-01-01 00:00:00'
        prev_act_end = ''#'2021-04-30 00:00:00'
        prev_year = 0
        prev_month = 0
        prev_clv_not_found = False

        print("llllllllllllllll*****************************************************************************************")
        print(act_start)
        print(act_end)
        print(month)
        print(year)
        print(summary_location)
        print("*****************************************************************************************")

        #Executive Summary Report

        #one time execution
        #thersold date        

        thersold_date = settings.THERSOLD_MAX_DATE        

        prediction_summary_clv = settings.PREDICTION_CLV_DATAFRAME.copy()
        thersold = thersold_date.split("-")
        thersold_yr = int(thersold[0])
        thersold_mnth = int(thersold[1])
        thersold_day = int(thersold[2])
        a_date = datetime.date(thersold_yr, thersold_mnth, thersold_day)
        prediction_summary_clv['thersold_day'] = a_date
        prediction_summary_clv['customer_age_days'] = pd.to_timedelta(prediction_summary_clv['customer_age'], unit='D')
        prediction_summary_clv['OrderDate'] = prediction_summary_clv['thersold_day'] - prediction_summary_clv['customer_age_days']
        prediction_summary_clv.drop(['thersold_day'],axis=1,inplace=True)
        prediction_summary_clv.drop(['customer_age_days'],axis=1,inplace=True)
        prediction_summary_clv_prev = prediction_summary_clv.copy()

        if(len(act_start.strip())):
            startDate = pd.to_datetime(act_start).date()
            startDate_tmp = pd.to_datetime(act_start).date()
            prev_start = str(startDate_tmp - pd.DateOffset(years=1))
            prev_start = pd.to_datetime(prev_start).date()
            prediction_summary_clv.drop(prediction_summary_clv[prediction_summary_clv['OrderDate'] < startDate].index, inplace = True)
            prediction_summary_clv_prev.drop(prediction_summary_clv_prev[prediction_summary_clv_prev['OrderDate'] < prev_start].index, inplace = True)
            
            if(len(act_end.strip())):
                endDate = pd.to_datetime(act_end).date()
                endDate_tmp = pd.to_datetime(act_end).date()
                prev_end = str(endDate_tmp - pd.DateOffset(years=1))
                prev_end = pd.to_datetime(prev_end).date()
                prediction_summary_clv.drop(prediction_summary_clv[prediction_summary_clv['OrderDate'] > endDate].index, inplace = True)
                prediction_summary_clv_prev.drop(prediction_summary_clv_prev[prediction_summary_clv_prev['OrderDate'] > prev_end].index, inplace = True)
                is_filter = True
            else:
                custom_filter = False
        else:
            custom_filter = False
            

        prediction_summary_clv['OrderDate'] = pd.to_datetime(prediction_summary_clv['OrderDate'])

        # extract date, month and year from dates
        prediction_summary_clv['Month'] = [i.month for i in prediction_summary_clv['OrderDate']]
        prediction_summary_clv['Year'] = [i.year for i in prediction_summary_clv['OrderDate']]
        prediction_summary_clv['day_of_week'] = [i.dayofweek for i in prediction_summary_clv['OrderDate']]
        prediction_summary_clv['day_of_year'] = [i.dayofyear for i in prediction_summary_clv['OrderDate']]

        r, c = prediction_summary_clv_prev.shape
        if r > 0 :
            prediction_summary_clv_prev['OrderDate'] = pd.to_datetime(prediction_summary_clv_prev['OrderDate'])

            # extract date, month and year from dates
            prediction_summary_clv_prev['Month'] = [i.month for i in prediction_summary_clv_prev['OrderDate']]
            prediction_summary_clv_prev['Year'] = [i.year for i in prediction_summary_clv_prev['OrderDate']]
            prediction_summary_clv_prev['day_of_week'] = [i.dayofweek for i in prediction_summary_clv_prev['OrderDate']]
            prediction_summary_clv_prev['day_of_year'] = [i.dayofyear for i in prediction_summary_clv_prev['OrderDate']]
        else:
            prev_clv_not_found = True  
            
        if not custom_filter:
            if(year > 0):
                prev_year = year - 1
                prediction_summary_clv.drop(prediction_summary_clv[prediction_summary_clv['Year'] != year].index, inplace = True)  
                prediction_summary_clv_prev.drop(prediction_summary_clv_prev[prediction_summary_clv_prev['Year'] != prev_year].index, inplace = True)        
                is_filter = True
                
                if(month > 0):
                    prev_month = month - 1
                    prediction_summary_clv.drop(prediction_summary_clv[prediction_summary_clv['Month'] != month].index, inplace = True)
                    prediction_summary_clv_prev.drop(prediction_summary_clv_prev[prediction_summary_clv_prev['Month'] != prev_month].index, inplace = True)        
                    is_filter = True
                    
        r, c = prediction_summary_clv_prev.shape
        if r == 0 :
            prev_clv_not_found = True

        #location based avg CLV of All customer
        prediction_summary_all_clv = prediction_summary_clv.groupby(['Location']).agg({"no_visit" : np.sum, "CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()

        #add, rename and remove the column
        prediction_summary_all_clv['Total_Transaction'] = round(prediction_summary_all_clv['Total_Transaction'],3)
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_ALL"})
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"Transactional Value" : "AVG_REVENUE_ALL"})
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"CtcID" : "TOTAL_USERS"})
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"CLV_3M" : "AVG_CLV_ALL"})
        prediction_summary_all_clv = prediction_summary_all_clv.rename(columns={"no_visit" : "Total_NO_ORDERS"})

        executive_summary_df = prediction_summary_all_clv

        #Repeat Customer
        prediction_summary_repeat_clv = prediction_summary_clv.copy()
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.loc[(prediction_summary_repeat_clv.no_visit >= 2)]
        prediction_summary_repeat_clv['Total_Transaction'] = prediction_summary_repeat_clv['Transactional Value']
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
        prediction_summary_repeat_clv['Total_Transaction'] = round(prediction_summary_repeat_clv['Total_Transaction'],3)
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_REPEAT"})
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.rename(columns={"Transactional Value" : "AVG_REVENUE_REPEAT"})
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.rename(columns={"CtcID" : "REPEAT_USERS"})
        prediction_summary_repeat_clv = prediction_summary_repeat_clv.rename(columns={"CLV_3M" : "AVG_CLV_REPEAT"})

        executive_summary_df = pd.merge(executive_summary_df, prediction_summary_repeat_clv, on=["Location"])

        #New Customer
        prediction_summary_new_clv = prediction_summary_clv.copy()
        prediction_summary_new_clv = prediction_summary_new_clv.loc[(prediction_summary_new_clv.no_visit == 1)]
        prediction_summary_new_clv['Total_Transaction'] = prediction_summary_new_clv['Transactional Value']
        prediction_summary_new_clv = prediction_summary_new_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
        prediction_summary_new_clv['Total_Transaction'] = round(prediction_summary_new_clv['Total_Transaction'],3)
        prediction_summary_new_clv = prediction_summary_new_clv.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_NEW"})
        prediction_summary_new_clv = prediction_summary_new_clv.rename(columns={"Transactional Value" : "AVG_REVENUE_NEW"})
        prediction_summary_new_clv = prediction_summary_new_clv.rename(columns={"CtcID" : "NEW_USERS"})
        prediction_summary_new_clv = prediction_summary_new_clv.rename(columns={"CLV_3M" : "AVG_CLV_NEW"})

        executive_summary_df = pd.merge(executive_summary_df, prediction_summary_new_clv, on=["Location"])

        #Churn Customer
        prediction_summary_churn_clv = prediction_summary_clv.copy()
        prediction_summary_churn_clv = prediction_summary_churn_clv.loc[(prediction_summary_churn_clv.no_visit > 1)]
        cus_total = prediction_summary_churn_clv['CtcID'].size
        cus_avg_freq = round(prediction_summary_churn_clv['frequency'].sum()/cus_total,3)
        prediction_summary_churn_clv = prediction_summary_churn_clv.loc[(prediction_summary_churn_clv.days_since_last_visit > (3 * cus_avg_freq))]
        prediction_summary_churn_clv['Total_Transaction'] = prediction_summary_churn_clv['Transactional Value']
        prediction_summary_churn_clv = prediction_summary_churn_clv.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
        prediction_summary_churn_clv['Total_Transaction'] = round(prediction_summary_churn_clv['Total_Transaction'],3)
        prediction_summary_churn_clv = prediction_summary_churn_clv.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_CHURN"})
        prediction_summary_churn_clv = prediction_summary_churn_clv.rename(columns={"Transactional Value" : "AVG_REVENUE_CHURN"})
        prediction_summary_churn_clv = prediction_summary_churn_clv.rename(columns={"CtcID" : "CHURN_USERS"})
        prediction_summary_churn_clv = prediction_summary_churn_clv.rename(columns={"CLV_3M" : "AVG_CLV_CHURN"})

        executive_summary_df = pd.merge(executive_summary_df, prediction_summary_churn_clv, on=["Location"])

        executive_summary_df['TOTAL_USERS'] = executive_summary_df['TOTAL_USERS'] - executive_summary_df['CHURN_USERS']

        executive_summary_df_final = executive_summary_df[['Location','Total_NO_ORDERS','AVG_REVENUE_ALL','AVG_CLV_ALL','TOTAL_REVENUE_ALL','TOTAL_USERS','REPEAT_USERS','NEW_USERS','CHURN_USERS']]


        if not prev_clv_not_found:
            #location based avg CLV of All customer
            prediction_all_clv_prev = prediction_summary_clv_prev.groupby(['Location']).agg({"no_visit" : np.sum, "CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()

            #add, rename and remove the column
            prediction_all_clv_prev['Total_Transaction'] = round(prediction_all_clv_prev['Total_Transaction'],3)
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_ALL"})
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"Transactional Value" : "AVG_REVENUE_ALL"})
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"CtcID" : "TOTAL_USERS"})
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"CLV_3M" : "AVG_CLV_ALL"})
            prediction_all_clv_prev = prediction_all_clv_prev.rename(columns={"no_visit" : "Total_NO_ORDERS"})

            executive_summary_df_prev = prediction_all_clv_prev

            #Repeat Customer
            prediction_repeat_clv_prev = prediction_summary_clv_prev.copy()
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.loc[(prediction_repeat_clv_prev.no_visit >= 2)]
            prediction_repeat_clv_prev['Total_Transaction'] = prediction_repeat_clv_prev['Transactional Value']
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
            prediction_repeat_clv_prev['Total_Transaction'] = round(prediction_repeat_clv_prev['Total_Transaction'],3)
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_REPEAT"})
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.rename(columns={"Transactional Value" : "AVG_REVENUE_REPEAT"})
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.rename(columns={"CtcID" : "REPEAT_USERS"})
            prediction_repeat_clv_prev = prediction_repeat_clv_prev.rename(columns={"CLV_3M" : "AVG_CLV_REPEAT"})

            executive_summary_df_prev = pd.merge(executive_summary_df_prev, prediction_repeat_clv_prev, on=["Location"])

            #New Customer
            prediction_new_clv_prev = prediction_summary_clv_prev.copy()
            prediction_new_clv_prev = prediction_new_clv_prev.loc[(prediction_new_clv_prev.no_visit == 1)]
            prediction_new_clv_prev['Total_Transaction'] = prediction_new_clv_prev['Transactional Value']
            prediction_new_clv_prev = prediction_new_clv_prev.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
            prediction_new_clv_prev['Total_Transaction'] = round(prediction_new_clv_prev['Total_Transaction'],3)
            prediction_new_clv_prev = prediction_new_clv_prev.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_NEW"})
            prediction_new_clv_prev = prediction_new_clv_prev.rename(columns={"Transactional Value" : "AVG_REVENUE_NEW"})
            prediction_new_clv_prev = prediction_new_clv_prev.rename(columns={"CtcID" : "NEW_USERS"})
            prediction_new_clv_prev = prediction_new_clv_prev.rename(columns={"CLV_3M" : "AVG_CLV_NEW"})

            executive_summary_df_prev = pd.merge(executive_summary_df_prev, prediction_new_clv_prev, on=["Location"])

            #Churn Customer
            prediction_churn_clv_prev = prediction_summary_clv_prev.copy()
            prediction_churn_clv_prev = prediction_churn_clv_prev.loc[(prediction_churn_clv_prev.no_visit > 1)]
            cus_total = prediction_churn_clv_prev['CtcID'].size
            cus_avg_freq = round(prediction_churn_clv_prev['frequency'].sum()/cus_total,3)
            prediction_churn_clv_prev = prediction_churn_clv_prev.loc[(prediction_churn_clv_prev.days_since_last_visit > (3 * cus_avg_freq))]
            prediction_churn_clv_prev['Total_Transaction'] = prediction_churn_clv_prev['Transactional Value']
            prediction_churn_clv_prev = prediction_churn_clv_prev.groupby(['Location']).agg({"CLV_3M" : np.mean, "Transactional Value" : np.mean, "Total_Transaction" : np.sum, "CtcID" : np.size}).reset_index()
            prediction_churn_clv_prev['Total_Transaction'] = round(prediction_churn_clv_prev['Total_Transaction'],3)
            prediction_churn_clv_prev = prediction_churn_clv_prev.rename(columns={"Total_Transaction" : "TOTAL_REVENUE_CHURN"})
            prediction_churn_clv_prev = prediction_churn_clv_prev.rename(columns={"Transactional Value" : "AVG_REVENUE_CHURN"})
            prediction_churn_clv_prev = prediction_churn_clv_prev.rename(columns={"CtcID" : "CHURN_USERS"})
            prediction_churn_clv_prev = prediction_churn_clv_prev.rename(columns={"CLV_3M" : "AVG_CLV_CHURN"})

            executive_summary_df_prev = pd.merge(executive_summary_df_prev, prediction_churn_clv_prev, on=["Location"])

            executive_summary_df_prev['TOTAL_USERS'] = executive_summary_df_prev['TOTAL_USERS'] - executive_summary_df_prev['CHURN_USERS']

            executive_summary_df_final_prev = executive_summary_df_prev[['Location','Total_NO_ORDERS','AVG_REVENUE_ALL','AVG_CLV_ALL','TOTAL_REVENUE_ALL','TOTAL_USERS','REPEAT_USERS','NEW_USERS','CHURN_USERS']]
            
            
            r1, c1 = executive_summary_df_final_prev.shape
            if r1>0 :
                executive_summary_df_final['Total_NO_ORDERS_Changes'] = round((executive_summary_df_final['Total_NO_ORDERS'] - executive_summary_df_final_prev['Total_NO_ORDERS'])/executive_summary_df_final_prev['Total_NO_ORDERS'],2)*100
                executive_summary_df_final['AVG_REVENUE_ALL_Changes'] = round((executive_summary_df_final['AVG_REVENUE_ALL'] - executive_summary_df_final_prev['AVG_REVENUE_ALL'])/executive_summary_df_final_prev['AVG_REVENUE_ALL'],2)*100
                executive_summary_df_final['AVG_CLV_ALL_Changes'] = round((executive_summary_df_final['AVG_CLV_ALL'] - executive_summary_df_final_prev['AVG_CLV_ALL'])/executive_summary_df_final_prev['AVG_CLV_ALL'],2)*100
                executive_summary_df_final['TOTAL_REVENUE_ALL_Changes'] = round((executive_summary_df_final['TOTAL_REVENUE_ALL'] - executive_summary_df_final_prev['TOTAL_REVENUE_ALL'])/executive_summary_df_final_prev['TOTAL_REVENUE_ALL'],2)*100
                executive_summary_df_final['TOTAL_USERS_Changes'] = round((executive_summary_df_final['TOTAL_USERS'] - executive_summary_df_final_prev['TOTAL_USERS'])/executive_summary_df_final_prev['TOTAL_USERS'],2)*100
                executive_summary_df_final['REPEAT_USERS_Changes'] = round((executive_summary_df_final['REPEAT_USERS'] - executive_summary_df_final_prev['REPEAT_USERS'])/executive_summary_df_final_prev['REPEAT_USERS'],2)*100
                executive_summary_df_final['NEW_USERS_Changes'] = round((executive_summary_df_final['NEW_USERS'] - executive_summary_df_final_prev['NEW_USERS'])/executive_summary_df_final_prev['AVG_REVENUE_ALL'],2)*100
                executive_summary_df_final['CHURN_USERS_Changes'] = round((executive_summary_df_final['CHURN_USERS'] - executive_summary_df_final_prev['CHURN_USERS'])/executive_summary_df_final_prev['CHURN_USERS'],2)*100
        else:
            executive_summary_df_final['Total_NO_ORDERS_Changes'] = '-'
            executive_summary_df_final['AVG_REVENUE_ALL_Changes'] = '-'
            executive_summary_df_final['AVG_CLV_ALL_Changes'] = '-'
            executive_summary_df_final['TOTAL_REVENUE_ALL_Changes'] = '-'
            executive_summary_df_final['TOTAL_USERS_Changes'] = '-'
            executive_summary_df_final['REPEAT_USERS_Changes'] = '-'
            executive_summary_df_final['NEW_USERS_Changes'] = '-'
            executive_summary_df_final['CHURN_USERS_Changes'] = '-'
            executive_summary_df_final_prev = pd.DataFrame()
            executive_summary_df_final_prev["Location"] = '-'
            executive_summary_df_final_prev["TOTAL_USERS"] = '-'
            executive_summary_df_final_prev["Total_NO_ORDERS"] = '-'
            executive_summary_df_final_prev["TOTAL_REVENUE_ALL"] = '-'
            executive_summary_df_final_prev["AVG_REVENUE_ALL"] = '-'
            executive_summary_df_final_prev["NEW_USERS"] = '-'
            executive_summary_df_final_prev["REPEAT_USERS"] = '-'
            executive_summary_df_final_prev["CHURN_USERS"] = '-'
            executive_summary_df_final_prev["AVG_CLV_ALL"] = '-'

        executive_summary_df_final.fillna('-', inplace=True)        
        executive_summary_dict = {}

        print(executive_summary_df_final_prev)
        print("*************************************************************************")
        print(executive_summary_df_final[executive_summary_df_final["Location"] == summary_location].size)
        print(executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location].size)
        print("*************************************************************************")
        

        if executive_summary_df_final[executive_summary_df_final["Location"] == summary_location].size != 0 :

            if executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location].size == 0 :
                executive_summary_dict["customers_prev"] = "-"   
                executive_summary_dict["customers_prev"] = "-"   
                executive_summary_dict["orders_prev"] = "-"   
                executive_summary_dict["revenue_prev"] = "-"   
                executive_summary_dict["aov_prev"] = "-"   
                executive_summary_dict["new_customers_prev"] = "-"   
                executive_summary_dict["repeat_customers_prev"] = "-"   
                executive_summary_dict["churn_customers_prev"] = "-"   
                executive_summary_dict["aclv_customers_prev"] = "-"   
            else:
                executive_summary_dict["customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["TOTAL_USERS"].tolist()[0]    
                executive_summary_dict["orders_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["Total_NO_ORDERS"].tolist()[0] 
                executive_summary_dict["revenue_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["TOTAL_REVENUE_ALL"].tolist()[0] 
                executive_summary_dict["aov_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["AVG_REVENUE_ALL"].tolist()[0] 
                executive_summary_dict["new_customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["NEW_USERS"].tolist()[0] 
                executive_summary_dict["repeat_customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["REPEAT_USERS"].tolist()[0]  
                executive_summary_dict["churn_customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["CHURN_USERS"].tolist()[0] 
                executive_summary_dict["aclv_customers_prev"] = executive_summary_df_final_prev[executive_summary_df_final_prev["Location"] == summary_location]["AVG_CLV_ALL"].tolist()[0] 


            executive_summary_dict["customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["TOTAL_USERS"].tolist()[0]
            executive_summary_dict["customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["TOTAL_USERS_Changes"].tolist()[0] 
                         

            executive_summary_dict["orders_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["Total_NO_ORDERS"].tolist()[0]  
            executive_summary_dict["orders_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["Total_NO_ORDERS_Changes"].tolist()[0]
            
            executive_summary_dict["revenue_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["TOTAL_REVENUE_ALL"].tolist()[0] 
            executive_summary_dict["revenue_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["TOTAL_REVENUE_ALL_Changes"].tolist()[0] 
            
            executive_summary_dict["aov_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["AVG_REVENUE_ALL"].tolist()[0]  
            executive_summary_dict["aov_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["AVG_REVENUE_ALL_Changes"].tolist()[0]  
            
            executive_summary_dict["new_customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["NEW_USERS"].tolist()[0]  
            executive_summary_dict["new_customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["NEW_USERS_Changes"].tolist()[0] 
            
            executive_summary_dict["repeat_customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["REPEAT_USERS"].tolist()[0]
            executive_summary_dict["repeat_customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["REPEAT_USERS_Changes"].tolist()[0]  
            
            executive_summary_dict["churn_customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["CHURN_USERS"].tolist()[0]  
            executive_summary_dict["churn_customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["CHURN_USERS_Changes"].tolist()[0]
            
            executive_summary_dict["aclv_customers_selected"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["AVG_CLV_ALL"].tolist()[0]  
            executive_summary_dict["aclv_customers_changes"] = executive_summary_df_final[executive_summary_df_final["Location"] == summary_location]["AVG_CLV_ALL_Changes"].tolist()[0] 
            
        result = {}
        result["executive_summary"] = executive_summary_dict
        return JsonResponse(result)


@api_view(["GET"])
def get_location(request):    
    
    location_list = ["The Sun Inn", "The Cricketers", "The Jobber\'s Rest", "Brasserie Blanc Winchester",
                 "Brasserie Blanc Portsmouth", "Brasserie Blanc Chichester","The Kings Head","The Boot",
                 "The Oaks","Brasserie Blanc Cheltenham","The Highwayman","Brasserie Blanc Leeds","Brasserie Blanc Threadneedle Street",
                 "Brasserie Blanc Southbank","The March Hare","The Queens Head","The White Bear","Brasserie Blanc Milton Keynes",
                 "Brasserie Blanc Chancery Lane","The Barley Mow","The Jolly Farmer","The Hare","Reigate","Brasserie Blanc Bournemouth",
                 "Brasserie Blanc Hale Barns","The Black Horse - Thame","The British Queen","Brasserie Blanc Beaconsfield","The Red Deer",
                 "The King\'s Arms","Brasserie Blanc Oxford","Fulham Reach","Brasserie Blanc Bath","Brasserie Blanc Tower of London",
                 "Brasserie Blanc Knutsford","The Victoria","The Oakwood","Brasserie Blanc Bristol","The Black Horse - Reigate"]               
    return JsonResponse(location_list,safe=False)


def login(request):        
    return render(request,'login.html')

def home(request):        
    return render(request,'home.html')