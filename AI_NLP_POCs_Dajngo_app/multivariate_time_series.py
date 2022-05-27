import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from ImageClassification.settings import BASE_DIR
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# fix random seed for reproducibility
# numpy.random.seed(7)
# look_back = 1
model = joblib.load('C:/Users/RNALAB/Documents/ts_multivar_cc')
#scaler = joblib.load('C:/Users/RNALAB/Documents/Minmax_scaler')
most_imp_cols = ['agent_headcount', 'busy_time', 'total_calls_duration',
       'after_call_work_time', 'utilization_rate']

def read_csv_data(csv_filename):
    if csv_filename.split(".")[-1] =='csv':
        data = pd.read_csv(csv_filename,sep =',')
        return data
    elif csv_filename.split(".")[-1] =='xlsx':
        data = pd.read_excel(csv_filename,sep =',')
        return data


def multivariate_ts_forecast(csv_file):


    ts_data = read_csv_data(csv_file)

    ts_data.dropna(axis=1, how='all', inplace=True)
    ts_data.dropna(axis=0, how='all', inplace=True)
    ts_data['interval'] = pd.to_datetime(ts_data['interval'])
    ts_data['interval'] = ts_data['interval'].apply(lambda x:
                                              pd.to_datetime(x, format='%Y%m%d', errors='ignore').date())
    daily = ts_data.groupby('interval').mean().copy()

    # Casting agent_headcount from flot to Int. Avoid error continuous in the prediction
    daily['agent_headcount'] = daily['agent_headcount'].astype('int64', copy=False)

    ts_data_filt = daily[most_imp_cols]

    rf_predictions = model.predict(ts_data_filt)
    daily["total_calls"] = np.round(np.array(daily["total_calls"]), 0)
    daily["Total_calls_Predicted"] = np.round(rf_predictions,0)

    excel_filename = "Multivariate_Time_Series_Forecasting" + ".xlsx"
    daily.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    return True, df_filepath



    # with PdfPages("C:/Users/RNALAB/Documents/ImageClassification_test/media/timeseries_forecast.pdf") as pdf:
    #     data_plot = data.copy()
    #     data_plot.set_index('Timestamp', inplace=True)
    #     plt.rcParams['figure.figsize'] = (22, 10)
    #     plt.plot(data_plot['Passengers(Actual_Values)'].iloc[100:120])
    #     plt.plot(data_plot['Passengers(Forcasted_Values)'].iloc[100:120])
    #     plt.legend()
    #     # plt.show()
    #     pdf.savefig()  # saves the current figure into a pdf page
    #     plt.close()
    #
    # #pdf_filename = "timeseries_forecast.pdf"
    # # df.to_excel(BASE_DIR+'/media/' + excel_filename)
    #
    # #df_pdf_filepath = '/media/' + pdf_filename
    #
    #
    # return True, df_filepath