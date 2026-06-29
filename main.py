import numpy as np
import pandas as pd
import os
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.chdir(r"C:\Users\Owner1\OneDrive\data_science_journey\data")
string = "stock_data"
##various question to format the time serie and prepare forecast : user provided. Eventually, the user  will have to provide a csv as well. 
"""
da_var = string(input("Provide the name of the date variable."))
freq = string(input("The frequency of data collection (D, B (Business day), W, M, Y)"))
format = string(input("Type of date format (timestamp of period)."))
"""
data = pd.read_csv(f"{string}.csv")
def datetime_manage(data, date_var , frequency , per_stamp = "per") : ##for business day periods periodicity = B
    try : 
        if per_stamp == "stamp":
            data["datetime"] = pd.to_datetime(data[date_var])##timestamp object
            data["datetime"] = data["datetime"].dt.to_period(frequency).dt.to_timestamp()
        else : 
            data["datetime"] = pd.PeriodIndex(pd.to_datetime(data[date_var]), freq=frequency)  # Use PeriodIndex for Series
        
        data = data.copy()  # Avoid mutating the original
        data.index = data["datetime"]
        data_final = data.drop(columns=["datetime", date_var])
    except (ValueError , AttributeError) as e  :
        print(e)
        return None
    
        
    return data_final
    
serie_copy = pd.Series
def series_diff(name_serie, seuil = 0.05) : ##series building. Differentiating the time serie until the trend between past value and current value no longer exists
    df = datetime_manage(data, date_var = "date", frequency = "B", per_stamp = "per")
    serie = df.loc[df["stock"] == "ADBE", name_serie] 
    last_value = serie.iloc[-1]
    list_data = []
    serie_copy = serie.copy()
    try : 
        result = adfuller(serie, autolag='AIC')
        j = result[1]
        index = 0
        new_serie = serie.copy()
        while j >seuil :
            new_serie = new_serie.diff().dropna()
            result = adfuller(new_serie, autolag='AIC')
            j= result[1]
            index += 1
            if index > 10 :
                print(f"Error :Too many differentiation required to produce a usable stationnary serie.")
                return None

        print(f"The serie in integrated of degree {index}")
    except ValueError as e:
        print(e)
        return None
    result_df = data.copy()  # ← list concat, not .append()
    
    result_df = new_serie
    
    return index, result_df, last_value, serie_copy
"""
brief description of computing object

input : 
pd.serie format needed. It takes a serie with the date value as index. 

LAG  lags included in the autoregressive model.

the object at initialization estimates the model at first and stock the value in self.model 

Method such as self.predict and self.bootstrapping makes future prediction about serie's value.  Errors are computed by bootstrapping method. 

"""    
class computing : 
    def __init__(self, serie, lag = 1) :
        self.serie = serie[1]
        self.index_serie = serie[0]
        self.last_value_serie = serie[2]
        self.serie_original_copy = serie[3]
        self.lag = lag
        self.model = 0
        self.fitted_model = 0 
        self.compute()
    def modelling(self, serie) : #serie must already be stationnary
        try : 
            model = AutoReg(self.serie, lags = self.lag)
            res = model.fit()
            ##test for heterosckedascity
            cov = "nonrobust" #this assumes homosckedasticity. For heterosckedascitic serie, "HAC" : Heteroskedasticity-autocorrelation robust covariance estimation
            res = model.fit(cov_type = cov)
            #print(res.summary())
            #print(f"aic ={res.aic}")
        except ValueError as e : 
            print(e)
            return None
        except AttributeError as a : 
            print(a)
            return None
        self.model = model
        self.fitted_model = res

    def compute(self):
        try : 
            serie= self.serie###à changer
            serie = self.serie.asfreq('B').fillna(0)
            self.modelling(serie)
        except AttributeError as a : 
            print(a)
            return None
    def predict(self, step = 10 ) :       
        try : 
            fitted_model = self.fitted_model
            pred_values = fitted_model.forecast(steps = step)
        except ValueError as e : 
            print(e)
            return None
        return pred_values    
    def bootstrapping(self, step = 10,alpha = 0.05, n_bootstrap = 1000): 
        resid = self.fitted_model.resid
        pred = self.predict(step = step)#list
        l_resid = len(resid)
        possible_ys = {i : [] for i in range(len(pred))}
        last_value = self.last_value_serie
        r_value = 0
        for _ in range(n_bootstrap) : 
            cumulative_error = 0
            
            for j, value in enumerate(pred): 
                random_resid = np.random.choice(resid)
                cumulative_error += random_resid         # accumulate errors across steps
                poss_val = value + cumulative_error
                possible_ys[j].append(poss_val)
        if self.index_serie > 0:
            reconstructed_means = []
            reconstructed_lower = []
            reconstructed_upper = []
            lv = self.last_value_serie
            for i in range(len(pred)):
                vals = np.array(possible_ys[i]) + lv
                reconstructed_means.append(np.mean(vals))
                reconstructed_lower.append(np.percentile(vals, 100 * alpha / 2))
                reconstructed_upper.append(np.percentile(vals, 100 * (1 - alpha / 2)))
                lv = np.mean(vals)
        return [max(0, x) for x in reconstructed_lower], reconstructed_upper, reconstructed_means
        


working_serie  = series_diff(name_serie = "close")

case = computing(working_serie, 3)
l, u, m = case.bootstrapping()
pred = m
    
##append orginal serie with new_values and dates index :
def append_predictions(serie: pd.Series, predictions: np.ndarray, freq: str = "B") -> pd.Series:
    k = len(predictions)
    last_date = serie.index[-1]
    if isinstance(last_date, pd.Period):
        last_date = last_date.to_timestamp(how="end")  # "end" = last moment of the period
    new_dates = pd.date_range(start=last_date, periods=k + 1, freq=freq)[1:]
    pred_serie = pd.Series(predictions, index=new_dates)
    pred_serie.index.name = serie.index.name
    
    return pd.concat([serie, pred_serie])
data_to_plot = append_predictions(working_serie[3], pred, "B")
fig, ax = plt.subplots()


data_to_plot.index = pd.PeriodIndex(data_to_plot.index).to_timestamp()
ax.plot(data_to_plot.iloc[-15:])

# Shaded uncertainty band for the last 2 points
k = len(pred)
x_band = data_to_plot.index[-k:]
y_band = data_to_plot.iloc[-k:]

ax.fill_between(
    x_band,
    l,   # lower bound
    u,   # upper bound     
    alpha=0.3,
    color='blue',
    label='Forecast range'
)

plt.legend()
plt.show()
#plt.savefig(key+".png", format = "png")
#plt.close()
