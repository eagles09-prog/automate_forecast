import numpy as np
import pandas as pd
import os
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


##various question to format the time serie and prepare forecast : user provided. Eventually, the user  will have to provide a csv as well. 
data = pd.read_csv(r"C:\Users\Owner1\OneDrive\data_science_journey\data\bdccpitrim_data.csv")
da_var = data.columns[0]
freq = "M"
dformat = "period"
endog_name = data.columns[-1]
print(endog_name)


def datetime_manage(data, date_var , frequency , per_stamp = "per") : ##for business day periods periodicity = B
    
    try:
        parsed = pd.to_datetime(data[date_var].astype(str), infer_datetime_format=True)

        if per_stamp == "stamp":
            data = data.copy()
            data["datetime"] = parsed.dt.to_period(frequency).dt.to_timestamp()
        else:
            data = data.copy()
            data["datetime"] = pd.PeriodIndex(parsed, freq=frequency)

        data.index = data["datetime"]
        data_final = data.drop(columns=["datetime", date_var])
        
    except (ValueError , AttributeError) as e  :
        print(e)
        return None
    
        
    return data_final
    
serie_copy = pd.Series
def series_diff(name_serie, seuil = 0.05) : ##series building. Differentiating the time serie until the trend between past value and current value no longer exists
    df = datetime_manage(data, date_var = da_var, frequency = freq, per_stamp = dformat)
    serie = df.loc[:, name_serie] 
    last_value = serie.iloc[-1]
    list_data = []
    serie_copy = serie.copy()
    print(serie.index)
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
        if serie is None:
            raise ValueError("series_diff() returned None — check stationarity/parsing errors above.")
        self.serie = serie[1]
        self.index_serie = serie[0]
        self.last_value_serie = serie[2]
        self.serie_original_copy = serie[3]
        self.lag = lag
        self.model = 0
        self.fitted_model = 0 
        self.compute()
    def modelling(self,serie): #serie must already be stationnary
        try : 
            model = AutoReg(serie, lags=self.lag)
            res = model.fit()

            # Breusch-Pagan test on the (homoskedastic) fit's residuals
            exog = res.model.exog  # includes constant + lagged regressors used in fitting
            exog_bp = sm.add_constant(res.fittedvalues)
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(res.resid, exog_bp)
            cov_type = "HAC" if bp_pvalue < 0.05 else "nonrobust"
            cov_kwds = {"maxlags": self.lag} if cov_type == "HAC" else None

            res = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
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
            serie = self.serie.asfreq(freq).fillna(0)
            self.modelling(serie)
            if self.fitted_model is None or self.fitted_model == 0:
                raise RuntimeError("Model fitting failed — see printed error above.")
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
    import numpy as np

    def bootstrapping(self, step=10, alpha=0.05, n_bootstrap=1000,clip_nonnegative=True):
        resid = np.asarray(self.fitted_model.resid)
        pred = np.asarray(self.predict(step=step))  # Predictions on d=2 stationary scale
        
        # Extract historical original series (adjust 'self.serie' to match your attribute name)
        y_history = np.asarray(self.serie_original_copy)  
        
        # Anchor points for d = 2
        last_y = y_history[-1]                  # Y_T
        last_delta_y = y_history[-1] - y_history[-2]  # Delta Y_T = Y_T - Y_{T-1}

        # Matrix to store all reconstructed bootstrap paths: shape = (n_bootstrap, step)
        bootstrapped_paths = np.zeros((n_bootstrap, step))
        
        for b in range(n_bootstrap):
            # 1. Sample residuals with replacement
            sampled_resids = np.random.choice(resid, size=step, replace=True)
            z_simulated = pred + sampled_resids
            
            # 2. Reverse differencing (d = 2)
            if self.index_serie == 2:
                # Step A: Integrate once to get Delta Y
                delta_y_sim = last_delta_y + np.cumsum(z_simulated)
                # Step B: Integrate second time to get original level Y
                y_simulated = last_y + np.cumsum(delta_y_sim)
                
            elif self.index_serie == 1:
                y_simulated = last_y + np.cumsum(z_simulated)
                
            else: # d = 0
                y_simulated = z_simulated
                
            bootstrapped_paths[b, :] = y_simulated

        # 3. Calculate percentiles across all bootstrap paths
        reconstructed_means = np.mean(bootstrapped_paths, axis=0)
        reconstructed_lower = np.percentile(bootstrapped_paths, 100 * (alpha / 2), axis=0)
        reconstructed_upper = np.percentile(bootstrapped_paths, 100 * (1 - alpha / 2), axis=0)
        
        # Apply lower bound non-negativity constraint
        if clip_nonnegative:
            reconstructed_lower = np.maximum(0, reconstructed_lower)
        

        return reconstructed_lower.tolist(), reconstructed_upper.tolist(), reconstructed_means.tolist()


working_serie  = series_diff(name_serie = endog_name)
if working_serie is None:
    raise SystemExit("Could not build a stationary series — see errors above.")
case = computing(working_serie, 3)
l, u, m = case.bootstrapping()
pred = m
    
##append orginal serie with new_values and dates index :
def append_predictions(serie: pd.Series, predictions: np.ndarray, freq: str = freq) -> pd.Series:
    k = len(predictions)
    last_period = serie.index[-1]  # already a Period if serie has a PeriodIndex

    if not isinstance(last_period, pd.Period):
        last_period = pd.Period(last_period, freq=freq)

    new_periods = pd.period_range(start=last_period + 1, periods=k, freq=freq)
    pred_serie = pd.Series(predictions, index=new_periods)
    pred_serie.index.name = serie.index.name

    return pd.concat([serie, pred_serie])
data_to_plot = append_predictions(working_serie[3], pred, freq)
data_to_plot.index = data_to_plot.index.to_timestamp()
fig, ax = plt.subplots(figsize = (12,6))



ax.plot(data_to_plot.iloc[-15:])
plt.ylabel("PIB")

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
    label='Prédictions à un degré de certitude de 95 %'
)
#plt.figure(figsize=(4,5))
plt.tight_layout(pad=0.5)

plt.legend()
plt.show()
#plt.savefig(key+".png", format = "png")
#plt.close()