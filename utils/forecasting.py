import pandas as pd
from datetime import datetime

try:
    from prophet import Prophet
    HAS_PROPHET = True
except:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_HW = True
except:
    HAS_HW = False


def forecast_series(ts, periods=4, freq='W'):
    # التأكد من أن العمود ds يحتوي على نوع datetime
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index, errors='coerce')

    df = ts.reset_index()
    df.columns = ['ds', 'y']
    df.dropna(inplace=True)

    # التحقق إذا كانت البيانات قليلة جداً
    if len(df) < 2:
        future = pd.date_range(df['ds'].max(), periods=periods + 1, freq=freq)[1:]
        return pd.DataFrame({'ds': future, 'yhat': [df['y'].iloc[-1]] * periods})

    # إذا كان مكتبة Prophet موجودة
    if HAS_PROPHET:
        try:
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=periods, freq=freq)
            fc = m.predict(future)
            return fc[['ds', 'yhat']].tail(periods)
        except Exception as e:
            print(f"Prophet failed: {e}")

    # إذا كانت مكتبة Holt-Winters موجودة
    if HAS_HW:
        model = ExponentialSmoothing(df['y'], trend='add')
        fit = model.fit()
        pred = fit.forecast(periods)
        future = pd.date_range(df['ds'].max(), periods=periods + 1, freq=freq)[1:]
        return pd.DataFrame({'ds': future, 'yhat': pred.values})

    return None
