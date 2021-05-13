import pandas as pd
from pytrends.request import TrendReq
from prophet import Prophet
import sys

def get_trends(key_words):

    pytrends = TrendReq()
    pytrends.build_payload(kw_list=key_words, timeframe='all', geo='PL')
    trends = pytrends.interest_over_time().resample('MS').sum() # .drop('isPartial', axis=1)

    return trends


def train(name, sells_ts=None, future_periods=12):

    if sells_ts is None:
        sells_ts = pd.read_csv('./data/sells_time_series.csv', index_col=0)
        sells_ts.index = pd.to_datetime(sells_ts.index)

    try:
        sells = sells_ts[name].to_frame()
    except KeyError:
        print("Nie ma takiej nazwy klastra!")
        sys.exit()

    # Prophet wymaga takich nazw
    sells.reset_index(inplace=True)
    sells.columns = ['ds', 'y']

    trends = get_trends(key_words=[name])

    # 1. Model wyuczony na samej sprzedaży
    m1 = Prophet(weekly_seasonality=True, daily_seasonality=True)
    m1.fit(sells)

    future = m1.make_future_dataframe(periods=future_periods, freq='MS')
    fcst1 = m1.predict(future)

    # 2. Regresja samych google trendsów
    # Tworzę osobną ramkę trends_history, która zawiera trendsy, ale z nazwami których wymaga Prophet
    trends_history = trends.reset_index()
    trends_history.columns = ['ds', 'y']

    m_trends = Prophet(weekly_seasonality=True, daily_seasonality=True)
    m_trends.fit(trends_history)
    fcst_trends = m_trends.predict(future)
    trends_trend = fcst_trends['trend']
    trends_seasonal = fcst_trends['yearly']

    # 3. Regresja sprzedaży na komponencie sezonowym google trendsów
    # sells = pd.concat([sells, trends.loc[sells['ds']].reset_index()[name]], axis=1)
    sells = pd.concat([sells, trends_seasonal[:-future_periods],
                       trends_trend[:-future_periods], trends.loc[sells['ds']].reset_index()[name]
                       ], axis=1)

    sells.rename(columns={'yearly': 'seasonal'}, inplace=True)

    m2 = Prophet(weekly_seasonality=True, daily_seasonality=True)
    m2.add_regressor('seasonal')
    m2.fit(sells)

    future['seasonal'] = trends_seasonal.values.squeeze()

    fcst2 = m2.predict(future)

    # 4. Regresja sprzedaży na komponencie sezonowym google trendsów oraz na trendzie google trendsów
    sells.rename(columns={'trend': "trends' trend"}, inplace=True)

    m3 = Prophet(weekly_seasonality=True, daily_seasonality=True)
    m3.add_regressor('seasonal')
    m3.add_regressor("trends' trend")
    m3.fit(sells)

    future["trends' trend"] = trends_trend.values.squeeze()
    fcst3 = m3.predict(future)

    # Regresja sprzedaży na raw google trendsach
    m4 = Prophet(weekly_seasonality=True, daily_seasonality=True)
    m4.add_regressor(name)
    m4.fit(sells)

    future[name] = fcst_trends['yhat']
    fcst4 = m4.predict(future)

    return (m1, fcst1), (m2, fcst2), (m3, fcst3), (m4, fcst4)


if __name__ == "__main__":

    # SAMPLE USAGE:

    sells_ts = pd.read_csv('../data/sells_time_series.csv', index_col=0)
    sells_ts.index = pd.to_datetime(sells_ts.index)
    m1, m2, m3, m4 = train('markiza', sells_ts=sells_ts)

    fig1 = m1[0].plot(m1[1])
    fig2 = m2[0].plot(m2[1])
    fig3 = m3[0].plot(m3[1])
    fig4 = m4[0].plot(m4[1])

    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
