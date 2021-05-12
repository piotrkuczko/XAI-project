import pandas as pd
from pytrends.request import TrendReq
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import xgboost


def return_series(df, name, key_words=None):
    """
    :param df: final_dataframe.
    :param name: nazwa wybrenej grupy (klastra) produktów
    :param key_words: key_words to lista słów dla których ma być wyszukany trend. Jeśli nie podacie nic, to domyślnie
                      do key_words trafi nazwa produktów name
    """

    if key_words is None:
        key_words = [name]

    pytrends = TrendReq()
    pytrends.build_payload(kw_list=key_words, timeframe='2018-02-01 2021-02-28', geo='PL')
    trends = pytrends.interest_over_time().resample('MS').sum().drop('isPartial', axis=1)[name]

    sells = df[name]
    return sells, trends


def lag_transform(s, lags_name, lags=12):
    """
    :param s: pandas.Series
    :param lag: ile kroków do tyłu
    """

    new_rows = []
    window = [i for i in range(len(s) - lags + 1)]
    for w in window:
        new_rows.append(s[w: w + 12].values.tolist())

    col_names = [lags_name + '_lag' + str(i) for i in range(lags, 0, -1)]
    lagged_data = pd.DataFrame(new_rows, columns=col_names)
    return lagged_data


def train(name, sells_ts):

    sells, trends = return_series(df=sells_ts, name='lampa')

    y = sells[11:].reset_index()
    y.columns = ['Data', 'sprzedaż']
    frames = [y, lag_transform(sells, 'sprzedaż'), lag_transform(trends, 'trendy')]
    df = pd.concat(frames, axis=1)

    index_for_preds = pd.date_range(start='2019/1/1', end='2021/3/1', freq="M").to_period('M')

    print('Fitting lasso...')
    # TODO parameter grid search
    lasso = linear_model.Lasso(alpha=1.)
    lasso.fit(X=df.drop(['sprzedaż', 'Data'], axis=1),
              y=df['sprzedaż'])
    lasso_y_hat = lasso.predict(df.drop(['sprzedaż', 'Data'], axis=1))
    lasso_train_error = mean_squared_error(lasso_y_hat, df['sprzedaż'], squared=False)
    print(f'Lasso training RMSE: {lasso_train_error}')
    lasso_y_hat = pd.Series(lasso_y_hat, index=index_for_preds)

    print('Fitting XGBoost...')
    # TODO parameter grid search
    xgb = xgboost.XGBRegressor()
    xgb.fit(X=df.drop(['sprzedaż', 'Data'], axis=1),
              y=df['sprzedaż'])
    xgb_y_hat = xgb.predict(df.drop(['sprzedaż', 'Data'], axis=1))
    xgb_train_error = mean_squared_error(xgb_y_hat, df['sprzedaż'], squared=False)
    print(f'XGBoost training RMSE: {xgb_train_error}')
    xgb_y_hat = pd.Series(xgb_y_hat, index=index_for_preds)

    results = dict()
    results['sells'] = sells
    results['trends'] = trends
    results['dataframe'] = df
    results['models'] = {'lasso': lasso,
                         'xgb': xgb}
    results['preds'] = {'lasso': lasso_y_hat,
                        'xgb': xgb_y_hat}

    return results
