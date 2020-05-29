import pandas as pd
from pytrends.request import TrendReq
from fbprophet import Prophet
import numpy as np
from scipy.signal import find_peaks

#Transformation du CSV d'entrée en liste de KW
def kw_to_vec(file) :
    """
    Fonction simple pour transformer un CSV contenant des mots-clés en liste de mots-clés
    :param file: un fichier CSV contenant des mots clés
    :return: une liste de mots clés
    """
    kw_list = pd.read_csv(file, sep = ";")
    kw_vec = list(kw_list["kw"])
    return kw_vec

#Récupération de l'ensemble des trends, 1 à 1
def get_data_trend(kw_list, country) :
    """
    La fonction qui récupère les TRENDS 1 à 1 pour chaque mot-clé présent dans la liste
    :param kw_list: liste de mots clés
    :param country: le pays ou se fait la recherche
    :return: une liste de dataframe - 1 DF par mot-clé
    """
    country = country
    datalist = [] #objet à retourner

    for kw in kw_list :
        # on initalise pytrends
        pytrends = TrendReq(hl = country, tz=360)
        # paramétrage de la requête Google Trend
        pytrends.build_payload(kw_list=[kw], cat=0, timeframe='today 5-y',
                               geo=country, gprop='')
        kw_over_time = pytrends.interest_over_time()
        # on récupère la data "Kewyword Over Time"
        kw_over_time["keyword"] = kw_over_time.columns[0]
        # On rajoute une colonne "mot clé" nécessaire à la suite des traitements
        kw_over_time = kw_over_time.rename(columns={kw_over_time.columns[0]: 'hits'})
        # on renomme la 1ère colonne
        datalist.append(kw_over_time)

    return datalist

#Prediction Prophet pour chaque trend
def prophet_kws(trends) :
    """
    Prédiction de l'année suivante, via la lib FB Prophet
    :param trends: une liste de dataframe contenant les trends pour 1 mot clé
    :return: une liste de dataframe
    """
    datalist = [] # la liste de DF que l'on va return en fin de process

    ###Debut du process
    # On parcours la liste de trend
    for trend in trends:
        kw = trend.keyword[0] # sauvegarde du KW
        print(kw)
        trend_no_index = trend.copy()
        trend_no_index.reset_index(inplace=True) #on transforme l'index de date en colonne de date
        kw_timeseries = trend_no_index[['hits', 'date']] # on ne conserve que les colonnes hits et date
        kw_timeseries = kw_timeseries.rename(columns={'date': 'ds', 'hits': 'y'}) # on les renomme pour répondre aux param de Prophet
        m = Prophet()
        m.fit(kw_timeseries)
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        kw_forecast = forecast.copy()
        kw_forecast['keyword'] = kw #on récupéère le KW en créant une nouvelle colonne
        kw_forecast['segment'] = np.where(kw_forecast['ds'] > pd.Timestamp.now(), 'forecast', 'actual') # nouvelle colonne segment, forecast si date prédite
        kw_forecast = kw_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'keyword', 'segment']] # on conserve les colonnes dont on a besoin
        kw_forecast = kw_forecast.merge(kw_timeseries, on='ds', how='left') # on récupère le trend (y) d'origine
        datalist.append(kw_forecast) #on feed notre liste
    return datalist

#Obtenir les max
def get_max(list_kws, nbmax) :
    datalist = []

    for df in list_kws :
        kw_forecast = df[df['segment'] == 'forecast']
        time_series = kw_forecast['yhat']
        indices = find_peaks(time_series)[0]
        datalist.append(indices)
    return datalist


