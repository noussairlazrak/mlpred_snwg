# -*- coding: utf-8 -*-
# ! /usr/bin/env python
""" 
mlpred.py

This file handles localized forecasts, based on GMAO's GEOS CF and OpenAQ data
.. codeauthor:: Christoph R Keller <christoph.a.keller@nasa.gov>
.. contributor:: Noussair Lazrak <noussair.lazrak@nyu.edu>
"""

import sys
import os
import re
import fsspec
import numpy as np
from scipy import stats
import pandas as pd
import datetime as dt
from datetime import timedelta
import time
import xarray as xr
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import pickle
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error, median_absolute_error
from math import sqrt
from tqdm import tqdm as tqdm
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import logging
import lightgbm as lgb
from pyod.models.iforest import IForest
import warnings
from IPython.display import HTML
sys.path.insert(1, 'MLpred')
from MLpred import mlpred
from geopy.distance import geodesic


warnings.filterwarnings("ignore")

ZARR_TEMPLATE = [
    "geos-cf/zarr/geos-cf.met_tavg_1hr_g1440x721_x1.zarr",
    "geos-cf/zarr/geos-cf.chm_tavg_1hr_g1440x721_v1.zarr",
]
ZARR_TEMPLATE = ["geos-cf/zarr/geos-cf-rpl.zarr"]
S3_TEMPLATE = "s3://dh-eis-fire-usw2-shared/geos-cf-rpl.zarr"
S3_FORECASTS_TEMPLATE = "s3://dh-eis-fire-usw2-shared/geos-cf-fcst-latest.zarr"
S3_REPLAY_TEMPLATE = "s3://dh-eis-fire-usw2-shared/geos-cf-ana-latest.zarr"
OPENDAP_TEMPLATE = "https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/fcast/met_tavg_1hr_g1440x721_x1.latest"
M2_TEMPLATE = "/home/ftei-dsw/Projects/SurfNO2/data/M2/{c}/small/*.{c}.%Y%m*.nc4"
M2_COLLECTIONS = ["tavg1_2d_flx_Nx", "tavg1_2d_lfo_Nx", "tavg1_2d_slv_Nx"]
OPENAQ_TEMPLATE = "https://api.openaq.org/v2//measurements?date_from={Y1}-{M1}-01T00%3A00%3A00%2B00%3A00&date_to={Y2}-{M2}-01T00%3A00%3A00%2B00%3A00&limit=10000&page=1&offset=0&sort=asc&radius=1000&location_id={ID}&parameter={PARA}&order_by=datetime"


DEFAULT_GASES = ["co", "hcho", "no", "no2", "noy", "o3"]

PPB2UGM3 = {"no2": 1.88, "o3": 1.97}
VVtoPPBV = 1.0e9


class ObsSiteList:
    def __init__(self, ifile=None):
        """
        Initialize ObsSiteList object (read from file if provided).
        """
        self._site_list = None
        if ifile is not None:
            self.load(ifile)

    def save(self, ofile="site_list.pkl"):
        """Write out a site_list, discarding all model and observation data beforehand (but keeping the trained XGBoost instances)
         `Generators` class: Contains medigan's public methods to facilitate users' automated sample generation workflows.
        Parameters
        ----------
        ofile: ofile
            Saves the list of sites to a pickle file
        """
        for isite in self._site_list:
            isite._obs = None
            isite._mod = None
        pickle.dump(self._site_list, open(ofile, "wb"), protocol=4)
        print("{} sites written to {}".format(len(self._site_list), ofile))
        return

    def load(self, ifile):
        """Reads a previously saved site list"""
        self._site_list = pickle.load(open(ifile, "rb"))
        print("Read {} sites from {}".format(len(self._site_list), ifile))
        return

    def filter_sites(self, year=2018, minobs=72, minvalue=15.0, silent=True):

        """Wrapper routine to get dataframe with average values for all sites with at least nobs observations for the first day of each month of the given year

        Parameters
        ----------
        year: year
            The year filter
        minobs: int
            The minimal observations to filter
        minvalue: float

        silent: bool
            Mute printed messages from the function

        Returns
        -------
        site_ids : pd.DataFrame
            A formatted observation data frame based on the filters
        """
        allmonths = []
        for imonth in tqdm(range(12)):
            testurl = "https://docs.openaq.org/v2/measurements?date_from={0:d}-{1:02d}-01T00%3A00%3A00%2B00%3A00&date_to={0:d}-{1:02d}-02T00%3A00%3A00%2B00%3A00&limit=100000&page=1&offset=0&sort=asc&parameter=no2&order_by=datetime".format(
                year, imonth + 1
            )
            allmonths.append(read_openaq(testurl, silent=silent))
        tmp = pd.concat(allmonths)
        cnt = tmp.groupby(["locationId", "unit"]).count().reset_index()
        sites = list(cnt.loc[cnt.value > minobs, "locationId"].values)
        subdf = tmp.loc[tmp["locationId"].isin(sites)].copy()
        meandf = subdf.groupby(["locationId", "unit"]).mean().reset_index()
        meandf.loc[meandf["unit"] == "µg/m³", "value"] = (
            meandf.loc[meandf["unit"] == "µg/m³", "value"] * 1.0 / 1.88
        )
        meandf.loc[meandf["unit"] == "ppm", "value"] = (
            meandf.loc[meandf["unit"] == "ppm", "value"] * 1000.0
        )
        site_ids = list(meandf.loc[meandf["value"] >= minvalue, "locationId"].values)
        print(
            "Found {} sites with average concentration above {} ppbv and more than {} observations".format(
                len(site_ids), minvalue, minobs
            )
        )
        self._minvalue = minvalue
        return site_ids

    def create_list(
        self,
        site_ids,
        minobs=240,
        silent=True,
        model_source="nc4",
        log=False,
        xgbparams={"booster": "gbtree", "eta": 0.5},
        **kwargs,
    ):
        """Create a list of observation sites by training all sites listed in site_ids that have at least minobs number of observations in the training window

        Parameters
        ----------
        site_ids: list
            list of OpenAQ site ids
        minobs: int
            The minimal observations to filter
        model_source: str
            Model sources
        silent: bool
            Mute printed messages from the function
        log: bool
            numpy.log for the training loop
        xgbparams: dict


        Returns
        -------
        site_ids : pd.DataFrame
            A formatted observation data frame based on the filters
        """
        self._site_list = []
        for i in tqdm(site_ids):
            isite = ObsSite(location_id=i, silent=silent, model_source=model_source)
            isite.read_obs(**kwargs)
            if isite._obs is None:
                if not isite._silent:
                    print("No observations found for site {}".format(i))
                continue
            if isite._obs.shape[0] < minobs:
                if not isite._silent:
                    print("Not enough observations found for site {}".format(i))
                continue
            # isite.read_mod()
            rc = isite.train(mindat=minobs, log=log, xgbparams=xgbparams)
            if rc == 0:
                self._site_list.append(isite)
        return

    def calc_ratios(self, start, end):

        """Get ratios between prediction and observation for each site in site_list'''


        Parameters
        ----------
        start: datetime
            The start date of training data set (GEOS-CF DATA and OpenAQ observation data)
        end: datetime
            The end date of training data set (GEOS-CF DATA and OpenAQ observation data)

        Returns
        -------
        dataframe
            a dataframe containing the ratios between prediction and observation for each site in site_list.
        """

        predictions = self.predict_sites(start, end)
        siteIds = []
        siteNames = []
        siteLats = []
        siteLons = []
        ratios = []
        meanObs = []
        meanPred = []
        for p in predictions:
            ip = predictions[p]
            idf = ip["prediction"]
            if idf is None:
                continue
            siteIds.append(p)
            siteNames.append(ip["name"])
            siteLats.append(ip["lat"])
            siteLons.append(ip["lon"])
            ratios.append(
                idf["observation"].values.mean() / idf["prediction"].values.mean()
            )
            meanObs.append(idf["observation"].values.mean())
            meanPred.append(idf["prediction"].values.mean())
        siteRatios = pd.DataFrame(
            {
                "Id": siteIds,
                "name": siteNames,
                "lat": siteLats,
                "lon": siteLons,
                "ratio": ratios,
                "obs": meanObs,
                "pred": meanPred,
            }
        )
        siteRatios["relChange"] = (siteRatios["ratio"] - 1.0) * 100.0
        return siteRatios

    def predict_sites(self, start, end):
        """Predict concentrations at all sites in the list of ObsSite objects

        Parameters
        ----------
        start: datetime
            The start date of training data set (GEOS-CF DATA and OpenAQ observation data)
        end: datetime
            The end date of training data set (GEOS-CF DATA and OpenAQ observation data)

        Returns
        -------
        list
            a list containing the prediction for each site
        """

        predictions = {}
        for isite in tqdm(self._site_list):
            isite.read_obs_and_mod(start=start, end=end)
            df = isite.predict(start=start, end=end)
            predictions[isite._id] = {
                "name": isite._name,
                "lat": isite._lat,
                "lon": isite._lon,
                "prediction": df,
            }
        return predictions

    def plot_deviation(
        self,
        siteRatios,
        title="NO2 deviation",
        minval=-30.0,
        maxval=30.0,
        mapbox_access_token=None,
    ):
        """Make global map showing deviation betweeen predictions and observations'''

        Parameters
        ----------
        siteRatios: list
            The site siteRatios betweeen predictions and observations for all sites
        title: str
            Map title
        minval: float

        maxval: float

        mapbox_access_token: str
            Mapbox token, Mapbox uses access tokens to associate API requests with your account. You can find your access tokens, create new ones, or delete existing ones on your Access Tokens page at mapbox.com


        Returns
        -------
        figure
            a map of deviation from all sites
        """

        siteRatios["text"] = [
            "{0:} (ID {1:}, Pred={2:.2f}ppbv, Deviation={3:.2f}%".format(i, j, k, l)
            for i, j, k, l in zip(
                siteRatios["name"],
                siteRatios["Id"],
                siteRatios["pred"],
                siteRatios["relChange"],
            )
        ]
        fig = go.Figure(
            data=go.Scattermapbox(
                lon=siteRatios["lon"],
                lat=siteRatios["lat"],
                text=siteRatios["text"],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=siteRatios["pred"],
                    sizemode="area",
                    color=siteRatios["relChange"],
                    cmin=minval,
                    cmax=maxval,
                    colorscale="RdBu",
                    opacity=0.8,
                    autocolorscale=False,
                    reversescale=True,
                    colorbar_title=title,
                ),
                # name = siteRatios['name'],
            )
        )
        # fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(
            hovermode="closest",
            mapbox_accesstoken=mapbox_access_token,
            mapbox_style="dark",
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig


class ObsSite:
    def __init__(
        self,
        location_id,
        read_obs=False,
        silent=False,
        model_source="nc4",
        species="no2",
        **kwargs,
    ):
        """
        Initialize ObsSite object.
        """
        self._init_site(location_id, species, silent, model_source)
        if read_obs:
            self.read_obs(**kwargs)

    def read_obs_and_mod(self, **kwargs):
        """Convenience wrapper to read both observations and model data"""
        self.read_obs(**kwargs)
        self.read_mod(**kwargs)
        return

    def read_obs(self, data=None, resample=None, **kwargs):
        """Wrapper routine to read observations

        Parameters
        ----------
        data: dataframe
            check of the observation dataframe is not empty, otherwise this method will read observations from OpenAQ
        resample: str
            This provides the ability to resample observation to daily, n Days mean value, example: ("5D" means 5 days mean value resample)
        """
        source = kwargs.get("source")

        if data is None:
            data = pd.DataFrame()
            if source == "local":
                url = kwargs.get("url")
                time_col = kwargs.get("time_col")
                unit = kwargs.get("unit")
                date_format = kwargs.get("date_format")
                value_collum = kwargs.get("value_collum")
                lat_col = kwargs.get("lat_col")
                lon_col = kwargs.get("lon_col")
                species = kwargs.get("species")
                lname = kwargs.get("lname")
                lat = kwargs.get("lat")
                lon = kwargs.get("lon")
                remove_outlier = kwargs.get("remove_outlier")
                start = kwargs.get("start")
                end = kwargs.get("end")

                data = read_local_obs(
                    obs_url=url,
                    time_col=time_col,
                    date_format=date_format,
                    value_collum=value_collum,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    species=species,
                    unit=unit,
                    lat=lat,
                    lon=lon,
                    start=start,
                    end=end,
                    remove_outlier = remove_outlier
                )

            elif source == "pandora":
                print("pandora")
                url = kwargs.get("url")
                time_col = kwargs.get("time_col")
                date_format = kwargs.get("date_format")
                value_collum = kwargs.get("value_collum")
                lat_col = kwargs.get("lat_col")
                lon_col = kwargs.get("lon_col")
                species = kwargs.get("species")
                lname = kwargs.get("lname")

                data = read_pandora(
                    url=url
                )


            else:
                data = pd.DataFrame()
                start_date = (
                    kwargs["start"] if "start" in kwargs else dt.datetime(2018, 1, 1)
                )
                end_date = kwargs["end"] if "end" in kwargs else dt.datetime(2024, 11, 11)

                month_difference = int((end_date.year - start_date.year) * 12 + end_date.month - start_date.month)


                three_month_periods = month_difference // 6

                for i in range(three_month_periods + 1):
                    period_start = start_date + relativedelta(months=6 * i)
                    
                    
                    period_end = min(start_date + relativedelta(months=6 * (i + 1)) - timedelta(days=1), end_date)
                    
                    print(f'getting openaq from {period_start} to {period_end}')
                    
                    
                    if not self._silent:
                        print(f"period retrieval {i + 1}: Start date - {period_start}, End date - {period_end}")

                    obs = self._read_openaq(start=period_start, end=period_end)
                    if obs is not None:
                        if data is None:
                            data = obs.copy()
                        else:
                            data = pd.concat([data, obs], ignore_index=True)


                folder_path = "obs/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                data.to_csv(f"{folder_path}{self._id}.csv", index=False)

        if data is None:
            if not self._silent:
                print("Warning: no observations found!")
            return
        if "lat" not in data.columns:
            if not self._silent:
                print(
                    "Warning: no latitude entry found in observation data - cannot process information"
                )
            return
        if resample is not None:
            data = data.set_index("time").resample(resample).mean().reset_index()
            print("Resampled observation data to: {}".format(resample))
        ilat = np.round(data["lat"].median(), 2)
        ilon = np.round(data["lon"].median(), 2)
        iname = data["location"].values[0]
        if not self._silent:
            print(
                "Found {:d} observations for {:} (lon={:.2f}; lat={:.2f})".format(
                    data.shape[0], iname, ilon, ilat
                )
            )
        self._lat = ilat if self._lat is None else self._lat
        self._lon = ilon if self._lon is None else self._lon
        assert ilat == self._lat
        assert ilon == self._lon
        self._name = iname if self._name is None else self._name
        if iname != self._name and not self._silent:
            print(
                "Warning: new station name is {}, vs. previously {}".format(
                    iname, self._name
                )
            )
            self._name = iname
        self._obs = (
            self._obs.merge(data, how="outer") if self._obs is not None else data
        )
        return

    def read_mod(self, **kwargs):
        """Wrapper routine to read model data"""

        lon = kwargs.pop('lon', self._lon)
        lat = kwargs.pop('lat', self._lat)
        assert lon is not None and lat is not None

        if "start" not in kwargs:
            min_time = self._obs["time"].min()
            if min_time < dt.datetime(2018, 1, 1):
                kwargs["start"] = dt.datetime(2018, 1, 1)
            else:
                kwargs["start"] = min_time

        if "end" not in kwargs:
            kwargs["end"] = self._obs["time"].max()

        mod = self._read_model(lon, lat, **kwargs)

        self._mod = self._mod.merge(mod, how="outer") if self._mod is not None else mod

        return

    def train(
        self,
        target_var="value",
        skipvar=["time", "location", "lat", "lon"],
        mindat=None,
        test_size=0.3,
        log=False,
        inc=False,
        xgbparams={"booster": "gbtree"},
        model_type="xgboost-tuned",
        **kwargs,
    ):

        """Train XGBoost model using data in memory

        Parameters
        ----------
        target_var: str
            the target to be predicted
        skipvar: list
            list of values to be skipped in the training dataset
        mindat: float

        test_size: float
            the size of the testing data set (default value is 0.3 (30%))

        log: bool

        inc:bool

            set to false to predict concentration if target species is not a feature input

        xgbparams: list
            list of xgboost model parameters

        model_type:
            default: "xgboost-tuned" to select the tuned model or default model
        """

        dat = self._merge(**kwargs)

        if dat is None:
            return -2
        if mindat is not None:
            if dat.shape[0] < mindat:
                print(
                    "Warning: not enough data - only {} rows vs. {} requested".format(
                        dat.shape[0], mindat
                    )
                )
                return -1
        yvar = [target_var]
        blacklist = yvar + skipvar
        xvar = [i for i in dat.columns if i not in blacklist]
        X = dat[xvar]
        y = dat[yvar]
        fvar = None
        if inc:
            fvar = "pm25_rh35_gcc" if self._species == "pm25" else self._species
            if fvar not in X:
                print(
                    "Warning: target species is not an input feature - cannot do increment ML (set inc=False to predict concentration instead)"
                )
                return -1
            y = y.values.flatten() - X[fvar].values.flatten()
        if log:
            y = np.log(y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size)

        if model_type == "Matrix":
            train = xgb.DMatrix(Xtrain, ytrain)
            if not self._silent:
                print("training model ...")
            bst = xgb.train(xgbparams, train)
            ptrain = bst.predict(xgb.DMatrix(Xtrain))
            ptest = bst.predict(xgb.DMatrix(Xtest))
            ytrainf = np.array(ytrain).flatten()
            ytestf = np.array(ytest).flatten()
            if log:
                ytrainf = np.exp(ytrainf)
                ytestf = np.exp(ytestf)
                ptrain = np.exp(ptrain)
                ptest = np.exp(ptest)
            if inc:
                ytrainf = ytrainf + np.array(Xtrain[fvar]).flatten()
                ytestf = ytestf + np.array(Xtest[fvar]).flatten()
                ptrain = ptrain + np.array(Xtrain[fvar]).flatten()
                ptest = ptest + np.array(Xtest[fvar]).flatten()
            if not self._silent:
                print("Training:")
                print("r2 = {:.2f}".format(r2_score(ytrainf, ptrain)))
                print("rmse = {:.2f}".format(mean_squared_error(ytrainf, ptrain)))
                print(
                    "nrmse = {:.2f}".format(
                        sqrt(mean_squared_error(ytrainf, ptrain)) / np.std(ytrainf)
                    )
                )
                print("nmb = {:.2f}".format(np.sum(ptrain - ytrainf) / np.sum(ytrainf)))
                print("Test:")
                print("r2 = {:.2f}".format(r2_score(ytestf, ptest)))
                print(
                    "nrmse = {:.2f}".format(
                        sqrt(mean_squared_error(ytestf, ptest)) / np.std(ytestf)
                    )
                )
                print("nmb = {:.2f}".format(np.sum(ptest - ytestf) / np.sum(ytestf)))

        if model_type == "xgboost-tuned":
            bst = xgb.XGBRegressor(
                colsample_bytree=0.3,
                learning_rate=0.01,
                max_depth=10,
                n_estimators=1000,
                verbosity=0,
            )
            prepared_model = bst.fit(Xtrain, ytrain)
            ypred = bst.predict(dat[X.columns])
            score = prepared_model.score(Xtest, ytest)
            target = prepared_model.predict(Xtest)

            if not self._silent:
                MSE = mean_squared_error(ytest, target)
                RMSE = mean_squared_error(ytest, target, squared=False)
                MAE = mean_absolute_error(ytest, target)
                r2 = r2_score(ytest, target)

                # RMSE Computation
                rmse_1 = np.sqrt(MSE_1(ytest, target))
                print("RMSE : % f" % (rmse_1))
                print("Score: ", score)
                print("mean square error", MSE)
                print("Root mean square error", RMSE)
                print("MAE", MAE)
                print("R2", r2)

                print("Train Accuracy:", prepared_model.score(Xtrain, ytrain))
                print("Test Accuracy:", prepared_model.score(Xtest, ytest))

        self._bst = bst
        self._x = X
        self._xcolumns = X.columns
        self._log = log
        self._inc = inc
        self._fvar = fvar
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        return 0

    def predict(self, add_obs=True, model_type="xgboost-tuned", **kwargs):

        """Make prediction for given time window and return predicted values along with observations

        Parameters
        ----------
        add_obs: dataframe
            add observation data to geos-cf model data
        model_type: str
            predict using a predefined model, or default model, the predefined model is hyperparameter tuned
        minval: float



        Returns
        -------
        dataframe
            a dataframe containing the predictions vs observation
        """

        if add_obs:
            dat = self._merge(**kwargs)
        else:
            start = kwargs["start"] if "start" in kwargs else dat["time"].min()
            end = kwargs["end"] if "end" in kwargs else dat["time"].max()
            dat = self._mod.loc[
                (self._mod["time"] >= start) & (self._mod["time"] <= end)
            ].copy()
            if "value" not in dat:
                dat["value"] = [np.nan for i in range(dat.shape[0])]
        if dat is None:
            return None
        if model_type == "xgboost-tuned":
            pred = self._bst.predict(dat[self._xcolumns])
        if model_type == "Matrix":
            pred = self._bst.predict(xgb.DMatrix(dat[self._xcolumns]))

        if self._log:
            pred = np.exp(pred)
        if self._inc:
            pred = pred + dat[self._fvar]

        df = dat[["time", "value"]].copy()
        df["prediction"] = pred
        df.rename(columns={"value": "observation"}, inplace=True)
        return df

    def plot(
        self,
        df,
        y=["observation", "prediction"],
        ylabel=r"$\text{NO}_{2}\,[\text{ppbv}]$",
        **kwargs,
    ):

        """Make plot of prediction vs. observation, as generated by self.predict()

        Parameters
        ----------
        df: dataframe
            dataframe containing the prediction and observation values generated by the predict() method
        y: list
            list of y-axis,
        ylabel: str
            y-axis label

        Returns
        -------
        figure
            a timeseries figure of the predictions vs observation
        """

        title = "Site = {0} ({1:.2f}N, {2:.2f}E)".format(
            self._name, self._lat, self._lon
        )
        fig = px.line(
            df, x="time", y=y, labels={"value": ylabel}, title=title, **kwargs
        )
        fig.update_layout(xaxis_title="Date (UTC)", yaxis_title=ylabel)
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0
            ),
            legend_title="",
        )

        return fig

    def explain(self, plot=False, feature=False):
        """plot SHAP values to explain how the features are driving the predictions

        Parameters
        ----------
        plot: bool
            specify the type of the plot, examples: "waterfall", "beeswarm", "scatter"
        feature: str
            if the plot type is "scatter", please define the input/ feature you want to get analysis for.


        Returns
        -------
        figure
            a Shap figure of the predictions
        """

        if self._bst:
            explainer = shap.Explainer(self._bst)
            shap_values = explainer(self._x)
            if plot == "waterfall":
                shap.plots.waterfall(shap_values[0])
            if plot == "beeswarm":
                shap.plots.beeswarm(shap_values)
            if plot == "scatter":
                if feature:
                    try:
                        shap.plots.scatter(shap_values[:, feature], color=shap_values)
                    except:
                        print("feature not found!")

        else:
            print("Please train the model first")
        return

    def _merge(
        self,
        start=None,
        end=None,
        mod_blacklist=["lat", "lon", "lev"],
        interpolation=True,
    ):

        """Merge model and observation and limit to given time window

        Parameters
        ----------
        start: datetime
            The start date of training data set (GEOS-CF DATA and OpenAQ observation data)
        end: datetime
            The end date of training data set (GEOS-CF DATA and OpenAQ observation data)

        mod_blacklist: list
            list of blacklisted features

        Returns
        -------
        dataframe
            a dataframe containing a merged dataframe of the model and observations
        """

        if self._mod is None or self._obs is None:
            if not self._silent:
                print("Warning: cannot merge because mod or obs is None")
            return None
        # toss model variables that are blacklisted.
        ivars = [i for i in self._mod.columns if i not in mod_blacklist]
        # interpolate model data to openaq time stamps
        mdat = (
            self._mod[ivars]
            .merge(self._obs, on=["time"], how="outer")
            .sort_values(by="time")
        )
        mdat = mdat.drop_duplicates(subset='time')
        mdat = mdat.sort_values(by='time')


        if interpolation:
            idat = mdat.set_index("time").interpolate(method="slinear").reset_index()
        else:
            idat = mdat

        dat = idat.loc[idat["time"].isin(self._obs["time"])].copy()
        start = start if start is not None else dat["time"].min()
        end = end if end is not None else dat["time"].max()
        testvar = ivars[-1]
        idat = dat.loc[
            (dat["time"] >= start) & (dat["time"] <= end) & (~np.isnan(dat[testvar]))
        ]
        if idat.shape[0] == 0:
            idat = None
        return idat

    def _init_site(self, location_id, species, silent, model_source):
        """Create an empty site object"""
        self._id = location_id
        self._species = species
        self._silent = silent
        self._modsrc = model_source
        self._lat = None
        self._lon = None
        self._name = None
        self._obs = None
        self._mod = None
        self._log = False
        self._inc = False
        self._fvar = None  # name of input feature corresponding to output variable. Only needed if _inc is True
        return

    def _read_model(
        self,
        ilon,
        ilat,
        start,
        end,
        resample=None,
        source=None,
        template=None,
        collections=None,
        remove_outlier=0,
        gases=DEFAULT_GASES,
        **kwargs,
    ):

        """Read model data

        Parameters
        ----------
        ilon: float
            site longitude
        ilat:
            site latitude
        start: datetime
            The start date of training data set (GEOS-CF DATA)

        end: datetime
            The end date of training data set (GEOS-CF DATA)

        resample: str
            This provides the ability to resample observation to daily, n Days mean value, example: ("5D" means 5 days mean value resample)

        source: str
            specify the source file format (e.g. "opendap", "nc4", "zarr" for compressed format)

        template: str
            the link template for each model data file type

        collections: list
            model collection (e.g. "tavg1_2d_flx_Nx")

        gases: list
            list with gas names used to identify fields that need to be converted from v/v to ppbv

        Returns
        -------
        dataframe
            a dataframe containing the geos-cf model data
        """

        dfs = []
        source = self._modsrc if source is None else source
        if source == "opendap":
            template = OPENDAP_TEMPLATE if template is None else template
            template = template if isinstance(template, type([])) else [template]
            for t in template:
                if not self._silent:
                    print("Reading {}...".format(t))
                ids = (
                    xr.open_dataset(t)
                    .sel(lon=ilon, lat=ilat, lev=1, method="nearest")
                    .sel(time=slice(start, end))
                    .load()
                    .to_dataframe()
                    .reset_index()
                )
                dfs.append(ids)
        if source == "nc4":
            template = M2_TEMPLATE if template is None else template
            collections = M2_COLLECTIONS if collections is None else collections
            for c in collections:
                itemplate = template.replace("{c}", c)
                ifiles = start.strftime(itemplate)
                if not self._silent:
                    print("Reading {}...".format(c))
                ids = (
                    xr.open_mfdataset(ifiles)
                    .sel(lon=ilon, lat=ilat, method="nearest")
                    .sel(time=slice(start, end))
                    .load()
                    .to_dataframe()
                    .reset_index()
                )
                dfs.append(ids)
        if source == "zarr" or source == "s3":
            if template is None:
                template = (
                    ZARR_TEMPLATE
                    if source == "zarr"
                    else [S3_TEMPLATE]
                )
            template = template if isinstance(template, type([])) else [template]
            for t in template:
                print("Reading {}...".format(t))
                ipath = fsspec.get_mapper(t)
                ds = xr.open_zarr(ipath)
                ids = (ds.sel(lon=ilon, lat=ilat, lev=1, method='nearest')
                         .load()
                         .to_dataframe()
                         .reset_index())

                ids['time'] = pd.to_datetime(ids['time'])

                ids = ids[(ids['time'] >= start) & (ids['time'] <= end)]
                if not ids.empty:
                    dfs.append(ids)

        if source == "local":
            url = kwargs.get("url")
            if not self._silent:
                print(f"Reading csv file: {url}")

            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            df = pd.read_csv(url)
            df[model_date_col] = pd.to_datetime(df[model_date_col])
            ids = df[(df[model_date_col] >= start) & (df[model_date_col] <= end)]
            if not ids.empty:
                dfs.append(ids)
                
        if source == "pandora":
            url = kwargs.get("url")
            if not self._silent:
                print(f"Reading csv file: {url}")

            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            df = pd.read_csv(url)
            df["time"] = pd.to_datetime(df["date"])
            df = df.drop(columns=['date'])
            ids = df[(df["time"] >= start) & (df["time"] <= end)]
            if not ids.empty:
                dfs.append(ids)

        if dfs:
            dfs_concatenated = pd.concat(dfs, ignore_index=True)
        else:
            dfs_concatenated = pd.DataFrame()

        mod = dfs_concatenated

        mod["time"] = [pd.to_datetime(i) for i in mod["time"]]
        mod["month"] = [i.month for i in mod["time"]]
        mod["hour"] = [i.hour for i in mod["time"]]
        mod["weekday"] = [i.weekday() for i in mod["time"]]

        mod["time"] = [pd.to_datetime(i) for i in mod["time"]]
        if resample is not None:
            mod = mod.set_index("time").resample(resample).mean().reset_index()
            print("Resampled model data to: {}".format(resample))
        mod["month"] = [i.month for i in mod["time"]]
        mod["hour"] = [i.hour for i in mod["time"]]
        mod["weekday"] = [i.weekday() for i in mod["time"]]
        # convert trace gases from v/v to ppbv
        for g in gases:
            if g in mod:
                if not self._silent:
                    print("Convert from v/v to ppbv: {}".format(g))
                mod[g] = mod[g] * VVtoPPBV
        return mod

    def _read_openaq(
        self=None, start=dt.datetime(2018, 1, 1), end=None, normalize=False, **kwargs
    ):
        """Read OpenAQ observations and convert to ppbv

        Parameters
        ----------
        self: object, optional
            The instance of the class (if used as a method)
        start: datetime
            The start date of training data set (GEOS-CF DATA)
        end: datetime
            The end date of training data set (GEOS-CF DATA)
        normalize: bool
            if True, normalize the observations values with standard deviation

        Returns
        -------
        dataframe
            a dataframe containing the observation data
        """

        # If self is None, remove all self and replace with args parameters
        if self is None:
            id_ = kwargs.get("_id", None)
            species = kwargs.get("_species", None)
            silent = kwargs.get("_silent", None)
        else:
            id_ = self._id
            species = self._species
            silent = self._silent

        end = start + relativedelta(years=1) if end is None else end
        url = (
            OPENAQ_TEMPLATE.replace("{ID}", str(id_))
            .replace("{PARA}", species)
            .replace("{Y1}", str(start.year))
            .replace("{M1}", "{:02d}".format(start.month))
            .replace("{D1}", "{:02d}".format(start.day))
            .replace("{Y2}", str(end.year))
            .replace("{M2}", "{:02d}".format(end.month))
            .replace("{D2}", "{:02d}".format(end.day))
        )
        allobs = read_openaq(url, silent=silent, **kwargs)

        if allobs is None:
            return None

        obs = allobs.loc[
            (allobs["parameter"] == species)
            & (~np.isnan(allobs["value"]))
            & (allobs["value"] >= 0.0)
        ].copy()

        # convert everything to ppbv
        if species != "pm25":
            assert species in PPB2UGM3

            conv_factor = PPB2UGM3[species]
            if not self._silent:
                print(f" converting to ppbv with conv_factor {conv_factor}")
            obs.loc[obs["unit"] == "ppm", "value"] = (
                obs.loc[obs["unit"] == "ppm", "value"] * 1000.0
            )
            obs.loc[obs["unit"] == "µg/m³", "value"] = (
                obs.loc[obs["unit"] == "µg/m³", "value"] * 1.0 / conv_factor
            )

        # subset to relevant columns
        outobs = obs[["time", "location", "value"]].copy()

        if normalize:
            outobs["value"] = (outobs["value"] - outobs["value"].mean()) / outobs[
                "value"
            ].std()

        if (
            "coordinates.latitude" in obs.columns
            and "coordinates.longitude" in obs.columns
        ):
            outobs["lat"] = obs["coordinates.latitude"]
            outobs["lon"] = obs["coordinates.longitude"]
        else:
            if not silent:
                print("Warning: no coordinates in dataset")

        return outobs

    def explain_model(model, X, plot, feature=False):
        """explain model via Shap values

        Parameters
        ----------
        model: model
            predefined model in memory

        X: dataframe
            dataframe in memory

        plot: str
            type of plot to be returned (e.g. "waterfall", "beeswarm", "scatter")

        feature:str
            when using scatter plot, please specify the feature to run shap analysis for

        Returns
        -------
        figure
            a Shap values plot
        """
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            if plot == "waterfall":
                shap.waterfall_plot(shap_values[0])
            if plot == "beeswarm":
                shap.beeswarm_plot(shap_values)
            if plot == "scatter":
                if feature:
                    shap.plots.scatter(shap_values[:, feature], color=shap_values)

        except:
            print("Warning: Model Error")

    def save_model(model, model_data=False, save_data=False, name=False):
        if name is False:
            name = "pretrained_model"
        pickle.dump(model, open(name + ".sav", "wb"))
        print("Model saved")
        if save_data:
            model_data.to_csv("model_data.csv")
            print("Model data saved")
        return

    def load_model(name=False, **kwargs):
        loaded_model = pickle.load(open(name, "rb"))
        return loaded_model

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def gridSerch(self, model, X_train, Y_train, **kwargs):
        print("Tunning the model hyper parameter for this location")
        params = {
            "max_depth": [3, 5, 6, 10, 15, 20],
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "subsample": np.arange(0.5, 1.0, 0.1),
            "colsample_bytree": np.arange(0.4, 1.0, 0.1),
            "colsample_bylevel": np.arange(0.4, 1.0, 0.1),
            "n_estimators": [100, 500, 1000],
        }
        clf = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            scoring="neg_mean_squared_error",
            n_iter=25,
            verbose=1,
        )
        clf.fit(X_train, Y_train)
        print("Best parameters:", clf.best_params_)
        print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))
        return clf

    def plot_intervals(
        self, predictions, mid=False, start=None, stop=None, title=None, **kwargs
    ):
        predictions = (
            predictions.loc[start:stop].copy()
            if start is not None or stop is not None
            else predictions.copy()
        )
        data = []

        """Lower Trace"""

        trace_low = go.Scatter(
            x=predictions.index,
            y=predictions["lower"],
            fill="tonexty",
            line=dict(color="darkblue"),
            fillcolor="rgba(173, 216, 230, 0.4)",
            showlegend=True,
            name="lower",
        )
        """Upper Trace"""
        trace_high = go.Scatter(
            x=predictions.index,
            y=predictions["upper"],
            fill=None,
            line=dict(color="orange"),
            showlegend=True,
            name="upper",
        )

        data.append(trace_high)
        data.append(trace_low)

        if mid:
            trace_mid = go.Scatter(
                x=predictions.index,
                y=predictions["mid"],
                fill=None,
                line=dict(color="green"),
                showlegend=True,
                name="mid",
            )
            data.append(trace_mid)

        """Actual Values Trace"""
        trace_actual = go.Scatter(
            x=predictions.index,
            y=predictions["mid"],
            fill=None,
            line=dict(color="black"),
            showlegend=True,
            name="middle",
        )
        data.append(trace_actual)

        """Observation Values Trace"""
        trace_actual = go.Scatter(
            x=predictions.index,
            y=predictions["value"],
            fill=None,
            line=dict(color="red"),
            showlegend=True,
            name="observation",
        )
        data.append(trace_actual)

        """prediction Values Trace"""
        bias_corrected = go.Scatter(
            x=predictions.index,
            y=predictions["bias_corrected"],
            fill=None,
            line=dict(color="blue"),
            showlegend=True,
            name="prediction",
        )
        data.append(bias_corrected)

        """Title and customization"""
        layout = go.Layout(
            height=900,
            width=1400,
            title=dict(text="Prediction Intervals" if title is None else title),
            yaxis=dict(title=dict(text="NO2 ppvb")),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=True),
                type="date",
            ),
        )

        fig = go.Figure(data=data, layout=layout)

        fig["layout"]["font"] = dict(size=20)
        fig.layout.template = "plotly_white"
        return fig

    def ConfidenceIntervals(
        self,
        LOWER_ALPHA=0.15,
        UPPER_ALPHA=0.85,
        N_ESTIMATORS=1000,
        MAX_DEPTH=5,
        LEARNING_RATE=0.01,
        colsample_bytree=0.3,
        OUTPUT="plot",
        **kwargs,
    ):
        """explain model via Shap values

        Parameters
        ----------
        model: model
            predefined model in memory

        X: dataframe
            dataframe in memory

        plot: str
            type of plot to be returned (e.g. "waterfall", "beeswarm", "scatter")

        feature:str
            when using scatter plot, please specify the feature to run shap analysis for

        Returns
        -------
        figure
            a Shap values plot
        """

        lower_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=LOWER_ALPHA,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
        )

        mid_model = xgb.XGBRegressor(
            loss="ls",
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            colsample_bytree=colsample_bytree,
            learning_rate=LEARNING_RATE,
            verbosity=0,
        )

        upper_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=UPPER_ALPHA,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
        )

        lower_model.fit(self.Xtrain, self.ytrain)
        mid_model.fit(self.Xtrain, self.ytrain)
        upper_model.fit(self.Xtrain, self.ytrain)

        predictions = pd.DataFrame(self.ytest)
        predictions["lower"] = lower_model.predict(self.Xtest)
        predictions["mid"] = mid_model.predict(self.Xtest)
        predictions["upper"] = upper_model.predict(self.Xtest)
        predictions["bias_corrected"] = self._bst.predict(self.Xtest)

        dat = self._merge()
        df = dat[["time", "value"]].copy()
        to_plot = df.merge(predictions)
        to_plot["timestamp"] = pd.to_datetime(to_plot["time"], unit="s")
        to_plot = to_plot.set_index(pd.DatetimeIndex(to_plot["timestamp"]))
        to_plot.dropna()
        to_plot.sort_index(inplace=True)

        if OUTPUT == "PLOT":
            to_plot.sort_index().dropna().resample("2D").mean()
            fig = self.plot_intervals(to_plot, start="2021-01-19", stop="2022-02-28")
            return fig
        if OUTPUT == "dataframe":
            return to_plot


def get_localised_forecast(**kwargs):
    """Routine to generate localised forecasts based on GEOS CF and local observation data.

    Parameters
    ----------
    site_settings: dict
        A dictionary containing site-specific settings.
        l_name : str
            Name of the location.
        species : str
            Type of species for which forecast is generated.
        silent : bool
            Flag to suppress notifications and warnings.
        lat : float
            Latitude of the location.
        lon : float
            Longitude of the location.
        model_src : str
            Source of the model for forecast generation.
        obs_src : str
            Source of the observation data.
        openaq_id : str
            ID associated with the OpenAQ data.
        model_url : str
            URL for accessing the GEOS CF model.
        obs_url : str
            URL for accessing the observation data.
        start : str
            Start date for the forecast.
        end : str
            End date for the forecast.
        resample : str
            Resampling frequency for the data.
        model_tuning : str
            Model tuning parameters.
        unit : str
            Unit of measurement for the forecast.
        interpolation : str
            Interpolation method for data processing.
        remove_outlier : int
            Threshold for removing outliers from observations.

    obs_settings: dict
        A dictionary containing observation-specific settings.
        time_col : str
            Column name for time in the observation data.
        date_format : str
            Format of the date in the observation data.
        obs_val_col : str
            Column name for observed values.
        lat_col : str
            Column name for latitude in the observation data.
        lon_col : str
            Column name for longitude in the observation data.
    """
    location_name = kwargs["site_settings"]["l_name"]
    species = kwargs["site_settings"]["species"]
    silent = kwargs["site_settings"]["silent"]
    lat = kwargs["site_settings"]["lat"]
    lon = kwargs["site_settings"]["lon"]
    model_source = kwargs["site_settings"]["model_src"]
    observation_source = kwargs["site_settings"]["obs_src"]
    openaq_id = kwargs["site_settings"]["openaq_id"]
    GEOS_CF = kwargs["site_settings"]["model_url"]
    OBS_URL = kwargs["site_settings"]["obs_url"]
    start = kwargs["site_settings"]["start"]
    end = kwargs["site_settings"]["end"]
    resample = kwargs["site_settings"]["resample"]
    model_tuning = kwargs["site_settings"]["model_tuning"]
    unit = kwargs["site_settings"]["unit"]
    interpolation = kwargs["site_settings"]["interpolation"]
    remove_outlier = kwargs["site_settings"]["remove_outlier"]

    time_col = kwargs["obs_settings"]["time_col"]
    date_format = kwargs["obs_settings"]["date_format"]
    obs_val_col = kwargs["obs_settings"]["obs_val_col"]
    lat_col = kwargs["obs_settings"]["lat_col"]
    lon_col = kwargs["obs_settings"]["lon_col"]
    remove_outlier = kwargs["obs_settings"]["remove_outlier"]
    ## Data preparation
    all_obs = pd.DataFrame()
    isite = ObsSite(
        openaq_id,
        model_source=model_source,
        species=species,
        observation_source=observation_source,
    )
    isite._silent = silent
    isite.read_obs(
        source=observation_source,
        url=OBS_URL,
        time_col=time_col,
        date_format=date_format,
        value_collum=obs_val_col,
        lat_col=lat_col,
        lon_col=lon_col,
        species=species,
        lat=lat,
        lon=lon,
        unit=unit,
        remove_outlier = remove_outlier,
        **kwargs,
    )

    all_obs = isite._obs
    isite.read_mod(source=model_source, url=GEOS_CF)
    all_data = isite._merge(interpolation=interpolation)
    all_data.dropna()

    yvar = "value"
    fvar = "pm25_rh35_gcc" if isite._species == "pm25" else isite._species.lower()
    
    if model_source == "pandora":
        fvar = "cf_sfcmr"
    
    ## Unusual difference between OBS and Model
    difference = all_data[fvar].mean() / all_data[yvar].mean()
    log_if_condition(
        (difference > 2),
        f"UNIT ERROR: GEOS CF IS HIGHER BY: {difference} IN LOCATION: {location_name} SPECIES: {species.lower()}",
    )

    skipvar = [
        "time",
        "location",
        "lat",
        "lon",
        "totcol_co",
        "totcol_hcho",
        "totcol_no2",
        "totcol_o3",
        "totcol_so2",
        "tropcol_co",
        "tropcol_hcho",
        "tropcol_o3",
        "tropcol_so2",
    ]
    blacklist = skipvar + [yvar]

    if remove_outlier:
        if not silent:
            print("outlier to be removed is selected")
        concentration_obs = all_data[yvar].values.reshape(-1, 1)
        model_IF = IForest(contamination=0.05)
        model_IF.fit(concentration_obs)
        anomalies = model_IF.predict(concentration_obs)
        all_data = all_data[anomalies != 1]

    xvar = [i for i in all_data.columns if i not in blacklist]

    x = all_data[xvar]
    y = all_data[yvar]


   

    # Preprocessing Data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(all_data[xvar].values.reshape(-1, 1))
    y_scaled = scaler.fit_transform(all_data[yvar].values.reshape(-1, 1))
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=7)

    # Baseline model using XGBoost
    baseline_model_xgb = xgb.XGBRegressor()
    baseline_model_xgb.fit(X_train, Y_train)
    baseline_model_score_xgb = baseline_model_xgb.score(X_test, Y_test)

    # Baseline model using LightGBM
    baseline_model_lgb = lgb.LGBMRegressor(verbosity=-1)
    baseline_model_lgb.fit(X_train, Y_train)
    
    
    baseline_model_score_lgb = baseline_model_lgb.score(X_test, Y_test)


    if baseline_model_score_xgb > baseline_model_score_lgb:
        baseline_model = baseline_model_xgb
        print("XGBoost is the baseline model.")
    else:
        baseline_model = baseline_model_lgb
        print("LightGBM is the baseline model.")

    eval_set = [(X_train, Y_train), (X_test, Y_test)]

    if model_tuning:
        # Tuning parameters for XGBoost
        param_grid_xgb = {
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 10, 15],
            "n_estimators": [50, 500, 1000],
        }
        grid_search_xgb = GridSearchCV(estimator=baseline_model_xgb,
                                       param_grid=param_grid_xgb,
                                       cv=5,
                                       scoring="neg_mean_squared_error")
        grid_search_xgb.fit(X_train, Y_train)

        # Tuning parameters for LightGBM
        param_grid_lgb = {
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 10, 15],
            "n_estimators": [50, 500, 1000],
            
        }
        
        grid_search_lgb = GridSearchCV(estimator=baseline_model_lgb,
                                   param_grid=param_grid_lgb,
                                   cv=5,
                                   scoring="neg_mean_squared_error",
                                   verbose=0) 
        grid_search_lgb.fit(X_train, Y_train)

        if grid_search_xgb.best_score_ > grid_search_lgb.best_score_:
            best_params = grid_search_xgb.best_params_
            best_model = xgb.XGBRegressor(**best_params)
            print("XGBoost is the best performer after tuning.")
        else:
            best_params = grid_search_lgb.best_params_
            best_model = lgb.LGBMRegressor(**best_params)
            print("LightGBM is the best performer after tuning.")

        best_model.fit(X_train, Y_train)
        selected_model = best_model
    
    else:
        selected_model = baseline_model

    selected_model.fit(
        X_train,
        Y_train,
        eval_set=eval_set,
        eval_metric="rmse",
    )
    Y_pred = selected_model.predict(X_test)
    rmse = round(mean_squared_error(Y_test, Y_pred, squared=False), 2)
    r2 = round(r2_score(Y_test, Y_pred), 2)
    mae = round(mean_absolute_error(Y_test, Y_pred), 2)

    log_if_condition(
        (r2 < 0.5),
        f"MODEL ERROR: MODEL RUNS POORLY IN THIS LOCATION: R2: {r2} ; RMSE: {rmse} IN LOCATION: {location_name} SPECIES: {species.lower()}",
    )

    ## Preparing the final dataframe with forecasts
    start = start if start is not None else all_obs["time"].min()
    end = end if end is not None else all_obs["time"].min()
    if not isite._silent:
        print(f"predictions started: {start}")
    model_data = isite._read_model(
        ilon=isite._lon,
        ilat=isite._lat,
        start=start,
        end=end,
        source=model_source,
        url=GEOS_CF,
    )
    model_data["time"] = [
        dt.datetime(i.year, i.month, i.day, i.hour, 0, 0) for i in model_data["time"]
    ]
    model_data["localised"] = selected_model.predict(model_data[x.columns])
    observation = isite._obs[["time", "value"]].copy()
    observation["time"] = [
        dt.datetime(i.year, i.month, i.day, i.hour, 0, 0) for i in observation["time"]
    ]
    
    merged_data = None
    all_validation_data = pd.DataFrame()

    if 'validation_sets' in kwargs:
        
        validation_sets = kwargs["validation_sets"]

        all_validation_data_list = []
        for dataset in validation_sets:
            all_obs = read_validation_set(**dataset)
            if all_obs is not None:
                if "value" in all_obs.columns:
                    all_obs.rename(columns={"value": dataset["name"]}, inplace=True)
                all_obs = all_obs[["time", dataset["name"]]]
                all_validation_data_list.append(all_obs)

        if all_validation_data_list:
            if len(all_validation_data_list) == 1:
                all_validation_data = all_validation_data_list[0]
            else:
                all_validation_data = pd.merge(all_validation_data_list[0], all_validation_data_list[1], on='time', how='outer')
                for i in range(2, len(all_validation_data_list)):
                    all_validation_data = pd.merge(all_validation_data, all_validation_data_list[i], on='time', how='outer')

            print(all_validation_data.columns)
        else:
            print("No validation data available.")
    else:
        print("No validation sets found.")

    
    if  all_validation_data.empty:
        merged_data = merge_dataframes(
            [model_data, observation], index_col="time", resample=resample, how ="outer")
        print("Validation set is empty")
    else:
        merged_data = merge_dataframes(
            [model_data, observation, all_validation_data],
            index_col="time",
            resample=resample, how ="outer")
    
    #merged_data = merged_data.dropna(subset=['value'])

   # HAQAST_data_product = merge_dataframes([model_data, observation], index_col="time", resample="1h")
    
    #export_to_gesdisc(HAQAST_DATA=HAQAST_data_product,location_name=location_name,species=species, unit=unit, lat=isite._lat, lon=isite._lon)
    location_plot(
        dataframe=merged_data,
        location_name=location_name,
        title=f"{location_name} ( {species} ) ({isite._lat}, {isite._lon} | {resample} resample)",
        species=species,
        unit=unit,
        model_info=f"(r2:{r2} | rmse:{rmse})",
        resample = resample
    )
    if isite._silent:
        print(f"forecasts generated for {fvar}")
    merged_data.to_csv(f"location_data/{location_name}_{species}_{resample}_resample.csv")
    return merged_data


## General Functions
def read_openaq(
    url, reference_grade_only=True, silent=False, remove_outlier=0, **kwargs
):
    """Helper routine to read OpenAQ via API (from given url) and create a dataframe of the data

    Parameters
    ----------
    url: str
        OpenAQ API url, with the location lat lon, sepcies and date. Please see: https://docs.openaq.org/docs

    reference_grade_only: bool
        Selects the reference grade values only from OpenAQ response

    silent: bool
        Display notifications and warnings from this method

    remove_outlier: int
        This allows removing outliers from observations
    """
    if not silent:
        print("Quering  {}".format(url))
    allobs = pd.DataFrame()
    retries = 0
    max_retries = 2

    while retries < max_retries:
        headers = {'X-API-Key': '849b4d760f2679b9e0cafea2a23a698578107984fb150a81d15640600bdb271f'}
        r = requests.get(url, headers=headers)
        
        if r.status_code == 200:
            allobs = pd.json_normalize(r.json()["results"])   
            break
        elif r.status_code == 429:
            retries += 1
            if retries < max_retries:
                time.sleep(5)
        else:
            print("there is a error pulling data from OpenAQ")
       
    if allobs.shape[0] == 0:
        if not silent:
            print("Warning: no OpenAQ data found for specified url")
        return None

    try:
        allobs = allobs.loc[
            (allobs["value"] >= 0.0) & (~np.isnan(allobs["value"]))
        ].copy()
        if reference_grade_only:
            allobs = allobs.loc[allobs["sensorType"] == "reference grade"].copy()
        allobs["time"] = [
            dt.datetime.strptime(i, "%Y-%m-%dT%H:%M:%S+00:00")
            for i in allobs["date.utc"]
        ]
        if remove_outlier > 0:
            std = allobs["value"].std()
            mn = allobs["value"].mean()
            minobs = mn - remove_outlier * std
            maxobs = mn + remove_outlier * std
            norig = allobs.shape[0]
            allobs = allobs.loc[
                (allobs["value"] >= minobs) & (allobs["value"] <= maxobs)
            ].copy()
            if not silent:
                nremoved = norig - allobs.shape[0]
                print(
                    "removed {:.0f} of {:.0f} values because considered outliers ({:.2f}%)".format(
                        nremoved, norig, np.float(nremoved) / np.float(norig) * 100.0
                    )
                )
        return allobs

    except:
        if not silent:
            print("Warning ...")
        return None

def read_local_obs(
    obs_url=None,
    time_col="Time",
    date_format="%m/%d/%Y %H:%M:%S",
    value_collum= None,
    lat_col=None,
    lon_col=None,
    species=None,
    silent=True,
    start = None,
    end = None,
    remove_outlier=False,
    rename_column=None,
    unit=None,
    lat=None,
    lon=None,
    **kwargs,
):

    col_name = rename_column if rename_column else "value"

    allobs = pd.read_csv(obs_url)

    allobs = allobs.loc[
        (allobs[value_collum] >= 0.0) & (~np.isnan(allobs[value_collum]))
    ].copy()

    allobs["time"] = [dt.datetime.strptime(i, date_format) for i in allobs[time_col]]

    # allobs[col_name] = allobs[value_collum]

    conversion_unit = "ppbv" if species != "pm25" else "ugm3"

    allobs[col_name] = convert_pollutant(
        species, allobs[value_collum], unit, conversion_unit
    )

    allobs["lat"] = allobs[lat_col] if lat_col else lat
    allobs["lon"] = allobs[lon_col] if lon_col else lon
    location_name = obs_url.split("/")[-1].split("_")[0]
    allobs["location"] = location_name
    allobs = allobs[["time", col_name, "lat", "lon", "location"]]

    if remove_outlier:
        if not silent:   
            print("removing outlier ....")
        std = allobs[col_name].std()
        mn = allobs[col_name].mean()
        minobs = mn - remove_outlier * std
        maxobs = mn + remove_outlier * std
        norig = allobs.shape[0]

        allobs = allobs.loc[
            (allobs[col_name] >= minobs) & (allobs[col_name] <= maxobs)
        ].copy()

 
        z_scores = np.abs(stats.zscore(allobs[col_name]))
        threshold = 3
        outlier_indices = np.where(z_scores > threshold)[0]  
        allobs = allobs.drop(allobs.index[outlier_indices]) 
        
        if start is not None:
            start = pd.to_datetime(start)
        if end is not None:
            end = pd.to_datetime(end)

        if start is not None and end is not None:
            allobs = allobs[(allobs['time'] >= start) & (allobs['time'] <= end)]

    return allobs

def extract_metadata(content):
    location_name = re.search(r'Full location name:\s*(.+)', content).group(1).strip()
    latitude = float(re.search(r'Location latitude \[deg\]:\s*([-\d.]+)', content).group(1))
    longitude = float(re.search(r'Location longitude \[deg\]:\s*([-\d.]+)', content).group(1))
    return {
        'location_name': location_name,
        'latitude': latitude,
        'longitude': longitude
    }


def read_pandora(url):
    response = requests.get(url)
    content = response.text
    metadata = extract_metadata(content)
    data_start_index = content.index("---------------------------------------------------------------------------------------") + len("---------------------------------------------------------------------------------------")
    
    # Read file with flexible column handling
    df = pd.read_csv(url, skiprows=data_start_index, header=None, sep='\s+', encoding='ISO-8859-1', on_bad_lines='skip')

    # Ensure only first 69 columns are used
    df = df.iloc[:, :69]
    df.columns = range(df.shape[1])
    
    # Convert time column
    df['time'] = pd.to_datetime(df[0], format="%Y%m%dT%H%M%S.%fZ")
    
    result_df = pd.DataFrame({
        'time': df['time'],
        'no2_surface_conc_mol_m3': df[55],
        'no2_surface_conc_uncertainty': df[56],
        'no2_surface_conc_index': df[57],
        'no2_heterogeneity_flag': df[58],
        'no2_stratospheric_column': df[59],
        'no2_tropospheric_column': df[61],
        'no2_layer1_height_km': df[67],
        'pressure_mbar': df[13],
        'temperature_k': df[14],
        'solar_zenith_angle': df[3],
        'quality_flag_no2': df[52],
        'integration_time_ms': df[30],
        'wavelength_shift_nm': df[28]
    })
    
    
    result_df['lat'] = float(metadata['latitude'])
    result_df['lon'] = float(metadata['longitude'])
    result_df['location'] = metadata['location_name']
    result_df['value'] = df[55]
    
    numeric_columns = [
        'no2_surface_conc_mol_m3', 
        'no2_surface_conc_uncertainty', 
        'pressure_mbar', 
        'temperature_k'
    ]
    
    result_df = result_df[
        (result_df['no2_surface_conc_mol_m3'] != -9e99) & 
        (result_df['quality_flag_no2'].isin([0, 1, 2, 10, 11, 12]))
    ]
    result_df[numeric_columns] = result_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    result_df['pandora'] = convert_no2_mol_m3_to_ppbv(df[55], df[14],df[13]) / 40

    
    result_df = result_df[(result_df['no2_surface_conc_index'].isin([1,2])) & 
    #(merged_data['quality_flag_no2'].isin([10])) &
    (result_df['no2_heterogeneity_flag'].isin([0,1])) &
    (result_df['pandora'] <= 50) &
    (result_df['pandora'] >= 0)]
    
    return result_df


def convert_no2_mol_m3_to_ppbv(no2_mol_m3, temperature_k, pressure_mbar):
    """
    Convert NO2 concentration from mol/m³ to ppbv.
    
    Args:
        no2_mol_m3 (float): NO2 concentration in mol/m³
        temperature_k (float): Temperature in Kelvin
        pressure_mbar (float): Pressure in millibars
    
    Returns:
        float: NO2 concentration in ppbv
    """
    pressure_atm = pressure_mbar / 1013.25
    return no2_mol_m3 * (24.45 * 1e9) / (0.0821 * temperature_k * pressure_atm)

def read_validation_set(
    openaq_id=None,
    source=None,
    name=None,
    url=None,
    time_col=None,
    date_format=None,
    obs_val_col=None,
    lat_col=None,
    lon_col=None,
    species=None,
    lat=999,
    lon=-999,
    unit=None,
    start=None,
    end=None,
    remove_outlier = None,
    **kwargs,
):

    all_obs = pd.DataFrame()
    isite = ObsSite(
        openaq_id, model_source=None, species=species, observation_source=source
    )
    isite._silent = True
    #print(start)
    isite.read_obs(
        source=source,
        url=url,
        time_col=time_col,
        date_format=date_format,
        value_collum=obs_val_col,
        lat_col=lat_col,
        lon_col=lon_col,
        species=species,
        lat=lat,
        lon=lon,
        unit=unit,
        start=start,
        end=end,
        rename_column = name,
        remove_outlier = remove_outlier,
        **kwargs,
    )

    return isite._obs




def convert_no2_to_ppbv(
    no2_concentration_mol_m3=None,
    volume_m3=1.0,
    pressure_pa=None,
    elevation_m=None,
    temperature_k=None,
):

    ideal_gas_constant = 8.314  # J/(mol·K)

    no2_concentration_mol_cm3 = no2_concentration_mol_m3 * 1.0e-6

    moles_no2 = no2_concentration_mol_cm3 * volume_m3

    pressure_conv = pressure_pa * 100
    pressure_at_sea_level = (
        pressure_conv
        * (1 - 0.0065 * elevation_m / (temperature_k + 0.0065 * elevation_m + 273.15))
        ** 5.2561
    )
    moles_no2_adjusted = moles_no2 * pressure_at_sea_level / pressure_conv

    # Convert adjusted concentration to ppbv
    ppbv = (moles_no2_adjusted / volume_m3) * 1.0e9  # 1 ppbv = 1e9 molecules/m³

    return ppbv


def nsites_by_threshold(df, maxconc=50):
    """Write number of sites with mean concentration above concentration threshold for concentrations ranging from 0 to maxconc ppbv"""
    concrange = np.arange(maxconc + 1) * 1.0
    ns = []
    for ival in concrange:
        nsit = df.loc[df.value > ival].shape[0]
        ns.append(nsit)
    nsites = pd.DataFrame()
    nsites["threshold"] = concrange
    nsites["nsites"] = ns
    return nsites


def plot_deviation_orig(siteRatios, title=None, minval=-30.0, maxval=30.0):
    """Make global map showing deviation betweeen predictions and observations"""
    siteRatios["text"] = [
        "{0:}, Deviation={1:.2f}%".format(i, j)
        for i, j in zip(siteRatios["name"], siteRatios["relChange"])
    ]
    fig = go.Figure(
        data=go.Scattergeo(
            lon=siteRatios["lon"],
            lat=siteRatios["lat"],
            text=siteRatios["text"],
            mode="markers",
            marker=dict(
                size=siteRatios["obs"],
                sizemode="area",
                color=siteRatios["relChange"],
                cmin=minval,
                cmax=maxval,
                colorscale="RdBu",
                autocolorscale=False,
                reversescale=True,
                line_color="rgb(40,40,40)",
                line_width=0.5,
                colorbar_title="NO2 deviation",
            ),
        )
    )
    fig.update_layout(
        title_text="Test",
        showlegend=False,
        height=300,
        geo=dict(landcolor="rgb(217,217,217)"),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    return fig


def merge_intervales_with_model(self, confidenceIntervals):
    """Merge intervals with actual model predictions"""
    all_intervals = confidenceIntervals[
        ["time", "value", "upper", "lower", "mid"]
    ].copy()
    all_intervals = all_intervals.resample("1H").mean()
    all_intervals = all_intervals.reset_index()
    all_intervals.columns = ["time", "value", "upper", "lower", "mid"]
    xg_preditions = self.predict()
    model_data = self._mod[["time", "pm25_rh35_gcc"]].copy()

    model_data["time"] = [
        dt.datetime(i.year, i.month, i.day, i.hour, 0, 0) for i in model_data["time"]
    ]
    xg_preditions = xg_preditions.merge(model_data)
    xg_preditions = xg_preditions.set_index("time").resample("1h").mean().reset_index()
    all_intervals = all_intervals.merge(xg_preditions)
    all_intervals = all_intervals.merge(xg_preditions)
    all_intervals = all_intervals.set_index("time").resample("1D").mean()
    return all_intervals



import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re

def location_plot(dataframe=None, location_name=None, title="Location x (lat, lon)", species="no2", unit="ppbv", model_info="", resample=None, directory="plots/"):
    """General plotting routine for all the forecasts with different observation sources"""

    location_name_cor = re.sub("[^0-9a-zA-Z]+", "_", location_name)
    dataframe["time"] = pd.to_datetime(dataframe["time"])
    final_df = dataframe.copy()
    fvar = "pm25_rh35_gcc" if species == "pm25" else species

    rows = 3 if any(col in final_df.columns for col in ["pandora_height", "zpbl"]) else 1
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    x = final_df["time"]

    unit_display = "ug/m3" if unit == "ugm3" else unit


    if "no2" in final_df.columns:
        fig.add_trace(go.Scatter(x=x, y=final_df["no2"], mode='lines', name="GEOS-CF Model",
                                 line=dict(color="gray", width=2, dash='dash')), row=1, col=1)

    if "pandora" in final_df.columns:
        fig.add_trace(go.Scatter(x=x, y=final_df["pandora"], mode='lines', name="Pandora Surface",
                                 line=dict(color="black", width=2)), row=1, col=1)

    if "corrected_pandora" in final_df.columns:
        fig.add_trace(go.Scatter(x=x, y=final_df["corrected_pandora"], mode='lines', name="Model + ML",
                                 line=dict(color="red", width=2)), row=1, col=1)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=f"{species} ({unit_display})",
        width=850,
        height=400,
        legend_title="Legend",
        xaxis_tickangle=-45,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white'
        
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(ticksuffix=" ppbv")

    if not os.path.exists(directory):
        os.makedirs(directory)

    fig.write_image(f"{directory}{location_name_cor}_{species}.svg", width=1600, height=600)

    return fig


def merge_dataframes(df_list, index_col, resample=None, how= 'outer'):
    """merge data routine"""
    merged_df = df_list[0]
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=index_col, how=how)
    if resample:
        numeric_cols = merged_df.select_dtypes(include='number').columns.tolist()
        merged_df = merged_df.resample(resample, on=index_col)[numeric_cols].mean().reset_index()
    return merged_df


def export_to_gesdisc(
    HAQAST_DATA=None,
    location_name=None,
    species=None,
    unit=None,
    start=dt.datetime.today() - relativedelta(years=1),
    end=dt.datetime.today(),
    lat=None,
    lon=None,
    IdentifierProductDOI=None,
):
    """Convert forecasts with GES DISC Formatting"""

    current_datetime = dt.datetime.now()
    current_time_GMT = dt.datetime.utcnow()
    current_time_GMT = time.mktime(current_time_GMT.timetuple())
    HAQAST_DATA["time"] = pd.to_datetime(HAQAST_DATA["time"])
    fvar = "pm25_rh35_gcc" if species == "pm25" else species

    first_timestamp = HAQAST_DATA["time"].min()
    last_timestamp = HAQAST_DATA["time"].max()
    min_year = HAQAST_DATA["time"].dt.year.min()
    max_year = HAQAST_DATA["time"].dt.year.max()
    location_name = location_name
    location_name = location_name.replace(" ", "_")
    lon = lon
    lat = lat
    parameter = species
    unit = unit
    VersionID = "1.0.0"
    Format = "ASCII"
    RangeBeginningDate = first_timestamp.strftime("%Y-%m-%d")
    RangeBeginningTime = first_timestamp.strftime("%H:%M:%S")
    RangeEndingDate = last_timestamp.strftime("%Y-%m-%d")
    RangeEndingTime = last_timestamp.strftime("%H:%M:%S")

    ProductionDateTime = current_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ProcessingLevel = "Level 4"
    Conventions = "ASCII"
    DataSetQuality = "A description of the bias-correction methodology and corresponding uncertainty estimates are provided in Keller et al. 2021 (https://doi.org/10.5194/acp-21-3555-2021)"
    title = "HAQAST localized ground-level concentration of nitrogen dioxide (NO2): model-observation fused,1-Hourly,Time-Averaged,Ground-Level (2m)"
    history = f"Original file generated: {current_time_GMT}"
    source = "GEOS-CF v1.0"
    institution = "NASA Global Modeling and Assimilation Office"
    references = "http://gmao.gsfc.nasa.gov"
    TemporalRange = f"{RangeBeginningDate} -> {RangeEndingDate}"
    filename = f"HAQAST_localized_concentration_{parameter}.L4.V1.{location_name}.{min_year}-{max_year}.txt"
    StationLatitude = lat
    StationLongitude = lon
    Contact = "http://gmao.gsfc.nasa.gov"
    (
        SouthernmostLatitude,
        NorthernmostLatitude,
        WesternmostLongitude,
        EasternmostLongitude,
    ) = calculate_extremes([(lat, lon)])
    SpatialCoverage = "point-source"

    if species == "no2":
        IdentifierProductDOI = "10.5067/R3MOD87DBR3E"
        shortName = "HAQLOCNO2"
    elif species == "pm25":
        IdentifierProductDOI = "10.5067/MGBETJN7JJCS"
        shortName = "HAQLOCPM25"
    elif species == "o3":
        IdentifierProductDOI = "10.5067/11JBPNUERB7L"
        shortName = "HAQLOCO3"
    else:
        IdentifierProductDOI = "10.5067/R3MOD87DBR3E"
        shortName = "HAQLOCNO2"

    HAQAST_DATA = HAQAST_DATA.rename(
        columns={
            "time": "ISO8601",
            "localised": "localized_model_value",
            fvar: "uncorrected_model_value",
        }
    )
    HAQAST_DATA["ISO8601"] = pd.to_datetime(HAQAST_DATA["ISO8601"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    HAQAST_DATA["location"] = location_name
    HAQAST_DATA["lat"] = lat
    HAQAST_DATA["lon"] = lon
    HAQAST_DATA["parameter"] = parameter
    HAQAST_DATA["unit"] = unit

    HAQAST_DATA = HAQAST_DATA[
        [
            "ISO8601",
            "location",
            "lat",
            "lon",
            "parameter",
            "unit",
            "localized_model_value",
            "uncorrected_model_value",
        ]
    ]

    # prepare HAQAST Metadata
    HAQAST_DATA["ISO8601"] = pd.to_datetime(HAQAST_DATA["ISO8601"], errors="coerce")

    folder_path = "HAQAST_localized_concentration_L4"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, filename)

    with open(file_path, "a") as file:
        file.truncate(0)
        file.write(
            "#######################################################################\n"
        )
        file.write(f'## GranuleID = "{filename}" \n')
        file.write(f'## ShortName = "{shortName} \n')
        file.write(f'## DOI = "{IdentifierProductDOI}S \n')
        file.write(
            f'## LongName = "HAQAST localized ground-level concentration of {parameter}: model-observation fused,1-Hourly,Time-Averaged,Ground-Level (2m)" \n'
        )
        file.write(f'## VersionID = "{VersionID}" \n')
        file.write(f'## Format = "{Format}" \n')
        file.write(f'## VersionID = "{VersionID}" \n')
        file.write(f'## RangeBeginningDate = "{RangeBeginningDate}" \n')
        file.write(f'## RangeBeginningTime = "{RangeBeginningTime}" \n')
        file.write(f'## RangeEndingDate = "{RangeEndingDate}" \n')
        file.write(f'## RangeEndingTime = "{RangeEndingTime}" \n')
        file.write(f'## IdentifierProductDOI = "{IdentifierProductDOI}" \n')
        file.write(f'## ProductionDateTime = "{ProductionDateTime}" \n')
        file.write(f'## ProcessingLevel = "{ProcessingLevel}" \n')
        file.write(f'## Conventions = "{Conventions}" \n')
        file.write(f'## DataSetQuality = "{DataSetQuality}" \n')
        file.write(f'## Title = "{title}" \n')
        file.write(f'## History = "{history}" \n')
        file.write(f'## Source = "{source}" \n')
        file.write(f'## Institution = "{institution}" \n')
        file.write(f'## references = "{references}" \n')
        file.write(f'## TemporalRange = "{TemporalRange}" \n')
        file.write(f'## Filename = "{filename}" \n')
        file.write(f'## StationLatitude = "{StationLatitude}" \n')
        file.write(f'## StationLongitude = "{StationLongitude}" \n')
        file.write(f'## Contact = "{Contact}" \n')
        file.write(f'## SouthernmostLatitude = "{SouthernmostLatitude}" \n')
        file.write(f'## NorthernmostLatitude = "{NorthernmostLatitude}" \n')
        file.write(f'## WesternmostLongitude = "{WesternmostLongitude}" \n')
        file.write(f'## EasternmostLongitude = "{EasternmostLongitude}" \n')
        file.write(f'## SpatialCoverage = "{SpatialCoverage}" \n')
        file.write(
            "#######################################################################\n"
        )
        HAQAST_DATA.to_csv(file, index=False)
        print("dataproduct is saved successfully ")


def get_site_information(site_id):
    """Collect location information from OpenAQ"""
    url = (
        "https://api.openaq.org/v2/locations/"
        + str(site_id)
        + "?limit=100&page=1&offset=0&sort=desc&radius=1000&order_by=lastUpdated&dumpRaw=false"
    )

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers).json()

    return response["results"][0]["name"]


def plot_actual_vs_predicted(actual_values, predicted_values, xlabel, ylabel, title, file_name, color='blue'):
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[mask]
    predicted_values = predicted_values[mask]
    
    if len(actual_values) == 0 or len(predicted_values) == 0:
        print("No valid data points to plot.")
        return
    
    X = actual_values.reshape(-1, 1)  
    y = predicted_values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    plt.scatter(actual_values, predicted_values, label='Estimated sfcmr', color=color, marker='o') 
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.plot(actual_values, model.predict(X), color='black', linewidth= 0.5, label=f'y = {slope:.2f}x + {intercept:.2f}')

   
    plt.plot(actual_values, actual_values, 'r--')
    
    max_val = max(np.max(actual_values), np.max(predicted_values))
    min_val = min(np.min(actual_values), np.min(predicted_values))

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.grid(True)
    plt.legend() 
    plt.savefig(f'plots/pandora_cf/{file_name}.png', dpi=1200)
    plt.show()
    

def scatter_plot_multiple_markers(df, x_column, y_columns, markers=None, title=None, xlabel=None, ylabel=None, marker_fill=False, file_name=None, directory = "plots/pandora_cf/"):
    colors = {
        "corrected_openaq": "orange",
        "openaq": "green",
        "corrected_pandora": "red",
        "no2": "black",
        "pandora": "blue",
        "estimated_sfcmr": "red",
    }

    if markers is None:
        markers = ['o', 's', '^', 'D', 'x', '+', '*', 'p', 'h', 'v']

    if not isinstance(x_column, str):
        raise ValueError("x_column must be a string.")

    if x_column not in df.columns:
        raise ValueError(f"x_column '{x_column}' not found in DataFrame.")

    if not all(isinstance(col, str) for col in y_columns):
        raise ValueError("All elements in y_columns must be strings.")

    valid_y_columns = [col for col in y_columns if col in df.columns]

    if not valid_y_columns:
        raise ValueError("No valid y_columns found in DataFrame.")

    fig, ax = plt.subplots()

    for i, y_col in enumerate(valid_y_columns):
        marker = markers[i % len(markers)]
        color = colors.get(y_col, 'b')
        
        if y_col == 'local_obs':
            marker_label ="Local Observation"
        elif y_col == 'openaq':
            marker_label ="OpenAQ"
        elif y_col == 'no2':
            marker_label ="GEOS-CF"
        elif y_col == 'value':
            marker_label ="Pandora"
        elif y_col == 'estimated_sfcmr':
            marker_label = "Hybrid (Pandora-CF)"
        else:
            marker_label = y_col
            
                

        if marker_fill:
            ax.scatter(df[x_column], df[y_col], marker=marker, color=color, label=marker_label)
        else:
            ax.scatter(df[x_column], df[y_col], marker=marker, edgecolors=color, facecolors='none', label=marker_label)


        data_min = min(df[x_column].min(), df[y_col].min())
        data_max = max(df[x_column].max(), df[y_col].max())
        ax.set_xlim(data_min, data_max)
        ax.set_ylim(data_min, data_max)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_aspect('equal', adjustable='box')

    plt.grid(True)
    plt.plot(df[x_column], df[x_column], color='black', linewidth=0.5)
    plt.legend()
    plt.savefig(f'{directory}/{file_name}_scatter_plot_{xlabel}.png', dpi=1200)
    print("scatter plot saved")
    plt.show()
    

    
def evaluate_time_series(actual, predicted):
    
    if len(actual) == 0 or len(predicted) == 0:
        print("No valid data points to plot.")
        return
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]


    residuals = actual - predicted

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    percentage_within_threshold = np.mean(np.abs(residuals) <= 0.1 * np.max(actual)) * 100  
    pearson_corr, _ = pearsonr(actual, predicted)
    spearman_corr, _ = spearmanr(actual, predicted)
    r_squared = r2_score(actual, predicted)
    autocorr_residuals = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    std_dev = np.std(residuals)
    mean_bias = np.mean(residuals)
    mean_actual = np.mean(actual)
    mean_predicted = np.mean(predicted)

    results = pd.DataFrame({
        'Statistic': ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error',
                      'Mean Absolute Percentage Error', 'Percentage Within Threshold',
                      'Pearson Correlation', 'Spearman Correlation', 'R-squared',
                      'Autocorrelation of Residuals', 'Standard Deviation of Residuals',
                      'Mean Bias', 'Mean Actual Value', 'Mean Predicted Value'],
        'Value': [mae, mse, rmse, mape, percentage_within_threshold,
                  pearson_corr, spearman_corr, r_squared, autocorr_residuals,
                  std_dev, mean_bias, mean_actual, mean_predicted]
    })

    return results
    
    
    
def calculate_extremes(coordinates):
    """Calculate coordinate extremes based on lat, lon of the location"""
    southernmost_lat = min(coordinates, key=lambda x: x[0])[0]
    northernmost_lat = max(coordinates, key=lambda x: x[0])[0]
    westernmost_long = min(coordinates, key=lambda x: x[1])[1]
    easternmost_long = max(coordinates, key=lambda x: x[1])[1]

    return southernmost_lat, northernmost_lat, westernmost_long, easternmost_long


def convert_pollutant(species, value, current_unit, conversion_unit):
    """Unit conversions routine"""
    
    PPB2UGM3 = {"no2": 1.88, "o3": 1.97}
    VVtoPPBV = 1.0e9

    if current_unit == "ugm3" and conversion_unit == "ppbv":
        print(f'converting from {current_unit} to {conversion_unit}')
        value = value * (1.0 / PPB2UGM3[species])
            
    if current_unit == conversion_unit:
        value = value
    
    return value 


def convert_ugm3_to_ppbv(value,species):

    PPB2UGM3 = {"no2": 1.88, "o3": 1.97}
    VVtoPPBV = 1.0e9
    conv_factor = PPB2UGM3[species]
    ppbv = value ** 1.0 / conv_factor

    return ppbv


def log_if_condition(condition, message, log_file="logs/locations_log.txt"):
    """Log management routine"""
    try:
        if condition:
            current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a") as log:
                log.write(f"{current_time} - {message}\n")
    except Exception as e:
        print(f"Error occurred while logging: {e}")

def get_location_info(url, location_code):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if location_code in data:
            location_data = data[location_code]
            info = {}
            
            for key, value in location_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                info[f'{key}_{sub_key}_{sub_sub_key}'] = sub_sub_value
                        else:
                            info[f'{key}_{sub_key}'] = sub_value
                else:
                    info[key] = value
            
            return info
        else:
            print(f"Location with code {location_code} not found")
            return None
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    
    
def plot_locations_map(locations, title='Location Map', output_path='locations_map', directory='pandora_cf/plots'):
    """
    Create map plot of locations with markers.
    
    Args:
        locations (list): List of dictionaries with location details
        title (str): Map title
        output_path (str): Base filename for output files
    
    Returns:
        plotly.graph_objs.Figure: Plotly figure object
    """
    # Validate input
    if not locations or not isinstance(locations, list):
        print("Invalid locations data")
        return None

    # Ensure output directory exists
    os.makedirs(directory, exist_ok=True)

    location_df = pd.DataFrame(locations)
    
    if 'lat' in location_df.columns and 'lon' in location_df.columns:
        center_lat = location_df['lat'].mean()
        center_lon = location_df['lon'].mean()
        print(f"Center Latitude: {center_lat}, Center Longitude: {center_lon}")
        center_lat = location_df['lat'].mean()
        center_lon = location_df['lon'].mean()


        colors = ['blue' if loc['type'] == 'OpenAQ' else 'red' for loc in locations]



        fig = go.Figure(go.Scattermapbox(
            mode='markers+text',
            lon=location_df['lon'],
            lat=location_df['lat'],
            marker=dict(
                size=10,
                color=colors,
                opacity=0.7
            ),
            text=location_df['name'],
            textposition='bottom center',
            hoverinfo='text',
            hovertext=[
                f"Name: {row['name']}<br>Type: {row.get('type', 'N/A')}" 
                for _, row in location_df.iterrows()
            ]
        ))

        fig.update_layout(
            title=title,
            mapbox=dict(
                style='open-street-map',
                center=dict(
                    lat=center_lat,
                    lon=center_lon
                ),
                zoom=5
            ),
            showlegend=False,
            height=800,
            width=1200
        )

        lat_range = location_df['lat'].max() - location_df['lat'].min()
        lon_range = location_df['lon'].max() - location_df['lon'].min()

        zoom_lat = max(2, min(10, 10 - lat_range))
        zoom_lon = max(2, min(10, 10 - lon_range))

        fig.update_layout(
            mapbox=dict(zoom=min(zoom_lat, zoom_lon))
        )

        try:
            fig.write_image(f'{directory}/{output_path}.png', scale=2)
            fig.write_image(f'{directory}/{output_path}.svg')
        except Exception as e:
            print(f"Error saving map: {e}")
        
        return fig
    else:
        print("Latitude or Longitude column missing.")


    




def get_no2_locations(lat, lon, radius_km):

    endpoint = 'https://api.openaq.org/v2/locations'
    

    params = {
        'latitude': lat,
        'longitude': lon,
        'radius': radius_km * 1000,  
        'parameter': 'no2',
        'limit': 1000
    }
    
    headers = {'X-API-Key': '849b4d760f2679b9e0cafea2a23a698578107984fb150a81d15640600bdb271f'}
    response = requests.get(endpoint, params=params, headers= headers)
    
    if response.status_code == 200:
        data = response.json()

        return [{'id': location['id'], 'name': location['name'], 'coordinates': location['coordinates']} for location in data['results']]
    else:
        print('Error:', response.status_code, response.text)
        return []
