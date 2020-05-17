# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 08:06:35 2020

@author: pribahsn
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import timedelta
from scipy.optimize import curve_fit
from typing import Dict
from SIR import SirParameter

def addmeasures(parameter: Dict, date, scaling: int) -> Dict:

    parameter['betagamma'] = parameter['betagamma'].append(pd.DataFrame({'beta': parameter['betagamma'].iloc[0]['beta']*scaling, 'gamma': parameter['betagamma'].iloc[0]['gamma']}, index=[date]), sort=True)

    return parameter


def estimate_sir_parameters(data, country, threshold=20, output=False, forecast=50, first_segment=15, last_segment=4):
    """"
    Estimates the parameters for the SIR model defined in
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model

    Estimation is conducted assuming exponential growth I(t) = I0 * exp(beta * t) of the infected at the early stage
    of the epidemics. It is performed by least-square regression  log I(t) ~ beta * t + log(I0)

    :param data: history
    :param country:
    :param threshold: history is taken into account from the day when confirmed cases exceed threshold
    :param output:
    :param forecast:
    :param first_segment:
    :param last_segment:
    """

    def linear(x, A, B):
        return A+B*x

    def exponential(x, A, B):
        return A*np.exp(B * x)

    # extract relevant history
    data_confirmed = data[country][data[country] > threshold]
    data_confirmed_start_date = data[country][data[country] > threshold].index[0]
    time_index = np.array(range(0, len(data_confirmed)))

    # fit parameters for first segment
    log_first_segment = np.log(data_confirmed.values[:first_segment])
    guess_log_I0 = log_first_segment[0]
    guess_growth_rate = ((log_first_segment[1:] - guess_log_I0) / time_index[1:first_segment]).mean()
    guessed_parameters = [guess_log_I0, guess_growth_rate]
    first_segment_fit, _ = curve_fit(linear, time_index[:first_segment], log_first_segment, p0=guessed_parameters)

    # gamma taken as average time till recovery
    # TODO: research and add estimation for gamma
    gamma = 1/15
    betagamma = pd.DataFrame({'beta': first_segment_fit[1] - gamma, 'gamma': gamma}, index=[data_confirmed_start_date])

    # estimate point in time when measures show effect and estimate parameters with measures
    if last_segment is not None:
        # estimate flow parameters for period with active measures
        last_segment_start = len(data_confirmed) - last_segment
        log_last_segment = np.log(data_confirmed.values[-last_segment:])
        guess_growth_rate = ((log_last_segment[1:] - log_last_segment[0]) / np.arange(1, len(log_last_segment))).mean()
        guess_log_I0 = log_last_segment[0] * np.exp(-guess_growth_rate * last_segment_start)
        guessed_parameters = [guess_log_I0, guess_growth_rate]
        last_segment_fit, _ = curve_fit(linear, time_index[-last_segment:], log_last_segment, p0=guessed_parameters)

        # compare prediction with and without measures and estimate time when measures start to show effects;
        # we estimate this as the day when the two curves cross; since the curve with measures is a flatter exponential
        # that starts at a higher level, this is the day when the prediction with measures is for the first time higher
        # than the prediction without measures
        prediction_first_fit = exponential(range(forecast), np.exp(first_segment_fit[0]), first_segment_fit[1])
        prediction_last_fit = exponential(range(forecast), np.exp(last_segment_fit[0]), last_segment_fit[1])
        no_curfew_days = int(sum(prediction_last_fit - prediction_first_fit > 0))
        if no_curfew_days < forecast:
            time_curfew = data_confirmed_start_date + timedelta(days=no_curfew_days)
            betagamma = betagamma.append(
                pd.DataFrame({'beta': last_segment_fit[1] - gamma, 'gamma': gamma}, index=[time_curfew])
            )

    if output:
        print(f'ɣ: {gamma}')
        print(f'β: {first_segment_fit[1]+gamma}')
        print(f'I₀: {np.exp(first_segment_fit[0])}')
        print(f'R₀: {(first_segment_fit[1]+gamma)/gamma}')
        print(f't₀: {data_confirmed_start_date}')

        #add time index to forecasts
        forecast_time_index = [data_confirmed_start_date + timedelta(days=k) for k in range(forecast)]
        prediction_first_fit = pd.Series(prediction_first_fit, index=forecast_time_index)
        prediction_last_fit = pd.Series(prediction_last_fit, index=forecast_time_index)


        plt.plot(prediction_first_fit)
        plt.plot(prediction_last_fit)
        plt.plot(data_confirmed)
        plt.yscale('log')
        plt.grid(which='both')
        plt.show()

    parameters: SirParameter = {'I0': np.exp(first_segment_fit[0]),
                                't0': data_confirmed_start_date,
                                'betagamma': betagamma}
    return parameters

