# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 08:33:58 2020

@author: pribahsn

Implements SIR model as defined in https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta, date
from scipy.integrate import odeint
from typing import Optional
from typing_extensions import TypedDict


class SirParameter(TypedDict):
    """
    Container Class for parameters for SIR model that are estimated from data
    :param I0: amount of infected at the beginning of the simulation
    :param t0: date of the beginning of simulation/pandemics
    :param betagamma: DataFrame with two columns for the parameters beta and gamma and a datetime index;
        index defines from when the parameters are valid; this allows to update the flow parameters to model changes
        when measures by the government are taken
    """
    I0: str
    t0: date
    betagamma: pd.DataFrame


def compute_sir_model_old(data: pd.DataFrame, country: str, parameter: SirParameter,
                      output: bool = False, forecast: int = 600):
    """
    Simulates the pandemics based on the SIR model
    """
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # compute SIR model without taking measures into account
    y0 = data['population'].at[country], parameter['I0'], 0
    t = np.linspace(0, forecast-1, forecast)

    dummy = odeint(deriv, y0, t, args=(data['population'].at[country],
                                       parameter['betagamma'].iloc[0]['beta'],
                                       parameter['betagamma'].iloc[0]['gamma']))
    SIR_data_regular = pd.DataFrame(data=dummy, columns=['S0', 'I0', 'R0'],
                                    index=[parameter['t0'] + timedelta(days=x) for x in range(forecast)])

    # compute SIR model taking measures into account
    SIR_data_curfew = pd.DataFrame(columns=['S', 'I', 'R'],
                                   index=[parameter['t0'] + timedelta(days=x) for x in range(forecast)])
    SIR_data_curfew.loc[parameter['t0']] = {'S': data['population'].at[country], 'I': parameter['I0'], 'R': 0}
    for i in range(len(parameter['betagamma'])):
        y0 = tuple(SIR_data_curfew.loc[parameter['betagamma'].index[i]].values)
        if i != len(parameter['betagamma'])-1:
            forecast_step = (parameter['betagamma'].index[i+1]-parameter['betagamma'].index[0]).days
            t = np.linspace(0, forecast_step, forecast_step+1)
        else:
            forecast_step = (parameter['betagamma'].index[0]+timedelta(days=forecast)-parameter['betagamma'].index[i]).days
            t = np.linspace(0, forecast_step, forecast_step+1)
        dummy = odeint(deriv, y0, t, args=(data['population'].at[country], parameter['betagamma'].iloc[i]['beta'], parameter['betagamma'].iloc[i]['gamma']))
        dummy = pd.DataFrame(columns=['S', 'I', 'R'], index=[parameter['betagamma'].index[i] + timedelta(days=x) for x in range(forecast_step+1)], data=dummy)
        SIR_data_curfew.update(dummy)

    SIR_data = pd.concat([SIR_data_regular, SIR_data_curfew], axis=1, join='inner')

    duration = (SIR_data['I'].loc[SIR_data['I'] == SIR_data['I'].max()].index[0]-SIR_data.index[0])*2
    duration = max(duration, timedelta(days=100))

    if duration.days < forecast:
        dates = [parameter['t0']+duration]
        for i in range(duration.days, forecast):
            dates.append(parameter['t0']+timedelta(days=i))
        SIR_data.drop(dates, inplace=True)

    if output:
        SIR_data.plot(title=country, grid=True)
        plt.show()

    return SIR_data


def compute_sir_model(population: int, parameter: SirParameter, recovered: Optional[int] = 0,
                      forecast: Optional[int] = 600) -> pd.DataFrame:
    """
    Simulates the pandemics based on the SIR model
    """

    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    y0 = population - parameter['I0'] - recovered, parameter['I0'], recovered
    t = np.linspace(0, forecast - 1, forecast)

    dummy = odeint(deriv, y0, t, args=(population,
                                       parameter['betagamma'].iloc[0]['beta'],
                                       parameter['betagamma'].iloc[0]['gamma']))
    SIR_data = pd.DataFrame(data=dummy, columns=['S', 'I', 'R'],
                            index=[parameter['t0'] + timedelta(days=x) for x in range(forecast)])
    return SIR_data


def compute_sir_model_with_measures(population: int, parameter: SirParameter, forecast: int = 600) -> pd.DataFrame:
    """
    Simulates the pandemics based on the SIR model, where parameters might change over time
    """
    #TODO: write comment
    #TODO: use cleaned function in main script
    simulation_starts = list(parameter["betagamma"].index)
    simulation_ends = simulation_starts[1:] + [simulation_starts[0] + timedelta(days=forecast + 1)]
    simulation_intervals = [{"start": start, "end": end} for start, end in zip(simulation_starts, simulation_ends)]

    inital_state = [population, parameter["I0"], 0]
    SIR_data = pd.DataFrame([inital_state], columns=["S", "I", "R"], index=[simulation_starts[0]])

    for interval in simulation_intervals:
        interval_parameter: SirParameter = {'I0': SIR_data.loc[interval["start"], "I"],
                                            't0': interval["start"],
                                            'betagamma': parameter["betagamma"].loc[[interval["start"]], :]}
        interval_duration = (interval["end"] - interval["start"]).days + 1
        interval_simulation = compute_sir_model(population=population, parameter=interval_parameter,
                                                recovered=SIR_data.loc[interval["start"], "R"],
                                                forecast=interval_duration)
        SIR_data = SIR_data.append(interval_simulation.drop(index=interval["start"]))

    return SIR_data


if __name__ == "__main__":
    population = 1000000
    data = pd.DataFrame(data = [population], columns=["population"], index=["Ctry1"])
    parameter: SirParameter = {'I0': 1,
                               't0': date(2020, 2, 1),
                               'betagamma': pd.DataFrame(data=[[0.4 + 1/15, 1/15],[0.2 + 1/15, 1/15]],
                                                        columns = ["beta", "gamma"],
                                                        index=[date(2020, 2, 1), date(2020, 3, 15)])}

    SIR_data_old = compute_sir_model_old(data=data, country="Ctry1", parameter=parameter, forecast=100)
    SIR_data_without_measures = compute_sir_model(population=population, parameter=parameter, forecast=100)
    SIR_data_with_measures = compute_sir_model_with_measures(population=population, parameter=parameter, forecast=100)

    #SIR_without_measures = SIR_data_old[["S0", "I0", "R0"]].join(SIR_data_without_measures)
    ax = SIR_data_old[["S0", "I0", "R0"]].plot(style=["r-", "g-", "b-"], linewidth=2.0)
    SIR_data_without_measures.plot(style=["g:", "b:", "r:"], linewidth=1.0, ax=ax)
    plt.yscale("log")
    plt.title("Simulation without Measures")
    plt.show()

    ax = SIR_data_old[["S", "I", "R"]].plot(style=["r-", "g-", "b-"], linewidth=2.0)
    SIR_data_with_measures.plot(style=["g:", "b:", "r:"], linewidth=1.0, ax=ax)
    plt.yscale("log")
    plt.title("Simulations with Measures")
    plt.show()