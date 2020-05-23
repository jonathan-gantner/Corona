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


def compute_sir_model(t0: date, I0: int, N: int, beta: float, gamma: float, R0: Optional[int] = 0,
                      forecast: Optional[int] = 600) -> pd.DataFrame:
    """
    Simulates the pandemics based on the SIR model for a given set of parameters
    :param t0: starting date of simulation
    :param I0: number of infected people at the beginning of the simulation
    :param beta: model parameter - average number of contacts per person per time unit  times the probability of
        disease transmission per contact between an infected and a susceptible person
    :param gamma: model parameter - gamma = 1/D where D is the number of days the an infected person in infectious
    :param R0: number of recovered people at the beginning of the simulation, default: 0
    :param forecast: number of days in the simulation

    :return: DataFrame with date-index and three columns 'S', 'I', and 'R' that contain the number of susceptible,
        infected, and recovered people for each day of the simulation

    """

    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    y0 = N - I0 - R0, I0, R0
    t = np.linspace(0, forecast - 1, forecast)

    dummy = odeint(deriv, y0, t, args=(N, beta, gamma))
    SIR_data = pd.DataFrame(data=dummy, columns=['S', 'I', 'R'],
                            index=[t0 + timedelta(days=x) for x in range(forecast)])
    return SIR_data


def compute_sir_model_with_measures(population: int, parameter: SirParameter, forecast: int = 600) -> pd.DataFrame:
    """
    Simulates the pandemics based on the SIR model, once ignoring and once implementing possible updates of the flow
    parameters beta and gamma in order to simulate the implementation of measures by the government

    :parm population: the size of the population
    :param parameter: a dict of type SirParameter that contains the model parameters, possibly including several
        updates of the flow parameters beta and gamma
    :param forecast: number of days in the forecast

    :return: DataFrame with date-index and six columns 'S0', 'I0', 'R0', 'S', 'I', and 'R', the first three columns
        contain the figures for susceptible, infected and recovered people in a simulation that ignores updates of the
        flow parameters beta and gamma, i.e. the entire time horizon is simulated with the parameters valid at t0 -
        this simulates an epidemics when the government does not implement any measures
        the last three columns implement contain the figures for susceptible, infected and recovered people in a
        simulation that updates the flow parameters beta and gamma as defined in parameter; this simulates an epidemics
        in which the government implements measures to slow down the propagation
    """
    #TODO: use cleaned function in main script
    simulation_starts = list(parameter["betagamma"].index)
    simulation_ends = simulation_starts[1:] + [simulation_starts[0] + timedelta(days=forecast + 1)]
    simulation_intervals = [{"start": start, "end": end} for start, end in zip(simulation_starts, simulation_ends)]

    inital_state = [population, parameter["I0"], 0]
    SIR_data = pd.DataFrame([inital_state], columns=["S", "I", "R"], index=[simulation_starts[0]])

    for interval in simulation_intervals:
        interval_duration = (interval["end"] - interval["start"]).days + 1
        interval_simulation = compute_sir_model(t0=interval["start"],
                                                I0=SIR_data.loc[interval["start"], "I"],
                                                N=population,
                                                beta=parameter["betagamma"].loc[interval["start"], "beta"],
                                                gamma=parameter["betagamma"].loc[interval["start"], "gamma"],
                                                R0=SIR_data.loc[interval["start"], "R"],
                                                forecast = interval_duration
                                                )
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
    SIR_data_without_measures = compute_sir_model(t0=parameter["t0"], I0=parameter["I0"], N=population,
                                                  beta=parameter["betagamma"].iloc[0, 0],
                                                  gamma=parameter["betagamma"].iloc[0, 1],
                                                  forecast=100)

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