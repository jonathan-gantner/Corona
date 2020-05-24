# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 08:33:58 2020

@author: pribahsn & jgantner

Implements SIR model as defined in https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta, date
from scipy.integrate import odeint
from typing import Optional

from scipy.optimize import curve_fit
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


def compute_sir_model_with_measures(population: int, parameter: SirParameter, forecast: int = 600,
                                    ignore_measures: bool = False) -> pd.DataFrame:
    """
    Simulates the pandemics based on the SIR model with possible updates of the flow
    parameters beta and gamma in order to simulate the implementation of measures by the government

    :param population: the size of the population
    :param parameter: a dict of type SirParameter that contains the model parameters, possibly including several
        updates of the flow parameters beta and gamma
    :param forecast: number of days in the forecast
    :param ignore_measures: bolean - if True, then measures are ignored and the entire time horizon is computed
        with the parameters given for the first time interval, i.e. at the beginning of the simulation

    :return: DataFrame with date-index and thre columns 'S', 'I', and 'R', that contain the figures for susceptible,
        infected and recovered people in the simulation
    """
    #TODO: use cleaned function in main script
    if ignore_measures:
        simulation_intervals = [{"start": parameter["betagamma"].index[0],
                                "end": parameter["betagamma"].index[0] + timedelta(days=forecast + 1)}]
    else:
        simulation_starts = list(parameter["betagamma"].index)
        simulation_ends = simulation_starts[1:] + [simulation_starts[0] + timedelta(days=forecast + 1)]
        if len(simulation_ends) > 1:
            assert all(t1 < t2 for t1, t2 in zip(simulation_ends[:-1], simulation_ends[1:])), \
                "Simulation was started with illegal parameter changes; change is after end of simulation"
        simulation_intervals = [{"start": start, "end": end} for start, end in zip(simulation_starts, simulation_ends)]

    initial_state = [population, parameter["I0"], 0]
    SIR_data = pd.DataFrame([initial_state], columns=["S", "I", "R"], index=[simulation_intervals[0]["start"]])

    for interval in simulation_intervals:
        interval_duration = (interval["end"] - interval["start"]).days + 1
        interval_simulation = compute_sir_model(t0=interval["start"],
                                                I0=SIR_data.loc[interval["start"], "I"],
                                                N=population,
                                                beta=parameter["betagamma"].loc[interval["start"], "beta"],
                                                gamma=parameter["betagamma"].loc[interval["start"], "gamma"],
                                                R0=SIR_data.loc[interval["start"], "R"],
                                                forecast=interval_duration
                                                )
        SIR_data = SIR_data.append(interval_simulation.drop(index=interval["start"]))

    return SIR_data


def estimate_sir_parameters(history_infected, threshold=250, output=False, forecast=50, first_segment=15, last_segment=20):
    """"
    Estimates the parameters for the SIR model defined

    Estimation is conducted assuming exponential growth I(t) = I0 * exp(beta * t) of the infected at the early stage
    of the epidemics. It is performed by least-square regression  log I(t) ~ beta * t + log(I0),

    :param history_infected: history of data on which the fit is conducted
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
    data_confirmed = history_infected[history_infected > threshold]
    data_confirmed_start_date = data_confirmed.index[0]
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

        # plot forecasts
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




if __name__ == "__main__":
    population = 1000000
    data = pd.DataFrame(data = [population], columns=["population"], index=["Ctry1"])
    parameter: SirParameter = {'I0': 1,
                               't0': date(2020, 2, 1),
                               'betagamma': pd.DataFrame(data=[[0.4 + 1/15, 1/15],[0.2 + 1/15, 1/15]],
                                                        columns = ["beta", "gamma"],
                                                        index=[date(2020, 2, 1), date(2020, 3, 15)])}

    SIR_data_old = compute_sir_model_old(data=data, country="Ctry1", parameter=parameter, forecast=100)
    SIR_data_without_measures = compute_sir_model_with_measures(population=population, parameter=parameter,
                                                                forecast=100, ignore_measures=True)

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