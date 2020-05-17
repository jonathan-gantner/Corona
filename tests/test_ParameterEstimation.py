from SIR import SirParameter
from ParameterEstimation import estimate_sir_parameters
import pandas as pd
from datetime import timedelta, date
import matplotlib.pyplot as plt
import numpy as np


def test_estimate_sir_parameters():
    start_date = date(2020, 2, 1)
    threshold = 15

    # create model data without measure
    I0 = 1
    gamma = 1 / 15
    beta = 0.4 + gamma
    date_index = [start_date + timedelta(days=k) for k in range(27)]
    data = [I0 * np.exp((beta + gamma) * k) for k in range(27)]
    history = pd.Series(data=data, index=date_index)

    # create model data with measure
    I0_measure = 200
    gamma_measure = 1 / 15
    beta_measure = 0.2 + gamma
    date_index_measure = [start_date + timedelta(days=k) for k in range(27, 50)]
    data_measure = [I0_measure * np.exp((beta_measure + gamma_measure) * k) for k in range(27, 50)]
    history_measure = pd.Series(data=data_measure, index=date_index_measure)

    history = history.append(history_measure)
    noise = pd.Series(np.random.normal(scale=0.005, size=50), index=history.index)
    history = history + noise * history

    t0 = history[history > threshold].index[0]
    betagamma = pd.DataFrame([[beta, gamma], [beta_measure, gamma]], columns=["beta", "gamma"],
                             index=[t0, start_date + timedelta(days=27)])


    estimated_parameters = estimate_sir_parameters(data=pd.DataFrame(history, columns=["Ctry"]), country="Ctry",
                                                   threshold=threshold, output=False, forecast=50,
                                                   first_segment=15, last_segment=15)
    errors = []
    if estimated_parameters["t0"] != t0:
        errors.append("t0 = " + str(t0) + " was determined wrongly as " + str(estimated_parameters["t0"]))
    if abs(estimated_parameters["I0"] - history[t0]) > 1:
        errors.append("Estimation for I0 = " + str(history[t0]) + " is " + str(estimated_parameters["I0"])
                      + ". Inacceptable much deviation!")

    if ((estimated_parameters["betagamma"] - betagamma).max().max() > 0.01 or
            (estimated_parameters["betagamma"] - betagamma).isna().any().any()):
        errors.append("Estimation for betagamma deviates too much: \n betagamma: \n" + str(betagamma) +
                      "\n estimation:\n" + str(estimated_parameters["betagamma"]))

    assert not errors, "Parameter estimation incorret! " + "\n".join(errors)
