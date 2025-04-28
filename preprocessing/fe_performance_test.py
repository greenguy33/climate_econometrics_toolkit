import pyfixest as pf
import pandas as pd
import time
import numpy as np
import statsmodels.api as sm
import time
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

nrows = 10000
nfes = [0,10,100]
total_times = []
for index, nfe in enumerate(nfes):
    print("num fixed effects", nfe)
    synth_data = pd.DataFrame()
    synth_data["target"] = np.random.normal(0, 1, nrows)
    synth_data["cov1"] = np.random.normal(0, 1, nrows)

    # run burn in demeaning to not count startup cost
    if index == 1:
        synth_data["fe1"] = np.random.choice(list(range(nfe)), nrows)
        pf.estimation.demean(
            np.array(synth_data[["target","cov1"]]), 
            np.array(synth_data[["fe1"]]), 
            np.ones(len(synth_data))
        )[0]

    start = time.time()
    
    if nfe != 0:
        synth_data["fe1"] = np.random.choice(list(range(nfe)), nrows)

        synth_data = pf.estimation.demean(
            np.array(synth_data[["target","cov1"]]), 
            np.array(synth_data[["fe1"]]), 
            np.ones(len(synth_data))
        )[0]
        synth_data = pd.DataFrame(synth_data, columns=["target","cov1"])

    reg = sm.OLS(synth_data["target"],synth_data[["cov1"]]).fit()

    end = time.time()
    print("total time:", end-start)
    total_times.append(end - start)
print(total_times)