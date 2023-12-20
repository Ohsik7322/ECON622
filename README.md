# ECON622
This repository is part of the replication of Brown (2019).

Sample_data.jl: Generates a sample of patients' characteristics and their choices.
In the first group, patients do not directly observe prices; instead, they make choices based on expectations.
In the second group, patients can accurately infer prices by utilizing the website.
Patients opt to use the website when their expected benefits surpass the associated costs.

MLE_estimation.jl: Estimates parameters using simulated maximum likelihood estimators based on the simulated data.

HMC_estimation.jl: Estimates parameters using Hamiltonian Monte Carlo (HMC). Currently, the code fails to execute successfully, possibly due to issues with the data-generating process (DGP) in the simulation sample or a misconfiguration of the initial values for HMC.