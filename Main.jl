include("sample_data.jl")
using Revise, TransformVariables, LogDensityProblems, DynamicHMC,  DynamicHMC.Diagnostics, Gadfly
using TransformedLogDensities,  LogDensityProblemsAD, SimpleUnPack
# # of observation
n = 500
# Set # of provider
n_p = 2
# Set # of loop
n_l =100
# Data after the internet is available
@unpack D_website, dw_ages, dw_gender, dw_income, dw_Charlson, dw_degree, dw_coinsurance, dw_ref, dw_dist, dw_price, d_pFE, W_n = dgp_website(n, n_p)
# Data before the internet is available
@unpack D_unknown, du_ages, du_gender, du_income, du_Charlson, du_degree, du_coinsurance, du_ref, du_dist, du_price, d_pFE, w_i, ϵ₀, ϵ₁, ϵ₂ = dgp_unknown(n, n_p)


include("MLE_estimation.jl")
include("HMC_estimation.jl")

# MLE estimation
# Initial values
θ_d_0 = [0.001; 0.001; -0.001; -0.037; 0.0003;2.52;rand(Normal(), n_p); 100.0; 0.01; 
1.00; -1.00;-1.00; -1.00;  0.01;0.01; -0.01; 1.00]
result_mle = optimize(θ_d->Q(θ_d), θ_d_0, Newton(); autodiff= :forward)

# HMC estimation
# results = mcmc_with_warmup(Random.default_rng(), ∇P, 1000) # Doesn't work well for now 