# HMC
struct HMCproblem
    D_website::Matrix{Int64}
    dw_ages::BitMatrix
    dw_gender::Vector{Int64}
    dw_income::Vector{Float64}
    dw_Charlson::Vector{Float64}
    dw_degree::Vector{Float64}
    dw_coinsurance::Vector{Float64}
    dw_ref::Matrix{Int64}
    dw_dist::Matrix{Float64}
    dw_price::Vector{Float64}
    W_n::Int64
    D_unknown::Matrix{Int64}
    du_ages::BitMatrix
    du_gender::Vector{Int64}
    du_income::Vector{Float64}
    du_Charlson::Vector{Float64}
    du_degree::Vector{Float64}
    du_coinsurance::Vector{Float64}
    du_ref::Matrix{Int64}
    du_dist::Matrix{Float64}
    du_price::Vector{Float64}
end

function(problem::HMCproblem)(θ_d)
    @unpack γ_est, σ_γ_est, ρ_est, α₁_est, α₂_est, refp_est, d_pFE_est, σ_est, θ_b_est, c_c_est, c_2_est, c_3_est, c_4_est, c_m_est, c_i_est, c_ba_est, c_ch_est = θ_d
    @unpack D_website, dw_ages, dw_gender, dw_income, dw_Charlson, dw_degree, dw_coinsurance, dw_ref, dw_dist, dw_price,
    W_n, D_unknown, du_ages, du_gender, du_income, du_Charlson, du_degree, du_coinsurance, du_ref, du_dist, du_price = problem

    sd_p = std(du_coinsurance *du_price', dims=2) # sd of price when price is unknown
    sd_pw = std(dw_coinsurance *dw_price', dims=2) # sd of price when using website
    e = rand(Normal(), n) # Error term of random coefficient when price is unknown
    e_w = rand(Normal(), n) # Error term of random coefficient when website is available
    
    # Log likelihood function
    
    # Draw Parameters
    γᵢ = γ_est .+ ρ_est .*du_coinsurance .+ σ_γ_est .* e # Random coefficient when price is unknown
    γᵢ_w = γ_est .+ ρ_est .*dw_coinsurance .+ σ_γ_est .* e_w # Random coefficient when website is available
    s = rand(Normal(), (n,n_p)) .* σ_est # Error term of price signal when price is unknown
    s_w = rand(Normal(), (n,n_p)) .* σ_est # Error term of price signal when website is available

    # Probability to choose providers when price is known (Website)
    eU_known = exp.(-γᵢ_w.* dw_coinsurance * dw_price' .+ α₁_est.*dw_dist .+ α₂_est.*dw_dist.^2 .+refp_est.*dw_ref .+ d_pFE_est') # Utility
    seU_known = sum(eU_known, dims=2)
    s_known_prob = eU_known./seU_known

    # Probability to choose providers when price is unknown (website)
    w_i_w = (sd_pw).^2 ./((sd_pw).^2 .+σ_est^2) # Information weight
    eU_unknown_w = exp.(-γᵢ.* w_i_w .* (dw_coinsurance * dw_price'.+s_w) .+ α₁_est.*dw_dist .+ α₂_est.*dw_dist.^2 .+refp_est.*dw_ref .+ d_pFE_est') # Utility
    seU_unknown_w = sum(eU_unknown_w, dims=2)
    s_unknown_w_prob = eU_unknown_w./seU_unknown_w

    # Probability to choose providers when price is unknown 
    w_i = (sd_p).^2 ./((sd_p).^2 .+σ_est^2) # Information weight
    eU_unknown = exp.(-γᵢ.* w_i .* (du_coinsurance * du_price'.+s) .+ α₁_est.*du_dist .+ α₂_est.*du_dist.^2 .+refp_est.*du_ref .+ d_pFE_est') # Utility
    seU_unknown = sum(eU_unknown, dims=2)
    s_unknown_prob = eU_unknown./seU_unknown
    
    # Probability to use website
    var_posterior = w_i_w.*σ_est^2
    exp_posterior = w_i_w.*(dw_coinsurance * dw_price' .+ s_w).+ (1 .-w_i_w).*mean(dw_coinsurance * dw_price', dims=2)
    #1 -γ_i * exp_posterior + δ
    num =exp.(-γᵢ_w.*exp_posterior .+  α₁_est.*dw_dist .+ α₂_est.*dw_dist.^2 .+refp_est.*dw_ref .+ d_pFE_est')
    #2 sum num
    snum = sum(num, dims=2)
    #3 sum - j
    num_j = snum .-num
    #4 benefit
    b = (γᵢ_w .*sum(var_posterior .* num .* num_j, dims=2))./ (2* snum.^2)

    # Probability to use website
    num_website = exp.(θ_b_est.*b .-[ones(n) dw_ages dw_gender dw_income dw_degree dw_Charlson] * [c_c_est; 0.0; c_2_est; c_3_est; c_4_est; c_m_est; c_i_est; c_ba_est; c_ch_est])
    prob_website = num_website./(1 .+num_website)
    prob_website[isnan.(prob_website)] .= 0.0
    s_website_prob_sum = mean(prob_website, dims=1)[1]

    # Probability to choose providers when website is available
    s_website_prob = s_unknown_w_prob.*prob_website .+ s_known_prob.*(1 .-prob_website)
    # Likelihood function
    s_unknown_prob = log.(s_unknown_prob)
    s_website_prob = log.(s_website_prob)
    llfs = 0.0
    for i in 1:n
        llfs = llfs+ s_unknown_prob[i, D_unknown[i]] + s_website_prob[i, D_website[i]]
    end
   # V = rand(Binomial(n, s_website_prob_sum))
   # llfp = log(factorial(big(n))*(s_website_prob_sum[1]^V) * ((1-s_website_prob_sum[1])^(n-V))/ (factorial(big(n-V))*factorial(big(V))))
    SL = llfs 
    # Loglikelihood of priors
    llfp = loglikelihood(Binomial(n, s_website_prob_sum), W_n)
    # Prior for γ
    l_γ_est = loglikelihood(Normal(0.0, 10), γ_est)
    # Prior for σ_γ
    l_σ_γ_est = loglikelihood(Normal(0.0, 1), σ_γ_est)
    # Prior for ρ
    l_ρ_est = loglikelihood(Normal(0.0, 1), ρ_est)
    # Prior for α₁
    l_α₁_est = loglikelihood(Normal(0.0, 1), α₁_est)
    # Prior for α₂
    l_α₂_est = loglikelihood(Normal(0.0, 1), α₂_est)
    # Prior for refp
    l_refp_est = loglikelihood(Normal(0.0, 1), refp_est)
    # Prior for d_pFE
    l_d_pFE_est = loglikelihood(Normal(0.0, 1), d_pFE_est)
    # Prior for σ
    l_σ_est = loglikelihood(Normal(0.0, 200), σ_est)
    # Prior for θ_b
    l_θ_b_est = loglikelihood(Normal(0.0, 1), θ_b_est)
    # Prior for c_c
    l_c_c_est = loglikelihood(Normal(0.0, 1), c_c_est)
    # Prior for c_2
    l_c_2_est = loglikelihood(Normal(0.0, 1), c_2_est)
    # Prior for c_3
    l_c_3_est = loglikelihood(Normal(0.0, 1), c_3_est)
    # Prior for c_4
    l_c_4_est = loglikelihood(Normal(0.0, 1), c_4_est)
    # Prior for c_m
    l_c_m_est = loglikelihood(Normal(0.0, 1), c_m_est)
    # Prior for c_i
    l_c_i_est = loglikelihood(Normal(0.0, 1), c_i_est)
    # Prior for c_ba
    l_c_ba_est = loglikelihood(Normal(0.0, 1), c_ba_est)
    # Prior for c_ch
    l_c_ch_est = loglikelihood(Normal(0.0, 1), c_ch_est)
    SL = SL + llfp+ l_γ_est + l_σ_γ_est + l_ρ_est + l_α₁_est + l_α₂_est + l_refp_est + l_d_pFE_est + l_σ_est + l_θ_b_est + l_c_c_est + l_c_2_est + l_c_3_est + l_c_4_est + l_c_m_est + l_c_i_est + l_c_ba_est + l_c_ch_est

    return SL
end


p= HMCproblem(D_website, dw_ages, dw_gender, dw_income, dw_Charlson, dw_degree, dw_coinsurance, dw_ref, dw_dist, dw_price, W_n,
D_unknown, du_ages, du_gender, du_income, du_Charlson, du_degree, du_coinsurance, du_ref, du_dist, du_price)

# Transform problem
function problem_transformation(p::HMCproblem)
    as((γ_est = as_positive_real, σ_γ_est = as_positive_real, ρ_est =as_negative_real, α₁_est = as_negative_real, α₂_est = as_positive_real, refp_est = as_positive_real, 
    d_pFE_est = as(Vector, n_p), σ_est = as_positive_real, θ_b_est = as_positive_real, c_c_est = as_real, c_2_est = as_real, c_3_est = as_real, c_4_est = as_real, 
    c_m_est = as_real, c_i_est = as_positive_real, c_ba_est = as_negative_real, c_ch_est = as_real))
end

t = problem_transformation(p)
P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, P)
# results = mcmc_with_warmup(Random.default_rng(), ∇P, 1000)
