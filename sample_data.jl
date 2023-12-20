using Pkg
Pkg.activate(".")
# Pkg.instantiate()
using DataFrames, Distributions, LinearAlgebra, Statistics, Random, Parameters, Optim



# Set seed for reproducibility
Random.seed!(1234)


# Demand Parameters
const γ = 0.12 # price mean coefficient
const σ_γ = 0.03 #  price sd coefficient
const ρ = -0.08 # Coef on price*coinsurance rate
const α₁ = -0.0351 # Coef on distance
const α₂ = 0.0027 # Coef on distance^2
const refp = 2.581 # Coef on referral
const σ = 150 #Price Signal

# Website Parameters
const θ_b = 0.05 # Coef on benefit of the internet usage
const c_c = 1.392 # Constant on cost of the internet usage
const c_2 = -1.421 # Coef on cost of the internet usage for age 19-35
const c_3 = -1.558 # Coef on cost of the internet usage for age 36-50
const c_4 = -1.739 # Coef on cost of the internet usage for age 51-64
const c_m = 0.085 # Coef on cost of the internet usage for male
const c_i = 0.023 # Coef on cost of the internet usage for income
const c_ba = -0.036 # Coef on cost of the internet usage for BA
const c_ch = 9.0 # Coef on cost of the internet usage for Charlson index

# DGP for observable & Provider FEs
function gen_data_obs(n, n_p)
    # 1 Age group: P((0-18), (19-35), (36-50), (51-64)) = [0.2, 0.18, 0.31, 0.31]
    age_probabilities = [0.2, 0.18, 0.31, 0.31]
    ages = rand(Categorical(age_probabilities), n)
    d_ages = permutedims([1, 2, 3, 4]) .== ages
    
    # 2 Gender: B(n, 0.47)
    d_gender = rand(Binomial(1, 0.47), n)
    
    # 3 Income: Truncated Normal (mean = 82.8, sd = 24.2), min = 22, max = 309
    d_income = rand(TruncatedNormal(82.8, 24.2, 22, 309), n)
   
    # 4 Charlson: Truncated Normal (Mean = 0.6, sd = 0.8), min = 0, max = 2
    d_Charlson = rand(TruncatedNormal(0.6, 0.8, 0, 2), n)
    
    # 5 BA: Truncated Normal (Mean = 33.8, sd = 14), min = 0, max = 100
    d_degree = rand(TruncatedNormal(33.8, 14, 0, 100), n)
    
    # 6 Coinsurance rate: U(0, 1)
    d_coinsurance = rand(Uniform(0, 1), n)
    
    # 7 Referral B(n, 0.33)
    # Conditional on referral, equal probability of each provider
    ref = rand(Binomial(1, 0.33), n)
    ref_con = rand(1:n_p, n)
    d_ref = permutedims(range(1, n_p, length = n_p)) .== ref_con
    d_ref = ref.*d_ref
    
    # 8 Distance to provider: N(29, 15)
    d_dist = zeros(n, n_p)
    for i in 1:n_p
        d_dist[:, i] .= rand(TruncatedNormal(29, 15, 0, 75), n)
    end
    
    # 9 Total negotiated price: Truncated N(700, 500) ?
    d_price = rand(TruncatedNormal(700, 500, 0, Inf), n_p)
    
    # 10 Provider FE: N(0, 1)
    d_pFE = rand(Normal(0, 1), n_p)

    return (d_ages=d_ages, d_gender=d_gender, d_income=d_income, d_Charlson=d_Charlson, d_degree=d_degree, d_coinsurance=d_coinsurance, d_ref=d_ref, d_dist=d_dist, d_price=d_price, d_pFE=d_pFE)
end


# DGP when price is known
function dgp_known(n, n_p)
    @unpack d_ages, d_gender, d_income, d_Charlson, d_degree, d_coinsurance, d_ref, d_dist, d_price, d_pFE = gen_data_obs(n, n_p)
    
    ϵ₀ = rand(GeneralizedExtremeValue(0, 1, 0), (n, n_p)) # Taste shock
    ϵ₁ = rand(Normal(), n) # Error term of random coefficient
    γᵢ = γ .+ ρ .*d_coinsurance .+ σ_γ .* ϵ₁ # Random coefficient
    U = -γᵢ.* d_coinsurance * d_price' .+ α₁.*d_dist .+ α₂.*d_dist.^2 .+refp.*d_ref .+ d_pFE' .+ϵ₀ # Utility
    D_known = mapslices(argmax, U, dims=2) # Choice 

    return (D_known=D_known, d_ages = d_ages, d_gender = d_gender, d_income = d_income, d_Charlson = d_Charlson, d_degree = d_degree, d_coinsurance = d_coinsurance, d_ref = d_ref, d_dist = d_dist, d_price = d_price, d_pFE = d_pFE)

end

# DGP when price is unknown
function dgp_unknown(n, n_p)
    @unpack d_ages, d_gender, d_income, d_Charlson, d_degree, d_coinsurance, d_ref, d_dist, d_price, d_pFE = gen_data_obs(n, n_p)
    ϵ₀ = rand(GeneralizedExtremeValue(0, 1, 0), (n, n_p)) # Taste shock
    ϵ₁ = rand(Normal(), n,n_p) .* σ # Signal 
    ϵ₂ = rand(Normal(), n) # Error term of random coefficient

    sd_p = std(d_coinsurance *d_price', dims=2) # sd of price
    w_i = (sd_p).^2 ./((sd_p).^2 .+σ^2)  # Information weight
    γᵢ = γ .+ ρ .*d_coinsurance .+ σ_γ .* ϵ₂ # Random coefficient
    U = -γᵢ.* w_i .* (d_coinsurance * d_price'.+ϵ₁) .+ α₁.*d_dist .+ α₂.*d_dist.^2 .+refp.*d_ref .+ d_pFE' .+ϵ₀ # Utility
    D_unknown = mapslices(argmax, U, dims=2) # Choice
    return (D_unknown = D_unknown, du_ages = d_ages, du_gender = d_gender, du_income = d_income, du_Charlson = d_Charlson, du_degree = d_degree, du_coinsurance = d_coinsurance, du_ref = d_ref, du_dist = d_dist, du_price = d_price, d_pFE = d_pFE, w_i=w_i, ϵ₀=ϵ₀, ϵ₁=ϵ₁, ϵ₂ = ϵ₂)
end

    # DGP when price website is available
function dgp_website(n, n_p)
    @unpack D_unknown, du_ages, du_gender, du_income, du_Charlson, du_degree, du_coinsurance, du_ref, du_dist, du_price, d_pFE, w_i, ϵ₀, ϵ₁, ϵ₂ = dgp_unknown(n, n_p)
    ϵ₃ = rand(GeneralizedExtremeValue(0, 1, 0), n) # Taste shock of the internet
    γᵢ = γ .+ ρ .*du_coinsurance .+ σ_γ .* ϵ₂ # Random coefficient
    
    #0 Variance and expectation of posterior beliefs
    var_posterior = w_i.*σ^2
    exp_posterior = w_i.*(du_coinsurance * du_price' .+ ϵ₁).+ (1 .-w_i).*mean(du_coinsurance * du_price', dims=2)
    
    #1 -γ_i * exp_posterior + δ
    num =exp.(-γᵢ.*exp_posterior .+ α₁.*du_dist .+ α₂.*du_dist.^2 .+refp.*du_ref .+ d_pFE')
    #2 sum num
    snum = sum(num, dims=2)
    #3 sum - j
    num_j = snum .-num
    #4 benefit
    b = (γᵢ .*sum(var_posterior .* num .* num_j, dims=2))./ (2* snum.^2)

    # susage
    W = θ_b .*b .-[ones(n) du_ages du_gender du_income du_degree du_Charlson] * [c_c; 0.0; c_2; c_3; c_4; c_m; c_i; c_ba; c_ch] .+ϵ₃
    W_u = [v >0.0 ? 1 : 0 for v in W] # Internet user
    W_n = length(W_u[W_u .== 1]) # # of internet user
   
    # Choice when price is known for the itnernet user
    U = -γᵢ.* du_coinsurance * du_price' .+ α₁.*du_dist .+ α₂.*du_dist.^2 .+refp.*du_ref .+ d_pFE' .+ϵ₀ # Utility
    D_known = mapslices(argmax, U, dims=2) # Choice
    D_website = D_known.*W_u .+ D_unknown.*(1 .-W_u)

    return (D_website = D_website, dw_ages = du_ages, dw_gender = du_gender, dw_income = du_income, dw_Charlson = du_Charlson, dw_degree = du_degree, dw_coinsurance = du_coinsurance, dw_ref = du_ref, dw_dist = du_dist, dw_price = du_price, d_pFE = d_pFE, W_n = W_n)
end


