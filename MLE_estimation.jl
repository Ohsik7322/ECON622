
# Parameters to be estimated
# γ_est # price mean coefficient
# σ_γ_est #  price sd coefficient
# ρ_est # Coef on price*coinsurance rate
# α₁_est # Coef on distance
# α₂_est # Coef on distance^2
# refp_est # Coef on referral
# d_p_FE_est # FE of provider
# σ_est #Price Signal

# θ_b_est # Coef on benefit of the internet usage
# c_c_est # Constant on cost of the internet usage
# c_2_est # Coef on cost of the internet usage for age 19-35
# c_3_est # Coef on cost of the internet usage for age 36-50
# c_4_est # Coef on cost of the internet usage for age 51-64
# c_m_est # Coef on cost of the internet usage for male
# c_i_est # Coef on cost of the internet usage for income
# c_ba_est # Coef on cost of the internet usage for BA
# c_ch_est # Coef on cost of the internet usage for Charlson index

# Choice probability when the price is known
function s_known(coinsurance, ref, dist, price, θ_d,n, n_p, n_l,e)
    γ_est = θ_d[1] # price mean coefficient
    σ_γ_est = θ_d[2] #  price sd coefficient
    ρ_est = θ_d[3] # Coef on price*coinsurance rate
    α₁_est = θ_d[4] # Coef on distance
    α₂_est = θ_d[5] # Coef on distance^2
    refp_est = θ_d[6] # Coef on referral
    d_pFE_est = θ_d[7:6+n_p] # FE of provider
    
    #1 Draw of random coefficient
    # e = rand(Normal(), (n,n_l)) # Error term of random coefficient
    s_known_prob = zeros(n,n_p)

    # Calculate choice probability
    for i in 1:n_l
        γᵢ = γ_est .+ ρ_est .*coinsurance .+ σ_γ_est .* e[:,i] # Random coefficient
        eU = exp.(-γᵢ.* coinsurance * price' .+ α₁_est.*dist .+ α₂_est.*dist.^2 .+refp_est.*ref .+ d_pFE_est') # Utility
        seU = sum(eU, dims=2)
        s_known_prob = s_known_prob .+ eU./seU 
    end
    s_known_prob=s_known_prob./n_l
    return s_known_prob
end


# Choice probability when the price is unknown
function s_unknown(coinsurance, ref, dist, price, θ_d,n, n_p, n_l)
    γ_est = θ_d[1] # price mean coefficient
    σ_γ_est = θ_d[2] #  price sd coefficient
    ρ_est = θ_d[3] # Coef on price*coinsurance rate
    α₁_est = θ_d[4] # Coef on distance
    α₂_est = θ_d[5] # Coef on distance^2
    refp_est = θ_d[6] # Coef on referral
    d_pFE_est = θ_d[7:6+n_p] # FE of provider
    σ_est = θ_d[7+n_p] #Price Signal
    
    #1 Draw of random coefficient
     e = rand(Normal(), (n,n_l)) # Error term of random coefficient
     s = rand(Normal(), (n,n_p,n_l)) .* σ_est # Error term of price signal
    s_unknown_prob = zeros(n,n_p)
    sd_p = std(coinsurance *price', dims=2) # sd of price

    # Calculate choice probability
    for i in 1:n_l
        γᵢ = γ_est .+ ρ_est .*coinsurance .+ σ_γ_est .* e[:,i] # Random coefficient
        w_i = (sd_p).^2 ./((sd_p).^2 .+σ_est^2)  # Information weight
        s_unknown_inside = zeros(n,n_p)
        for j in 1:n_l
            eU = exp.(-γᵢ.* w_i .* (coinsurance * price'.+s[:,:,j]) .+ α₁_est.*dist .+ α₂_est.*dist.^2 .+refp_est.*ref .+ d_pFE_est') # Utility
            seU = sum(eU, dims=2)
            s_unknown_inside = eU./seU .+ s_unknown_inside
        end
        s_unknown_inside = s_unknown_inside./n_l
        s_unknown_prob = s_unknown_inside.+s_unknown_prob
    end
    s_unknown_prob=s_unknown_prob./n_l
    return s_unknown_prob
end

function s_unknown_web(coinsurance, ref, dist, price, θ_d,n, n_p, n_l,e,s)
    γ_est = θ_d[1] # price mean coefficient
    σ_γ_est = θ_d[2] #  price sd coefficient
    ρ_est = θ_d[3] # Coef on price*coinsurance rate
    α₁_est = θ_d[4] # Coef on distance
    α₂_est = θ_d[5] # Coef on distance^2
    refp_est = θ_d[6] # Coef on referral
    d_pFE_est = θ_d[7:6+n_p] # FE of provider
    σ_est = θ_d[7+n_p] #Price Signal
    
    #1 Draw of random coefficient
    # e = rand(Normal(), (n,n_l)) # Error term of random coefficient
    # s = rand(Normal(), (n,n_p,n_l)) .* σ_est # Error term of price signal
    s_unknown_prob = zeros(n,n_p)
    sd_p = std(coinsurance *price', dims=2) # sd of price
    for i in 1:n_l
        γᵢ = γ_est .+ ρ_est .*coinsurance .+ σ_γ_est .* e[:,i] # Random coefficient
        w_i = (sd_p).^2 ./((sd_p).^2 .+σ_est^2)  # Information weight
        s_unknown_inside = zeros(n,n_p)
        for j in 1:n_l
            eU = exp.(-γᵢ.* w_i .* (coinsurance * price'.+s[:,:,j]) .+ α₁_est.*dist .+ α₂_est.*dist.^2 .+refp_est.*ref .+ d_pFE_est') # Utility
            seU = sum(eU, dims=2)
            s_unknown_inside = eU./seU .+ s_unknown_inside
        end
        s_unknown_inside = s_unknown_inside./n_l
        s_unknown_prob = s_unknown_inside.+s_unknown_prob
    end
    s_unknown_prob=s_unknown_prob./n_l
    return s_unknown_prob
end

function s_website(ages, gender, income, degree, Charlson, coinsurance, ref, dist, price, θ_d,n, n_p, n_l)
    γ_est = θ_d[1] # price mean coefficient
    σ_γ_est = θ_d[2] #  price sd coefficient
    ρ_est = θ_d[3] # Coef on price*coinsurance rate
    α₁_est = θ_d[4] # Coef on distance
    α₂_est = θ_d[5] # Coef on distance^2
    refp_est = θ_d[6] # Coef on referral
    d_pFE_est = θ_d[7:6+n_p] # FE of provider
    σ_est = θ_d[7+n_p] #Price Signal
    θ_b_est = θ_d[8+n_p]  # Coef on benefit of the internet usage
    c_c_est = θ_d[9+n_p] # Constant on cost of the internet usage
    c_2_est = θ_d[10+n_p] # Coef on cost of the internet usage for age 19-35
    c_3_est = θ_d[11+n_p] # Coef on cost of the internet usage for age 36-50
    c_4_est = θ_d[12+n_p] # Coef on cost of the internet usage for age 51-64
    c_m_est = θ_d[13+n_p] # Coef on cost of the internet usage for male
    c_i_est = θ_d[14+n_p] # Coef on cost of the internet usage for income
    c_ba_est = θ_d[15+n_p] # Coef on cost of the internet usage for BA
    c_ch_est = θ_d[16+n_p] # Coef on cost of the internet usage for Charlson index

    # Probability to use website
    sd_p = std(coinsurance *price', dims=2) # sd of price
    #1 Draw of random coefficient
    e = rand(Normal(), (n,n_l)) # Error term of random coefficient
    s = rand(Normal(), (n,n_p,n_l)) .* σ_est # Error term of price signal
    b = zeros(n)
    for i in 1:n_l
        γᵢ = γ_est .+ ρ_est .*coinsurance .+ σ_γ_est .* e[:,i] # Random coefficient
        w_i = (sd_p).^2 ./((sd_p).^2 .+σ_est^2)  # Information weight
        var_posterior = w_i.*σ_est^2
        exp_posterior = zeros(n,n_p)
        for j in 1:n_l
            exp_posterior = exp_posterior.+ w_i.*(coinsurance * price' .+ s[:,:,j]).+ (1 .-w_i).*mean(coinsurance * price', dims=2)
        end
        exp_posterior = exp_posterior./n_l
         #1 -γ_i * exp_posterior + δ
         num =exp.(-γᵢ.*exp_posterior .+  α₁_est.*dist .+ α₂_est.*dist.^2 .+refp_est.*ref .+ d_pFE_est')
         #2 sum num
         snum = sum(num, dims=2)
         #3 sum - j
         num_j = snum .-num
         #4 benefit
         b = b.+ (γᵢ .*sum(var_posterior .* num .* num_j, dims=2))./ (2* snum.^2)
    end
    b = b./n_l
    # numerator
    num_website = exp.(θ_b_est.*b .-[ones(n) ages gender income degree Charlson] * [c_c_est; 0.0; c_2_est; c_3_est; c_4_est; c_m_est; c_i_est; c_ba_est; c_ch_est])
    prob_website = num_website./(1 .+num_website)
    #prob_website[isnan.(prob_website)] .= 0.0
    s_unknown_prob = s_unknown_web(coinsurance, ref, dist, price, θ_d,n, n_p, n_l,e,s)
    s_known_prob = s_known(coinsurance, ref, dist, price, θ_d,n, n_p, n_l,e)
    s_website_prob = s_unknown_prob.*prob_website .+ s_known_prob.*(1 .-prob_website)
    s_website_prob_sum = sum(prob_website, dims=1)/n
    return (s_website_prob= s_website_prob, s_website_prob_sum = s_website_prob_sum)
end

# Log likelihood function
function LLF(D_website, dw_ages, dw_gender, dw_income, dw_Charlson, dw_degree, dw_coinsurance, dw_ref, dw_dist, dw_price, W_n, 
    D_unknown, du_coinsurance, du_ref, du_dist, du_price, θ_d,n, n_p, n_l)
    s_unknown_prob = -log.(s_unknown(du_coinsurance, du_ref, du_dist, du_price, θ_d,n, n_p, n_l))
    @unpack s_website_prob, s_website_prob_sum = s_website(dw_ages, dw_gender, dw_income, dw_degree, dw_Charlson, dw_coinsurance, dw_ref, dw_dist, dw_price, θ_d,n, n_p, n_l)
    s_website_prob = -log.(s_website_prob)
    llfs = 0.0
    for i in 1:n
        llfs = llfs+ s_unknown_prob[i, D_unknown[i]] + s_website_prob[i, D_website[i]]
    end
    #W_n_2=ceil(Int,round(W_n/n*100))
    #llfp = log(factorial(big(100))/ (factorial(big(100-W_n_2))*factorial(big(W_n_2))) * (s_website_prob_sum[1]^W_n_2) * ((1-s_website_prob_sum[1])^(100-W_n_2)))
    llfp = -log(factorial(big(n))*(s_website_prob_sum[1]^W_n) * ((1-s_website_prob_sum[1])^(n-W_n))/ (factorial(big(n-W_n))*factorial(big(W_n))))
    SL = llfs + llfp
    return SL
end
