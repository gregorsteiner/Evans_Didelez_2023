functions {
  real normal_copula(real u, real v, real rho) {
    real rho_sq = square(rho);

    return (-0.5*log(1-rho_sq) + (2*rho*inv_Phi(u)*inv_Phi(v) - rho_sq * (inv_Phi(u)^2 + inv_Phi(v)^2)) / (2*(1-rho_sq)));
  }
}
data {
  int<lower=0> N;
  real Y[N];
  int X[N];
  real Z[N];
}
parameters {
  real beta;
  real<lower=0> sigma;
  real mu_Z;
  real<lower=0> sigma_Z;
  real<lower=0, upper=1> p_X;
  real alpha_phi;
  real beta_phi;
}
model {
  // priors
  mu_Z ~ normal(0, 5);
  beta ~ normal(0, 5);
  p_X ~ beta(1, 1);
  sigma_Z ~ exponential(0.1);
  sigma ~ exponential(0.1);
  alpha_phi ~ normal(0, 2);
  beta_phi ~ normal(0, 5);
  
  // model
  Z ~ normal(mu_Z, sigma_Z);
  X ~ bernoulli(p_X);
  for (i in 1:N)
    Y[i] ~ normal(beta * X[i], sigma);
  for (i in 1:N)
    target += normal_copula(normal_cdf(Z[i], mu_Z, sigma_Z), normal_cdf(Y[i], beta * X[i], sigma), 2 * inv_logit(alpha_phi + beta_phi * X[i]) - 1);
    
  
}

