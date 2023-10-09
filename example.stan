functions {
  real normal_copula(real u, real v, real rho) {
    real rho_sq = square(rho);

    return (-0.5*log(1-rho_sq) + (2*rho*inv_Phi(u)*inv_Phi(v) - rho_sq * (inv_Phi(u)^2 + inv_Phi(v)^2)) / (2*(1-rho_sq)));
  }
}
data {
  int<lower=0> N;
  vector[N] Y;
  int X[N];
  vector[N] Z;
}
parameters {
  real mu_Z;
  real<lower=0> sigma_Z;
  real beta;
  real<lower=0> sigma;
  real<lower=0, upper=1> p_X;
  real alpha_phi;
  real beta_phi;
}
model {
  target += normal_lpdf(Z | mu_Z, sigma_Z);
  target += bernoulli_lpmf(X | p_X);
  target += normal_lpdf(Y | beta, sigma);
  for (i in 1:N)
    target += normal_copula(normal_cdf(Z[i], mu_Z, sigma_Z), normal_cdf(Y[i], beta * X[i], sigma), 2 * inv_logit(alpha_phi + beta_phi * X[i]) - 1);
    
  // priors
  target += normal_lpdf(mu_Z | 0, 5);
  target += normal_lpdf(beta | 0, 5);
  target += beta_lpdf(p_X | 1, 1);
  target += exponential_lpdf(sigma_Z | 1);
  target += exponential_lpdf(sigma | 1);
  target += normal_lpdf(alpha_phi | 0, 5);
  target += normal_lpdf(beta_phi | 0, 5);
}

