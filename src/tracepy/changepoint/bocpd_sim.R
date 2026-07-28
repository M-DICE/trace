## bocpd_sim.R
## Simulation wrapper for BOCPD
## Run all replications for one increment
## entirely in R to avoid Python<->R bridge overhead per time step.
##
## ---- IMPORTANT ----
## BOCPD_functions.R MUST be sourced first.

.sim_env <- new.env(parent = emptyenv())

## Number of workers available for mclapply on this machine.
##
## mclapply relies on forking, which is unavailable on Windows: R there accepts
## mc.cores = 1 only (see ?mclapply) and errors otherwise.  detectCores() may
## also return NA, which ?detectCores warns is unsuitable for mc.cores, so the
## unknown case falls back to serial too.  Replications seed themselves, so a
## serial run reproduces the parallel result exactly.
.detect_sim_cores <- function() {
  if (identical(.Platform$OS.type, "windows")) {
    return(1L)
  }
  n <- parallel::detectCores(logical = FALSE)
  if (is.na(n)) 1L else max(1L, as.integer(n))
}

.sim_env$n_cores <- .detect_sim_cores()

cat(sprintf(
  "  Running R %s on %s (OS.type=%s) - %d core(s) available%s\n",
  getRversion(), R.version$platform, .Platform$OS.type, .sim_env$n_cores,
  if (.sim_env$n_cores == 1L) " (serial execution)" else " (parallel execution)"
))


run_bocpd_r <- function(x, prior_alpha, prior_sigma2, sig_prior, pm, ptr, maxp, np, msl) {
  N <- length(x)
  nmodel <- 1L

  a_prior       <- c(prior_alpha[1], prior_alpha[2],
                     0.5 / prior_sigma2, 0.5 / prior_sigma2)
  params_ini_list <- list(list(alpha = prior_alpha, sigma = prior_sigma2))
  prior_list      <- list(list(a_prior = a_prior, sig_prior = sig_prior))
  intfun_list     <- list(integrate.fun5)
  global_params   <- list(pm, ptr)

  cdf_msl <- if (msl > 1) {
    1 - sum((1 - ptr)^(seq_len(msl - 1) - 1L) * ptr)
  } else {
    1
  }
  runl_Gx_v <- function(ts) (1 - (1 - ptr)^(ts - msl)) * (1 - ptr)^(msl - 1L) / cdf_msl

  Gxt0   <- runl_Gx_v(msl:N)
  Gxt1   <- runl_Gx_v((msl - 1L):(N - 1L))
  stay   <- c(rep(1, msl - 1L), (1 - Gxt0) / (1 - Gxt1))
  change <- c(rep(0, msl - 1L), 1 - (1 - Gxt0) / (1 - Gxt1))

  hmm <- list(
    x          = matrix(x[seq_len(2L * msl - 1L)], ncol = 1L),
    Nx         = 2L * msl - 1L,
    pModels    = pm,
    transition = cbind(stay[seq_len(2L * msl - 1L)],
                       change[seq_len(2L * msl - 1L)]),
    intPar     = vector("list", N),
    intExtra   = vector("list", N),
    forward    = vector("list", N),
    ParsID     = vector("list", N),
    tmpCost    = vector("list", N)
  )

  hmm$forward[[msl - 1L]] <- 1

  intPar_m <- integrate.par(hmm, msl - 1L, 0L, nmodel,
                             params_ini_list, prior_list, intfun_list)
  hmm$intPar[[msl - 1L]] <- list(unlist(intPar_m))

  gx_init <- ((1 - ptr)^(seq(msl, 2L * msl - 1L) - 1L)) * ptr / cdf_msl
  for (i in seq(msl, 2L * msl - 1L)) {
    intPar_m <- integrate.par(hmm, i, 0L, nmodel,
                               params_ini_list, prior_list, intfun_list)
    intPar <- unlist(intPar_m)
    hmm$intPar[[i]] <- list(matrix(intPar, nrow = nmodel, ncol = 1L))
    hmm$forward[[i]] <- 1
    hmm$ParsID[[i]]  <- i
    hmm$tmpCost[[i]] <- intPar[1L] + log(pm) + log(gx_init[i - msl + 1L])
  }

  cpt_breaks_len <- 1L
  i <- 2L * msl
  cpt_map <- NULL

  # Stop at first detected changepoint
  while (cpt_breaks_len == 1L && i <= N) {
    hmm$x          <- matrix(x[seq_len(i)], ncol = 1L)
    hmm$Nx         <- i
    hmm$transition <- cbind(stay[seq_len(i)], change[seq_len(i)])

    Integrate         <- forward.cpt.int(hmm, i, nmodel,
                                          params_ini_list, prior_list, intfun_list, msl)
    hmm               <- Integrate$hmm
    hmm$intPar[[i]]   <- Integrate$intPar.list
    hmm$intExtra[[i]] <- Integrate$intExtra.list

    hmm <- forward.cpt.calc.resample(hmm, i, nmodel, maxp, np, TRUE, 5L, ptr, msl)

    cpt_map        <- map.cpt(hmm, nmodel, global_params, runl.fun2, msl)
    cpt_breaks_len <- length(cpt_map$breaks)
    i <- i + 1L
  }

  if (is.null(cpt_map) || cpt_breaks_len == 1L) {
    return(list(cpt_est = NA_integer_, time_est = NA_integer_))
  }

  list(cpt_est = as.integer(cpt_map$breaks[2L]),
       time_est = as.integer(i - 1L))
}


run_bocpd_increment <- function(
  simN, n_trends, trend_idx,
  npre, npost_max, level, trend_control, trend_inc, sigma,
  delay_max,
  prior_alpha, prior_sigma2, sig_prior_shape, sig_prior_rate,
  pm, ptr, maxp, np, msl
) {
  sig_prior    <- c(sig_prior_shape, sig_prior_rate)
  trend_interv <- trend_control + trend_inc

  cpt_est_vec  <- rep(NA_integer_, simN)
  time_est_vec <- rep(NA_integer_, simN)
  delays       <- integer(simN)
  errors       <- numeric(0)
  det_times    <- numeric(0)

  for (s in seq_len(simN)) {
    seed_val <- s * n_trends + trend_idx
    set.seed(seed_val)

    delay       <- sample.int(delay_max, 1L)
    npre_delay  <- npre + delay
    npost_delay <- npost_max - delay
    nt          <- npre_delay + npost_delay

    # Simulate control and intervention series
    t_vec <- seq_len(nt)
    y_ctr <- level + trend_control * t_vec + rnorm(nt, 0, sigma)

    y_itv                      <- numeric(nt)
    pre_idx                    <- seq_len(npre_delay)
    y_itv[pre_idx]             <- level + trend_control * pre_idx + rnorm(npre_delay, 0, sigma)
    if (npost_delay > 0L) {
      post_idx                   <- seq_len(npost_delay)
      y_itv[(npre_delay + 1L):nt] <- level + trend_control * npre_delay +
                                      trend_interv * post_idx + rnorm(npost_delay, 0, sigma)
    }

    x   <- y_itv - y_ctr
    res <- run_bocpd_r(x, prior_alpha, prior_sigma2, sig_prior, pm, ptr, maxp, np, msl)

    cpt_est_vec[s]  <- res$cpt_est
    time_est_vec[s] <- res$time_est
    delays[s]       <- delay

    if (!is.na(res$cpt_est)) {
      errors    <- c(errors,    res$cpt_est - npre_delay)
      det_times <- c(det_times, res$time_est)
    }
  }

  n_det <- length(errors)
  list(
    cpt_est             = cpt_est_vec,
    time_est            = time_est_vec,
    delay               = delays,
    errors              = errors,
    detection_rate      = n_det / simN,
    mean_error          = if (n_det > 0L) mean(errors) else NaN,
    mean_time_to_detect = if (length(det_times) > 0L) mean(det_times) else NaN
  )
}


.ws1d <- function(a, b) mean(abs(sort(a) - sort(b)))

run_bocpd_distribution_increment <- function(
  simN, n_trends, trend_idx,
  npre, npost_max, mu, sigma_dist, ns,
  trend_mu, delay_max,
  prior_alpha, prior_sigma2, sig_prior_shape, sig_prior_rate,
  pm, ptr, maxp, np, msl
) {
  sig_prior <- c(sig_prior_shape, sig_prior_rate)

  run_one <- function(s) tryCatch(.run_one_dist(s), error = function(e) NULL)

  .run_one_dist <- function(s) {
    seed_val <- s * n_trends + trend_idx
    set.seed(seed_val)

    delay       <- sample.int(delay_max, 1L)
    npre_delay  <- npre + delay
    npost_delay <- npost_max - delay
    nt          <- npre_delay + npost_delay

    # Control: constant N(mu, sigma_dist) at every time point
    sample_ctr <- matrix(rnorm(ns * nt, mu, sigma_dist), nrow = ns, ncol = nt)

    # Intervention: one rnorm call + add drift to post-period columns (replaces for loop)
    sample_itv <- matrix(rnorm(ns * nt, mu, sigma_dist), nrow = ns, ncol = nt)
    if (npost_delay > 0L) {
      drift_mu <- trend_mu * seq_len(npost_delay)
      sample_itv[, (npre_delay + 1L):nt] <-
        sample_itv[, (npre_delay + 1L):nt] + rep(drift_mu, each = ns)
    }

    # Wasserstein: sort columns once, then vectorised colMeans (replaces vapply loop)
    sorted_ctr <- apply(sample_ctr, 2, sort)
    sorted_itv <- apply(sample_itv, 2, sort)
    x <- colMeans(abs(sorted_ctr - sorted_itv))

    res <- run_bocpd_r(x, prior_alpha, prior_sigma2, sig_prior, pm, ptr, maxp, np, msl)
    list(cpt_est = res$cpt_est, time_est = res$time_est, delay = delay)
  }

  n_cores <- max(1L, min(as.integer(simN), .sim_env$n_cores))
  raw <- parallel::mclapply(seq_len(simN), run_one, mc.cores = n_cores)

  cpt_est_vec  <- rep(NA_integer_, simN)
  time_est_vec <- rep(NA_integer_, simN)
  delays       <- integer(simN)
  errors       <- numeric(0)
  det_times    <- numeric(0)

  for (s in seq_len(simN)) {
    r <- raw[[s]]
    if (inherits(r, "try-error") || is.null(r)) next

    delay      <- r$delay
    npre_delay <- npre + delay
    cpt_val    <- r$cpt_est
    ttd_val    <- r$time_est

    cpt_est_vec[s]  <- if (!is.null(cpt_val) && !is.na(cpt_val)) cpt_val else NA_integer_
    time_est_vec[s] <- if (!is.null(ttd_val) && !is.na(ttd_val)) ttd_val else NA_integer_
    delays[s]       <- delay

    if (!is.na(cpt_est_vec[s])) {
      errors    <- c(errors,    cpt_est_vec[s] - npre_delay)
      det_times <- c(det_times, time_est_vec[s])
    }
  }

  n_det <- length(errors)
  list(
    cpt_est             = cpt_est_vec,
    time_est            = time_est_vec,
    delay               = delays,
    errors              = errors,
    detection_rate      = n_det / simN,
    mean_error          = if (n_det > 0L) mean(errors) else NaN,
    mean_time_to_detect = if (length(det_times) > 0L) mean(det_times) else NaN
  )
}
