"""
BOCPD (Bayesian Online Changepoint Detection)
This is the Python port of SimRewilding/Rewild_trend_change_BOCPD.R.

Requires R (>= 4.x) and rpy2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import rpy2.robjects as ro
    from rpy2.rinterface_lib.sexp import NAIntegerType as _NAIntegerType
    from rpy2.robjects.vectors import FloatVector

    _HAS_RPY2 = True
    _RPY2_ERR: str | None = None
except ImportError as _e:
    _HAS_RPY2 = False
    _RPY2_ERR = str(_e)
    FloatVector = None  # type: ignore[misc,assignment]
    _NAIntegerType = None  # type: ignore[misc,assignment]

_R_SOURCED = False
_SIM_R_SOURCED = False


def _require_rpy2() -> None:
    if not _HAS_RPY2:
        raise ImportError(
            "rpy2 is required but could not be imported. "
            "Ensure R (>= 4.x) is installed on your system and rpy2 is available.\n"
            f"  (original error: {_RPY2_ERR})"
        )


def _find_r_source() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        candidate = p / "SimRewilding" / "BOCPD_functions.R"
        if candidate.exists():
            return candidate
        p = p.parent
    raise FileNotFoundError("Cannot locate SimRewilding/BOCPD_functions.R")


def _find_sim_r_source() -> Path:
    return Path(__file__).resolve().parent / "bocpd_sim.R"


def _ensure_sourced() -> None:
    global _R_SOURCED
    if not _R_SOURCED:
        _require_rpy2()
        r_path = str(_find_r_source()).replace("\\", "/")
        ro.r["source"](r_path)
        _R_SOURCED = True


def _ensure_sim_sourced() -> None:
    global _SIM_R_SOURCED
    if not _SIM_R_SOURCED:
        _ensure_sourced()
        sim_path = str(_find_sim_r_source()).replace("\\", "/")
        ro.r["source"](sim_path)
        _SIM_R_SOURCED = True


# ============================================================================
# rpy2 helpers
# ============================================================================


def _R(name: str):
    return ro.r[name]


def _r_list(*args, **kwargs):
    return _R("list")(*args, **kwargs)


def _r_matrix(data, ncol=None, nrow=None):
    kwargs = {}
    if ncol is not None:
        kwargs["ncol"] = ncol
    if nrow is not None:
        kwargs["nrow"] = nrow
    return _R("matrix")(data, **kwargs)


def _r_cbind(a, b):
    return _R("cbind")(a, b)


def _r_unlist(x):
    return _R("unlist")(x)


def _set_field(rlist, name: str, value):
    """Functional ``rlist$name <- value``."""
    return ro.baseenv["$<-"](rlist, name, value)


def _set_at(rlist, idx: int, value):
    """Functional ``rlist[[idx]] <- value`` — extends the list as needed."""
    return ro.baseenv["[[<-"](rlist, idx, value)


def _set_sub(rlist, field: str, idx: int, value):
    """Functional ``rlist$field[[idx]] <- value``."""
    sub = rlist.rx2(field)
    sub = _set_at(sub, idx, value)
    return _set_field(rlist, field, sub)


def _runl_Gx(t: int, ptr: float, msl: int) -> float:
    return float(_R("runl.fun2")(t, ptr, msl).rx2("Gx")[0])


def _runl_gx(t: int, ptr: float, msl: int) -> float:
    return float(_R("runl.fun2")(t, ptr, msl).rx2("gx")[0])


# ============================================================================
# BOCPD Functions used for simulations
# ============================================================================


def run_bocpd(
    x: np.ndarray,
    *,
    prior_alpha: tuple[float, float] = (0.0, 0.05),
    prior_sigma2: float = 0.0025,
    sig_prior: tuple[float, float] = (100.0, 1.0),
    pm: float = 1.0,
    ptr: float = 0.02,
    maxp: int = 50,
    np_: int = 40,
    msl: int = 10,
    nmodel: int = 1,
) -> dict[str, int | None]:
    """
    Run online BOCPD on a univariate series, stopping at the first detected
    changepoint.
    This is ported from Rewild_trend_change_BOCPD.R.

    Parameters
    ----------
    x : array-like
        Observed series (typically y_itv − y_ctr difference time series).
    prior_alpha : (float, float)
        Prior means for (intercept, slope). Corresponds to ``alpha.ini`` in R.
    prior_sigma2 : float
        Prior variance scale. Corresponds to ``sigma.ini`` in R.
    sig_prior : (float, float)
        ``(shape u0, rate v0)`` of the inverse-gamma prior on σ².
    pm : float
        Prior model probability.
    ptr : float
        Geometric run-length hazard rate.
    maxp : int
        Maximum candidate changepoints before stratified resampling.
    np_ : int
        Particles to retain after resampling.
    msl : int
        Minimum segment length.
    nmodel : int
        Number of segment models (1 for the linear trend model).

    Returns
    -------
    dict
        ``{'cpt_est': int | None, 'time_est': int | None}``

        * ``cpt_est``  — MAP changepoint location (1-indexed, as in R), or
          ``None`` if no changepoint was detected within the series.
        * ``time_est`` — Step index when the changepoint was first declared,
          or ``None`` if no changepoint was detected.
    """
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter):
        return _run_bocpd_impl(
            x,
            prior_alpha=prior_alpha,
            prior_sigma2=prior_sigma2,
            sig_prior=sig_prior,
            pm=pm,
            ptr=ptr,
            maxp=maxp,
            np_=np_,
            msl=msl,
            nmodel=nmodel,
        )


def _run_bocpd_impl(
    x: np.ndarray,
    *,
    prior_alpha: tuple[float, float],
    prior_sigma2: float,
    sig_prior: tuple[float, float],
    pm: float,
    ptr: float,
    maxp: int,
    np_: int,
    msl: int,
    nmodel: int,
) -> dict[str, int | None]:
    _ensure_sourced()

    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 2 * msl:
        raise ValueError(f"Series length {N} is shorter than 2*msl = {2 * msl}")

    integrate_par = _R("integrate.par")
    forward_cpt_int = _R("forward.cpt.int")
    forward_cpt_calc_resample = _R("forward.cpt.calc.resample")
    map_cpt = _R("map.cpt")
    runl_fun = _R("runl.fun2")
    integrate_fun5 = _R("integrate.fun5")

    # Param / prior R structures
    a_prior = FloatVector([prior_alpha[0], prior_alpha[1], 0.5 / prior_sigma2, 0.5 / prior_sigma2])
    sig_prior_vec = FloatVector(list(sig_prior))
    params_ini_list = _r_list(
        _r_list(alpha=FloatVector(list(prior_alpha)), sigma=float(prior_sigma2))
    )
    prior_list = _r_list(_r_list(a_prior=a_prior, sig_prior=sig_prior_vec))
    intfun_list = _r_list(integrate_fun5)
    global_params = _r_list(float(pm), float(ptr))

    # stay / change run-length probabilities
    Gxt0 = np.array([_runl_Gx(t, ptr, msl) for t in range(msl, N + 1)])
    Gxt1 = np.array([_runl_Gx(t, ptr, msl) for t in range(msl - 1, N)])
    stay = np.concatenate([np.ones(msl - 1), (1.0 - Gxt0) / (1.0 - Gxt1)])
    change = np.concatenate([np.zeros(msl - 1), 1.0 - (1.0 - Gxt0) / (1.0 - Gxt1)])

    # Initial hmm
    hmm = _r_list(
        x=_r_matrix(FloatVector(x[: 2 * msl - 1].tolist()), ncol=1),
        Nx=int(2 * msl - 1),
        pModels=float(pm),
        transition=_r_cbind(
            FloatVector(stay[: 2 * msl - 1].tolist()),
            FloatVector(change[: 2 * msl - 1].tolist()),
        ),
        intPar=_r_list(),
        intExtra=_r_list(),
        forward=_r_list(),
        ParsID=_r_list(),
        tmpCost=_r_list(),
    )

    # hmm$forward[[msl-1]] <- 1
    hmm = _set_sub(hmm, "forward", msl - 1, 1.0)

    # Init at cpt = msl-1
    intPar_m = integrate_par(hmm, msl - 1, 0, 1, params_ini_list, prior_list, intfun_list)
    hmm = _set_sub(hmm, "intPar", msl - 1, _r_list(_r_unlist(intPar_m)))

    # Initial iterations msl..2*msl-1
    for i in range(msl, 2 * msl):
        intPar_m = integrate_par(hmm, i, 0, 1, params_ini_list, prior_list, intfun_list)
        intPar = _r_unlist(intPar_m)
        hmm = _set_sub(hmm, "intPar", i, _r_list(_r_matrix(intPar, nrow=nmodel, ncol=1)))
        hmm = _set_sub(hmm, "forward", i, 1.0)
        hmm = _set_sub(hmm, "ParsID", i, int(i))

        intPar_arr = np.asarray(intPar)
        tmpCost_val = float(intPar_arr[0] + np.log(pm) + np.log(_runl_gx(i, ptr, msl)))
        hmm = _set_sub(hmm, "tmpCost", i, tmpCost_val)

    # Main loop: stop at first detected changepoint
    cpt_breaks_len = 1
    i = 2 * msl
    cpt_map = None
    while cpt_breaks_len == 1 and i <= N:
        hmm = _set_field(hmm, "x", _r_matrix(FloatVector(x[:i].tolist()), ncol=1))
        hmm = _set_field(hmm, "Nx", int(i))
        hmm = _set_field(
            hmm,
            "transition",
            _r_cbind(
                FloatVector(stay[:i].tolist()),
                FloatVector(change[:i].tolist()),
            ),
        )

        Integrate = forward_cpt_int(hmm, i, nmodel, params_ini_list, prior_list, intfun_list, msl)
        hmm = Integrate.rx2("hmm")
        hmm = _set_sub(hmm, "intPar", i, Integrate.rx2("intPar.list"))
        hmm = _set_sub(hmm, "intExtra", i, Integrate.rx2("intExtra.list"))

        hmm = forward_cpt_calc_resample(hmm, i, nmodel, maxp, np_, True, 5, ptr, msl)

        cpt_map = map_cpt(hmm, 1, global_params, runl_fun, msl)
        cpt_breaks_len = len(cpt_map.rx2("breaks"))
        i += 1

    if cpt_map is None or cpt_breaks_len == 1:
        return {"cpt_est": None, "time_est": None}

    breaks = list(cpt_map.rx2("breaks"))
    return {"cpt_est": int(breaks[1]), "time_est": int(i) - 1}


def _r_result_to_dict(result) -> dict:
    """Convert the standard rpy2 result list from an R bocpd increment call to a Python dict."""
    import math

    def _int_or_none(v) -> int | None:
        if _NAIntegerType is not None and isinstance(v, _NAIntegerType):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    me = float(result.rx2("mean_error")[0])
    mttd = float(result.rx2("mean_time_to_detect")[0])
    return {
        "cpt_est": [_int_or_none(v) for v in result.rx2("cpt_est")],
        "time_est": [_int_or_none(v) for v in result.rx2("time_est")],
        "delay": [int(v) for v in result.rx2("delay")],
        "errors": [float(v) for v in result.rx2("errors")],
        "detection_rate": float(result.rx2("detection_rate")[0]),
        "mean_error": me if not math.isnan(me) else float("nan"),
        "mean_time_to_detect": mttd if not math.isnan(mttd) else float("nan"),
    }


def run_bocpd_increment(
    *,
    trend_inc: float,
    trend_idx: int,
    n_trends: int,
    simN: int,
    npre: int,
    npost_max: int,
    level: float,
    trend_control: float,
    sigma: float,
    delay_max: int,
    prior_alpha: tuple[float, float],
    prior_sigma2: float,
    sig_prior: tuple[float, float],
    pm: float,
    ptr: float,
    maxp: int,
    np_: int,
    msl: int,
) -> dict:
    """Run all *simN* BOCPD replications for one trend increment inside R."""
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter):
        _ensure_sim_sourced()
        result = _R("run_bocpd_increment")(
            simN=int(simN),
            n_trends=int(n_trends),
            trend_idx=int(trend_idx),
            npre=int(npre),
            npost_max=int(npost_max),
            level=float(level),
            trend_control=float(trend_control),
            trend_inc=float(trend_inc),
            sigma=float(sigma),
            delay_max=int(delay_max),
            prior_alpha=FloatVector(list(prior_alpha)),
            prior_sigma2=float(prior_sigma2),
            sig_prior_shape=float(sig_prior[0]),
            sig_prior_rate=float(sig_prior[1]),
            pm=float(pm),
            ptr=float(ptr),
            maxp=int(maxp),
            np=int(np_),
            msl=int(msl),
        )
        return _r_result_to_dict(result)


def run_bocpd_distribution_increment(
    *,
    trend_inc: float,
    trend_idx: int,
    n_trends: int,
    simN: int,
    npre: int,
    npost_max: int,
    mu: float,
    sigma: float,
    ns: int,
    delay_max: int,
    prior_alpha: tuple[float, float],
    prior_sigma2: float,
    sig_prior: tuple[float, float],
    pm: float,
    ptr: float,
    maxp: int,
    np_: int,
    msl: int,
) -> dict:
    """Run all *simN* distribution-BOCPD replications for one mean-shift increment inside R."""
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter):
        _ensure_sim_sourced()
        result = _R("run_bocpd_distribution_increment")(
            simN=int(simN),
            n_trends=int(n_trends),
            trend_idx=int(trend_idx),
            npre=int(npre),
            npost_max=int(npost_max),
            mu=float(mu),
            sigma_dist=float(sigma),
            ns=int(ns),
            trend_mu=float(trend_inc),
            delay_max=int(delay_max),
            prior_alpha=FloatVector(list(prior_alpha)),
            prior_sigma2=float(prior_sigma2),
            sig_prior_shape=float(sig_prior[0]),
            sig_prior_rate=float(sig_prior[1]),
            pm=float(pm),
            ptr=float(ptr),
            maxp=int(maxp),
            np=int(np_),
            msl=int(msl),
        )
        return _r_result_to_dict(result)
