"""CoolProp refrigerant properties and JAX-traceable (p, h) tables.

CoolProp is not JAX-traceable. At plant construction we flash a dense
(p, h) grid with CoolProp for whatever HEOS fluid the user named, then
the residual only interpolates. Two-phase density uses the Zivi void
fraction so slip stays consistent with the finite-volume DAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import CoolProp.CoolProp as CP
import jax.numpy as jnp
import numpy as np
from jax import Array

# Common HVAC names → CoolProp HEOS identifiers. Any other CoolProp fluid
# name is accepted as-is after a PropsSI probe.
_ALIASES = {
    "r-32": "R32",
    "r32": "R32",
    "hfc-32": "R32",
    "hfc32": "R32",
    "r-410a": "R410A",
    "r410a": "R410A",
    "410a": "R410A",
    "r-134a": "R134a",
    "r134a": "R134a",
    "134a": "R134a",
    "r-290": "R290",
    "r290": "R290",
    "propane": "R290",
    "r-1234yf": "R1234yf",
    "r1234yf": "R1234yf",
    "r-1234ze": "R1234ze(E)",
    "r1234ze": "R1234ze(E)",
    "r1234zee": "R1234ze(E)",
    "r1234ze(e)": "R1234ze(E)",
    "r-22": "R22",
    "r22": "R22",
    "r-407c": "R407C",
    "r407c": "R407C",
    "r-404a": "R404A",
    "r404a": "R404A",
    "r-507a": "R507A",
    "r507a": "R507A",
    "r-717": "Ammonia",
    "r717": "Ammonia",
    "ammonia": "Ammonia",
    "nh3": "Ammonia",
    "r-744": "CO2",
    "r744": "CO2",
    "co2": "CO2",
    "carbondioxide": "CO2",
    "r-600a": "R600a",
    "r600a": "R600a",
    "isobutane": "R600a",
    "r-123": "R123",
    "r123": "R123",
    "r-245fa": "R245fa",
    "r245fa": "R245fa",
}

COMMON_REFRIGERANTS = (
    "R32",
    "R410A",
    "R134a",
    "R290",
    "R1234yf",
    "R1234ze(E)",
    "R22",
    "R407C",
    "R404A",
    "R600a",
    "Ammonia",
)


class PHState(NamedTuple):
    """Mass-specific properties from a (p, h) query."""

    T: Array
    rho: Array
    x: Array
    mu: Array
    k: Array
    cp: Array
    drho_dp: Array
    drho_dh: Array
    Tsat: Array
    hf: Array
    hg: Array
    T_bubble: Array


@dataclass(frozen=True)
class FluidInfo:
    """CoolProp constants for one HEOS fluid (not used inside the JIT residual)."""

    name: str
    Tc: float
    pc: float
    Tmin: float
    Ttriple: float
    M: float
    rhoc: float

    @property
    def R(self) -> float:
        return 8.314462618 / self.M


@dataclass(frozen=True)
class PropertyTables:
    fluid: str
    pc: float
    Tc: float
    p: Array
    h: Array
    T: Array
    rho: Array
    x: Array
    mu: Array
    k: Array
    cp: Array
    Tsat: Array
    T_bubble: Array
    T_dew: Array
    hf: Array
    hg: Array
    rhof: Array
    rhog: Array


def resolve_fluid(name: str) -> str:
    """Map a user refrigerant name to a CoolProp HEOS identifier."""
    raw = name.strip()
    if not raw:
        raise ValueError("refrigerant name is empty")
    key = raw.lower().replace(" ", "").replace("_", "-")
    candidates = []
    if key in _ALIASES:
        candidates.append(_ALIASES[key])
    candidates.extend(
        [
            raw,
            raw.upper(),
            raw.replace("-", ""),
            raw.replace("-", "").upper(),
        ]
    )
    seen: set[str] = set()
    last_err = None
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            CP.PropsSI("Tcrit", cand)
            return cand
        except Exception as exc:  # noqa: BLE001 — CoolProp raises many types
            last_err = exc
    raise ValueError(
        f"Unknown refrigerant {name!r} ({last_err}). "
        f"Try a CoolProp HEOS name such as {', '.join(COMMON_REFRIGERANTS)}."
    )


def fluid_info(name: str) -> FluidInfo:
    fluid = resolve_fluid(name)
    return FluidInfo(
        name=fluid,
        Tc=float(CP.PropsSI("Tcrit", fluid)),
        pc=float(CP.PropsSI("pcrit", fluid)),
        Tmin=float(CP.PropsSI("Tmin", fluid)),
        Ttriple=float(CP.PropsSI("Ttriple", fluid)),
        M=float(CP.PropsSI("M", fluid)),
        rhoc=float(CP.PropsSI("rhomass_critical", fluid)),
    )


def list_refrigerants() -> list[str]:
    """CoolProp fluids that can usually run a subcritical room-heating cycle."""
    known = set(CP.get_global_param_string("FluidsList").split(","))
    out = [f for f in COMMON_REFRIGERANTS if f in known]
    extra = sorted(
        f
        for f in known
        if f.startswith("R") and f not in out and not f.endswith(".mix")
    )
    return out + extra


def make_state(fluid: str) -> CP.AbstractState:
    return CP.AbstractState("HEOS", resolve_fluid(fluid))


def zivi_density(x: float | np.ndarray, rhof: float, rhog: float) -> float | np.ndarray:
    """Slip-corrected two-phase density (Zivi void fraction)."""
    x = np.clip(np.asarray(x, dtype=float), 1e-12, 1.0 - 1e-12)
    sl = ((1.0 - x) / x) * np.power(max(rhog / max(rhof, 1e-8), 1e-8), 2.0 / 3.0)
    alpha = 1.0 / (1.0 + sl)
    return alpha * rhog + (1.0 - alpha) * rhof


def _transport(AS: CP.AbstractState, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        mu = float(AS.viscosity())
        k = float(AS.conductivity())
        cp = float(AS.cpmass())
        if not (np.isfinite(mu) and np.isfinite(k) and np.isfinite(cp)):
            raise ValueError("non-finite transport")
        return mu, k, max(cp, 400.0)
    except Exception:  # noqa: BLE001
        return fallback


def sat_at_p(fluid: str | CP.AbstractState, p: float) -> dict[str, float]:
    """Bubble / dew envelope at pressure p [Pa]."""
    AS = fluid if isinstance(fluid, CP.AbstractState) else make_state(fluid)
    p = float(p)
    AS.update(CP.PQ_INPUTS, p, 0.0)
    T_bubble = float(AS.T())
    hf = float(AS.hmass())
    rhof = float(AS.rhomass())
    mu_f, k_f, cp_f = _transport(AS, (1.6e-4, 0.12, 1400.0))
    AS.update(CP.PQ_INPUTS, p, 1.0)
    T_dew = float(AS.T())
    hg = float(AS.hmass())
    rhog = float(AS.rhomass())
    mu_g, k_g, cp_g = _transport(AS, (1.2e-5, 0.012, 1000.0))
    return {
        "p": p,
        "T_bubble": T_bubble,
        "T_dew": T_dew,
        "Tsat": T_dew,
        "hf": hf,
        "hg": hg,
        "rhof": rhof,
        "rhog": rhog,
        "mu_f": mu_f,
        "k_f": k_f,
        "cp_f": cp_f,
        "mu_g": mu_g,
        "k_g": k_g,
        "cp_g": cp_g,
    }


def sat_at_T(fluid: str | CP.AbstractState, T: float, q: float = 0.0) -> dict[str, float]:
    """Saturation state at temperature T [K] and quality q (0=bubble, 1=dew)."""
    AS = fluid if isinstance(fluid, CP.AbstractState) else make_state(fluid)
    AS.update(CP.QT_INPUTS, float(q), float(T))
    return sat_at_p(AS, float(AS.p()))


def p_sat(fluid: str | CP.AbstractState, T: float, q: float = 0.0) -> float:
    AS = fluid if isinstance(fluid, CP.AbstractState) else make_state(fluid)
    AS.update(CP.QT_INPUTS, float(q), float(T))
    return float(AS.p())


def flash_ph(fluid: str | CP.AbstractState, p: float, h: float, info: FluidInfo | None = None) -> dict[str, float]:
    """NumPy (p, h) flash used to populate tables (CoolProp + Zivi density)."""
    AS = fluid if isinstance(fluid, CP.AbstractState) else make_state(fluid)
    p = float(p)
    h = float(h)
    env = sat_at_p(AS, p)
    hf, hg = env["hf"], env["hg"]
    span = max(hg - hf, 1.0e3)
    if hf < h < hg:
        x = float(np.clip((h - hf) / span, 0.0, 1.0))
        try:
            AS.update(CP.HmassP_INPUTS, h, p)
            T = float(AS.T())
        except Exception:  # noqa: BLE001
            T = (1.0 - x) * env["T_bubble"] + x * env["T_dew"]
        rho = float(zivi_density(x, env["rhof"], env["rhog"]))
        mu = (1.0 - x) * env["mu_f"] + x * env["mu_g"]
        k = (1.0 - x) * env["k_f"] + x * env["k_g"]
        cp = 2.5e4
    else:
        x = 0.0 if h <= hf else 1.0
        try:
            AS.update(CP.HmassP_INPUTS, h, p)
            T = float(AS.T())
            rho = float(AS.rhomass())
            mu, k, cp = _transport(
                AS,
                (env["mu_f"], env["k_f"], env["cp_f"])
                if x < 0.5
                else (env["mu_g"], env["k_g"], env["cp_g"]),
            )
        except Exception:  # noqa: BLE001
            R = info.R if info is not None else 188.0
            if x < 0.5:
                T = env["T_bubble"] + (h - hf) / max(env["cp_f"], 800.0)
                rho = env["rhof"] * (1.0 + 4.5e-7 * (p - env["p"]))
                mu, k, cp = env["mu_f"], env["k_f"], env["cp_f"]
            else:
                T = env["T_dew"] + (h - hg) / max(env["cp_g"], 600.0)
                rho = max(p / max(R * T, 1.0), 0.3)
                mu, k, cp = env["mu_g"], env["k_g"], env["cp_g"]
    return {
        "T": T,
        "rho": float(np.clip(rho, 0.2, 2000.0)),
        "x": float(np.clip(x, 0.0, 1.0)),
        "mu": float(max(mu, 1e-6)),
        "k": float(max(k, 5e-4)),
        "cp": float(max(cp, 400.0)),
        **{k: env[k] for k in ("T_bubble", "T_dew", "Tsat", "hf", "hg", "rhof", "rhog")},
    }


def _table_pressure_span(AS: CP.AbstractState, info: FluidInfo) -> tuple[float, float]:
    T_lo = max(info.Tmin + 10.0, 225.0)
    for _ in range(50):
        try:
            p_lo = float(p_sat(AS, T_lo, 0.0))
        except Exception:  # noqa: BLE001
            T_lo += 3.0
            continue
        if p_lo >= 8.0e4:
            break
        T_lo += 2.0
    else:
        p_lo = 8.0e4
    T_hi = min(info.Tc - 5.0, 348.0)
    if T_hi <= T_lo + 12.0:
        raise ValueError(
            f"{info.name} critical temperature {info.Tc:.1f} K is too low "
            "for a subcritical heat-pump table."
        )
    p_hi = min(0.92 * info.pc, float(p_sat(AS, T_hi, 0.0)))
    p_min = max(8.0e4, 1.02 * p_lo)
    if p_hi <= p_min * 1.35:
        raise ValueError(
            f"{info.name} saturation span is too narrow for a subcritical table "
            f"({p_min/1e5:.2f}–{p_hi/1e5:.2f} bar)."
        )
    return p_min, p_hi


def _bilinear_corners(xg: Array, yg: Array, tab: Array, x: Array, y: Array):
    ix = jnp.clip(jnp.searchsorted(xg, x, side="right") - 1, 0, xg.size - 2)
    iy = jnp.clip(jnp.searchsorted(yg, y, side="right") - 1, 0, yg.size - 2)
    x0, x1 = xg[ix], xg[ix + 1]
    y0, y1 = yg[iy], yg[iy + 1]
    tx = jnp.clip((x - x0) / jnp.maximum(x1 - x0, 1e-30), 0.0, 1.0)
    ty = jnp.clip((y - y0) / jnp.maximum(y1 - y0, 1e-30), 0.0, 1.0)
    f00 = tab[ix, iy]
    f10 = tab[ix + 1, iy]
    f01 = tab[ix, iy + 1]
    f11 = tab[ix + 1, iy + 1]
    val = (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11
    d_dx = ((1 - ty) * (f10 - f00) + ty * (f11 - f01)) / jnp.maximum(x1 - x0, 1e-30)
    d_dy = ((1 - tx) * (f01 - f00) + tx * (f11 - f10)) / jnp.maximum(y1 - y0, 1e-30)
    return val, d_dx, d_dy


def _bilinear(xg: Array, yg: Array, tab: Array, x: Array, y: Array) -> Array:
    val, _, _ = _bilinear_corners(xg, yg, tab, x, y)
    return val


def _interp1(xg: Array, yg: Array, x: Array) -> Array:
    return jnp.interp(x, xg, yg, left=yg[0], right=yg[-1])


def eval_ph(tables: PropertyTables, p: Array, h: Array) -> PHState:
    """JAX (p, h) property evaluation (scalar p, h).

    ``T``, ``x``, ``ρ``, ``μ``, ``k``, and ``c_p`` are interpolated
    independently on the flashed grid. Density slopes come only from the
    ``ρ(p,h)`` surface. Off-dome, the interpolated ``(T,x,ρ)`` do not
    necessarily satisfy a single CoolProp flash.
    """
    p = jnp.clip(p, tables.p[0], tables.p[-1])
    h = jnp.clip(h, tables.h[0], tables.h[-1])
    rho, drho_dp, drho_dh = _bilinear_corners(tables.p, tables.h, tables.rho, p, h)
    return PHState(
        T=_bilinear(tables.p, tables.h, tables.T, p, h),
        rho=rho,
        x=jnp.clip(_bilinear(tables.p, tables.h, tables.x, p, h), 0.0, 1.0),
        mu=_bilinear(tables.p, tables.h, tables.mu, p, h),
        k=_bilinear(tables.p, tables.h, tables.k, p, h),
        cp=_bilinear(tables.p, tables.h, tables.cp, p, h),
        drho_dp=drho_dp,
        drho_dh=drho_dh,
        Tsat=_interp1(tables.p, tables.Tsat, p),
        hf=_interp1(tables.p, tables.hf, p),
        hg=_interp1(tables.p, tables.hg, p),
        T_bubble=_interp1(tables.p, tables.T_bubble, p),
    )


def sat_from_tables(tables: PropertyTables, p: float) -> dict[str, float]:
    """NumPy saturation envelope from the 1-D table (initial state, design)."""
    pg = np.asarray(tables.p)
    p = float(np.clip(p, pg[0], pg[-1]))
    return {
        "Tsat": float(np.interp(p, pg, np.asarray(tables.Tsat))),
        "T_bubble": float(np.interp(p, pg, np.asarray(tables.T_bubble))),
        "T_dew": float(np.interp(p, pg, np.asarray(tables.T_dew))),
        "hf": float(np.interp(p, pg, np.asarray(tables.hf))),
        "hg": float(np.interp(p, pg, np.asarray(tables.hg))),
        "rhof": float(np.interp(p, pg, np.asarray(tables.rhof))),
        "rhog": float(np.interp(p, pg, np.asarray(tables.rhog))),
    }


def p_sat_from_tables(tables: PropertyTables, T: float, kind: str = "dew") -> float:
    """Invert Tsat(p) from the table (monotonic in p)."""
    pg = np.asarray(tables.p)
    Tg = np.asarray(tables.T_dew if kind == "dew" else tables.T_bubble)
    T = float(np.clip(T, Tg.min(), Tg.max()))
    return float(np.interp(T, Tg, pg))


def build_tables(fluid: str = "R32", n_p: int = 48, n_h: int = 72) -> PropertyTables:
    """Build a dense (p, h) interpolant for ``fluid`` (any CoolProp HEOS name)."""
    return _build_tables(resolve_fluid(fluid), int(n_p), int(n_h))


@lru_cache(maxsize=8)
def _build_tables(fluid: str, n_p: int, n_h: int) -> PropertyTables:
    info = fluid_info(fluid)
    AS = make_state(fluid)
    p_min, p_max = _table_pressure_span(AS, info)
    p_grid = np.geomspace(p_min, p_max, n_p)

    env0 = sat_at_p(AS, float(p_grid[0]))
    env1 = sat_at_p(AS, float(p_grid[-1]))
    h_min = float(min(env0["hf"], env1["hf"]) - 8.0e4)
    h_max = float(max(env0["hg"], env1["hg"]) + 2.0e5)
    h_grid = np.linspace(h_min, h_max, n_h)

    T = np.zeros((n_p, n_h))
    rho = np.zeros((n_p, n_h))
    x = np.zeros((n_p, n_h))
    mu = np.zeros((n_p, n_h))
    k = np.zeros((n_p, n_h))
    cp = np.zeros((n_p, n_h))
    Tsat = np.zeros(n_p)
    T_bubble = np.zeros(n_p)
    T_dew = np.zeros(n_p)
    hf = np.zeros(n_p)
    hg = np.zeros(n_p)
    rhof = np.zeros(n_p)
    rhog = np.zeros(n_p)

    for i, p in enumerate(p_grid):
        env = sat_at_p(AS, float(p))
        Tsat[i] = env["Tsat"]
        T_bubble[i] = env["T_bubble"]
        T_dew[i] = env["T_dew"]
        hf[i], hg[i] = env["hf"], env["hg"]
        rhof[i], rhog[i] = env["rhof"], env["rhog"]
        for j, h in enumerate(h_grid):
            pr = flash_ph(AS, float(p), float(h), info)
            T[i, j] = pr["T"]
            rho[i, j] = pr["rho"]
            x[i, j] = pr["x"]
            mu[i, j] = pr["mu"]
            k[i, j] = pr["k"]
            cp[i, j] = pr["cp"]

    return PropertyTables(
        fluid=fluid,
        pc=info.pc,
        Tc=info.Tc,
        p=jnp.asarray(p_grid),
        h=jnp.asarray(h_grid),
        T=jnp.asarray(T),
        rho=jnp.asarray(rho),
        x=jnp.asarray(x),
        mu=jnp.asarray(mu),
        k=jnp.asarray(k),
        cp=jnp.asarray(cp),
        Tsat=jnp.asarray(Tsat),
        T_bubble=jnp.asarray(T_bubble),
        T_dew=jnp.asarray(T_dew),
        hf=jnp.asarray(hf),
        hg=jnp.asarray(hg),
        rhof=jnp.asarray(rhof),
        rhog=jnp.asarray(rhog),
    )
