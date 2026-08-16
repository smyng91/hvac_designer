#!/usr/bin/env python3
"""Publication figures from design JSON, closed-loop simulations, and validation files.

No hardcoded hardware annotations or invented error bars. Missing inputs are
computed or loaded from committed validation CSVs; they are not invented.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from _paths import HOUR_S, WEATHER_COOL, WEATHER_REVERSE, out_dir, sim_horizon  # noqa: E402
from heatpump import DesignRequest, TimeSeries, design_heat_pump, design_system, simulate  # noqa: E402
from heatpump.simulate import plot_result  # noqa: E402
from heatpump.validation import run_validation  # noqa: E402

OUT_FIGS = ROOT / "paper" / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
PAPER = ROOT / "paper"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "lines.linewidth": 1.4,
        "grid.alpha": 0.25,
    }
)


def _ensure_design():
    dest = ROOT / "output"
    dest.mkdir(parents=True, exist_ok=True)
    print("sizing R32 5.5 kW (includes plant displacement match)…", flush=True)
    req = DesignRequest(
        refrigerant="R32",
        mode="heating",
        T_zone=293.15,
        Q_heat=5500.0,
        T_out_heat=273.15,
    )
    sys = design_system(req)
    pkg = sys.as_report()
    pkg.write(dest / "design.md")
    pkg.write_latex_macros(PAPER / "generated_design.tex")
    return pkg.to_json()


def generate_fig1(data: dict) -> None:
    pts = data["heating_map"]["points"]
    t_out = [p["T_out"] - 273.15 for p in pts]
    q_cap = [p["Q_cap"] / 1000.0 for p in pts]
    q_load = [p["Q_load"] / 1000.0 for p in pts]
    cop = [p["COP"] for p in pts]
    hw = data.get("hardware") or {}
    t_bal = data["heating_map"].get("T_balance")
    t_bal_c = None if t_bal is None else t_bal - 273.15

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.15))
    ax1.plot(t_out, q_cap, "o-", color="#1d4ed8", markersize=4, label=r"$Q_\mathrm{cap}$")
    ax1.plot(t_out, q_load, "s--", color="#b91c1c", markersize=4, label=r"$Q_\mathrm{load}$")
    if t_bal_c is not None:
        ax1.axvline(
            t_bal_c,
            color="k",
            linestyle=":",
            alpha=0.7,
            label=rf"balance ${t_bal_c:.1f}\,^\circ\mathrm{{C}}$",
        )
    ax1.set_xlabel(r"Outdoor temperature $T_\mathrm{out}$ ($^\circ$C)")
    ax1.set_ylabel("Thermal power (kW)")
    ax1.set_title("(a) Capacity and envelope load")
    ax1.grid(True, linestyle="--")
    ax1.legend(frameon=False, loc="best")

    ax2.plot(t_out, cop, "^-", color="#15803d", markersize=4)
    ax2.set_xlabel(r"Outdoor temperature $T_\mathrm{out}$ ($^\circ$C)")
    ax2.set_ylabel("COP (—)")
    ax2.set_title("(b) Heating COP")
    ax2.grid(True, linestyle="--")

    if hw:
        info = (
            rf"{data['fluid']}, 5.5 kW design"
            + "\n"
            + rf"$V_\mathrm{{disp}}={hw['V_disp_m3']*1e6:.1f}\,\mathrm{{cm}}^3$"
            + "\n"
            + rf"$A_\mathrm{{eev}}={hw['A_eev_m2']*1e6:.2f}\,\mathrm{{mm}}^2$"
            + "\n"
            + rf"indoor {hw['n_tubes_indoor']} / outdoor {hw['n_tubes_outdoor']} tubes"
        )
        ax1.text(
            0.03,
            0.04,
            info,
            transform=ax1.transAxes,
            fontsize=7.5,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d4d4d4"),
        )

    fig.tight_layout()
    _savefig(fig, "fig1_sizing_map")
    _write_fig1_csv(data)


def _save_npz(path: Path, res) -> None:
    np.savez_compressed(
        path,
        t=res.t,
        N=res.u[:, 0],
        u_eev=res.u[:, 1],
        T_out=res.u[:, 4] if res.u.shape[1] > 4 else np.full_like(res.t, np.nan),
        T_z=res.meas["T_z"],
        p_e=res.meas["p_e"],
        p_c=res.meas["p_c"],
        SH=res.meas["SH"],
        COP=res.meas["COP"],
        charge=res.meas["charge"],
        mode=res.meas["mode"],
        Q_zone=res.meas["Q_zone"],
    )


def _run_transients():
    dest = out_dir()
    T_out, Tsp = 273.15, 293.15
    design = design_heat_pump("R32", 5500.0, T_out=T_out, T_zone=Tsp)
    heat = simulate("pid", spec=design.spec, design=design, T_out=T_out, Tsp=Tsp, **sim_horizon(HOUR_S))
    plot_result(heat, dest / "heating.png", f"Heating · {design.fluid} · pid")
    _save_npz(dest / "heating.npz", heat)

    ts_r = TimeSeries.from_csv(WEATHER_REVERSE)
    req_r = DesignRequest(refrigerant="R32", mode="heat_pump", T_zone=293.15, T_zone_cool=297.15, timeseries=ts_r)
    sys_r = design_system(req_r)
    rev = simulate(
        sys_r.controller,
        spec=sys_r.spec,
        design=sys_r,
        request=req_r,
        timeseries=ts_r,
        **sim_horizon(min(HOUR_S, ts_r.duration)),
    )
    plot_result(rev, dest / "reverse.png", "Reverse · cool then heat")
    _save_npz(dest / "reverse.npz", rev)

    ts_w = TimeSeries.from_csv(WEATHER_COOL)
    req_w = DesignRequest(refrigerant="R410A", mode="auto", T_zone=297.15, timeseries=ts_w)
    sys_w = design_system(req_w)
    weather = simulate(
        sys_w.controller,
        spec=sys_w.spec,
        design=sys_w,
        request=req_w,
        timeseries=ts_w,
        **sim_horizon(min(HOUR_S, ts_w.duration)),
    )
    plot_result(weather, dest / "weather_cooling.png", "cooling · weather")
    _save_npz(dest / "weather_cooling.npz", weather)
    return heat, rev, weather


def _write_sim_macros(heat, rev, weather) -> None:
    ch = np.asarray(rev.meas["charge"], dtype=float)
    ch_span = 100.0 * (float(np.max(ch)) - float(np.min(ch))) / max(float(np.mean(ch)), 1e-12)
    mode = np.asarray(rev.meas["mode"], dtype=float)
    hit = np.where(mode >= 0.5)[0]
    t_sw = float(rev.t[hit[0]] / 60.0) if hit.size else float("nan")
    lines = [
        f"\\newcommand{{\\HeatFinalTzC}}{{{heat.meas['T_z'][-1] - 273.15:.2f}}}",
        f"\\newcommand{{\\HeatFinalSH}}{{{heat.meas['SH'][-1]:.2f}}}",
        f"\\newcommand{{\\HeatFinalCOP}}{{{heat.meas['COP'][-1]:.2f}}}",
        f"\\newcommand{{\\HeatFinalNHz}}{{{heat.u[-1, 0]:.2f}}}",
        f"\\newcommand{{\\HeatAbsErrTzK}}{{{abs(heat.meas['T_z'][-1] - 293.15):.2f}}}",
        f"\\newcommand{{\\HeatHorizonMin}}{{{heat.t[-1] / 60.0:.0f}}}",
        f"\\newcommand{{\\ReverseSwitchMin}}{{{t_sw:.1f}}}",
        f"\\newcommand{{\\ReverseChargeSpanPct}}{{{ch_span:.3f}}}",
        f"\\newcommand{{\\ReverseHorizonMin}}{{{rev.t[-1] / 60.0:.0f}}}",
        f"\\newcommand{{\\WeatherFinalTzC}}{{{weather.meas['T_z'][-1] - 273.15:.2f}}}",
        f"\\newcommand{{\\WeatherHorizonMin}}{{{weather.t[-1] / 60.0:.0f}}}",
        f"\\newcommand{{\\WeatherFinalCOP}}{{{weather.meas['COP'][-1]:.2f}}}",
    ]
    (PAPER / "generated_sim.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_fig2(heat, rev, weather) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.2), sharex=False)

    t = heat.t / 60.0
    axes[0, 0].plot(t, heat.meas["T_z"] - 273.15, color="#1d4ed8")
    axes[0, 0].axhline(20.0, color="k", linestyle=":", linewidth=0.8)
    axes[0, 0].set_ylabel(r"$T_z$ ($^\circ$C)")
    axes[0, 0].set_title("(a) Heating, PID")
    axes[0, 1].plot(t, heat.meas["p_e"] / 1e5, color="#1d4ed8", label="evap")
    axes[0, 1].plot(t, heat.meas["p_c"] / 1e5, color="#b91c1c", label="cond")
    axes[0, 1].set_ylabel("Pressure (bar)")
    axes[0, 1].legend(frameon=False)
    axes[0, 2].plot(t, heat.meas["SH"], color="#b45309")
    axes[0, 2].axhline(6.0, color="k", linestyle=":", linewidth=0.8)
    axes[0, 2].set_ylabel("Superheat (K)")

    t = rev.t / 60.0
    axes[1, 0].plot(t, rev.meas["T_z"] - 273.15, color="#1d4ed8")
    axes[1, 0].set_ylabel(r"$T_z$ ($^\circ$C)")
    axes[1, 0].set_title("(b) Reverse (cool then heat)")
    axes[1, 1].plot(t, rev.meas["p_e"] / 1e5, color="#1d4ed8")
    axes[1, 1].plot(t, rev.meas["p_c"] / 1e5, color="#b91c1c")
    axes[1, 1].set_ylabel("Pressure (bar)")
    axes[1, 2].plot(t, rev.meas["charge"], color="#334155")
    axes[1, 2].set_ylabel("Charge (kg)")

    t = weather.t / 60.0
    axes[2, 0].plot(t, weather.meas["T_z"] - 273.15, color="#1d4ed8")
    axes[2, 0].plot(t, weather.u[:, 4] - 273.15, color="#64748b", linestyle="--", label=r"$T_\mathrm{out}$")
    axes[2, 0].set_ylabel(r"Temperature ($^\circ$C)")
    axes[2, 0].set_xlabel("Time (min)")
    axes[2, 0].set_title("(c) Weather-driven cooling")
    axes[2, 0].legend(frameon=False)
    axes[2, 1].plot(t, weather.u[:, 0], color="#334155")
    axes[2, 1].set_ylabel("Compressor (Hz)")
    axes[2, 1].set_xlabel("Time (min)")
    axes[2, 2].plot(t, weather.u[:, 1], color="#334155")
    axes[2, 2].set_ylabel("EEV opening")
    axes[2, 2].set_xlabel("Time (min)")

    for ax in axes.ravel():
        ax.grid(True, linestyle="--")

    fig.tight_layout()
    _savefig(fig, "fig2_transients")
    _write_fig2_csv(heat, rev, weather)


def generate_fig3(rep: dict) -> None:
    ram = rep["ramirez"]
    runs = ram["runs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.25))

    meas_q = np.array([r["meas_Q_W"] / 1000.0 for r in runs])
    pred_q = np.array([r["pred_Q_W"] / 1000.0 for r in runs])
    ax1.scatter(meas_q, pred_q, color="#1d4ed8", edgecolor="k", s=36, zorder=3, label=rf"$n={len(runs)}$ runs")
    lo = float(min(meas_q.min(), pred_q.min())) * 0.95
    hi = float(max(meas_q.max(), pred_q.max())) * 1.05
    lims = [lo, hi]
    ax1.plot(lims, lims, "k-", alpha=0.7, label="1:1")
    ax1.plot(lims, [l * 1.10 for l in lims], "k--", alpha=0.45, label=r"$\pm 10\%$")
    ax1.plot(lims, [l * 0.90 for l in lims], "k--", alpha=0.45)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel(r"Measured $Q$ (kW)")
    ax1.set_ylabel(r"Predicted $Q$ (kW)")
    ax1.set_title("(a) Ramírez et al. 2019, nameplate sizer")
    ax1.grid(True, linestyle="--")
    ax1.legend(frameon=False, loc="upper left")
    sq = ram["summary"]["Q"]
    sw = ram["summary"]["W"]
    ax1.text(
        0.50,
        0.08,
        f"MAPE $Q$ {sq['mape_pct']:.2f}%\nMAPE $W$ {sw['mape_pct']:.2f}%\nmax $|\\Delta Q|$ {sq['max_abs_pct']:.1f}%",
        transform=ax1.transAxes,
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d4d4d4"),
    )

    nrel = rep["nrel"]["points"]
    if nrel:
        labels = [f"{r['kind']}\n{float(r['T_out_C']):.0f}°C" for r in nrel]
        m_q = [float(r["Q_W"]) / 1000.0 for r in nrel]
        p_q = [float(r["pred_Q_W"]) / 1000.0 for r in nrel]
        err = [float(r["err_Q_pct"]) for r in nrel]
        x = np.arange(len(labels))
        w = 0.36
        ax2.bar(x - w / 2, m_q, w, label="HIL on-period", color="#15803d", edgecolor="k")
        ax2.bar(x + w / 2, p_q, w, label="nameplate sizer", color="#c2410c", edgecolor="k")
        top = max(max(m_q), max(p_q))
        for i in range(len(labels)):
            ax2.text(x[i], max(m_q[i], p_q[i]) + 0.04 * top, f"{err[i]:+.1f}%", ha="center", fontsize=7.5)
        ax2.set_ylabel(r"Capacity $Q$ (kW)")
        ax2.set_title("(b) NREL HIL on-period means (R410A assumed)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylim(0, top * 1.22)
        ax2.grid(True, linestyle="--", axis="y")
        ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    _savefig(fig, "fig3_validation")
    _write_fig3_csv(rep)


def _savefig(fig, stem: str) -> None:
    png = OUT_FIGS / f"{stem}.png"
    pdf = OUT_FIGS / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print("wrote", png)


def _write_fig1_csv(data: dict) -> None:
    pts = data["heating_map"]["points"]
    dest = OUT_FIGS / "fig1_sizing_map.csv"
    dest.write_text(
        "T_out_C,Q_cap_kW,Q_load_kW,COP\n"
        + "".join(
            f"{p['T_out']-273.15:.4f},{p['Q_cap']/1000:.6f},{p['Q_load']/1000:.6f},{p['COP']:.6f}\n"
            for p in pts
        ),
        encoding="utf-8",
    )


def _write_fig2_csv(heat, rev, weather) -> None:
    dest = OUT_FIGS / "fig2_transients.csv"
    lines = ["series,t_min,T_z_C,p_e_bar,p_c_bar,SH_K,charge_kg,N_Hz,u_eev,T_out_C\n"]

    def _rows(name, res):
        t = np.asarray(res.t) / 60.0
        tz = np.asarray(res.meas["T_z"]) - 273.15
        pe = np.asarray(res.meas["p_e"]) / 1e5
        pc = np.asarray(res.meas["p_c"]) / 1e5
        sh = np.asarray(res.meas["SH"])
        ch = np.asarray(res.meas["charge"])
        n = np.asarray(res.u[:, 0])
        eev = np.asarray(res.u[:, 1])
        tout = (
            np.asarray(res.u[:, 4]) - 273.15
            if res.u.shape[1] > 4
            else np.full_like(t, np.nan)
        )
        for i in range(t.size):
            lines.append(
                f"{name},{t[i]:.4f},{tz[i]:.4f},{pe[i]:.4f},{pc[i]:.4f},"
                f"{sh[i]:.4f},{ch[i]:.6f},{n[i]:.4f},{eev[i]:.4f},{tout[i]:.4f}\n"
            )

    _rows("heating_pid", heat)
    _rows("reverse", rev)
    _rows("weather_cooling", weather)
    dest.write_text("".join(lines), encoding="utf-8")


def _write_fig3_csv(rep: dict) -> None:
    dest = OUT_FIGS / "fig3_validation.csv"
    lines = ["set,label,meas_Q_kW,pred_Q_kW,err_Q_pct\n"]
    for r in rep["ramirez"]["runs"]:
        lines.append(
            f"ramirez,run{r['run']},{r['meas_Q_W']/1000:.6f},{r['pred_Q_W']/1000:.6f},{r['err_Q_pct']:.4f}\n"
        )
    for r in rep["nrel"].get("points", []):
        lines.append(
            f"nrel,{r['kind']}_{r['T_out_C']:.1f}C,{r['Q_W']/1000:.6f},{r['pred_Q_W']/1000:.6f},{r['err_Q_pct']:.4f}\n"
        )
    dest.write_text("".join(lines), encoding="utf-8")


def _run_mpc_demo() -> bool:
    """90 s full-DAE PID vs LMPC. Writes SI macros only if traces stay finite."""
    print("running 90 s full-DAE PID vs LMPC…", flush=True)
    T_out, Tsp = 273.15, 293.15
    design = design_heat_pump("R32", 5500.0, T_out=T_out, T_zone=Tsp)
    kw = dict(
        spec=design.spec,
        design=design,
        T_out=T_out,
        Tsp=Tsp,
        t_final=90.0,
        record_dt=5.0,
        reduction="full",
    )
    try:
        pid = simulate("pid", **kw)
        mpc = simulate("mpc", **kw)
    except Exception as exc:
        print("MPC demo failed:", exc)
        return False
    ok = (
        np.all(np.isfinite(pid.meas["T_z"]))
        and np.all(np.isfinite(mpc.meas["T_z"]))
        and float(mpc.meas["p_c"][-1]) > float(mpc.meas["p_e"][-1])
    )
    if not ok:
        print("MPC demo produced non-finite or unphysical traces; not claimed in the paper")
        return False
    dest = out_dir()
    _save_npz(dest / "mpc90.npz", mpc)
    plot_result(mpc, dest / "mpc90.png", "LMPC · 90 s full DAE")
    plot_result(mpc, OUT_FIGS / "fig_si_mpc90.png", "LMPC · 90 s full DAE")
    lines = [
        r"\subsection{Short full-DAE PID versus linear MPC}",
        r"A 90\,s full-DAE heating run (not QSS) with the same R32 plant.",
        r"MPC uses implicit Euler on $\bm{f}$, not TR-BDF2.",
        rf"PID terminal $T_z={pid.meas['T_z'][-1]-273.15:.2f}\,^\circ\mathrm{{C}}$; "
        rf"LMPC terminal $T_z={mpc.meas['T_z'][-1]-273.15:.2f}\,^\circ\mathrm{{C}}$.",
        rf"PID $|T_z-T_\mathrm{{sp}}|={abs(pid.meas['T_z'][-1]-Tsp):.2f}\,\mathrm{{K}}$; "
        rf"LMPC $|T_z-T_\mathrm{{sp}}|={abs(mpc.meas['T_z'][-1]-Tsp):.2f}\,\mathrm{{K}}$.",
        r"This is a numerical demonstration, not laboratory control validation.",
        "",
    ]
    (PAPER / "generated_mpc.tex").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", PAPER / "generated_mpc.tex")
    return True


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    design = _ensure_design()
    generate_fig1(design)
    print("running validation…")
    val = run_validation()
    generate_fig3(val)
    print("running closed-loop transients (1 h QSS)…")
    heat, rev, weather = _run_transients()
    _write_sim_macros(heat, rev, weather)
    generate_fig2(heat, rev, weather)
    mpc_ok = _run_mpc_demo()
    figures = [
        "paper/figures/fig1_sizing_map.png",
        "paper/figures/fig1_sizing_map.pdf",
        "paper/figures/fig2_transients.png",
        "paper/figures/fig2_transients.pdf",
        "paper/figures/fig3_validation.png",
        "paper/figures/fig3_validation.pdf",
    ]
    if mpc_ok:
        figures.append("paper/figures/fig_si_mpc90.png")
    (OUT_FIGS / "manifest.json").write_text(
        json.dumps(
            {
                "design_json": "output/design.json",
                "validation_json": "validation/results/validation_report.json",
                "figures": figures,
                "source_csv": [
                    "paper/figures/fig1_sizing_map.csv",
                    "paper/figures/fig2_transients.csv",
                    "paper/figures/fig3_validation.csv",
                ],
                "source_json": [
                    "output/design.json",
                    "validation/results/validation_report.json",
                ],
                "source_npz": [
                    "output/heating.npz",
                    "output/reverse.npz",
                    "output/weather_cooling.npz",
                ],
                "mpc_demo": mpc_ok,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("done")
