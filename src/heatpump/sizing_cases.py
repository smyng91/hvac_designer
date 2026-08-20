"""Automated sizing case studies at published rating and example conditions.

Hardware comes from ``design_system`` with ``match_plant=False`` (the same
algebraic inversion used for the public-data scores). Capacity ratios are
``capacity_at`` of that geometry at the case temperatures, divided by the
stated duty. No catalog is fitted and no AHRI/ACCA certification is claimed.

Published dry-bulb conditions (wet-bulb is not used by the residual)::

    ISO 5151:2017 T1 cooling: 27 °C indoor / 35 °C outdoor
    AHRI 210/240 cooling A: 80 °F indoor / 95 °F outdoor
    AHRI 210/240 heating H1: 70 °F indoor / 47 °F outdoor
    AHRI 210/240 heating H3: 70 °F indoor / 17 °F outdoor

Manual S ceilings and ASHRAE 90.1 Appendix G oversize factors are cited
constants, not outputs of this sizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from heatpump.capacity import capacity_at
from heatpump.design import design_system
from heatpump.gates import DesignGateError
from heatpump.plant import apply_operating_mode
from heatpump.requirements import DesignRequest, TimeSeries, cooling_tons_to_w

# Exact Fahrenheit → kelvin for AHRI 210/240 dry-bulb points.
def _f_to_k(deg_f: float) -> float:
    return (float(deg_f) - 32.0) * 5.0 / 9.0 + 273.15


ISO_T1_TZ_K = 27.0 + 273.15
ISO_T1_TOUT_K = 35.0 + 273.15
AHRI_A_TZ_K = _f_to_k(80.0)
AHRI_A_TOUT_K = _f_to_k(95.0)
AHRI_H1_TZ_K = _f_to_k(70.0)
AHRI_H1_TOUT_K = _f_to_k(47.0)
AHRI_H3_TZ_K = _f_to_k(70.0)
AHRI_H3_TOUT_K = _f_to_k(17.0)

# ANSI/ACCA 3 Manual S-2014 equipment-selection checklist ceilings
# (ACCA Manual S brochure): total cooling ≤ 115% of design cooling load;
# total heating ≤ 140% of design heating load.
MANUAL_S_COOL_MAX = 1.15
MANUAL_S_HEAT_MAX = 1.40

# ANSI/ASHRAE/IES 90.1-2016 §G3.1.2.2 / IC 90.1-2016-14: baseline *modeling*
# oversize, not a field selection requirement.
APP_G_COOL_OVERSIZE = 1.15
APP_G_HEAT_OVERSIZE = 1.25

_REPO = Path(__file__).resolve().parents[2]
_WEATHER_REVERSE = _REPO / "examples" / "weather_reverse.csv"


@dataclass(frozen=True)
class ModeScore:
    kind: str
    Q_duty_W: float
    T_zone_K: float
    T_out_K: float
    Q_cap_W: float
    COP: float
    ratio: float
    feasible: bool


@dataclass(frozen=True)
class SizingCaseResult:
    key: str
    label: str
    mode: str
    fluid: str
    conditions: str
    V_disp_m3: float
    A_eev_m2: float
    n_tubes_in: int
    n_tubes_out: int
    plant_match_scale: float
    heating: ModeScore | None
    cooling: ModeScore | None
    notes: tuple[str, ...]
    rejected: bool = False
    gate_detail: str = ""
    Q_heat_W: float | None = None
    Q_cool_W: float | None = None
    T_zone_heat_K: float | None = None
    T_out_heat_K: float | None = None
    T_zone_cool_K: float | None = None
    T_out_cool_K: float | None = None


def _score(spec, fluid: str, kind: str, T_out: float, T_zone: float, Q_duty: float) -> ModeScore:
    spec_m = apply_operating_mode(spec, kind)
    pt = capacity_at(
        fluid=fluid,
        kind=kind,
        T_out=T_out,
        T_zone=T_zone,
        spec=spec_m,
    )
    q = float(pt.Q_cap) if pt.feasible else float("nan")
    cop = float(pt.COP) if pt.feasible else float("nan")
    ratio = q / float(Q_duty) if pt.feasible and Q_duty > 0.0 else float("nan")
    return ModeScore(
        kind=kind,
        Q_duty_W=float(Q_duty),
        T_zone_K=float(T_zone),
        T_out_K=float(T_out),
        Q_cap_W=q,
        COP=cop,
        ratio=ratio,
        feasible=bool(pt.feasible),
    )


def _from_system(key: str, label: str, conditions: str, sys) -> SizingCaseResult:
    req = sys.request
    heat = cool = None
    if sys.heating is not None:
        heat = _score(
            sys.spec,
            sys.fluid,
            "heating",
            req.T_out_heat,
            req.T_zone_heat,
            req.Q_heat,
        )
    if sys.cooling is not None:
        cool = _score(
            sys.spec,
            sys.fluid,
            "cooling",
            req.T_out_cool,
            req.T_zone_cool,
            req.Q_cool,
        )
    indoor = sys.spec.indoor.n_tubes if sys.spec.indoor is not None else sys.spec.n_tubes_c
    outdoor = sys.spec.outdoor.n_tubes if sys.spec.outdoor is not None else sys.spec.n_tubes_e
    scale = 1.0
    if sys.heating is not None:
        scale = max(scale, float(sys.heating.plant_match_scale))
    if sys.cooling is not None:
        scale = max(scale, float(sys.cooling.plant_match_scale))
    return SizingCaseResult(
        key=key,
        label=label,
        mode=req.mode,
        fluid=sys.fluid,
        conditions=conditions,
        V_disp_m3=float(sys.spec.V_disp),
        A_eev_m2=float(sys.spec.A_eev),
        n_tubes_in=int(indoor),
        n_tubes_out=int(outdoor),
        plant_match_scale=float(scale),
        heating=heat,
        cooling=cool,
        notes=tuple(sys.notes),
        rejected=False,
        gate_detail="",
        Q_heat_W=req.Q_heat,
        Q_cool_W=req.Q_cool,
        T_zone_heat_K=req.T_zone_heat,
        T_out_heat_K=req.T_out_heat,
        T_zone_cool_K=req.T_zone_cool,
        T_out_cool_K=req.T_out_cool,
    )


def _from_gate_error(
    key: str,
    label: str,
    conditions: str,
    req: DesignRequest,
    err: DesignGateError,
) -> SizingCaseResult:
    failed = err.gates.hard_failures
    detail = "; ".join(g.detail for g in failed) if failed else str(err)
    return SizingCaseResult(
        key=key,
        label=label,
        mode=req.mode,
        fluid=req.refrigerant,
        conditions=conditions,
        V_disp_m3=float("nan"),
        A_eev_m2=float("nan"),
        n_tubes_in=0,
        n_tubes_out=0,
        plant_match_scale=float("nan"),
        heating=None,
        cooling=None,
        notes=(f"Sizer rejected the duty: {detail}",),
        rejected=True,
        gate_detail=detail,
        Q_heat_W=req.Q_heat,
        Q_cool_W=req.Q_cool,
        T_zone_heat_K=req.T_zone_heat,
        T_out_heat_K=req.T_out_heat,
        T_zone_cool_K=req.T_zone_cool,
        T_out_cool_K=req.T_out_cool,
    )


def _request(**kwargs) -> DesignRequest:
    kwargs.setdefault("match_plant", False)
    return DesignRequest(**kwargs)


def case_definitions(weather_reverse: Path | None = None) -> list[dict]:
    """Case metadata and ``DesignRequest`` kwargs. Duties and temperatures are stated, not inferred, except the optional weather row."""
    ton3 = cooling_tons_to_w(3.0)
    cases = [
        {
            "key": "Ciso",
            "label": "C1",
            "conditions": "ISO 5151 T1",
            "request": _request(
                refrigerant="R410A",
                mode="cooling",
                Q_cool=3500.0,
                T_zone=ISO_T1_TZ_K,
                T_out_cool=ISO_T1_TOUT_K,
            ),
        },
        {
            "key": "Cahri",
            "label": "C2",
            "conditions": "AHRI 210/240 A",
            "request": _request(
                refrigerant="R410A",
                mode="cooling",
                Q_cool=ton3,
                T_zone=AHRI_A_TZ_K,
                T_out_cool=AHRI_A_TOUT_K,
            ),
        },
        {
            "key": "Cex",
            "label": "C3",
            "conditions": "example",
            "request": _request(
                refrigerant="R410A",
                mode="cooling",
                Q_cool=6200.0,
                T_zone=24.0 + 273.15,
                T_out_cool=35.0 + 273.15,
            ),
        },
        {
            "key": "Hex",
            "label": "Hx",
            "conditions": "example",
            "request": _request(
                refrigerant="R32",
                mode="heating",
                Q_heat=5500.0,
                T_zone=20.0 + 273.15,
                T_out_heat=0.0 + 273.15,
            ),
        },
        {
            "key": "Hahri",
            "label": "H47",
            "conditions": "AHRI 210/240 H1",
            "request": _request(
                refrigerant="R410A",
                mode="heating",
                Q_heat=5500.0,
                T_zone=AHRI_H1_TZ_K,
                T_out_heat=AHRI_H1_TOUT_K,
            ),
        },
        {
            "key": "Hrej",
            "label": "H17",
            "conditions": "AHRI 210/240 H3",
            "request": _request(
                refrigerant="R32",
                mode="heating",
                Q_heat=5500.0,
                T_zone=AHRI_H3_TZ_K,
                T_out_heat=AHRI_H3_TOUT_K,
            ),
        },
        {
            "key": "Rdual",
            "label": "R1",
            "conditions": "example, both duties",
            "request": _request(
                refrigerant="R32",
                mode="heat_pump",
                Q_heat=5500.0,
                Q_cool=6200.0,
                T_zone=20.0 + 273.15,
                T_zone_heat=20.0 + 273.15,
                T_zone_cool=24.0 + 273.15,
                T_out_heat=0.0 + 273.15,
                T_out_cool=35.0 + 273.15,
            ),
        },
        {
            "key": "Rahri",
            "label": "R2",
            "conditions": "AHRI A + H1",
            "request": _request(
                refrigerant="R410A",
                mode="heat_pump",
                Q_heat=ton3,
                Q_cool=ton3,
                T_zone=AHRI_H1_TZ_K,
                T_zone_heat=AHRI_H1_TZ_K,
                T_zone_cool=AHRI_A_TZ_K,
                T_out_heat=AHRI_H1_TOUT_K,
                T_out_cool=AHRI_A_TOUT_K,
            ),
        },
    ]
    path = weather_reverse if weather_reverse is not None else _WEATHER_REVERSE
    if path.is_file():
        ts = TimeSeries.from_csv(path)
        cases.append(
            {
                "key": "Rwx",
                "label": "R3",
                "conditions": "inferred from example CSV",
                "request": _request(
                    refrigerant="R32",
                    mode="auto",
                    T_zone=293.15,
                    timeseries=ts,
                ),
            }
        )
    return cases


def run_cases(weather_reverse: Path | None = None) -> list[SizingCaseResult]:
    out: list[SizingCaseResult] = []
    for spec in case_definitions(weather_reverse):
        try:
            sys = design_system(spec["request"])
        except DesignGateError as err:
            out.append(
                _from_gate_error(
                    spec["key"], spec["label"], spec["conditions"], spec["request"], err
                )
            )
            continue
        out.append(_from_system(spec["key"], spec["label"], spec["conditions"], sys))
    return out


def _fmt(x: float, nd: int = 2) -> str:
    if x != x:  # NaN
        return "---"
    return f"{x:.{nd}f}"


def _duty_cell(c: SizingCaseResult) -> str:
    if c.mode in ("heat_pump",) or (
        c.Q_heat_W is not None and c.Q_cool_W is not None and c.Q_heat_W >= 200.0 and c.Q_cool_W >= 200.0
    ):
        return f"{c.Q_heat_W/1e3:.2f} / {c.Q_cool_W/1e3:.2f}"
    if c.heating is not None or (c.Q_heat_W is not None and c.mode == "heating"):
        q = c.heating.Q_duty_W if c.heating is not None else c.Q_heat_W
        return f"{q/1e3:.2f}"
    q = c.cooling.Q_duty_W if c.cooling is not None else c.Q_cool_W
    return f"{q/1e3:.2f}"


def _temp_cell(c: SizingCaseResult) -> str:
    if c.mode == "heat_pump" or (
        c.Q_heat_W is not None
        and c.Q_cool_W is not None
        and c.Q_heat_W >= 200.0
        and c.Q_cool_W >= 200.0
    ):
        th = c.T_zone_heat_K - 273.15
        toh = c.T_out_heat_K - 273.15
        tc = c.T_zone_cool_K - 273.15
        toc = c.T_out_cool_K - 273.15
        return f"{th:.1f}/{toh:.1f}; {tc:.1f}/{toc:.1f}"
    if c.mode == "heating" or c.heating is not None:
        tz = (c.heating.T_zone_K if c.heating is not None else c.T_zone_heat_K) - 273.15
        to = (c.heating.T_out_K if c.heating is not None else c.T_out_heat_K) - 273.15
        return f"{tz:.1f} / {to:.1f}"
    tz = (c.cooling.T_zone_K if c.cooling is not None else c.T_zone_cool_K) - 273.15
    to = (c.cooling.T_out_K if c.cooling is not None else c.T_out_cool_K) - 273.15
    return f"{tz:.1f} / {to:.1f}"


def _ratio_cell(c: SizingCaseResult) -> str:
    if c.rejected:
        return "rejected"
    if c.heating is not None and c.cooling is not None:
        return f"{_fmt(c.heating.ratio)} / {_fmt(c.cooling.ratio)}"
    s = c.heating or c.cooling
    if s is None:
        return "---"
    return _fmt(s.ratio)


def _mode_tex(c: SizingCaseResult) -> str:
    if c.mode == "heat_pump":
        return "reverse"
    return c.mode


def latex_table_rows(cases: list[SizingCaseResult]) -> str:
    lines = []
    for c in cases:
        if c.rejected:
            vdisp, aeev, tubes = "---", "---", "---"
            ratio = "rejected"
            cond = c.conditions + " (gate)"
        else:
            vdisp = f"{c.V_disp_m3*1e6:.1f}"
            aeev = f"{c.A_eev_m2*1e6:.2f}"
            tubes = f"{c.n_tubes_in}/{c.n_tubes_out}"
            ratio = _ratio_cell(c)
            cond = c.conditions
        lines.append(
            f"{c.label} & {_mode_tex(c)} & {c.fluid} & {_duty_cell(c)} & "
            f"{_temp_cell(c)} & {cond} & "
            f"{vdisp} & {aeev} & {tubes} & {ratio} \\\\"
        )
    return "\n".join(lines)


def latex_si_rows(cases: list[SizingCaseResult]) -> str:
    lines = []
    for c in cases:
        if c.rejected:
            if c.mode == "heating" and c.Q_heat_W:
                qtxt = f"{c.Q_heat_W/1e3:.2f}"
            elif c.mode == "cooling" and c.Q_cool_W:
                qtxt = f"{c.Q_cool_W/1e3:.2f}"
            else:
                qh = f"{c.Q_heat_W/1e3:.2f}" if c.Q_heat_W else "---"
                qc = f"{c.Q_cool_W/1e3:.2f}" if c.Q_cool_W else "---"
                qtxt = f"{qh}/{qc}"
            lines.append(
                f"{c.label} rejected & {c.fluid} & {qtxt} & --- & --- & --- & --- & --- & --- \\\\"
            )
            continue
        if c.heating is not None:
            h = c.heating
            lines.append(
                f"{c.label} heat & {c.fluid} & {h.Q_duty_W/1e3:.2f} & "
                f"{h.T_zone_K-273.15:.1f} & {h.T_out_K-273.15:.1f} & "
                f"{h.Q_cap_W/1e3:.2f} & {_fmt(h.COP)} & {_fmt(h.ratio)} & "
                f"{c.V_disp_m3*1e6:.1f} \\\\"
            )
        if c.cooling is not None:
            k = c.cooling
            lines.append(
                f"{c.label} cool & {c.fluid} & {k.Q_duty_W/1e3:.2f} & "
                f"{k.T_zone_K-273.15:.1f} & {k.T_out_K-273.15:.1f} & "
                f"{k.Q_cap_W/1e3:.2f} & {_fmt(k.COP)} & {_fmt(k.ratio)} & "
                f"{c.V_disp_m3*1e6:.1f} \\\\"
            )
    return "\n".join(lines)


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def write_latex(cases: list[SizingCaseResult], dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        _macro("ManualSCoolMax", f"{MANUAL_S_COOL_MAX:.2f}"),
        _macro("ManualSHeatMax", f"{MANUAL_S_HEAT_MAX:.2f}"),
        _macro("AppGCoolOversize", f"{APP_G_COOL_OVERSIZE:.2f}"),
        _macro("AppGHeatOversize", f"{APP_G_HEAT_OVERSIZE:.2f}"),
        _macro("SizingNcases", str(len(cases))),
        f"\\newcommand{{\\SizingCaseRows}}{{%\n{latex_table_rows(cases)}\n}}",
        f"\\newcommand{{\\SizingSIRows}}{{%\n{latex_si_rows(cases)}\n}}",
    ]
    for c in cases:
        p = c.key
        lines.append(_macro(f"Case{p}Fluid", c.fluid))
        if c.rejected:
            lines.append(_macro(f"Case{p}Vdisp", "---"))
            lines.append(_macro(f"Case{p}Aeev", "---"))
            detail = (
                c.gate_detail.replace("%", r"\%")
                .replace("_", r"\_")
                .replace("°C", r"$^\circ$C")
            )
            lines.append(_macro(f"Case{p}Gate", detail))
            continue
        lines.append(_macro(f"Case{p}Vdisp", f"{c.V_disp_m3*1e6:.1f}"))
        lines.append(_macro(f"Case{p}Aeev", f"{c.A_eev_m2*1e6:.2f}"))
        if c.heating is not None:
            lines.append(_macro(f"Case{p}HeatRatio", _fmt(c.heating.ratio)))
            lines.append(_macro(f"Case{p}HeatQcapkW", f"{c.heating.Q_cap_W/1e3:.2f}"))
        if c.cooling is not None:
            lines.append(_macro(f"Case{p}CoolRatio", _fmt(c.cooling.ratio)))
            lines.append(_macro(f"Case{p}CoolQcapkW", f"{c.cooling.Q_cap_W/1e3:.2f}"))
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _json_num(x: float) -> float | None:
    if x != x:
        return None
    return float(x)


def to_json(cases: list[SizingCaseResult]) -> dict:
    rows = []
    for c in cases:
        rec = {
            "key": c.key,
            "label": c.label,
            "mode": c.mode,
            "fluid": c.fluid,
            "conditions": c.conditions,
            "rejected": c.rejected,
            "gate_detail": c.gate_detail,
            "V_disp_cm3": _json_num(c.V_disp_m3 * 1e6) if not c.rejected else None,
            "A_eev_mm2": _json_num(c.A_eev_m2 * 1e6) if not c.rejected else None,
            "n_tubes_in": c.n_tubes_in if not c.rejected else None,
            "n_tubes_out": c.n_tubes_out if not c.rejected else None,
            "plant_match_scale": _json_num(c.plant_match_scale) if not c.rejected else None,
            "notes": list(c.notes),
        }
        if c.heating is not None:
            rec["heating"] = {
                "Q_duty_W": c.heating.Q_duty_W,
                "T_zone_C": c.heating.T_zone_K - 273.15,
                "T_out_C": c.heating.T_out_K - 273.15,
                "Q_cap_W": c.heating.Q_cap_W,
                "COP": c.heating.COP,
                "ratio": c.heating.ratio,
                "feasible": c.heating.feasible,
            }
        if c.cooling is not None:
            rec["cooling"] = {
                "Q_duty_W": c.cooling.Q_duty_W,
                "T_zone_C": c.cooling.T_zone_K - 273.15,
                "T_out_C": c.cooling.T_out_K - 273.15,
                "Q_cap_W": c.cooling.Q_cap_W,
                "COP": c.cooling.COP,
                "ratio": c.cooling.ratio,
                "feasible": c.cooling.feasible,
            }
        rows.append(rec)
    return {
        "match_plant": False,
        "manual_s_cool_max": MANUAL_S_COOL_MAX,
        "manual_s_heat_max": MANUAL_S_HEAT_MAX,
        "app_g_cool_oversize": APP_G_COOL_OVERSIZE,
        "app_g_heat_oversize": APP_G_HEAT_OVERSIZE,
        "cases": rows,
    }


def write_outputs(
    *,
    tex: Path,
    json_path: Path | None = None,
    weather_reverse: Path | None = None,
) -> list[SizingCaseResult]:
    cases = run_cases(weather_reverse)
    write_latex(cases, tex)
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(to_json(cases), indent=2) + "\n", encoding="utf-8")
    return cases


def main(argv: list[str] | None = None) -> None:
    del argv
    tex = _REPO / "paper" / "generated_sizing_cases.tex"
    js = _REPO / "output" / "sizing_cases.json"
    cases = write_outputs(tex=tex, json_path=js)
    print(f"wrote {tex} and {js} ({len(cases)} cases)")
    for c in cases:
        print(
            f"  {c.label} {c.mode} {c.fluid}  V={c.V_disp_m3*1e6:.1f} cm3  "
            f"ratio {_ratio_cell(c)}"
        )


if __name__ == "__main__":
    main()
