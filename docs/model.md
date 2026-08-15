# Transient two-phase vapor-compression plant

This note describes the air-source heat-pump / air-conditioner model in
this repository: a subcritical vapor-compression cycle with distributed
two-phase heat exchangers, closed by a scroll compressor and an
isenthalpic electronic expansion valve (EEV), coupled to a single dry
zone. The residual is written in JAX so the implicit integrator and
model-predictive controllers share one plant.

The implementation lives under `src/heatpump/`. SI units are used
throughout (Pa, K, J/kg, kg/s, W, m).

## 1. Scope

The plant is a **single-stage, subcritical, two-phase** machine. Any
CoolProp HEOS fluid can be named; the sizer rejects duties that would
require transcritical condensation (typical room heating with CO2).

Included:

- evaporator and condenser as 1-D finite-volume channels
- scroll / recip compressor with clearance volumetric efficiency
- isenthalpic EEV
- indoor / outdoor coil remap for heating vs cooling
- lumped zone air capacitance
- load-based hardware sizing
- zone + superheat controllers (PID cascade, hysteresis, bang-bang, linear and nonlinear MPC)

Not included:

- moisture / latent load, frosting, or defrost
- ducts, fans as dynamic states, or multi-zone buildings
- reversing mid-integration (run heating and cooling as two simulations)
- transcritical gas coolers, flash tanks, or economizers
- oil, charge migration into inactive volumes, or piping inertia

## 2. Architecture

```
  outdoor air ── outdoor coil ── refrigerant loop ── indoor coil ── zone
                      │                                    │
                   evap (heat)                          cond (heat)
                   cond (cool)                          evap (cool)
```

Modules:

| Module | Role |
|---|---|
| `thermo` | CoolProp flashes → JAX \((p,h)\) tables |
| `components` | compressor, EEV, HTC, air-side march |
| `plant` | finite-volume DAE residual \( \dot y = f(t,y,u) \) |
| `solver` | TR-BDF2 + damped Newton with `jacfwd` |
| `design` | CoolProp design cycle and hardware sizing |
| `requirements` | `DesignRequest`, constraints, timeseries |
| `control` | zone / superheat laws and MPC |
| `simulate` | closed-loop integration and CLI |

CoolProp is **not** called from the JIT residual. At setup the fluid is
flashed onto a dense \((p,h)\) grid; the residual only interpolates.

## 3. Refrigerant properties

### 3.1 CoolProp flashes

`resolve_fluid` maps common HVAC names (`R-32`, `410a`, `propane`, …)
onto CoolProp HEOS identifiers. Saturation envelopes use \( (p,q) \)
inputs. Single-phase states use \( (h,p) \). Transport properties that
fail near the critical point fall back to saturated-liquid or
saturated-vapor values.

### 3.2 Two-phase density (Zivi)

Homogeneous density under-predicts slip in evaporators. Two-phase
density uses Zivi’s void fraction

\[
\alpha = \left[ 1 + \frac{1-x}{x} \left(\frac{\rho_g}{\rho_f}\right)^{2/3} \right]^{-1},
\qquad
\rho = \alpha\,\rho_g + (1-\alpha)\,\rho_f.
\]

Quality is \( x = (h - h_f)/(h_g - h_f) \) inside the dome.

### 3.3 JAX tables

`build_tables(fluid)` builds a log-spaced pressure grid from just above
the fluid’s usable saturation floor to \( 0.92\,p_c \), and a linear
enthalpy grid from \( h_f^\mathrm{min} - 80\,\mathrm{kJ/kg} \) to
\( h_g^\mathrm{max} + 200\,\mathrm{kJ/kg} \). Each node stores
\( T, \rho, x, \mu, k, c_p \). One-dimensional saturation columns store
\( T_\mathrm{bubble}(p) \), \( T_\mathrm{dew}(p) \), \( h_f \), \( h_g \),
\( \rho_f \), \( \rho_g \).

`eval_ph` bilinearly interpolates the 2-D fields. Density derivatives
\( (\partial\rho/\partial p)_h \) and \( (\partial\rho/\partial h)_p \)
are the analytic slopes of that interpolant — they are **not** stored
as separate tables. Superheat uses dew temperature; subcooling uses
bubble temperature (zeotropic glide is therefore consistent).

The residual must stay finite under `jacfwd`. Quality used in the HTC
is clipped away from \( 0 \) and \( 1 \); pressures and enthalpies are
projected onto the table box after every Newton update.

## 4. Plant DAE

### 4.1 State and inputs

Each coil has a **single pressure** (acoustic equilibrium) and a
distributed enthalpy / wall-temperature field. The state is

\[
y = \bigl[ p_e,\; h_e^{(1:n_e)},\; T_{w,e}^{(1:n_e)},\;
           p_c,\; h_c^{(1:n_c)},\; T_{w,c}^{(1:n_c)},\; T_z \bigr].
\]

Default mesh is \( n_e = n_c = 6 \) (31 states). Inputs are

\[
u = \bigl[ N,\; u_\mathrm{eev},\; \phi_i,\; \phi_o,\; T_\mathrm{out},\; Q_\mathrm{gain} \bigr]
\]

with compressor speed \( N \) in Hz, EEV opening in \( [0,1] \), indoor
/ outdoor fan fractions, outdoor dry-bulb, and heat **into** the zone
(W) on top of the envelope term.

### 4.2 Finite-volume heat exchanger

This is a method-of-lines HVAC model in the family of Bendapudi,
Rasmussen, and Qiao: one pressure per coil, upwind enthalpy cells, and
a linear internal mass-flow profile between the known port flows.

Mass and energy in integral form are

\[
\frac{\partial\rho}{\partial t} + \frac{\partial(\rho v)}{\partial z} = 0,
\qquad
\frac{\partial(\rho h)}{\partial t} + \frac{\partial(\rho v h)}{\partial z}
  = \frac{\partial p}{\partial t} + \frac{P}{A} q''.
\]

Per cell, with well-mixed energy and upwind inlet enthalpy \( h^\mathrm{up} \),

\[
\rho V \frac{\mathrm{d}h}{\mathrm{d}t}
  = \dot m_\mathrm{in}(h^\mathrm{up} - h) + Q + V \frac{\mathrm{d}p}{\mathrm{d}t}.
\]

Interface mass flow is the linear blend

\[
\dot m(z) = \dot m_\mathrm{in}\,(1-\xi) + \dot m_\mathrm{out}\,\xi,
\qquad \xi \in [0,1].
\]

Evaporator ports: inlet = EEV, outlet = compressor suction.
Condenser ports: inlet = compressor discharge, outlet = EEV.

Overall mass closes the pressure ODE. Inventory includes the header
volume \( V_h \) at the mean cell density:

\[
M = \sum_i \rho_i V_i + V_h \langle\rho\rangle,
\qquad
\mathrm{d}\rho = \Bigl(\frac{\partial\rho}{\partial p}\Bigr)_h \mathrm{d}p
               + \Bigl(\frac{\partial\rho}{\partial h}\Bigr)_p \mathrm{d}h.
\]

Omitting \( (\partial\rho/\partial h)_p \) on the header (or the cells)
produces a secular charge drift. Both terms are kept.

A **moving-boundary** coil (explicit SH / two-phase / SC zones) is not
used. Finite volume lets the same residual handle startup, flooding,
and dry-out without switching index sets, which matters for Newton and
`jacfwd`. Superheat, two-phase, and subcooling still appear as the
enthalpy profile along the mesh.

### 4.3 Compressor

Semi-empirical scroll / recip map. Pressure ratio
\( \Pi = p_d / p_s \). Volumetric efficiency is the clearance form

\[
\eta_v = 1 - C\bigl(\Pi^{1/\gamma} - 1\bigr).
\]

Mass flow is \( \dot m = \eta_v\,\rho_s\,V_\mathrm{disp}\,N_\mathrm{eff} \),
with a smooth cutoff \( N_\mathrm{eff} = N\,\sigma(1.5(N-4)) \) so the
machine is off near 0 Hz. Discharge enthalpy uses a polytropic
isentropic rise (no entropy inversion inside the residual)

\[
\Delta h_\mathrm{is}
  = \frac{\gamma}{\gamma-1}\frac{p_s}{\rho_s}\bigl(\Pi^{(\gamma-1)/\gamma}-1\bigr),
\qquad
h_d = h_s + \Delta h_\mathrm{is}/\eta_\mathrm{is},
\]

and shaft power \( W = \dot m\,(h_d - h_s) \). \( \gamma \) is taken
from the CoolProp suction state at design and held constant in the
transient. \( \eta_\mathrm{is} \) is the user-supplied isentropic
efficiency (constant).

### 4.4 Expansion valve

Isenthalpic orifice, \( h_\mathrm{eev} = h_{c,\mathrm{out}} \):

\[
\dot m = C_d\,A_\mathrm{max}\,u\sqrt{2\rho\,\Delta p_+}.
\]

\( A = A_\mathrm{max}\,u \) is the geometric opening. \( \Delta p_+ \)
is a \( C^1 \) soft-plus so the Jacobian exists at zero pressure
difference.

### 4.5 Heat transfer

Refrigerant-side HTC is Dittus–Boelter when quality is 0 or 1,

\[
\mathrm{Nu} = 0.023\,\mathrm{Re}^{0.8}\,\mathrm{Pr}^{n},
\qquad n = 0.4\ \text{(evap)},\ 0.3\ \text{(cond)},
\]

and the Shah two-phase multiplier \( F(x,p_r) \) inside the dome.

Air is quasi-steady. Each wall cell sees the incoming air temperature;
heat to the wall is \( q = h_a A_a (T_a - T_w) \), and the air
temperature is marched downstream with \( c_{p,\mathrm{air}} \) from
CoolProp dry air at the coil inlet temperature. Fan fraction scales
both \( h_a \) and \( \dot m_a \). Design \( h_a \) is Zhukauskas
cross-flow over a tube bank.

Wall energy: \( C_w \dot T_w = -Q_\mathrm{ref} + Q_\mathrm{air} \).

### 4.6 Zone

One dry capacitance:

\[
C_z \dot T_z = Q_\mathrm{zone} + Q_\mathrm{gain} + UA\,(T_\mathrm{out} - T_z).
\]

Sign convention: \( Q_\mathrm{air} \) is heat from air to the coil wall.

- Heating: indoor coil is the condenser, \( Q_\mathrm{zone} = -\sum Q_{\mathrm{air},c} \) (positive when the coil heats the room).
- Cooling: indoor coil is the evaporator, \( Q_\mathrm{zone} = -\sum Q_{\mathrm{air},e} \) (negative when the coil cools the room).

COP uses useful capacity: \( Q_\mathrm{zone} \) in heating,
\( |Q_\mathrm{zone}| \) in cooling, divided by compressor power.

### 4.7 Operating mode

Indoor and outdoor geometry are stored as `CoilSpec` objects. Heating
maps outdoor → evaporator, indoor → condenser. Cooling swaps them.
`apply_operating_mode` copies the active pair onto the `*_e` / `*_c`
fields the residual reads. Mid-run reversing is not modeled.

## 5. Design

`design_cycle` builds a subcritical four-point cycle from CoolProp:

1. suction at dew + superheat
2. discharge at \( h_1 + (h_{2s}-h_1)/\eta_\mathrm{is} \)
3. liquid at bubble − subcool
4. EEV outlet at \( h_3 \), \( p_e \)

Heating: evaporator air is outdoor, condenser air is the zone.
Cooling: the opposite. Evaporating / condensing temperatures are
offset from those air temperatures by the design approaches
`DT_evap` / `DT_cond` (defaults 10 K / 12 K). Condensing above
\( p_\mathrm{crit} \) is rejected (subcritical plant).

Mass flow follows the useful duty (condenser heat in heating,
evaporator heat in cooling). Displacement is

\[
V_\mathrm{disp} = \frac{\dot m}{\eta_v\,\rho_1\,N_\mathrm{design}},
\qquad
\eta_v = 1 - C\bigl(\Pi^{1/\gamma}-1\bigr).
\]

EEV area inverts the orifice law at the design opening
\( u = 0.40 \). Tube count is iterated until the ε-NTU heat rate
equals the cycle duty, with refrigerant HTC from Dittus–Boelter /
Shah at the design mass flux and air-side HTC from Zhukauskas.
Air mass flow is closed from \( Q = \varepsilon\,\dot m_a c_p\,\Delta T \).
Wall capacitance is \( \rho_\mathrm{cu} c_{p,\mathrm{cu}} A t_w \).
Charge is the Zivi / flashed density on the design enthalpy profile
times internal volume.

Envelope \( UA = Q_\mathrm{load}/|T_z-T_\mathrm{out}| \). Zone
capacitance is \( \rho_\mathrm{air} c_p V \) (pass `V_zone` or
`C_zone`; default volume 50 m³).

A reversible unit (`mode=heat_pump`) sizes both duties and takes the
harder compressor, EEV, and coil of the two.

### 5.1 Sizing from a timeseries

If `DesignRequest` has a `TimeSeries` and no nameplate, design duty is
the peak \( |Q_\mathrm{gain}| \) that must be cancelled to hold the
setpoint:

- cooling capacity \( \max(Q_\mathrm{gain}, 0) \)
- heating capacity \( \max(-Q_\mathrm{gain}, 0) \)

Outdoor design temperature is the ambient **at that peak hour**, not
the extreme of the whole record. When the profile is the complete
load, envelope \( UA \) is set to zero so the CSV is not double
counted. `mode=auto` becomes heating, cooling, or heat_pump from the
signs of those peaks.

CSV columns (case-insensitive): time, outdoor temperature, load;
optional setpoint. `Q_kW` is taken as kilowatts. `load_kind` is
`gain` (default, positive heats the zone), `cooling_load`, or
`heating_load`.

### 5.2 Feasibility gates

After the cycle and geometry are closed, `evaluate_gates` checks a
hard envelope. Failure raises `DesignGateError`:

- \( p_c \le f\,p_\mathrm{crit} \) (user material limit; default \( f = 0.90 \))
- \( T_\mathrm{disch} \le T_\mathrm{disch,max} \) (default 115 °C)
- pressure ratio \( \le \Pi_\mathrm{max} \) (default 7.5)
- superheat in \( [4,10] \) K, subcooling \( \ge 0 \)
- optional mass-flux cap if `Constraints.G_max` is set

These are user envelope limits, not substitutes for the EOS.

### 5.3 Capacity versus outdoor temperature

A *fixed* machine (sized \( V_\mathrm{disp} \) and coil geometry) is
re-closed on a \( T_\mathrm{out} \) grid: \( T_e \) and \( T_c \) are
solved so refrigerant \( \dot m\,\Delta h \) equals the ε-NTU coil
heat rate. Mass flow follows the clearance map. Useful capacity is
condenser heat in heating and evaporator heat in cooling.

The load line is Newton cooling \( UA\,|T_z - T_\mathrm{out}| \), or
the timeseries \( Q \) interpolated on outdoor temperature. The
**balance point** is the zero of \( Q_\mathrm{cap}(T) - Q_\mathrm{load}(T) \).
Design margin is \( Q_\mathrm{cap}/Q_\mathrm{load} \) at the design
outdoor temperature.

### 5.4 Cooling psychrometrics

The transient plant remains dry. The design package computes indoor
wet-bulb and dew point from CoolProp humid air. The coil is wet when
\( T_e \) is below the indoor dew point; leaving humidity is then
saturation at \( T_e \). Latent heat is
\( \dot m_a (W_\mathrm{in}-W_\mathrm{out}) h_{fg}(T) \) with water
\( h_{fg} \) from CoolProp. SHR is \( Q_\mathrm{sens}/Q_\mathrm{coil} \),
an output, not an input.

### 5.5 Design package

`SystemDesign.as_report()` writes markdown + JSON: gates, hardware,
charge from the enthalpy profile, shaft current \( I = W/V \)
(or \( W/(V\sqrt{3}) \)), capacity tables with closed \( T_e,T_c \),
and psychrometrics. Motor efficiency is applied only when supplied.

## 6. Controllers

All laws share the same plant measurements (zone temperature,
superheat, …). The inner loop is a superheat EEV: low SH closes the
valve; a compressor-speed feedforward plus a slow PID trim set the
opening, rate-limited so inventory can move.

**PID cascade.** ISA PID on zone temperature with back-calculation
anti-windup and derivative on measurement. Gains flip sign in cooling
and scale with \( C_z \) and \( N_\mathrm{max} \). Compressor speed is
rate-limited.

**Hysteresis.** On/off compressor with deadband and minimum on/off
times (mode-aware: heat when cold, cool when hot).

**Bang-bang.** Deadband without cycle timers (chatters on a fast
plant; included as a baseline).

**Linear MPC.** Affine model \( y_{k+1} = A y_k + B u_k + c \) from
`jacfwd` of the residual, discretized with implicit Euler to match the
plant solver. Decision variables are \( (N, u_\mathrm{eev}) \). The QP
is dense least-squares on the stacked inputs, then projected onto box
constraints.

**Nonlinear MPC.** Shooting: implicit-Euler rollout of the full
residual, projected gradient on the input sequence.

`controller=auto` picks hysteresis if the allowed speed band is
narrower than 15 Hz, otherwise PID.

## 7. Time integration

TR-BDF2 (Hosea & Shampine, 1996; MATLAB `ode23tb`): a trapezoidal
stage to \( t+\gamma h \) and a BDF2 completion,
\( \gamma = 2-\sqrt{2} \). The method is stiffly accurate and
L-stable, which the pressure DAE needs. Each stage is damped Newton
with an exact `jacfwd` Jacobian and a four-point line search. The
embedded trapezoidal solution estimates local error; the step is
rejected and cut when the residual stalls or the error exceeds
tolerance.

NMPC uses the same damped Newton as a differentiable implicit Euler
step (unrolled, no adaptive \( h \)).

States are projected onto the property table and a temperature box
after every accepted step.

## 8. Closed-loop simulation

`simulate` sizes the plant if no `PlantSpec` is given, builds tables,
forms \( f(t,y,u) \), and integrates with the chosen controller.
Exogenous \( T_\mathrm{out}(t) \), \( Q_\mathrm{gain}(t) \), and
optional \( T_\mathrm{sp}(t) \) come from a `TimeSeries` or from
constants. Controls are held for `record_dt` (zero-order hold).

Typical heating check (R32, 5.5 kW, 0 °C outdoor, 20 °C setpoint,
PID, 1 h): zone settles at the setpoint, superheat near 6 K, COP
around 3.3, charge drift well below 1 %.

## 9. Limitations and intended use

The model is a **controls-oriented plant**, not a rating-software
digital twin. Coil HTCs are published correlations, not circuit-resolved
fin-and-tube CFD. The compressor is a clearance / polytropic map, not
a manufacturer 10-coefficient polynomial. Charge is conserved by the
DAE but is not a fitted nameplate charge. Air is dry.

Use it to:

- size a first-cut machine for a refrigerant and a load profile
- test zone / superheat controllers, including differentiable MPC
- study transients (pull-down, flooding, speed ramps) that moving-boundary models handle poorly

Do not use it for AHRI / EN rating claims, refrigerant-charge
optimization against a real cabinet, or transcritical CO2.

## 10. Notation

| Symbol | Meaning |
|---|---|
| \( p_e, p_c \) | evaporator / condenser pressure |
| \( h \) | specific enthalpy |
| \( x \) | vapor quality |
| \( T_w, T_z \) | wall / zone temperature |
| \( \rho, \alpha \) | density, void fraction |
| \( \dot m \) | mass flow |
| \( N \) | compressor electrical frequency (Hz) |
| \( V_\mathrm{disp} \) | displacement per revolution |
| \( UA, C_z \) | envelope conductance, zone capacitance |
| \( Q_\mathrm{gain} \) | exogenous heat into the zone |

## 11. References

- S. Bendapudi, J. E. Braun, and E. A. Groll, “A comparison of moving-boundary and finite-volume formulations for transients in centrifugal chillers,” *Int. J. Refrigeration*, 2008.
- B. P. Rasmussen, “Dynamic modeling for vapor compression systems — Part I / II,” *HVAC&R Research*, 2012.
- H. Qiao, V. Aute, and R. Radermacher, “Transient modeling of a flash tank vapor injection heat pump system,” *Int. J. Refrigeration*, 2015.
- M. E. Hosea and L. F. Shampine, “Analysis and implementation of TR-BDF2,” *Applied Numerical Mathematics*, 1996.
- S. M. Zivi, “Estimation of steady-state steam void-fraction by means of the principle of minimum entropy production,” *J. Heat Transfer*, 1964.
- M. M. Shah, “A general correlation for heat transfer during film condensation inside pipes,” *Int. J. Heat Mass Transfer*, 1979.
- J. C. Chen, “Correlation for boiling heat transfer to saturated fluids in convective flow,” *I&EC Process Design and Development*, 1966.
- I. H. Bell, J. Wronski, S. Quoilin, and V. Lemort, “Pure and pseudo-pure fluid thermophysical property evaluation and the open-source thermophysical property library CoolProp,” *Ind. Eng. Chem. Res.*, 2014.
