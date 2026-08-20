# Transient two-phase vapor-compression plant

Subcritical air-source heat pump / air conditioner: distributed
two-phase coils, clearance or AHRI 540 compressor, isenthalpic EEV,
one lumped zone. The JAX residual $`\dot y=f(t,y,u)`$ is shared by the
integrator and MPC.

Implementation: `src/heatpump/`. Usage: [quickstart](quickstart.md),
[examples](../examples/README.md), [README](../README.md). Rendered
(with equations): [wiki](https://github.com/smyng91/hvac_designer/wiki/Model).
Example and CLI results go to `output/`. SI throughout (Pa, K, J/kg,
kg/s, W, m).

## 1. Scope

Single-stage, subcritical, two-phase. Any CoolProp HEOS fluid; duties
that would require transcritical condensation are rejected.

Included: 1-D finite-volume evaporator and condenser; clearance or
cited AHRI 540 compressor; isenthalpic EEV; indoor/outdoor remap;
lumped zone; optional humidity and frost; load-based sizing; PID,
hysteresis, bang-bang, LMPC, NMPC.

Not included: automatic defrost or frost derate tables; ducts or
multi-zone; transcritical CO2, flash tanks, economizers; oil or
piping inertia.

## 2. Architecture

```
  outdoor air ── outdoor coil ── refrigerant loop ── indoor coil ── zone
                      │                                    │
                   evap (heat)                          cond (heat)
                   cond (cool)                          evap (cool)
```

| Module | Role |
|---|---|
| `thermo` | CoolProp flashes $`\to`$ JAX $`(p,h)`$ tables |
| `psychro` | humid-air tables; design-package SHR |
| `components` | clearance map, orifice, Shah, air march |
| `devices` | optional compressor, EEV, HTC, air, zone, fan, frost |
| `plant` | residual $`\dot y=f(t,y,u)`$ |
| `solver` | TR-BDF2; QSS for hours/days |
| `design` | cycle close and hardware sizing |
| `capacity` | off-design $`Q(T_\mathrm{out})`$ |
| `control` | zone / superheat laws and MPC |
| `simulate` | closed-loop integration and CLI |

Empty `PlantSpec` slots call `components`. Retrofit by assignment
(`replace(spec, compressor=MyMap(...))`). CoolProp is not called from
the JIT residual.

## 3. State and inputs

Each coil has one pressure (acoustic equilibrium) and a distributed
enthalpy / wall field.

```math
\begin{aligned}
y=\bigl[&p_e,\; h_e^{(1:n_e)},\; T_{w,e}^{(1:n_e)},\\
&p_c,\; h_c^{(1:n_c)},\; T_{w,c}^{(1:n_c)},\; T_z,\\
&W_z^{(\mathrm{opt})},\; m_\mathrm{fr}^{(\mathrm{opt})}\bigr].
\end{aligned}
```

Default mesh $`n_e=n_c=6`$ (27 states, dry). $`W_z`$ and $`m_\mathrm{fr}`$
are appended only with user `moist` / `frost` and user RH.

```math
u=\bigl[N,\; u_\mathrm{eev},\; \phi_i,\; \phi_o,\; T_\mathrm{out},\; Q_\mathrm{gain}\bigr]
```

optional extras $`\bigl[W_\mathrm{gain},\; \mathrm{defrost},\; \mathrm{RH}_\mathrm{out}\bigr]`$.
Omitted $`W_\mathrm{gain}`$ is zero, not invented infiltration.

| Symbol | Unit | Meaning |
|---|---|---|
| $`p_e,p_c`$ | Pa | evaporator / condenser pressure |
| $`h_e^{(i)},h_c^{(i)}`$ | J/kg | cell enthalpy |
| $`T_{w,e}^{(i)},T_{w,c}^{(i)}`$ | K | cell wall temperature |
| $`T_z`$ | K | zone dry-bulb |
| $`W_z`$ | kg/kg | zone humidity ratio (moist) |
| $`m_\mathrm{fr}`$ | kg | outdoor frost mass (frost) |
| $`N`$ | Hz | compressor electrical frequency |
| $`u_\mathrm{eev}`$ | — | EEV opening in $`[0,1]`$ |
| $`\phi_i,\phi_o`$ | — | indoor / outdoor fan fraction |
| $`T_\mathrm{out}`$ | K | outdoor dry-bulb |
| $`Q_\mathrm{gain}`$ | W | exogenous heat **into** the zone |
| $`W_\mathrm{gain}`$ | kg/s | exogenous vapor into the zone |
| $`\mathrm{defrost}`$ | — | melt flag; melt only if $`W_\mathrm{defrost}>0`$ |
| $`\mathrm{RH}_\mathrm{out}`$ | — | outdoor relative humidity $`0`$–$`1`$ |

## 4. Refrigerant properties

`resolve_fluid` maps HVAC names onto CoolProp HEOS ids. Saturation:
$`(p,q)`$. Single-phase: $`(h,p)`$.

Quality inside the dome and Zivi void fraction:

```math
\begin{aligned}
x&=\frac{h-h_f}{h_g-h_f},\\
\alpha&=\left[1+\frac{1-x}{x}\left(\frac{\rho_g}{\rho_f}\right)^{2/3}\right]^{-1},\\
\rho&=\alpha\rho_g+(1-\alpha)\rho_f.
\end{aligned}
```

`build_tables` uses a log-$`p`$ grid to $`0.92\,p_c`$ and a linear $`h`$
grid. Each node stores $`T,\rho,x,\mu,k,c_p`$. Saturation columns:
$`T_\mathrm{bub}(p)`$, $`T_\mathrm{dew}(p)`$, $`h_f,h_g,\rho_f,\rho_g`$.
`eval_ph` bilinearly interpolates; $`(\partial\rho/\partial p)_h`$ and
$`(\partial\rho/\partial h)_p`$ are the interpolant slopes. Superheat
uses dew $`T`$; subcooling uses bubble $`T`$.

## 5. Plant DAE

### 5.1 Finite-volume coils

Method of lines (Bendapudi / Rasmussen / Qiao): one pressure per coil,
upwind enthalpy, linear internal mass-flow profile.

```math
\begin{aligned}
\frac{\partial\rho}{\partial t}+\frac{\partial(\rho v)}{\partial z}&=0,\\
\frac{\partial(\rho h)}{\partial t}+\frac{\partial(\rho v h)}{\partial z}
&=\frac{\partial p}{\partial t}+\frac{P}{A}q''.
\end{aligned}
```

Per cell, well-mixed energy, upwind inlet $`h^\mathrm{up}`$:

```math
\rho V\frac{\mathrm{d}h}{\mathrm{d}t}
=\dot m_\mathrm{in}(h^\mathrm{up}-h)+Q+V\frac{\mathrm{d}p}{\mathrm{d}t}.
```

```math
\dot m(\xi)=\dot m_\mathrm{in}(1-\xi)+\dot m_\mathrm{out}\,\xi,\qquad \xi\in[0,1].
```

Evaporator ports: EEV $`\to`$ suction. Condenser ports: discharge $`\to`$ EEV.
Inventory includes header volume $`V_h`$ at mean cell density:

```math
M=\sum_i\rho_i V_i+V_h\langle\rho\rangle,\qquad
\mathrm{d}\rho=\Bigl(\frac{\partial\rho}{\partial p}\Bigr)_h\mathrm{d}p
+\Bigl(\frac{\partial\rho}{\partial h}\Bigr)_p\mathrm{d}h.
```

The residual pressure ODE is the discrete mass balance on that inventory
(uniform cells; header terms use mean cell slopes):

```math
\dot h_i^\mathrm{rhs}=\frac{\dot m_i^\mathrm{in}(h_i^\mathrm{up}-h_i)+Q_i}{\rho_i V_i}.
```

```math
\begin{aligned}
\mathrm{num}
&=(\dot m_\mathrm{in}-\dot m_\mathrm{out})
-\sum_i V_i(\partial\rho_i/\partial h)_p\dot h_i^\mathrm{rhs}\\
&\quad-V_h\langle(\partial\rho/\partial h)_p\dot h^\mathrm{rhs}\rangle,\\
\mathrm{den}
&=\sum_i V_i(\partial\rho_i/\partial p)_h
+V_h\langle(\partial\rho/\partial p)_h\rangle\\
&\quad+\sum_i V_i\rho_i^{-1}(\partial\rho_i/\partial h)_p
+V_h\langle\rho^{-1}(\partial\rho/\partial h)_p\rangle,\\
\frac{\mathrm{d}p}{\mathrm{d}t}&=\frac{\mathrm{num}}{\mathrm{den}},\qquad
\frac{\mathrm{d}h_i}{\mathrm{d}t}=\dot h_i^\mathrm{rhs}+\rho_i^{-1}\frac{\mathrm{d}p}{\mathrm{d}t}.
\end{aligned}
```

Both density derivatives are kept (omitting the $`h`$ term drifts charge).
The discrete mass balance is consistent with $`p(t)`$ on the mesh; it does
not imply that numerical inventory is conserved to machine precision.
A moving-boundary coil is not used.

Internal volumes:

```math
V=n_\mathrm{tubes}\,\tfrac{\pi}{4}D^2 L,\qquad
A_r=n_\mathrm{tubes}\,\pi D L,\qquad
A_a=A_r\cdot\mathrm{fin}.
```

### 5.2 Compressor

Clearance volumetric efficiency and polytropic isentropic rise
($`\Pi=p_d/p_s`$):

```math
\begin{aligned}
\eta_v&=1-C\bigl(\Pi^{1/\gamma}-1\bigr),\\
N_\mathrm{eff}&=N\,\sigma\bigl(1.5(N-4)\bigr),\\
\dot m&=\eta_v\,\rho_s\,V_\mathrm{disp}\,N_\mathrm{eff}.
\end{aligned}
```

$`\sigma`$ is a logistic sigmoid. Then

```math
\begin{aligned}
\Delta h_\mathrm{is}&=\frac{\gamma}{\gamma-1}\frac{p_s}{\rho_s}\bigl(\Pi^{(\gamma-1)/\gamma}-1\bigr),\\
h_d&=h_s+\Delta h_\mathrm{is}/\eta_\mathrm{is},\\
W&=\dot m(h_d-h_s).
\end{aligned}
```

$`\gamma`$ is taken at the design suction state and held. $`\eta_\mathrm{is}`$
is constant. $`C`$ is clearance.

AHRI 540 (only if a cited file is supplied), $`T_s,T_d`$ dew points in °C:

```math
\begin{aligned}
X&=C_1+C_2 T_s+C_3 T_d+C_4 T_s^2+C_5 T_s T_d+C_6 T_d^2\\
&\quad+C_7 T_s^3+C_8 T_s^2 T_d+C_9 T_s T_d^2+C_{10} T_d^3.
\end{aligned}
```

$`X`$ is power (W) or mass flow. Hermetic close: $`h_d=h_s+W/\dot m`$.
Lee 2021 Table 5 is in `data/maps/`. No default polynomial.

### 5.3 Expansion valve

Isenthalpic, $`h_\mathrm{eev}=h_{c,\mathrm{out}}`$:

```math
\begin{aligned}
\dot m&=C_d A_\mathrm{max}u\sqrt{2\rho\,\Delta p_+},\\
\Delta p_+&=\tfrac12\bigl(\Delta p+\sqrt{\Delta p^2+\varepsilon^2}\bigr),
\end{aligned}
```

with $`\Delta p=p_\mathrm{in}-p_\mathrm{out}`$ and $`\varepsilon=20\,\mathrm{kPa}`$.
The $`C^1`$ soft-plus keeps the Jacobian defined at $`\Delta p=0`$.

### 5.4 Heat transfer

Single-phase Dittus–Boelter; two-phase Shah multiplier $`F(x,p_r)`$:

```math
\mathrm{Nu}=0.023\,\mathrm{Re}^{0.8}\,\mathrm{Pr}^{n},\qquad
n=0.4\ \text{(evap)},\ 0.3\ \text{(cond)}.
```

Inside the dome a Shah-type multiplier $`F(x,p_r)`$ is applied to that
single-phase coefficient (not the full Shah 1979 correlation). With
$`x_s=\mathrm{clip}(x,10^{-3},1-10^{-3})`$ and $`p_r=p/p_\mathrm{crit}`$,

```math
X_{tt}=\bigl((1-x_s)/x_s\bigr)^{0.9}p_r^{0.15}.
```

```math
\begin{aligned}
F_\mathrm{e}&=1+1.8\,X_{tt}^{-0.7}+8x_s(1-x_s),\\
F_\mathrm{c}&=(1-x_s)^{0.8}+3.8\,x_s^{0.76}(1-x_s)^{0.04}p_r^{-0.38}.
\end{aligned}
```

Two-phase HTC is $`h_\mathrm{sp}\max(F,1)`$. Air is quasi-steady. Series UA:

```math
\frac{1}{UA}=\frac{1}{h_r A_r}+\frac{1}{h_a A_a}.
```

Air is marched cell-wise against the **refrigerant** temperature with
that series $`UA`$ (not against $`T_w`$). Heat from air to cell $`i`$ is
$`q_i=UA_i(T_{\mathrm{air},i}-T_{\mathrm{ref},i})`$, and
$`T_{\mathrm{air},i+1}=T_{\mathrm{air},i}-q_i/(\dot m_a c_{p,a})`$.
That equilibrium $`Q`$ is the heat to the refrigerant. A slaved wall ODE
is integrated for frost and diagnostics; $`T_w`$ does not enter the
air-to-refrigerant energy close:

```math
\begin{aligned}
T_w^\mathrm{ss}&=T_\mathrm{ref}+Q/(h_r A_r),\\
\dot T_w&=(T_w^\mathrm{ss}-T_w)/\tau,\\
\tau&=\max\bigl(C_w/(n\cdot 400),\,2\,\mathrm{s}\bigr).
\end{aligned}
```

The floor on $`\tau`$ keeps the wall equation integrable; it is not a
capacity derate. Fan fraction scales $`h_a`$ and $`\dot m_a`$. Design
$`h_a`$: Zhukauskas $`\mathrm{Nu}=0.27\,\mathrm{Re}^{0.63}\,\mathrm{Pr}^{0.36}`$.

Sign: $`Q_\mathrm{air}`$ is heat from air to the coil.

```math
Q_\mathrm{zone}=\begin{cases}
-\sum Q_{\mathrm{air},c} & \text{heating (indoor = cond)}\\
-\sum(Q_{\mathrm{air},e}+Q_{\mathrm{lat},e}) & \text{cooling (indoor = evap)}
\end{cases}
```

Dry plants have $`Q_\mathrm{lat}=0`$. COP is useful capacity over shaft
power: $`Q_\mathrm{zone}/W`$ in heating, $`|Q_\mathrm{zone}|/W`$ in cooling.
`diagnostics()` reports the same $`Q_\mathrm{zone}`$, plus $`Q_\mathrm{lat}`$
and $`\delta_\mathrm{fr}`$ when those flags are on.

### 5.5 Zone

Dry-air capacitance at the design setpoint (default $`V=50\,\mathrm{m}^3`$):

```math
C_z=\rho_\mathrm{air}c_p V,\qquad
C_z\dot T_z=Q_\mathrm{zone}+Q_\mathrm{gain}+UA(T_\mathrm{out}-T_z).
```

At $`20^\circ\mathrm{C}`$, $`C_z\approx 60.6\,\mathrm{kJ/K}`$; at
$`24^\circ\mathrm{C}`$, $`\approx 59.8\,\mathrm{kJ/K}`$. Pass `V_zone`
or `C_zone` to change it. Furniture and walls are not included.

When the weather CSV is the complete load, $`UA=0`$ so the record is
not double-counted.

### 5.6 Humidity and frost (off by default)

Require user `RH_out` and, if moist, `RH_zone0`. Leaving-coil humidity
is saturation at refrigerant $`T`$ when that $`T`$ is below the local
dew point.

```math
Q_\mathrm{lat}=\dot m_a(W_\mathrm{in}-W_\mathrm{out})h_{fg},\qquad
\rho V\,\dot W_z=\dot m_{a,i}(W_\mathrm{coil,out}-W_z)+W_\mathrm{gain}.
```

Frost (requires moist) grows on the outdoor coil when the mean wall is
below freezing. The humidity sink is the wet-coil march (leaving humidity
is saturation at refrigerant $`T`$), not $`W_\mathrm{sat}(T_w)`$:

```math
\begin{aligned}
\dot m_\mathrm{fr}&=\dot m_{a,o}\max(W_\mathrm{in}-W_\mathrm{out},0)
\quad(T_w<273.15\,\mathrm{K}),\\
\delta&=m_\mathrm{fr}/(\rho_\mathrm{fr}A).
\end{aligned}
```

Hayashi (1977) density and Sanders (1974) conductivity
($`T_s`$ is wall temperature in °C, used as the frost-surface proxy;
$`\rho`$ in kg/m³):

```math
\rho_\mathrm{fr}=650\exp(0.277\,T_s),\qquad
k_\mathrm{fr}=0.001202\,\rho_\mathrm{fr}^{0.963}.
```

Closure `ice` uses IAPWS ice Ih at $`0^\circ\mathrm{C}`$
($`\rho=916.7\,\mathrm{kg/m}^3`$, $`k=2.22\,\mathrm{W/m\cdot K}`$).
Extra resistance $`r=\delta/k`$ sits in series with $`h_a`$. Melt only
if the defrost flag is set **and** $`W_\mathrm{defrost}>0`$.

### 5.7 Operating mode

Heating: outdoor $`\to`$ evap, indoor $`\to`$ cond. Cooling swaps.
`apply_operating_mode` copies the active pair onto `*_e` / `*_c`.
Mid-run reverse remaps state (coils keep inventory). The time-series `mode`
($`1`$=heat, $`0`$=cool) schedules the valve; without it, load sign
plus deadband and minimum dwell. The controller flips sign and clears
windup.

## 6. Design

Four-point CoolProp cycle: (1) dew + SH, (2) polytropic discharge
enthalpy matching the residual in section 5.2,
(3) bubble − SC, (4) $`h_3`$ at $`p_e`$. Pass `compression='heos'` to
use CoolProp $`h(p_c,s_1)`$ instead.

```math
T_e=T_{\mathrm{air},e}-\Delta T_\mathrm{evap},\qquad
T_c=T_{\mathrm{air},c}+\Delta T_\mathrm{cond}.
```

Defaults $`\Delta T_\mathrm{evap}=10\,\mathrm{K}`$,
$`\Delta T_\mathrm{cond}=12\,\mathrm{K}`$. Condensing above $`p_c`$ is
rejected. Useful duty sets $`\dot m`$ (condenser heat in heating,
evaporator heat in cooling).

```math
\begin{aligned}
V_\mathrm{disp}&=\frac{\dot m}{\eta_v\rho_1 N_\mathrm{design}},\\
A_\mathrm{max}&=\frac{\dot m}{C_d u_\mathrm{des}\sqrt{2\rho_3(p_c-p_e)}},\qquad
u_\mathrm{des}=0.40.
\end{aligned}
```

Tube count is iterated until ε-NTU $`Q`$ equals cycle duty. Air flow
from $`Q=\varepsilon\dot m_a c_p\Delta T`$. Wall capacitance
$`\rho_\mathrm{cu}c_{p,\mathrm{cu}}A t_w`$. Charge is Zivi / flashed
density on the design profile times internal volume.

Default $`n_e=n_c=6`$ then scales $`V_\mathrm{disp}`$ and $`A_\mathrm{eev}`$
so a short implicit-Euler settle of the residual (zone $`T`$ held) meets
$`Q_\mathrm{load}`$ at $`N_\mathrm{des}`$. That scale is a sizer–plant
consistency step, not a laboratory fit. Pass `match_plant=False` to keep
the algebraic inversion only. Coarse meshes (`n_e<6`) skip it unless
`match_plant=True`.

From a time series with no nameplate:

```math
Q_\mathrm{cool}=\max(Q_\mathrm{gain},0),\qquad
Q_\mathrm{heat}=\max(-Q_\mathrm{gain},0).
```

Design outdoor $`T`$ is the ambient **at that peak**, not the record
extreme. `mode=auto` becomes heating, cooling, or heat_pump from those
peaks. A reversible unit takes the harder compressor, EEV, and coils.

Off-design: a fixed machine is re-closed on a $`T_\mathrm{out}`$ grid
so $`\dot m\Delta h`$ equals ε-NTU $`Q`$. Balance point:
$`Q_\mathrm{cap}(T)-Q_\mathrm{load}(T)=0`$. Design-package SHR is
$`Q_\mathrm{sens}/Q_\mathrm{coil}`$, an output.
`examples/design.py` writes the package to `output/` (no transient).

Gates (raise `DesignGateError`): $`p_c\le f p_\mathrm{crit}`$,
$`T_\mathrm{disch}\le T_\mathrm{disch,max}`$, $`\Pi\le\Pi_\mathrm{max}`$,
SH in $`[4,10]\,\mathrm{K}`$, $`\mathrm{SC}\ge 0`$, optional $`G_\mathrm{max}`$.

## 7. Controllers

Inner loop: superheat EEV. Low SH closes the valve; speed feedforward
plus a slow PID trim, rate-limited.

**PID cascade.** Load feedforward then ISA trim (derivative on
measurement, back-calculation anti-windup). Integral $`\mathrm{d}t`$
clamped to $`5\,\mathrm{s}`$ so QSS ZOH does not wind up.

```math
Q_\mathrm{need}=\begin{cases}
UA(T_\mathrm{out}-T_\mathrm{sp})+Q_\mathrm{gain} & \text{cooling}\\
UA(T_\mathrm{sp}-T_\mathrm{out})-Q_\mathrm{gain} & \text{heating}
\end{cases}
```

```math
N_\mathrm{ff}=N_\mathrm{design}\,\mathrm{clip}(Q_\mathrm{need}/Q_\mathrm{ref},0,1.4).
```

```math
N=\mathrm{sat}\bigl(N_\mathrm{ff}+k_p e+\textstyle\int k_i e\,\mathrm{d}t-k_d\dot T_z\bigr),\qquad
e=T_\mathrm{sp}-T_z.
```

Gains scale with $`N_\mathrm{max}/70`$; signs flip in cooling.
`set_mode` flips signs and clears windup.

**Hysteresis.** On/off with deadband and min on/off times.
**Bang-bang.** Deadband, no timers.
**LMPC.** Implicit Euler on the residual: $`A=(I-\Delta t\,\partial f/\partial y)^{-1}`$,
then $`y_{k+1}=A y_k+B u_k+c`$ with $`B`$ and $`c`$ from `jacfwd`.
Decision $`(N,u_\mathrm{eev})`$.
**NMPC.** Implicit-Euler shooting of the residual.
`controller=auto` selects PID.

## 8. Time integration

TR-BDF2 (Hosea & Shampine, 1996), $`\gamma=2-\sqrt{2}`$: trapezoidal
stage of length $`\gamma h`$, then BDF2 completion to $`t+h`$.

```math
\begin{aligned}
y_{n+\gamma}-y_n&=\tfrac12\gamma h\bigl(f(t,y_n)+f(t+\gamma h,y_{n+\gamma})\bigr),\\
y_{n+1}&=a y_{n+\gamma}+b y_n+c h\,f(t+h,y_{n+1}),
\end{aligned}
```

```math
a=\frac{1}{\gamma(2-\gamma)},\quad
b=-\frac{(1-\gamma)^2}{\gamma(2-\gamma)},\quad
c=\frac{1-\gamma}{2-\gamma}.
```

Damped Newton with `jacfwd`. Embedded trapezoidal error; reject and cut,
or accept implicit Euler at $`\Delta t_\mathrm{min}`$. Default
$`\Delta t\in[5\,\mathrm{ms},\,8\,\mathrm{s}]`$.
States are projected onto the property table after each accepted step.

`reduction="qss"` (`auto` when $`t_\mathrm{final}\ge 3600\,\mathrm{s}`$):
short DAE warmup, then slow ODEs ($`T_z`$, optional $`W_z,m_\mathrm{fr}`$)
with periodic refrigerant relax. `u_of_t` uses **absolute** time.
`reduction="full"` keeps the DAE for the whole horizon.

## 9. Parameters

Defaults below are designer / controller defaults. Sized plants overwrite
geometry ($`V_\mathrm{disp}`$, tubes, $`A_\mathrm{eev}`$, $`C_z`$, $`UA`$).

### Design request

| Symbol / field | Default | Meaning |
|---|---|---|
| refrigerant | required | CoolProp HEOS name |
| mode | `heating` | `heating`, `cooling`, `heat_pump`, `auto` |
| $`T_z`$ | $`294.15\,\mathrm{K}`$ | zone setpoint |
| $`Q_\mathrm{heat},Q_\mathrm{cool}`$ | — | nameplate duties; else peak of the CSV |
| $`T_{\mathrm{out},\mathrm{heat}}`$ | $`273.15\,\mathrm{K}`$ | heating design outdoor |
| $`T_{\mathrm{out},\mathrm{cool}}`$ | $`308.15\,\mathrm{K}`$ | cooling design outdoor |
| $`\Delta T_\mathrm{evap},\Delta T_\mathrm{cond}`$ | $`10,12\,\mathrm{K}`$ | design approaches |
| SH, SC | $`6,4\,\mathrm{K}`$ | design superheat / subcool |
| $`N_\mathrm{design}`$ | $`50\,\mathrm{Hz}`$ | design compressor frequency |
| $`n_e,n_c`$ | $`6`$ | cells per coil |
| $`V`$ | $`50\,\mathrm{m}^3`$ | zone volume if `C_zone` omitted |
| indoor RH | $`0.50`$ | design-package psychrometrics only |
| voltage, phases | $`230\,\mathrm{V}`$, 1 | shaft current $`I=W/V`$ or $`W/(V\sqrt{3})`$ |
| $`\eta_\mathrm{motor}`$ | — | applied only if supplied |

### Feasibility / actuator limits

| Symbol | Default | Meaning |
|---|---|---|
| $`\mathrm{SH}_\mathrm{sp}`$ | $`6\,\mathrm{K}`$ | EEV superheat setpoint |
| $`\mathrm{SH}`$ band | $`[4,10]\,\mathrm{K}`$ | gate |
| $`\mathrm{SC}_\mathrm{min}`$ | $`0\,\mathrm{K}`$ | gate |
| $`T_\mathrm{disch,max}`$ | $`388.15\,\mathrm{K}`$ | gate |
| $`\Pi_\mathrm{max}`$ | $`7.5`$ | gate |
| $`f=p_c/p_\mathrm{crit}`$ | $`0.90`$ | gate |
| $`N_\mathrm{max}`$ | $`70\,\mathrm{Hz}`$ | compressor clip |
| $`u_\mathrm{eev}`$ band | $`[0.10,0.72]`$ | EEV clip |
| min on / off | $`60,90\,\mathrm{s}`$ | hysteresis |
| $`T_z`$ band | $`0.5\,\mathrm{K}`$ | hysteresis half-width scale |
| $`G_\mathrm{max}`$ | — | optional mass-flux cap |

### Plant geometry (overwritten by the sizer)

| Symbol | Typical role |
|---|---|
| $`D,L,n_\mathrm{tubes},\mathrm{fin}`$ | tube OD, length, count, air-area multiplier |
| $`V_h`$ | header volume |
| $`V_\mathrm{disp},C,\eta_\mathrm{is},\gamma`$ | displacement, clearance, isentropic efficiency, polytropic $`\gamma`$ |
| $`A_\mathrm{max},C_d`$ | EEV area and discharge coefficient |
| $`C_w`$ | coil wall thermal mass |
| $`C_z,UA`$ | zone capacitance and envelope conductance |
| $`\dot m_{a0},h_a,c_{p,a}`$ | design air flow, air HTC, air specific heat |
| $`W_\mathrm{defrost}`$ | electric defrost power; $`0`$ = no heater |
| frost closure | `hayashi` or `ice` |

### Integrator

| Symbol | Default | Meaning |
|---|---|---|
| rtol, atol | $`10^{-3},10^{-5}`$ | TR-BDF2 tolerances |
| $`\Delta t_\mathrm{min},\Delta t_\mathrm{max}`$ | $`5\,\mathrm{ms},8\,\mathrm{s}`$ | step bounds |
| record_dt | $`2\,\mathrm{s}`$ (QSS: $`30`$–$`60\,\mathrm{s}`$) | ZOH / output grid |
| QSS warmup | $`\sim 180\,\mathrm{s}`$ | full DAE before reduction |
| $`n_\mathrm{relax}`$ | $`12`$ | implicit-Euler cycle refreshes |

## 10. Limitations

Control-oriented plant, not a rating twin. HTCs are published
correlations. Charge is conserved, not fitted. Air is dry unless moist
is enabled. Use for first-cut sizing, controller tests, and transients
that moving-boundary models handle poorly. Do not use for AHRI/EN
claims, charge optimization against a real cabinet, or transcritical CO2.

Unfitted literature comparisons: [validation/](../validation/README.md).

## 11. References

- S. Bendapudi, J. E. Braun, and E. A. Groll, *Int. J. Refrigeration*, 2008.
- B. P. Rasmussen, *HVAC&R Research*, 2012.
- H. Qiao, V. Aute, and R. Radermacher, *Int. J. Refrigeration*, 2015.
- M. E. Hosea and L. F. Shampine, *Applied Numerical Mathematics*, 1996.
- S. M. Zivi, *J. Heat Transfer*, 1964.
- M. M. Shah, *Int. J. Heat Mass Transfer*, 1979 (cited for the two-phase
  multiplier family; the residual uses a simplified Shah-type factor on
  Dittus–Boelter, not the full 1979 correlation).
- I. H. Bell et al., *Ind. Eng. Chem. Res.*, 2014 (CoolProp).
- H. Ramírez, J. Jiménez-Cabas, and A. Bula, *Data in Brief*, 2019, doi:10.1016/j.dib.2019.104316.
- S. Ramaraj and B. Sparn, NLR Data Catalog, 2024, doi:10.7799/2440214.
- C.-Y. Lee, T. Cao, Y. Hwang, R. Radermacher, and S. Shaffer, *IOP Conf. Ser.: Mater. Sci. Eng.* 1180 (2021) 012041.
- Y. Hayashi et al., *J. Heat Transfer*, 1977.
- C. T. Sanders, Ph.D. thesis, Delft University of Technology, 1974.
