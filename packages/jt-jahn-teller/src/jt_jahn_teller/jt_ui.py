import numpy as np
from ipywidgets import VBox, HBox, Layout, Dropdown, FloatSlider, IntSlider, Accordion, Output, Checkbox, Label
import ipywidgets as widgets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
from IPython.display import display

plt.ioff()

def fresh_fig(figsize=(5,4)):
    fig = plt.figure(figsize=figsize)
    return fig

# --- Styling constants ---
FIG_DPI   = 120
TITLE_FS  = 16               # uniform titles
LABEL_FS  = 14               # base axis label size
TICK_FS   = 12               # base tick size

# Panel-specific font scales
P2_SCALE  = 1.15
P3_SCALE  = 1.11
P2_LABEL  = LABEL_FS * P2_SCALE
P2_TICK   = TICK_FS  * P2_SCALE
P3_LABEL  = LABEL_FS * P3_SCALE
P3_TICK   = TICK_FS  * P3_SCALE
P2_PER_LEVEL_LABEL = P2_LABEL * 0.90  # slightly smaller right-side labels

plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS
})

# --- Colors (CPK-inspired) ---
COLOR_C = '#4A4A4A'        # Carbon: dark gray
COLOR_H_FACE = '#FFFFFF'   # Hydrogen: white
COLOR_H_EDGE = '#222222'   # Hydrogen edge
COLOR_BOND_MC = '#6E6E6E'  # Metal–Carbon bond
COLOR_BOND_CH = '#BFBFBF'  # C–H bond


# --- Crystal-field model (exact minimization + optional soft-min smoothing) ---

ORBITALS = ['dx2-y2', 'dz2', 'dxy', 'dxz', 'dyz']

def crystal_field_energies(Dq, Ds, Dt):
    return {
        'dz2':    6*Dq - 2*Ds - 6*Dt,   # a1g
        'dx2-y2': 6*Dq + 2*Ds - Dt,     # b1g
        'dxy':   -4*Dq + 2*Ds - Dt,     # b2g
        'dxz':   -4*Dq - Ds + 4*Dt,     # eg
        'dyz':   -4*Dq - Ds + 4*Dt,     # eg
    }

def map_Q_to_DsDt(Q, ks=0.8, kt=-0.10):
    return ks*Q, kt*Q

def enumerate_occupancies(n_electrons):
    occs = []
    for o1 in range(3):
        for o2 in range(3):
            for o3 in range(3):
                for o4 in range(3):
                    for o5 in range(3):
                        if o1+o2+o3+o4+o5 == n_electrons:
                            occs.append([o1,o2,o3,o4,o5])
    return occs

def config_energy(occ_vec, energies, pairing_energy):
    E = 0.0
    n_pairs = 0
    for o, orb in zip(occ_vec, ORBITALS):
        E += o * energies[orb]
        if o == 2:
            n_pairs += 1
    return E + pairing_energy * n_pairs

def config_degeneracy(occ_vec):
    n_single = sum(1 for o in occ_vec if o == 1)
    return 2**n_single

def minimize_electronic_energy(energies, n_electrons, pairing_energy):
    best_E = None
    best_occ = None
    for occ in enumerate_occupancies(n_electrons):
        E = config_energy(occ, energies, pairing_energy)
        if (best_E is None) or (E < best_E - 1e-12):
            best_E = E
            best_occ = occ
    n_unpaired = sum(1 for o in best_occ if o == 1)
    occ_dict = {orb: o for orb, o in zip(ORBITALS, best_occ)}
    return occ_dict, best_E, n_unpaired

def softmin_electronic_energy(energies, n_electrons, pairing_energy, T_soft):
    """Stabilized soft-min using log-sum-exp trick to avoid overflow/underflow.
    Returns: -T * log(sum_g exp(-(E - E0)/T)) + E0, where E0=min(E) over configs.
    """
    if T_soft <= 0:
        return minimize_electronic_energy(energies, n_electrons, pairing_energy)[1]
    E_list = []
    g_list = []
    for occ in enumerate_occupancies(n_electrons):
        E = config_energy(occ, energies, pairing_energy)
        g = config_degeneracy(occ)
        E_list.append(E)
        g_list.append(g)
    if not E_list:
        return 0.0
    E0 = min(E_list)
    # compute Z' = sum g * exp(-(E - E0)/T)
    terms = [g*np.exp(-(E - E0)/max(T_soft, 1e-12)) for E, g in zip(E_list, g_list)]
    Zp = sum(terms) + 1e-300
    return E0 - max(T_soft, 1e-12) * np.log(Zp)

def total_energy_vs_Q(Q_grid, n_elec, Dq, P_pair, k_lattice, ks, kt, a3, a4, T_soft=0.0):
    E_tot = []
    for Q in Q_grid:
        Ds, Dt = map_Q_to_DsDt(Q, ks=ks, kt=kt)
        e = crystal_field_energies(Dq, Ds, Dt)
        if T_soft > 0:
            E_elec = softmin_electronic_energy(e, n_elec, P_pair, T_soft)
        else:
            E_elec = minimize_electronic_energy(e, n_elec, P_pair)[1]
        E_tot.append(E_elec + 0.5 * k_lattice * (Q**2) + a3*(Q**3) + a4*(Q**4))
    return np.array(E_tot)

def ion_to_dcount(label):
    mapping = {
        "Ti(III)  d1": 1,
        "V(III)   d2": 2,
        "Cr(III)  d3": 3,
        "Mn(III)  d4": 4,
        "Fe(III)  d5": 5,
        "Fe(II)   d6": 6,
        "Co(II)   d7": 7,
        "Co(III)  d6": 6,
        "Ni(II)   d8": 8,
        "Cu(II)   d9": 9,
        "Zn(II)  d10": 10,
        "Custom…": None,
    }
    return mapping.get(label, None)



# --- Panel 1: Atomic Structure ---

MOLECULE_SCALE = 2.0   # per user
VIEW_LIM = 1.5         # per user

def _norm(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n if n>1e-9 else 1.0)

def _perp_basis(v):
    v = _norm(v)
    ref = np.array([0,0,1.0]) if abs(v[2]) < 0.9 else np.array([1.0,0,0])
    u = np.cross(v, ref); u = _norm(u)
    w = np.cross(v, u);  w = _norm(w)
    return u, w

def _draw_small_cap(ax, center, vdir, ch_len, ch_tilt_deg):
    # Realistic tetrahedral-ish CH geometry:
    # theta = angle between CH bond and +vdir (outward axis)
    theta = np.deg2rad(ch_tilt_deg)
    ring_r = ch_len * np.sin(theta)
    out_s  = ch_len * np.cos(theta)
    n_subs = 3

    vhat = _norm(vdir)
    u, w = _perp_basis(vhat)
    ring_center = center + out_s * vhat
    for k in range(n_subs):
        ang = 2*np.pi * k / n_subs
        p = ring_center + ring_r * (np.cos(ang)*u + np.sin(ang)*w)
        ax.plot([center[0], p[0]], [center[1], p[1]], [center[2], p[2]],
                linewidth=1.5*MOLECULE_SCALE, color=COLOR_BOND_CH)
        ax.scatter([p[0]], [p[1]], [p[2]], s=95*(MOLECULE_SCALE**2),
                   facecolors=COLOR_H_FACE, edgecolors=COLOR_H_EDGE, linewidths=0.8)

def draw_panel1(Q, ch_len, ch_tilt_deg):
    fig = fresh_fig((5.8,5.6))
    ax = fig.add_subplot(111, projection='3d')
    r_eq = 1.0
    r_ax = 1.0 + Q
    donors = np.array([
        [ r_eq, 0.0, 0.0],
        [-r_eq, 0.0, 0.0],
        [ 0.0,  r_eq, 0.0],
        [ 0.0, -r_eq, 0.0],
        [ 0.0, 0.0,  r_ax],
        [ 0.0, 0.0, -r_ax],
    ])

    ax.scatter([0], [0], [0], s=260*(MOLECULE_SCALE**2))
    ax.scatter(donors[:,0], donors[:,1], donors[:,2], s=170*(MOLECULE_SCALE**2), c=COLOR_C)

    for p in donors:
        ax.plot([0,p[0]], [0,p[1]], [0,p[2]], linewidth=2.0*MOLECULE_SCALE, color=COLOR_BOND_MC)

    for p in donors:
        _draw_small_cap(ax, p, p - np.array([0.0,0.0,0.0]), ch_len, ch_tilt_deg)

    lim = VIEW_LIM
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    try: ax.set_box_aspect([1,1,1])
    except Exception: pass

    try: ax.set_axis_off()
    except Exception:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
        try: ax._axis3don = False
        except Exception: pass

    ax.set_title('Molecular Structure', fontsize=TITLE_FS, y=1.02)
    plt.show()



# --- Panel 2: Electronic Structure (stable arrows, fixed symmetry labels) ---
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import transforms

def _deoverlap_min_gap(sorted_items, gap):
    # Given items sorted by target y (descending), return y positions that
    # respect a minimum vertical gap without changing order.
    # Each item is (name, y0, label_text).
    ys = []
    prev = None
    for _, y0, _ in sorted_items:
        if prev is None:
            y = y0
        else:
            # enforce min gap while preserving order; top-to-bottom clamp
            y = min(y0, prev - gap)
        ys.append(y)
        prev = y
    return ys

def draw_panel2(energies, occ, n_unpaired,
                label_delta_factor=0.11, label_xoffset=0.012,
                arrow_half_factor=0.055, arrow_dx_pair=0.040):
    # Fallback fonts if globals not set
    global P2_LABEL, P2_TICK, P2_PER_LEVEL_LABEL, TITLE_FS
    try: P2_LABEL
    except NameError: P2_LABEL = 12
    try: P2_TICK
    except NameError: P2_TICK = 10
    try: P2_PER_LEVEL_LABEL
    except NameError: P2_PER_LEVEL_LABEL = 11
    try: TITLE_FS
    except NameError: TITLE_FS = 14

    # Colors (fallbacks)
    try: LEVEL_T2G
    except NameError: LEVEL_T2G = '#1f77b4'
    try: LEVEL_EG
    except NameError: LEVEL_EG = '#d62728'

    # Fixed horizontal positions for all degeneracies
    orbital_xbase = {
        'dx2-y2': 0.50,
        'dz2':    0.30,
        'dxy':    0.40,
        'dxz':    0.20,
        'dyz':    0.60,
    }

    # Taller figure so it matches/exceeds Panel 3 height
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    # Energetics
    order_desc = sorted(energies.items(), key=lambda kv: kv[1], reverse=True)
    yvals = [E for _, E in order_desc]
    y_min, y_max = min(yvals), max(yvals)
    yr = (y_max - y_min) if (y_max - y_min) > 1e-9 else 1.0

    # Asymmetric padding: more headroom at top so e_g doesn't crowd the frame
    pad_bot = 0.12 * yr
    pad_top = 0.22 * yr
    ax.set_ylim(y_min - pad_bot, y_max + pad_top)

    # If sliders exist, lock y-range across the Q range using same asymmetric padding
    try:
        Q_min = float(getattr(Q_sl, 'min', -0.8))
        Q_max = float(getattr(Q_sl, 'max',  0.8))
        ks_val = float(ks_sl.value) if 'ks_sl' in globals() else 0.8
        kt_val = float(kt_sl.value) if 'kt_sl' in globals() else -0.10
        Dq_val = float(Dq_sl.value) if 'Dq_sl' in globals() else 1.2
        def cf_energies(Dq, Q):
            Ds = ks_val * Q; Dt = kt_val * Q
            return {
                'dz2':     6*Dq - 2*Ds - 6*Dt,
                'dx2-y2':  6*Dq + 2*Ds - 1*Dt,
                'dxy':    -4*Dq + 2*Ds - 1*Dt,
                'dxz':    -4*Dq - 1*Ds + 4*Dt,
                'dyz':    -4*Dq - 1*Ds + 4*Dt,
            }
        e_min = cf_energies(Dq_val, Q_min); e_max = cf_energies(Dq_val, Q_max)
        y_floor = min(min(e_min.values()), min(e_max.values()))
        y_ceil  = max(max(e_min.values()), max(e_max.values()))
        yr_full = max(y_ceil - y_floor, 1e-6)
        pad_bot = 0.12 * yr_full
        pad_top = 0.22 * yr_full
        ax.set_ylim(y_floor - pad_bot, y_ceil + pad_top)
        yr = (y_ceil - y_floor) if (y_ceil - y_floor) > 1e-9 else 1.0
    except Exception:
        pass

    # X range and label anchor
    x_left, x_right = 0.10, 0.70
    xlim_max = 1.16
    ax.set_xlim(-0.05, xlim_max)
    x_label = min(x_right + float(label_xoffset), xlim_max - 0.02)

    # Level lines (colored by family)
    ORBITAL_GROUP = {'dx2-y2':'eg', 'dz2':'eg', 'dxy':'t2g', 'dxz':'t2g', 'dyz':'t2g'}
    for orb, E in order_desc:
        c = LEVEL_EG if ORBITAL_GROUP[orb] == 'eg' else LEVEL_T2G
        ax.plot([x_left, x_right], [E, E], color=c, linewidth=2.0)

    # Right-side labels with minimum vertical gap; dxz/dyz combined
    base_delta = float(label_delta_factor) * yr
    candidates = [
        ('dx2-y2', energies['dx2-y2'], r'$d_{x^2-y^2}\,(b_{1g})$'),
        ('dz2',    energies['dz2'],    r'$d_{z^2}\,(a_{1g})$'),
        ('dxy',    energies['dxy'],    r'$d_{xy}\,(b_{2g})$'),
        ('dxz_dyz', energies['dxz'],   r'$d_{xz},\,d_{yz}\,(e_g)$'),
    ]
    candidates.sort(key=lambda t: t[1], reverse=True)  # top to bottom
    y_clamped = _deoverlap_min_gap(candidates, base_delta)
    for (name, y0, text), ylab in zip(candidates, y_clamped):
        ax.text(x_label, ylab, text, va='center', fontsize=P2_PER_LEVEL_LABEL, clip_on=True)

    # Brackets and family labels at left
    Et = [energies['dxy'], energies['dxz'], energies['dyz']]
    Ee = [energies['dz2'],  energies['dx2-y2']]
    y0_t, y1_t = min(Et), max(Et)
    y0_e, y1_e = min(Ee), max(Ee)
    x_br, tick = x_left - 0.015, 0.015
    # t2g (blue)
    ax.plot([x_br, x_br], [y0_t, y1_t], color=LEVEL_T2G, linewidth=1.2)
    ax.plot([x_br, x_br+tick], [y0_t, y0_t], color=LEVEL_T2G, linewidth=1.2)
    ax.plot([x_br, x_br+tick], [y1_t, y1_t], color=LEVEL_T2G, linewidth=1.2)
    ax.text(x_br - 0.006, 0.5*(y0_t+y1_t), r'$t_{2g}$', ha='right', va='center',
            fontsize=P2_LABEL, color=LEVEL_T2G, clip_on=True)
    # eg (red)
    ax.plot([x_br, x_br], [y0_e, y1_e], color=LEVEL_EG, linewidth=1.2)
    ax.plot([x_br, x_br+tick], [y0_e, y0_e], color=LEVEL_EG, linewidth=1.2)
    ax.plot([x_br, x_br+tick], [y1_e, y1_e], color=LEVEL_EG, linewidth=1.2)
    ax.text(x_br - 0.006, 0.5*(y0_e+y1_e), r'$e_g$', ha='right', va='center',
            fontsize=P2_LABEL, color=LEVEL_EG, clip_on=True)

    # Electron arrows (stable positions)
    half   = float(arrow_half_factor) * yr
    dxpair = float(arrow_dx_pair)
    for (orb, E) in order_desc:
        n = occ.get(orb, 0)
        if n == 0:
            continue
        xb = orbital_xbase.get(orb, 0.40)
        if n == 1:
            ax.annotate('', xy=(xb, E + half), xytext=(xb, E - half),
                        arrowprops=dict(arrowstyle='-|>', lw=1.2, color='black'))
        elif n == 2:
            xL, xR = xb - dxpair/2.0, xb + dxpair/2.0
            ax.annotate('', xy=(xL, E + half), xytext=(xL, E - half),
                        arrowprops=dict(arrowstyle='-|>', lw=1.2, color='black'))
            ax.annotate('', xy=(xR, E - half), xytext=(xR, E + half),
                        arrowprops=dict(arrowstyle='-|>', lw=1.2, color='black'))

    # Axes cosmetics
    ax.set_yticks(sorted(set([round(v,3) for v in energies.values()])))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks([])
    ax.set_ylabel('Energy (eV)', fontsize=P2_LABEL)
    ax.set_xlabel(f'Levels + Spins (unpaired = {n_unpaired})', fontsize=P2_LABEL)
    ax.tick_params(axis='both', labelsize=P2_TICK)
    ax.set_title('Electronic Structure', fontsize=TITLE_FS, y=1.02)

    # Symmetry labels fixed inside frame at y=0.92
    bt = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(x_br,    0.92, r'$O_h$',    transform=bt, ha='center', va='bottom', fontsize=P2_LABEL, fontweight='bold')
    ax.text(x_label, 0.92, r'$D_{4h}$', transform=bt, ha='center', va='bottom', fontsize=P2_LABEL, fontweight='bold')

    plt.show()



# --- Panel 3: Stability (match Panel 2 height) ---

def draw_panel3(Q_current, n_elec, Dq, P_pair, k_lattice, ks, kt, a3, a4,
                smooth_enabled=False, T_soft=0.02, Q_span=(-0.6, 0.6), N=241):
    fig = fresh_fig((6.0, 5.5))  # match Panel 2 size
    ax = fig.add_subplot(111)
    Q_grid = np.linspace(Q_span[0], Q_span[1], N)
    E_curve = total_energy_vs_Q(Q_grid, n_elec, Dq, P_pair, k_lattice, ks, kt, a3, a4,
                                T_soft if smooth_enabled else 0.0)
    ax.plot(Q_grid, E_curve)

    Ds, Dt = map_Q_to_DsDt(Q_current, ks=ks, kt=kt)
    e = crystal_field_energies(Dq, Ds, Dt)
    if smooth_enabled and T_soft > 0:
        E_elec = softmin_electronic_energy(e, n_elec, P_pair, T_soft)
    else:
        E_elec = minimize_electronic_energy(e, n_elec, P_pair)[1]
    E_now = E_elec + 0.5 * k_lattice * (Q_current**2) + a3*(Q_current**3) + a4*(Q_current**4)
    ax.plot(Q_current, E_now, marker='o')

    ax.set_xlabel('Distortion Q', fontsize=P3_LABEL)
    ax.set_ylabel('Total energy E(Q) (eV)', fontsize=P3_LABEL)
    ax.tick_params(axis='both', labelsize=P3_TICK)
    ax.set_title('Stability', fontsize=TITLE_FS, y=1.02)
    plt.show()



def make_dashboard():
    """Construct the identical interactive UI and return a VBox.
    Mirrors the current notebook layout (header, row1, Accordion, panels, preset buttons).
    """
    global ion_dd, dcount_sl, Q_sl, Dq_sl, P_sl, k_sl, ks_sl, kt_sl, a3_sl, a4_sl, smooth_chk, Tsoft_sl, ch_len_sl, ch_tilt_sl, arrow_len_sl, arrow_dx_sl, label_dy_sl, label_dx_sl, out1, out2, out3
    # --- Dashboard wiring ---

    ion_dd = Dropdown(
        options=["Cu(II)   d9", "Mn(III)  d4", "Ti(III)  d1", "V(III)   d2", "Cr(III)  d3",
                 "Fe(III)  d5", "Fe(II)   d6", "Co(II)   d7", "Co(III)  d6", "Ni(II)   d8",
                 "Zn(II)  d10", "Custom…"],
        value="Cu(II)   d9", description="Ion:", layout=Layout(width='280px')
    )
    dcount_sl = IntSlider(value=9, min=0, max=10, step=1, description='d electrons:', layout=Layout(width='330px'))

    Q_sl  = FloatSlider(value=0.25, min=-0.6, max=0.6, step=0.01, description='Q:', layout=Layout(width='320px'))
    Dq_sl = FloatSlider(value=1.2, min=0.0, max=3.0, step=0.05, description='10Dq (eV):', layout=Layout(width='340px'))
    P_sl  = FloatSlider(value=0.8, min=0.0, max=3.0, step=0.05, description='Pairing P (eV):', layout=Layout(width='340px'))
    k_sl  = FloatSlider(value=2.0, min=0.0, max=6.0, step=0.1, description='Lattice k:', layout=Layout(width='320px'))

    # Advanced mapping
    ks_sl = FloatSlider(value=0.8,  min=-2.0, max= 2.0, step=0.05, description='k_s (Ds/Q):', layout=Layout(width='300px'))
    kt_sl = FloatSlider(value=-0.10, min=-2.0, max= 2.0, step=0.05, description='k_t (Dt/Q):', layout=Layout(width='300px'))

    # Advanced energy shaping
    a3_sl = FloatSlider(value=-0.10, min=-3.0, max=3.0, step=0.01, description='a3 · Q^3:', layout=Layout(width='300px'))
    a4_sl = FloatSlider(value= 1.20, min= 0.0, max=8.0, step=0.05, description='a4 · Q^4:', layout=Layout(width='300px'))
    smooth_chk = Checkbox(value=True, description='Smooth energy (soft-min)')
    Tsoft_sl   = FloatSlider(value=0.03, min=0.0, max=0.15, step=0.005, description='T (eV):', readout=True, layout=Layout(width='300px'))

    # New: Visual tuning
    ch_len_sl   = FloatSlider(value=0.25*MOLECULE_SCALE, min=0.18*MOLECULE_SCALE, max=0.40*MOLECULE_SCALE, step=0.01, description='C–H length', layout=Layout(width='300px'))
    ch_tilt_sl  = FloatSlider(value=70.53, min=40.0, max=85.0, step=0.1, description='CH tilt (°)', layout=Layout(width='300px'))
    arrow_len_sl= FloatSlider(value=0.055, min=0.030, max=0.120, step=0.002, description='Arrow half (×ΔE)', layout=Layout(width='300px'))
    arrow_dx_sl = FloatSlider(value=0.040, min=0.020, max=0.080, step=0.002, description='Arrow Δx', layout=Layout(width='300px'))
    label_dy_sl = FloatSlider(value=0.11,  min=0.06,  max=0.18,  step=0.005, description='Label Δy (×ΔE)', layout=Layout(width='300px'))
    label_dx_sl = FloatSlider(value=0.012, min=0.006, max=0.040, step=0.001, description='Label x offset', layout=Layout(width='300px'))

    adv_map_box    = VBox([ks_sl, kt_sl])
    adv_energy_box = VBox([a3_sl, a4_sl, smooth_chk, Tsoft_sl])
    adv_visual_box = VBox([ch_len_sl, ch_tilt_sl, Label('— Spins —'), arrow_len_sl, arrow_dx_sl, Label('— Labels —'), label_dy_sl, label_dx_sl])

    acc = Accordion(children=[adv_map_box, adv_energy_box, adv_visual_box])
    acc.set_title(0, 'Advanced: mapping Q → Ds, Dt')
    acc.set_title(1, 'Advanced: energy shaping + smoothing')
    acc.set_title(2, 'Advanced: visual tuning (CH3, arrows, labels)')
    acc.layout = Layout(margin='0 0 8px 0')  # small spacer below Advanced

    # Output widgets per user
    out1 = Output(layout=Layout(width='360px', margin='0 6px 0 0'))
    out2 = Output(layout=Layout(width='530px', margin='0 30px 0 6px'))
    out3 = Output(layout=Layout(width='510px', margin='0 0 0 0'))

    def on_controls_change(*args):
        d_from_ion = ion_to_dcount(ion_dd.value)
        if d_from_ion is not None:
            dcount_sl.disabled = True
            dcount_sl.value = d_from_ion
        else:
            dcount_sl.disabled = False

        Dq = Dq_sl.value
        Ds, Dt = map_Q_to_DsDt(Q_sl.value, ks=ks_sl.value, kt=kt_sl.value)
        e = crystal_field_energies(Dq, Ds, Dt)
        occ, E_elec, n_unp = minimize_electronic_energy(e, dcount_sl.value, P_sl.value)

        with out1:
            out1.clear_output(wait=True)
            draw_panel1(Q_sl.value, ch_len_sl.value, ch_tilt_sl.value)
        with out2:
            out2.clear_output(wait=True)
            draw_panel2(e, occ, n_unp,
                        label_delta_factor=label_dy_sl.value,
                        label_xoffset=label_dx_sl.value,
                        arrow_half_factor=arrow_len_sl.value,
                        arrow_dx_pair=arrow_dx_sl.value)
        with out3:
            out3.clear_output(wait=True)
            draw_panel3(Q_sl.value, dcount_sl.value, Dq_sl.value, P_sl.value,
                        k_sl.value, ks_sl.value, kt_sl.value, a3_sl.value, a4_sl.value,
                        smooth_enabled=smooth_chk.value, T_soft=Tsoft_sl.value)

    for w in [ion_dd, dcount_sl, Q_sl, Dq_sl, P_sl, k_sl, ks_sl, kt_sl, a3_sl, a4_sl, smooth_chk, Tsoft_sl,
              ch_len_sl, ch_tilt_sl, arrow_len_sl, arrow_dx_sl, label_dy_sl, label_dx_sl]:
        w.observe(on_controls_change, names='value')

    # Initial render
    on_controls_change()

    header = HBox([ion_dd, dcount_sl], layout=Layout(align_items='center', justify_content='flex-start'))
    row1   = HBox([Q_sl, Dq_sl, P_sl, k_sl], layout=Layout(flex_flow='row wrap', justify_content='flex-start', gap='8px'))
    panels = HBox([out1, out2, out3], layout=Layout(justify_content='flex-start', align_items='flex-start', gap='0px', margin='6px 0 0 0'))


    # --- Presets and Reset Button ---

    def apply_weak_field():
        Dq_sl.value = 0.50
        P_sl.value = 1.50
        k_sl.value = 2.0

    def apply_strong_field():
        Dq_sl.value = 2.20
        P_sl.value = 0.30
        k_sl.value = 2.0

    def reset_parameters():
        Dq_sl.value = 1.20
        P_sl.value = 0.80
        k_sl.value = 2.0
        ks_sl.value = 0.80
        kt_sl.value = -0.10
        a3_sl.value = -0.10
        a4_sl.value = 1.20
        Tsoft_sl.value = 0.03
        smooth_chk.value = True
        Q_sl.value = 0.25
        ion_dd.value = "Cu(II)   d9"

    weak_btn = widgets.Button(description="Weak-Field Ligand", button_style='info')
    strong_btn = widgets.Button(description="Strong-Field Ligand", button_style='warning')
    reset_btn = widgets.Button(description="Reset Parameters", button_style='success')

    weak_btn.on_click(lambda b: apply_weak_field())
    strong_btn.on_click(lambda b: apply_strong_field())
    reset_btn.on_click(lambda b: reset_parameters())

    preset_controls = HBox([weak_btn, strong_btn, reset_btn], layout=Layout(margin='8px 0 0 0'))
    container = VBox([header, row1, acc, panels])
    container = VBox([container, preset_controls])
    return container

