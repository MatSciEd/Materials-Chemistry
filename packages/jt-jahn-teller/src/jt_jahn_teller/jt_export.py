
# --- MP4 Exporter (robust v17) ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FFMpegWriter
from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

FONT_SCALE = 1.15   # +15% fonts
TITLE_Y    = 1.02   # common title baseline
WSPACE     = 0.40   # inter-panel spacing

LEVEL_T2G = '#1f77b4'
LEVEL_EG  = '#d62728'

ORBITAL_LATEX = {'dx2-y2': r'd_{x^2-y^2}', 'dz2': r'd_{z^2}', 'dxy': r'd_{xy}', 'dxz': r'd_{xz}', 'dyz': r'd_{yz}'}
ORBITAL_GROUP = {'dx2-y2':'eg', 'dz2':'eg', 'dxy':'t2g', 'dxz':'t2g', 'dyz':'t2g'}

# ---- Exact occupancy enumeration for consistency with Panel 3 ----
ORBITALS = ['dx2-y2','dz2','dxy','dxz','dyz']

def enumerate_occupancies(n):
    occs = []
    # All 5 orbitals, values 0..2, sum == n
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    e = n - (a+b+c+d)
                    if 0 <= e <= 2:
                        occs.append((a,b,c,d,e))
    return occs

def config_energy(occ, energies, P_pair):
    E = 0.0; pairs = 0
    for o, orb in zip(occ, ORBITALS):
        E += energies[orb] * o
        if o >= 2:
            pairs += 1
    return E + P_pair * pairs

def config_degeneracy(occ):
    up = sum(1 for o in occ if o == 1)
    return 2**up

def minimize_electronic_energy(energies, n_electrons, pairing_energy):
    best_E = float('inf'); best_occ = None
    for occ in enumerate_occupancies(int(n_electrons)):
        E = config_energy(occ, energies, pairing_energy)
        if E < best_E:
            best_E, best_occ = E, occ
    occ_dict = {orb:o for o, orb in zip(best_occ, ORBITALS)}
    n_unp = sum(1 for o in best_occ if o == 1)
    return occ_dict, best_E, n_unp

def softmin_electronic_energy(energies, n_electrons, pairing_energy, T_soft):
    if T_soft <= 0:
        return minimize_electronic_energy(energies, n_electrons, pairing_energy)[1]
    E_list = []; g_list = []
    for occ in enumerate_occupancies(int(n_electrons)):
        E_list.append(config_energy(occ, energies, pairing_energy))
        g_list.append(config_degeneracy(occ))
    if not E_list:
        return 0.0
    E0 = min(E_list)
    T = max(T_soft, 1e-12)
    Zp = sum(g*np.exp(-(E - E0)/T) for E,g in zip(E_list, g_list)) + 1e-300
    return E0 - T*np.log(Zp)

def _norm(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v/(n if n > 1e-9 else 1.0)

def _perp_basis(v):
    v = _norm(v)
    ref = np.array([0,0,1.0]) if abs(v[2]) < 0.9 else np.array([1.0,0,0])
    u = _norm(np.cross(v, ref))
    w = _norm(np.cross(v, u))
    return u, w

def _cf_energies(Dq, ks, kt, Q):
    Ds = ks * Q
    Dt = kt * Q
    return {
        'dz2':     6*Dq - 2*Ds - 6*Dt,
        'dx2-y2':  6*Dq + 2*Ds - 1*Dt,
        'dxy':    -4*Dq + 2*Ds - 1*Dt,
        'dxz':    -4*Dq - 1*Ds + 4*Dt,
        'dyz':    -4*Dq - 1*Ds + 4*Dt,
    }

def _fill_occupancy_aufbau_hund(e, dcount, P, tol=None):
    # Group energies with tolerance (relative to span)
    Es = list(e.values())
    span = max(Es) - min(Es) if Es else 0.0
    if tol is None:
        tol = max(1e-5, 1e-3*span)
    levels = sorted(e.items(), key=lambda kv: kv[1])  # ascending
    groups = []
    for orb, E in levels:
        if not groups or abs(E - groups[-1][0][1]) > tol:
            groups.append([(orb, E)])
        else:
            groups[-1].append((orb, E))

    occ = {k:0 for k in e}
    total = 0.0
    rem = int(dcount)

    # First pass: singly-occupy within each group (maximize spin)
    for grp in groups:
        if rem <= 0: break
        m = min(len(grp), rem)
        for i in range(m):
            orb = grp[i][0]
            occ[orb] += 1
            total += e[orb]
        rem -= m

    # Second pass: add second electrons (pairing) bottom -> top
    for grp in groups:
        if rem <= 0: break
        for i in range(len(grp)):
            if rem <= 0: break
            orb = grp[i][0]
            if occ[orb] >= 1:
                occ[orb] += 1
                total += e[orb] + P
                rem -= 1

    n_unp = sum(1 for v in occ.values() if v == 1)
    return total, occ, n_unp


def total_energy_and_occ(Dq, P, k, ks, kt, a3, a4, dcount, Q):
    # Prefer using the same mapping functions as Panel 3 if available
    cf = globals().get('crystal_field_energies', None)
    mapfn = globals().get('map_Q_to_DsDt', None)
    if callable(mapfn) and callable(cf):
        Ds, Dt = mapfn(Q, ks=ks, kt=kt)
        e = cf(Dq, Ds, Dt)
    else:
        Ds = ks * Q; Dt = kt * Q
        e = {'dz2': 6*Dq - 2*Ds - 6*Dt,
             'dx2-y2': 6*Dq + 2*Ds - Dt,
             'dxy': -4*Dq + 2*Ds - Dt,
             'dxz': -4*Dq - Ds + 4*Dt,
             'dyz': -4*Dq - Ds + 4*Dt}
    # Exact minimization for occupancy to match interactive panel
    occ_dict, E_el, n_unp = minimize_electronic_energy(e, dcount, P)
    E_lat = a3*Q**3 + a4*Q**4 + k*(Q**2)
    return E_el + E_lat, e, occ_dict, n_unp


def render_atomic(ax, Q, *, metal_color="#3A6EA5", atom_s_factor=0.56, bond_w_factor=0.80,
                  ion_label=None, font_scale=FONT_SCALE, title_y=TITLE_Y):
    ax.cla()
    r_eq, r_ax = 1.0, 1.0 + Q
    donors = np.array([[ r_eq,0,0],[-r_eq,0,0],[0, r_eq,0],[0,-r_eq,0],[0,0, r_ax],[0,0,-r_ax]]) * 1.15
    scale = 2.0
    metal_s    = 260*(scale**2) * 0.65 * atom_s_factor
    carbon_s   = 170*(scale**2) * 0.65 * atom_s_factor
    hydrogen_s =  95*(scale**2) * 0.65 * atom_s_factor
    w_mc       = 2.0*scale * 0.90 * bond_w_factor
    w_ch       = 1.5*scale * 0.90 * bond_w_factor
    COLOR_C = "#444444"; COLOR_BOND_MC = "#888888"; COLOR_BOND_CH = "#aaaaaa"
    COLOR_H_FACE = "#ffffff"; COLOR_H_EDGE = "#999999"

    ax.scatter([0],[0],[0], s=metal_s, c=metal_color, alpha=0.98)
    ax.scatter(donors[:,0], donors[:,1], donors[:,2], s=carbon_s, c=COLOR_C, alpha=0.98)
    for p in donors:
        ax.plot([0,p[0]],[0,p[1]],[0,p[2]], linewidth=w_mc, color=COLOR_BOND_MC, alpha=0.98)

    theta = np.deg2rad(35)
    ring_r = 0.55*np.sin(theta); out_s = 0.55*np.cos(theta)
    for c in donors:
        vhat = _norm(c); u,w = _perp_basis(vhat); center = c + out_s*vhat
        for kidx in range(3):
            ang = 2*np.pi*kidx/3
            p = center + ring_r*(np.cos(ang)*u + np.sin(ang)*w)
            ax.plot([c[0],p[0]],[c[1],p[1]],[c[2],p[2]], linewidth=w_ch, color=COLOR_BOND_CH, alpha=0.98)
            ax.scatter([p[0]],[p[1]],[p[2]], s=hydrogen_s, facecolors=COLOR_H_FACE, edgecolors=COLOR_H_EDGE, linewidths=0.8, alpha=0.98)

    lim = 1.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    try: ax.set_box_aspect([1,1,1])
    except Exception: pass
    try: ax.set_axis_off()
    except Exception:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Title and ion label
    ax.set_title('Molecular Structure', fontsize=int(14*font_scale), y=1.175*title_y)
    if ion_label:
        ax.text2D(0.5, 1.05, str(ion_label), transform=ax.transAxes,
                  ha='center', va='top', fontsize=int(14*font_scale*0.90))

def render_electronic(ax, e, occ, n_unp, *, xlim_max=1.16, label_x_offset=0.024, label_dy_factor=0.11,
                      p2_label_scale=0.90, p2_tick_scale=0.90, p2_perlevel_scale=0.90, font_scale=FONT_SCALE):
    ax.cla()
    P2_LABEL = int(12*font_scale); P2_TICK = int(10*font_scale); P2_PER = int(11*font_scale)
    x_left, x_right = 0.10, 0.70
    order_desc = sorted(e.items(), key=lambda kv: kv[1], reverse=True)
    yvals = [E for _, E in order_desc]
    y_min, y_max = min(yvals), max(yvals)
    yr = (y_max - y_min) if (y_max - y_min) > 1e-9 else 1.0
    pad_bot = 0.12 * yr; pad_top = 0.22 * yr
    ax.set_ylim(y_min - pad_bot, y_max + pad_top)
    ax.set_xlim(-0.05, xlim_max)
    x_label = min(x_right + float(label_x_offset), xlim_max - 0.02)

    for orb, E in order_desc:
        c = LEVEL_EG if ORBITAL_GROUP[orb]=='eg' else LEVEL_T2G
        ax.plot([x_left, x_right], [E, E], color=c, linewidth=2.0)

    base_delta = float(label_dy_factor) * yr
    candidates = [
        ('dx2-y2', e['dx2-y2'], r'$d_{x^2-y^2}\,(b_{1g})$'),
        ('dz2',    e['dz2'],    r'$d_{z^2}\,(a_{1g})$'),
        ('dxy',    e['dxy'],    r'$d_{xy}\,(b_{2g})$'),
        ('dxz_dyz', e['dxz'],   r'$d_{xz},\,d_{yz}\,(e_g)$'),
    ]
    candidates.sort(key=lambda t: t[1], reverse=True)
    y_adj = []; prev = None
    for _, y0, _ in candidates:
        y = y0 if prev is None else min(y0, prev - base_delta)
        y_adj.append(y); prev = y
    for (_, _, txt), ylab in zip(candidates, y_adj):
        ax.text(x_label, ylab, txt, va='center', fontsize=int(P2_PER*p2_perlevel_scale), clip_on=True, ha='left')

    Et = [e['dxy'], e['dxz'], e['dyz']]; Ee = [e['dz2'], e['dx2-y2']]
    y0_t, y1_t = min(Et), max(Et); y0_e, y1_e = min(Ee), max(Ee)
    x_br, tick = x_left - 0.015, 0.015
    ax.plot([x_br, x_br], [y0_t, y1_t], color=LEVEL_T2G, linewidth=1.2)
    ax.plot([x_br, x_br+tick], [y0_t, y0_t], color=LEVEL_T2G, linewidth=1.2)
    ax.plot([x_br, x_br+tick], [y1_t, y1_t], color=LEVEL_T2G, linewidth=1.2)
    ax.text(x_br - 0.006, 0.5*(y0_t+y1_t), r'$t_{2g}$', ha='right', va='center',
            fontsize=int(P2_LABEL*p2_label_scale), color=LEVEL_T2G, clip_on=True)
    ax.plot([x_br, x_br], [y0_e, y1_e], color=LEVEL_EG, linewidth=1.2)
    ax.plot([x_br, x_br+tick], [y0_e, y0_e], color=LEVEL_EG, linewidth=1.2)
    ax.plot([x_br, x_br+tick], [y1_e, y1_e], color=LEVEL_EG, linewidth=1.2)
    ax.text(x_br - 0.006, 0.5*(y0_e+y1_e), r'$e_g$', ha='right', va='center',
            fontsize=int(P2_LABEL*p2_label_scale), color=LEVEL_EG, clip_on=True)

    half   = 0.055 * yr
    dxpair = 0.040
    xbase = {'dx2-y2':0.50,'dz2':0.30,'dxy':0.40,'dxz':0.20,'dyz':0.60}
    for orb, E in order_desc:
        n = occ.get(orb, 0)
        xb = xbase.get(orb, 0.40)
        if n == 1:
            ax.annotate('', xy=(xb, E + half), xytext=(xb, E - half), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='black'))
        elif n == 2:
            xL, xR = xb - dxpair/2.0, xb + dxpair/2.0
            ax.annotate('', xy=(xL, E + half), xytext=(xL, E - half), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='black'))
            ax.annotate('', xy=(xR, E - half), xytext=(xR, E + half), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='black'))

    ax.set_yticks(sorted(set([round(v,3) for v in e.values()])))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks([])
    ax.set_ylabel('Energy (eV)', fontsize=int(P2_LABEL*p2_label_scale))
    ax.set_xlabel(f'Levels + Spins (unpaired = {n_unp})', fontsize=int(P2_LABEL*p2_label_scale))
    ax.tick_params(axis='both', labelsize=int(P2_TICK*p2_tick_scale))
    ax.set_title('Electronic Structure', fontsize=int(14*font_scale), y=TITLE_Y)

    bt = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(x_right + float(label_x_offset), 0.92, r'$D_{4h}$', transform=bt, ha='left',
            fontsize=int(P2_LABEL*p2_label_scale), fontweight='bold')
    ax.text(x_br, 0.92, r'$O_h$', transform=bt, ha='center',
            fontsize=int(P2_LABEL*p2_label_scale), fontweight='bold')

def render_stability(ax, Qs, E_curve, q_now, E_now, *, font_scale=FONT_SCALE, title_y=TITLE_Y):
    ax.cla()
    ax.plot(Qs, E_curve, lw=2.0)
    ax.plot([q_now], [E_now], 'o', ms=6)
    ax.set_xlabel('Distortion Q', fontsize=int(12*font_scale))
    ax.set_ylabel('Total Energy (arb.)', fontsize=int(12*font_scale))
    ax.set_title('Stability', fontsize=int(14*font_scale), y=title_y)

def _read_params_from_widgets():
    # Attempt to pull current UI state; fallback to defaults
    try:
        par = dict(Dq=float(Dq_sl.value), P=float(P_sl.value), k=float(k_sl.value),
                   ks=float(ks_sl.value), kt=float(kt_sl.value),
                   a3=float(a3_sl.value), a4=float(a4_sl.value),
                   dcount=int(dcount_sl.value))
        Q_min, Q_max = float(Q_sl.min), float(Q_sl.max)
        # Ion label
        ion_label = None
        for cand in ['ion_dd','ion_sel','ion_dropdown','metal_ion_dd','tm_dd']:
            if cand in globals():
                ion_label = globals()[cand].value
                break
        return par, Q_min, Q_max, ion_label
    except Exception:
        # Safe defaults
        par = dict(Dq=1.20, P=0.80, k=2.0, ks=0.80, kt=-0.10, a3=-0.10, a4=1.20, dcount=9)
        return par, -0.6, 0.6, None

def export_jt_video(path, include_atomic=True, fps=30, seconds=15,
                                   xlim_max=1.16, label_x_offset=0.024, label_dy_factor=0.11,
                                   p2_label_scale=0.90, p2_tick_scale=0.90, p2_perlevel_scale=0.90,
                                   metal_color="#3A6EA5", atom_s_factor=0.56, bond_w_factor=0.80,
                                   wspace=WSPACE, font_scale=FONT_SCALE, title_y=TITLE_Y):
    par, Q_min, Q_max, ion_label = _read_params_from_widgets()
    nframes = int(fps * seconds)
    Qs = np.linspace(Q_min, Q_max, nframes)
    Qs_curve = np.linspace(Q_min, Q_max, 400)
    smooth_enabled = False; T_soft = 0.0
    try:
        smooth_enabled = bool(smooth_chk.value)
        T_soft = float(Tsoft_sl.value)
    except Exception:
        pass
    E_curve = _energy_curve_from_params(par, Qs_curve, smooth_enabled=smooth_enabled, T_soft=T_soft)

    fig = plt.figure(figsize=(16.5, 5.1), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0, 0], projection='3d') if include_atomic else None
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    fig.subplots_adjust(wspace=wspace)

    writer = FFMpegWriter(fps=fps, metadata=dict(artist='JT-Notebook'), bitrate=2400)
    with writer.saving(fig, path, dpi=160):
        for q in Qs:
            E_now, e, occ, n_unp = total_energy_and_occ(**par, Q=q)
            if include_atomic:
                render_atomic(ax1, q, metal_color=metal_color, atom_s_factor=atom_s_factor, bond_w_factor=bond_w_factor,
                              ion_label=(ion_label if ion_label else f'd^{par.get("dcount", 0)}'),
                              font_scale=font_scale, title_y=title_y)
            render_electronic(ax2, e, occ, n_unp,
                              xlim_max=xlim_max, label_x_offset=label_x_offset, label_dy_factor=label_dy_factor,
                              p2_label_scale=p2_label_scale, p2_tick_scale=p2_tick_scale, p2_perlevel_scale=p2_perlevel_scale,
                              font_scale=font_scale)
            render_stability(ax3, Qs_curve, E_curve, q, E_now, font_scale=font_scale, title_y=title_y)
            writer.grab_frame()
    plt.close(fig)
    print("Saved:", path)

def _energy_curve_from_params(par, Qs_curve, smooth_enabled=False, T_soft=0.0):
    cf = globals().get('crystal_field_energies', None)
    mapfn = globals().get('map_Q_to_DsDt', None)
    use_global = callable(cf) and callable(mapfn)
    E_vals = []
    for Q in Qs_curve:
        if use_global and smooth_enabled:
            Ds, Dt = mapfn(Q, ks=par['ks'], kt=par['kt'])
            e = cf(par['Dq'], Ds, Dt)
            E_el = softmin_electronic_energy(e, par['dcount'], par['P'], T_soft)
        elif use_global and not smooth_enabled:
            Ds, Dt = mapfn(Q, ks=par['ks'], kt=par['kt'])
            e = cf(par['Dq'], Ds, Dt)
            _, E_el, _ = minimize_electronic_energy(e, par['dcount'], par['P'])
        else:
            # Fallback to local mapping + minimization
            E_el, _, _, _ = total_energy_and_occ(**par, Q=Q)
            # total_energy_and_occ includes lattice already; but for curve we want total energy:
            E_vals.append(E_el)
            continue
        # Add lattice part
        E_lat = par['a3']*Q**3 + par['a4']*Q**4 + par['k']*(Q**2)
        E_vals.append(E_el + E_lat)
    return np.array(E_vals)

# Call export_jt_video after all functions are defined, uncomment for use:
"""export_jt_video(
    "jt_demo.mp4",
    include_atomic=True,
    fps=30,
    seconds=10,
    xlim_max=1.16,
    label_x_offset=0.024,
    label_dy_factor=0.07,
    p2_label_scale=1.0,
    p2_tick_scale=1.0,
    p2_perlevel_scale=1.00,
    metal_color="#3A6EA5",
    atom_s_factor=0.56,
    bond_w_factor=0.80,
)"""