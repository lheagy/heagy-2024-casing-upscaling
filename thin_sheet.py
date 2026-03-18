"""
thin_sheet.py
=============
Analytic impedance functions and upscaling-inversion utilities for
electromagnetic thin sheets, slabs, and cylinders.

Functions
---------
Core EM primitives
    admittivity, gamma, eta_tm

Impedance models
    wave_impedance
    impedance_sheet, impedance_sheet_deriv_sigma, impedance_sheet_deriv_mu
    impedance_cyl_symmetric
    input_impedance_slab_symmetric, impedance_slab_dsigma, impedance_slab_dmu
    impedance_solid_cyl, impedance_solid_cyl_dsigma, impedance_solid_cyl_dmu
    impedance_open_cyl

Loss surfaces
    loss_sheet, loss_cylinder

Thin-sheet inversion (σ only)
    solve_sigma_eff_one_freq, invert_sigma_eff_vs_freq

Thin-sheet inversion (σ + μ)
    solve_sigma_mu_eff_one_freq_sheet, invert_sigma_mu_eff_vs_freq_sheet

Slab-in-medium inversion (σ only)
    solve_sigma_eff_one_freq_slab_in_medium

Solid-cylinder inversion (σ only)
    solve_sigma_eff_one_freq_solid_cyl, invert_sigma_eff_vs_freq_solid_cyl

Solid-cylinder inversion (σ + μ)
    solve_sigma_mu_eff_one_freq_solid_cyl, invert_sigma_mu_eff_vs_freq_solid_cyl

Debye models
    sigma_debye, debye_one, debye_basis
    fit_debye_to_sigma
    predict_Z_sheet_debye, fit_debye_sigma_mu_to_impedance
    predict_Z_sheet_debye_decomp, eval_sigma_mu_from_p,
    fit_debye_decomp_sigma_mu_to_impedance
"""

import numpy as np
from scipy.special import iv, kv
from scipy.constants import mu_0
from scipy.optimize import least_squares
from discretize import tests

# ──────────────────────────────────────────────────────────────────────────────
# 1. Core EM primitives
# ──────────────────────────────────────────────────────────────────────────────

def admittivity(omega, sigma, eps=0.0):
    return sigma + 1j * omega * eps


def gamma(omega, mu, sigma, eps=0.0):
    """γ = sqrt(iωμ(σ + iωε))"""
    return np.sqrt(1j * omega * mu * admittivity(omega, sigma, eps))


def eta_tm(omega, mu, sigma, eps=0.0):
    """η = iωμ/γ  (TM impedance)"""
    g = gamma(omega, mu, sigma, eps)
    return 1j * omega * mu / g


# ──────────────────────────────────────────────────────────────────────────────
# 2. Impedance models
# ──────────────────────────────────────────────────────────────────────────────

def wave_impedance(omega, mu, sigma, eps=0.0):
    """Intrinsic wave impedance Z = iωμ/γ."""
    g = gamma(omega, mu, sigma, eps)
    return 1j * omega * mu / g


def impedance_sheet(omega, t, mu2, sigma2, eps2=0.0):
    """Thin-sheet impedance: Z = η₂ coth(γ₂ t)."""
    g2 = gamma(omega, mu2, sigma2, eps2)
    eta2 = 1j * omega * mu2 / g2
    return eta2 / np.tanh(g2 * t)


def impedance_sheet_deriv_sigma(omega, t, mu, sigma, eps=0.0):
    """∂Z/∂σ for thin-sheet impedance."""
    iwm = 1j * omega * mu
    g = np.sqrt(iwm * (sigma + 1j * omega * eps))
    x = g * t
    cothx = np.cosh(x) / np.sinh(x)
    csch2x = 1.0 / np.sinh(x) ** 2
    return -(iwm ** 2) / 2.0 * (cothx / g ** 3 + t * csch2x / g ** 2)


def impedance_sheet_deriv_mu(omega, t, mu, sigma, eps=0.0):
    """∂Z/∂μ for thin-sheet impedance."""
    iwm = 1j * omega * mu
    g = np.sqrt(iwm * (sigma + 1j * omega * eps))
    x = g * t
    cothx = np.cosh(x) / np.sinh(x)
    csch2x = 1.0 / np.sinh(x) ** 2
    return (1j * omega / 2.0) * (cothx / g - t * csch2x)


def impedance_cyl_symmetric(omega, a, b, mu1, sigma1, mu2, sigma2, eps=0):
    """Full 3-region cylindrical surface impedance (symmetric background medium)."""
    g1 = gamma(omega, mu1, sigma1)
    g2 = gamma(omega, mu2, sigma2)
    eta1 = 1j * omega * mu1 / g1
    I0b, I1b = iv(0, g2 * b), iv(1, g2 * b)
    K0b, K1b = kv(0, g2 * b), kv(1, g2 * b)
    I1a, K1a = iv(1, g2 * a), kv(1, g2 * a)
    num = I0b * (eta1 * g1 * K1a) + K0b * (eta1 * g1 * I1a)
    den = I1b * (eta1 * g1 * K1a) - K1b * (eta1 * g1 * I1a)
    return (1j * omega * mu2 / g2) * (num / den)


def impedance_hollow_cyl(omega, a, b, mu_bore, sigma_bore, mu_shell, sigma_shell, eps=0):
    """
    Input impedance at r=b for a hollow cylindrical shell (a < r < b).

    The bore (r < a) contains a conducting medium (mu_bore, sigma_bore) and
    the shell (a < r < b) has material (mu_shell, sigma_shell).

    Unlike impedance_cyl_symmetric / impedance_open_cyl, this formula correctly
    accounts for the bore loading: it reduces to impedance_open_cyl when
    sigma_bore → 0 and to impedance_solid_cyl when sigma_bore = sigma_shell.

    Derivation: match E_z and H_phi at r=a to express the K/I amplitude ratio
    r in the shell in terms of Z_L = η_bore · I₀(γ_bore·a)/I₁(γ_bore·a), then
    evaluate the shell impedance at r=b.
    """
    g1 = gamma(omega, mu_bore, sigma_bore, eps)
    g2 = gamma(omega, mu_shell, sigma_shell, eps)
    eta1 = 1j * omega * mu_bore / g1
    eta2 = 1j * omega * mu_shell / g2

    # Bore loading impedance (field regular at origin → I₀ solution only)
    Z_L = eta1 * iv(0, g1 * a) / iv(1, g1 * a)

    # Bessel functions of shell wavenumber evaluated at inner (a) and outer (b) radii
    I0a, I1a = iv(0, g2 * a), iv(1, g2 * a)
    I0b, I1b = iv(0, g2 * b), iv(1, g2 * b)
    K0a, K1a = kv(0, g2 * a), kv(1, g2 * a)
    K0b, K1b = kv(0, g2 * b), kv(1, g2 * b)

    # Amplitude ratio r = C/B from matching Z_shell(a) = Z_L
    r = (Z_L * I1a - eta2 * I0a) / (eta2 * K0a + Z_L * K1a)

    return eta2 * (I0b + r * K0b) / (I1b - r * K1b)


def propagate_impedance_outward(omega, r_in, r_out, Z_in, mu_bg, sigma_bg, eps=0):
    """
    Propagate an impedance Z_in at r=r_in outward to r=r_out through a
    homogeneous background medium (mu_bg, sigma_bg).

    The field in r_in < r < r_out is written as
        E_z = D·I₀(γr) + E·K₀(γr),   H_phi = (γ/iωμ)·(D·I₁(γr) − E·K₁(γr))
    Matching Z_in at r=r_in determines E/D, then Z is evaluated at r=r_out.
    """
    g = gamma(omega, mu_bg, sigma_bg, eps)
    eta = 1j * omega * mu_bg / g

    I0i, I1i = iv(0, g * r_in),  iv(1, g * r_in)
    K0i, K1i = kv(0, g * r_in),  kv(1, g * r_in)
    I0o, I1o = iv(0, g * r_out), iv(1, g * r_out)
    K0o, K1o = kv(0, g * r_out), kv(1, g * r_out)

    rho = (Z_in * I1i - eta * I0i) / (eta * K0i + Z_in * K1i)
    return eta * (I0o + rho * K0o) / (I1o - rho * K1o)


def input_impedance_slab_symmetric(omega, t, mu_s, sigma_s, eps_s, mu_b, sigma_b, eps_b):
    """Input impedance of a slab embedded symmetrically in a background medium."""
    g_s = gamma(omega, mu_s, sigma_s, eps_s)
    Z_s = 1j * omega * mu_s / g_s
    Z_b = wave_impedance(omega, mu_b, sigma_b, eps_b)
    T = np.tanh(g_s * t)
    return Z_s * (Z_b + Z_s * T) / (Z_s + Z_b * T)


def impedance_slab_dsigma(omega, t, mu_s, sigma_s, eps_s, mu_b, sigma_b, eps_b):
    """∂Z_in/∂σ_s for a slab in a symmetric background."""
    g = gamma(omega, mu_s, sigma_s, eps_s)
    Zs = 1j * omega * mu_s / g
    Zb = wave_impedance(omega, mu_b, sigma_b, eps_b)
    T = np.tanh(g * t)
    dg = (1j * omega * mu_s) / (2.0 * g)
    dZs = -(1j * omega * mu_s) ** 2 / (2.0 * g ** 3)
    dT = t * (1 - T ** 2) * dg
    N = Zs * (Zb + Zs * T)
    D = Zs + Zb * T
    dN = dZs * (Zb + Zs * T) + Zs * (dZs * T + Zs * dT)
    dD = dZs + Zb * dT
    return (dN * D - N * dD) / D ** 2


def impedance_slab_dmu(omega, t, mu_s, sigma_s, eps_s, mu_b, sigma_b, eps_b):
    """∂Z_in/∂μ_s for a slab in a symmetric background."""
    g = gamma(omega, mu_s, sigma_s, eps_s)
    Zs = 1j * omega * mu_s / g
    Zb = wave_impedance(omega, mu_b, sigma_b, eps_b)
    T = np.tanh(g * t)
    dg = g / (2.0 * mu_s)
    dZs = (1j * omega) / (2.0 * g)
    dT = t * (1 - T ** 2) * dg
    N = Zs * (Zb + Zs * T)
    D = Zs + Zb * T
    dN = dZs * (Zb + Zs * T) + Zs * (dZs * T + Zs * dT)
    dD = dZs + Zb * dT
    return (dN * D - N * dD) / D ** 2


def impedance_solid_cyl(omega, b, mu, sigma, eps=0.0):
    """Impedance of a solid cylinder: Z = (iωμ/γ) I₀(γb)/I₁(γb)."""
    g = gamma(omega, mu, sigma, eps)
    x = g * b
    return (1j * omega * mu / g) * (iv(0, x) / iv(1, x))


def impedance_solid_cyl_dsigma(omega, b, mu, sigma, eps=0.0):
    """∂Z/∂σ for solid cylinder impedance."""
    g = gamma(omega, mu, sigma, eps)
    x = g * b
    I0, I1 = iv(0, x), iv(1, x)
    R = I0 / I1
    eta = 1j * omega * mu / g
    dRdx = 1.0 - R ** 2 + R / x
    dg_ds = (1j * omega * mu) / (2.0 * g)
    deta_ds = -(1j * omega * mu) ** 2 / (2.0 * g ** 3)
    return deta_ds * R + eta * dRdx * b * dg_ds


def impedance_solid_cyl_dmu(omega, b, mu, sigma, eps=0.0):
    """∂Z/∂μ for solid cylinder impedance."""
    g = gamma(omega, mu, sigma, eps)
    x = g * b
    I0, I1 = iv(0, x), iv(1, x)
    R = I0 / I1
    eta = 1j * omega * mu / g
    dRdx = 1.0 - R ** 2 + R / x
    dg_dm = g / (2.0 * mu)
    deta_dm = (1j * omega) / (2.0 * g)
    return deta_dm * R + eta * dRdx * b * dg_dm


def impedance_open_cyl(omega, a, b, mu, sigma, eps=0):
    """
    Hollow cylinder with open bore at r=a (H_φ(a) ≈ 0).

    Z = (iωμ/γ) [I₀(γb)K₁(γa) + K₀(γb)I₁(γa)] / [I₁(γb)K₁(γa) − K₁(γb)I₁(γa)]
    """
    g = gamma(omega, mu, sigma, eps)
    I0b, I1b = iv(0, g * b), iv(1, g * b)
    K0b, K1b = kv(0, g * b), kv(1, g * b)
    I1a, K1a = iv(1, g * a), kv(1, g * a)
    num = I0b * K1a + K0b * I1a
    den = I1b * K1a - K1b * I1a
    return (1j * omega * mu / g) * (num / den)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Loss surfaces
# ──────────────────────────────────────────────────────────────────────────────

def loss_sheet(omega, t, mu, z_data, sigma_r, sigma_i, eps=0.0):
    """|Z_sheet(σ) − z_data|² over a grid of (σ_r, σ_i)."""
    z = impedance_sheet(omega, t, mu, sigma_r + 1j * sigma_i, eps2=eps)
    r = z - z_data
    return r.real ** 2 + r.imag ** 2


def loss_cylinder(omega, b, mu, z_data, sigma_r, sigma_i, eps=0.0):
    """|Z_solid_cyl(σ) − z_data|² over a grid of (σ_r, σ_i)."""
    z = impedance_solid_cyl(omega, b, mu, sigma_r + 1j * sigma_i, eps=eps)
    r = z - z_data
    return r.real ** 2 + r.imag ** 2


# ──────────────────────────────────────────────────────────────────────────────
# 4. Thin-sheet inversion: conductivity only
# ──────────────────────────────────────────────────────────────────────────────

def solve_sigma_eff_one_freq(omega, z_data, t_eff, mu=mu_0, sigma0=5e6 + 0j, eps=0.0, test=False):
    """
    Invert thin-sheet impedance for σ_eff at one frequency.

    Returns: sigma_hat, z_pred, success, message
    """
    x0 = np.array([sigma0.real, sigma0.imag], dtype=float)

    def F(x):
        r = impedance_sheet(omega, t_eff, mu, x[0] + 1j * x[1], eps2=eps) - z_data
        return np.array([r.real, r.imag], dtype=float)

    def J(x):
        dz = impedance_sheet_deriv_sigma(omega, t_eff, mu, x[0] + 1j * x[1], eps=eps)
        return np.column_stack([[dz.real, dz.imag], [(1j * dz).real, (1j * dz).imag]])

    if test:
        tests.check_derivative(lambda x: (F(x), J(x)), x0=x0)

    sol = least_squares(F, x0=x0, jac=J, method="lm")
    sigma_hat = sol.x[0] + 1j * sol.x[1]
    return sigma_hat, impedance_sheet(omega, t_eff, mu, sigma_hat, eps2=eps), sol.success, sol.message


def invert_sigma_eff_vs_freq(w, z_data, t_eff, mu=mu_0, sigma_init=5e6, eps=0.0, return_messages=False):
    """
    Frequency-by-frequency thin-sheet conductivity inversion.

    Returns: sigma_eff, z_pred, success  (and messages if return_messages=True)
    """
    w = np.asarray(w)
    z_data = np.asarray(z_data, dtype=complex)
    sigma_eff = np.empty_like(z_data, dtype=complex)
    z_pred = np.empty_like(z_data, dtype=complex)
    success = np.zeros(w.shape, dtype=bool)
    messages = []
    sigma0 = sigma_init + 0j
    for k, omega in enumerate(w):
        s, zh, ok, msg = solve_sigma_eff_one_freq(
            omega, z_data[k], t_eff=t_eff, mu=mu, sigma0=sigma0, eps=eps
        )
        sigma_eff[k], z_pred[k], success[k] = s, zh, ok
        if return_messages:
            messages.append(msg)
        if ok:
            sigma0 = s
    if return_messages:
        return sigma_eff, z_pred, success, messages
    return sigma_eff, z_pred, success


# ──────────────────────────────────────────────────────────────────────────────
# 5. Thin-sheet inversion: conductivity + permeability
# ──────────────────────────────────────────────────────────────────────────────

def solve_sigma_mu_eff_one_freq_sheet(
    omega, z_data, t_eff,
    sigma0=5e6 + 0j, mu0=mu_0 + 0j, eps=0.0,
    complex_values=True, test=False,
    method="trf", x_scale="jac", max_nfev=5000,
):
    """
    Jointly invert thin-sheet impedance for σ and μ at one frequency.

    If complex_values=True:  unknowns = [σ_r, σ_i, μ_r, μ_i]  (2 eq / 4 unknowns)
    If complex_values=False: unknowns = [σ_r, μ_r]             (2 eq / 2 unknowns)

    Returns: sigma_hat, mu_hat, z_pred, success, message
    """
    if complex_values:
        x0 = np.array([sigma0.real, sigma0.imag, mu0.real, mu0.imag], dtype=float)

        def unpack(x):
            return x[0] + 1j * x[1], x[2] + 1j * x[3]

        def F(x):
            s, m = unpack(x)
            r = impedance_sheet(omega, t_eff, m, s, eps2=eps) - z_data
            return np.array([r.real, r.imag], dtype=float)

        def J(x):
            s, m = unpack(x)
            ds = impedance_sheet_deriv_sigma(omega, t_eff, m, s, eps=eps)
            dm = impedance_sheet_deriv_mu(omega, t_eff, m, s, eps=eps)
            return np.column_stack([
                [ds.real, ds.imag], [(1j * ds).real, (1j * ds).imag],
                [dm.real, dm.imag], [(1j * dm).real, (1j * dm).imag],
            ])
    else:
        x0 = np.array([sigma0.real, mu0.real], dtype=float)

        def unpack(x):
            return x[0] + 0j, x[1] + 0j

        def F(x):
            s, m = unpack(x)
            r = impedance_sheet(omega, t_eff, m, s, eps2=eps) - z_data
            return np.array([r.real, r.imag], dtype=float)

        def J(x):
            s, m = unpack(x)
            ds = impedance_sheet_deriv_sigma(omega, t_eff, m, s, eps=eps)
            dm = impedance_sheet_deriv_mu(omega, t_eff, m, s, eps=eps)
            return np.column_stack([[ds.real, ds.imag], [dm.real, dm.imag]])

    if test:
        tests.check_derivative(lambda x: (F(x), J(x)), x0=x0, dx=np.random.randn(len(x0)) * 1e-4)

    sol = least_squares(
        F, x0=x0, jac=J, method=method, x_scale=x_scale,
        max_nfev=max_nfev, ftol=1e-14, xtol=1e-14, gtol=1e-14,
    )
    sigma_hat, mu_hat = unpack(sol.x)
    return sigma_hat, mu_hat, impedance_sheet(omega, t_eff, mu_hat, sigma_hat, eps2=eps), sol.success, sol.message


def invert_sigma_mu_eff_vs_freq_sheet(
    w, z_data, t_eff,
    sigma_init=5e6 + 0j, mu_init=mu_0 + 0j, eps=0.0,
    complex_values=True, method="trf", x_scale="jac", max_nfev=5000,
):
    """
    Frequency-by-frequency thin-sheet σ + μ inversion.

    Returns: sigma_eff, mu_eff, z_pred, ok
    """
    w = np.asarray(w, float)
    z_data = np.asarray(z_data, complex)
    n = w.size
    sigma_eff = np.empty(n, dtype=complex)
    mu_eff = np.empty(n, dtype=complex)
    z_pred = np.empty(n, dtype=complex)
    ok = np.zeros(n, dtype=bool)
    sigma0, mu0 = sigma_init + 0j, mu_init + 0j
    for k, omega in enumerate(w):
        s, m, zh, success, _ = solve_sigma_mu_eff_one_freq_sheet(
            omega, z_data[k], t_eff, sigma0=sigma0, mu0=mu0, eps=eps,
            complex_values=complex_values, method=method, x_scale=x_scale, max_nfev=max_nfev,
        )
        sigma_eff[k], mu_eff[k], z_pred[k], ok[k] = s, m, zh, success
        if success:
            sigma0, mu0 = s, m
    return sigma_eff, mu_eff, z_pred, ok


# ──────────────────────────────────────────────────────────────────────────────
# 6. Slab-in-medium inversion: conductivity only
# ──────────────────────────────────────────────────────────────────────────────

def solve_sigma_eff_one_freq_slab_in_medium(
    omega, z_data, t, mu_s, sigma0=5e6 + 0j, eps_s=0.0,
    mu_b=None, sigma_b=0.0, eps_b=0.0, test=False,
):
    """
    Invert slab-in-medium impedance for σ_s at one frequency.

    Returns: sigma_hat, z_pred, success, message
    """
    if mu_b is None:
        mu_b = mu_s
    x0 = np.array([sigma0.real, sigma0.imag], dtype=float)

    def Z_of_sigma(sigma):
        return input_impedance_slab_symmetric(
            omega, t, mu_s=mu_s, sigma_s=sigma, eps_s=eps_s,
            mu_b=mu_b, sigma_b=sigma_b, eps_b=eps_b,
        )

    def F(x):
        r = Z_of_sigma(x[0] + 1j * x[1]) - z_data
        return np.array([r.real, r.imag], dtype=float)

    def J(x):
        dz = impedance_slab_dsigma(
            omega, t, mu_s=mu_s, sigma_s=x[0] + 1j * x[1], eps_s=eps_s,
            mu_b=mu_b, sigma_b=sigma_b, eps_b=eps_b,
        )
        return np.column_stack([[dz.real, dz.imag], [(1j * dz).real, (1j * dz).imag]])

    if test:
        tests.check_derivative(lambda x: (F(x), J(x)), x0=x0)

    sol = least_squares(F, x0=x0, jac=J, method="lm", x_scale="jac",
                        ftol=1e-14, xtol=1e-14, gtol=1e-14)
    sigma_hat = sol.x[0] + 1j * sol.x[1]
    return sigma_hat, Z_of_sigma(sigma_hat), sol.success, sol.message


# ──────────────────────────────────────────────────────────────────────────────
# 7. Solid cylinder inversion: conductivity only
# ──────────────────────────────────────────────────────────────────────────────

def solve_sigma_eff_one_freq_solid_cyl(
    omega, z_data, b, mu, sigma0=5e6 + 0j, eps=0.0, test=False,
):
    """
    Invert solid-cylinder impedance for σ at one frequency.

    Returns: sigma_hat, z_pred, success, message
    """
    x0 = np.array([sigma0.real, sigma0.imag], dtype=float)

    def F(x):
        r = impedance_solid_cyl(omega, b, mu, x[0] + 1j * x[1], eps=eps) - z_data
        return np.array([r.real, r.imag], dtype=float)

    def J(x):
        dz = impedance_solid_cyl_dsigma(omega, b, mu, x[0] + 1j * x[1], eps=eps)
        return np.column_stack([[dz.real, dz.imag], [(1j * dz).real, (1j * dz).imag]])

    if test:
        tests.check_derivative(lambda x: (F(x), J(x)), x0=x0)

    sol = least_squares(F, x0=x0, jac=J, method="lm", x_scale="jac",
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    sigma_hat = sol.x[0] + 1j * sol.x[1]
    return sigma_hat, impedance_solid_cyl(omega, b, mu, sigma_hat, eps=eps), sol.success, sol.message


def invert_sigma_eff_vs_freq_solid_cyl(
    w, z_data, b, mu, sigma_init=5e6 + 0j, eps=0.0, test_index=None,
):
    """
    Frequency-by-frequency conductivity inversion for solid cylinder.

    Returns: sigma_eff, z_pred, ok
    """
    w = np.asarray(w, float)
    z_data = np.asarray(z_data, dtype=complex)
    n = w.size
    sigma_eff = np.empty(n, dtype=complex)
    z_pred = np.empty(n, dtype=complex)
    ok = np.zeros(n, dtype=bool)
    sigma0 = sigma_init + 0j
    for k, omega in enumerate(w):
        sig, zh, success, _ = solve_sigma_eff_one_freq_solid_cyl(
            omega, z_data[k], b=b, mu=mu, sigma0=sigma0, eps=eps,
            test=(test_index is not None and k == test_index),
        )
        sigma_eff[k], z_pred[k], ok[k] = sig, zh, success
        if success:
            sigma0 = sig
    return sigma_eff, z_pred, ok


# ──────────────────────────────────────────────────────────────────────────────
# 8. Solid cylinder inversion: conductivity + permeability
# ──────────────────────────────────────────────────────────────────────────────

def solve_sigma_mu_eff_one_freq_solid_cyl(
    omega, z_data, b,
    sigma0=5e6 + 0j, mu0=mu_0 + 0j, eps=0.0,
    complex_values=True, test=False,
    method="trf", x_scale="jac", max_nfev=5000,
    bounds=(-np.inf, np.inf),
):
    """
    Jointly invert solid-cylinder impedance for σ and μ at one frequency.

    If complex_values=True:  unknowns = [σ_r, σ_i, μ_r, μ_i]  (2 eq / 4 unknowns)
    If complex_values=False: unknowns = [σ_r, μ_r]             (2 eq / 2 unknowns)

    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on the unknowns, passed to least_squares.
        For complex_values=True  the order is [σ_r, σ_i, μ_r, μ_i].
        For complex_values=False the order is [σ_r, μ_r].
        Scalars are broadcast to all parameters.  Default: no bounds.

    Returns: sigma_hat, mu_hat, z_pred, success, message
    """
    if complex_values:
        x0 = np.array([sigma0.real, sigma0.imag, mu0.real, mu0.imag], dtype=float)

        def unpack(x):
            return x[0] + 1j * x[1], x[2] + 1j * x[3]

        def F(x):
            s, m = unpack(x)
            r = impedance_solid_cyl(omega, b, m, s, eps=eps) - z_data
            return np.array([r.real, r.imag], dtype=float)

        def J(x):
            s, m = unpack(x)
            ds = impedance_solid_cyl_dsigma(omega, b, m, s, eps=eps)
            dm = impedance_solid_cyl_dmu(omega, b, m, s, eps=eps)
            return np.column_stack([
                [ds.real, ds.imag], [(1j * ds).real, (1j * ds).imag],
                [dm.real, dm.imag], [(1j * dm).real, (1j * dm).imag],
            ])
    else:
        x0 = np.array([sigma0.real, mu0.real], dtype=float)

        def unpack(x):
            return x[0] + 0j, x[1] + 0j

        def F(x):
            s, m = unpack(x)
            r = impedance_solid_cyl(omega, b, m, s, eps=eps) - z_data
            return np.array([r.real, r.imag], dtype=float)

        def J(x):
            s, m = unpack(x)
            ds = impedance_solid_cyl_dsigma(omega, b, m, s, eps=eps)
            dm = impedance_solid_cyl_dmu(omega, b, m, s, eps=eps)
            return np.column_stack([[ds.real, ds.imag], [dm.real, dm.imag]])

    if test:
        tests.check_derivative(lambda x: (F(x), J(x)), x0=x0, dx=np.random.randn(len(x0)) * 1e-4)

    sol = least_squares(
        F, x0=x0, jac=J, method=method, x_scale=x_scale,
        max_nfev=max_nfev, ftol=1e-14, xtol=1e-14, gtol=1e-14,
        bounds=bounds,
    )
    sigma_hat, mu_hat = unpack(sol.x)
    return sigma_hat, mu_hat, impedance_solid_cyl(omega, b, mu_hat, sigma_hat, eps=eps), sol.success, sol.message


def invert_sigma_mu_eff_vs_freq_solid_cyl(
    w, z_data, b,
    sigma_init=5e6 + 0j, mu_init=mu_0 + 0j, eps=0.0,
    complex_values=True, method="trf", x_scale="jac", max_nfev=5000,
    bounds=(-np.inf, np.inf),
):
    """
    Frequency-by-frequency σ + μ inversion for solid cylinder.

    bounds : 2-tuple of array_like, optional
        Passed directly to solve_sigma_mu_eff_one_freq_solid_cyl.
        For complex_values=True  the order is [σ_r, σ_i, μ_r, μ_i].
        For complex_values=False the order is [σ_r, μ_r].

    Returns: sigma_eff, mu_eff, z_pred, ok
    """
    w = np.asarray(w, float)
    z_data = np.asarray(z_data, complex)
    n = w.size
    sigma_eff = np.empty(n, dtype=complex)
    mu_eff = np.empty(n, dtype=complex)
    z_pred = np.empty(n, dtype=complex)
    ok = np.zeros(n, dtype=bool)
    sigma0, mu0 = sigma_init + 0j, mu_init + 0j
    for k, omega in enumerate(w):
        s, m, zh, success, _ = solve_sigma_mu_eff_one_freq_solid_cyl(
            omega, z_data[k], b=b, sigma0=sigma0, mu0=mu0, eps=eps,
            complex_values=complex_values, method=method, x_scale=x_scale, max_nfev=max_nfev,
            bounds=bounds,
        )
        sigma_eff[k], mu_eff[k], z_pred[k], ok[k] = s, m, zh, success
        if success:
            sigma0, mu0 = s, m
    return sigma_eff, mu_eff, z_pred, ok


# ──────────────────────────────────────────────────────────────────────────────
# 9. Debye models
# ──────────────────────────────────────────────────────────────────────────────

def sigma_debye(omega, sigma_inf, delta_sigma, tau):
    """Single-term Debye conductivity: σ(ω) = σ_∞ + Δσ/(1 + iωτ)."""
    return sigma_inf + delta_sigma / (1.0 + 1j * omega * tau)


def debye_one(omega, x_inf, dx, tau):
    """Single-term Debye dispersion: x(ω) = x_∞ + Δx/(1 + iωτ)."""
    return x_inf + dx / (1.0 + 1j * omega * tau)


def debye_basis(omega, tau):
    """
    Debye basis matrix B[i, k] = 1/(1 + iω_i τ_k).

    omega : (N,), tau : (K,) → returns (N, K).
    """
    omega = np.asarray(omega, float)[:, None]
    tau = np.asarray(tau, float)[None, :]
    return 1.0 / (1.0 + 1j * omega * tau)


def fit_debye_to_sigma(omega, sigma_eff, wts=None, p0=None, bounds=None):
    """
    Fit single-term Debye model to complex σ_eff(ω).

    Returns: p_hat = [sigma_inf, delta_sigma, tau], result
    """
    omega = np.asarray(omega, float)
    y = np.asarray(sigma_eff, complex)
    if wts is None:
        wts = np.ones_like(omega)
    wts = np.asarray(wts, float)

    def residual(p):
        sw = np.sqrt(wts)
        r = sigma_debye(omega, *p) - y
        return np.r_[sw * r.real, sw * r.imag]

    if p0 is None:
        n = len(omega)
        hi = slice(int(0.9 * n), n) if n >= 10 else slice(max(0, n - 3), n)
        sigma_inf0 = np.median(y.real[hi])
        lo = slice(0, max(3, int(0.1 * n)))
        delta0 = np.median(y.real[lo]) - sigma_inf0
        kpk = np.argmax(np.abs(y.imag))
        tau0 = 1.0 / max(omega[kpk], 1e-30)
        p0 = np.array([sigma_inf0, delta0, tau0], float)

    if bounds is None:
        bounds = ([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf])

    res = least_squares(residual, x0=p0, bounds=bounds, method="trf")
    return res.x, res


def predict_Z_sheet_debye(omega, t_eff, p, eps=0.0):
    """
    Z(ω) for thin sheet with Debye σ(ω) and μ(ω).

    p = [σ_∞, Δσ, τ_σ, μ_∞, Δμ, τ_μ]
    """
    sigma_inf, d_sigma, tau_sigma, mu_inf, d_mu, tau_mu = p
    return impedance_sheet(
        omega, t_eff,
        debye_one(omega, mu_inf, d_mu, tau_mu),
        debye_one(omega, sigma_inf, d_sigma, tau_sigma),
        eps2=eps,
    )


def fit_debye_sigma_mu_to_impedance(
    omega, z_data, t_eff,
    p0=None, bounds=None, eps=0.0, weights=None,
    residual_mode="reim",
):
    """
    Fit Debye σ(ω) and μ(ω) to complex impedance data Z(ω).

    residual_mode: "reim" or "logmag_phase"
    Returns: p_hat, result, success
    """
    omega = np.asarray(omega, float)
    z_data = np.asarray(z_data, complex)
    wts = np.ones_like(omega) if weights is None else np.asarray(weights, float)

    if p0 is None:
        sigma_inf0 = max(np.real(1.0 / (t_eff * z_data[0])), 1.0)
        p0 = np.array([sigma_inf0, 0.0, 1e-3, mu_0, 0.0, 1e-3], dtype=float)

    if bounds is None:
        lb = np.array([0.0, -np.inf, 1e-12, 0.0, -np.inf, 1e-12], dtype=float)
        ub = np.full(6, np.inf)
        bounds = (lb, ub)

    def wrap_phase(phi):
        return (phi + np.pi) % (2 * np.pi) - np.pi

    def residual(p):
        z_pred = predict_Z_sheet_debye(omega, t_eff, p, eps=eps)
        if residual_mode == "reim":
            r = z_pred - z_data
            return np.r_[wts * r.real, wts * r.imag]
        elif residual_mode == "logmag_phase":
            r_mag = np.log(np.abs(z_pred)) - np.log(np.abs(z_data))
            r_phs = wrap_phase(np.angle(z_pred) - np.angle(z_data))
            return np.r_[wts * r_mag, wts * r_phs]
        else:
            raise ValueError("residual_mode must be 'reim' or 'logmag_phase'")

    res = least_squares(residual, p0, method="trf", bounds=bounds, x_scale="jac",
                        ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=5000)
    return res.x, res, res.success


def predict_Z_sheet_debye_decomp(omega, t_eff, p, tau, eps=0.0):
    """
    Z for thin sheet using multi-term Debye decomposition.

    p = [σ_∞, Δσ[0:K], μ_∞, Δμ[0:K]]
    tau: (K,) fixed relaxation times
    """
    omega = np.asarray(omega, float)
    K = len(tau)
    B = debye_basis(omega, tau)
    sigma_w = p[0] + B @ p[1:1 + K]
    mu_w = p[1 + K] + B @ p[2 + K:2 + 2 * K]
    return impedance_sheet(omega, t_eff, mu_w, sigma_w, eps2=eps)


def eval_sigma_mu_from_p(omega, p, tau):
    """Evaluate σ(ω) and μ(ω) from multi-term Debye parameter vector p."""
    K = len(tau)
    B = debye_basis(omega, tau)
    sigma_w = p[0] + B @ p[1:1 + K]
    mu_w = p[1 + K] + B @ p[2 + K:2 + 2 * K]
    return sigma_w, mu_w


def fit_debye_decomp_sigma_mu_to_impedance(
    omega, z_data, t_eff, tau,
    p0=None, eps=0.0, weights=None,
    residual_mode="logmag_phase",
    enforce_nonneg=True,
    sigma_inf_bounds=(0.0, np.inf),
    mu_inf_bounds=(0.0, np.inf),
    max_nfev=50000,
):
    """
    Fit multi-term Debye σ(ω) and μ(ω) decompositions to complex impedance data.

    Returns: p_hat, result
    """
    omega = np.asarray(omega, float)
    z_data = np.asarray(z_data, complex)
    K = len(tau)
    wts = np.ones_like(omega) if weights is None else np.asarray(weights, float)

    if p0 is None:
        p0 = np.r_[1e6, np.zeros(K), mu_0, np.zeros(K)].astype(float)

    if enforce_nonneg:
        lb = np.r_[sigma_inf_bounds[0], np.zeros(K), mu_inf_bounds[0], np.zeros(K)]
        ub = np.r_[sigma_inf_bounds[1], np.full(K, np.inf), mu_inf_bounds[1], np.full(K, np.inf)]
    else:
        lb = np.r_[sigma_inf_bounds[0], np.full(K, -np.inf), mu_inf_bounds[0], np.full(K, -np.inf)]
        ub = np.r_[sigma_inf_bounds[1], np.full(K, np.inf), mu_inf_bounds[1], np.full(K, np.inf)]

    def wrap_phase(phi):
        return (phi + np.pi) % (2 * np.pi) - np.pi

    def residual(p):
        z_pred = predict_Z_sheet_debye_decomp(omega, t_eff, p, tau, eps=eps)
        if residual_mode == "reim":
            r = z_pred - z_data
            return np.r_[wts * r.real, wts * r.imag]
        elif residual_mode == "logmag_phase":
            r_mag = np.log(np.abs(z_pred)) - np.log(np.abs(z_data))
            r_phs = wrap_phase(np.angle(z_pred) - np.angle(z_data))
            return np.r_[wts * r_mag, wts * r_phs]
        else:
            raise ValueError("residual_mode must be 'reim' or 'logmag_phase'")

    res = least_squares(residual, p0, method="lm", x_scale="jac",
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=max_nfev)
    return res.x, res
