"""
Tests for thin_sheet.py impedance functions.
Run with:  pytest tests/test_thin_sheet.py -v
"""

import numpy as np
import pytest
from scipy.constants import mu_0

from thin_sheet import (
    impedance_solid_cyl, impedance_hollow_cyl, impedance_open_cyl,
    impedance_sheet, propagate_impedance_outward,
)

# Frequency sweep used across tests
OMEGAS = 2 * np.pi * np.logspace(1, 4, 30)

# Typical casing material
SIGMA_CAS = 1e6   # S/m
MU_CAS    = mu_0  # non-magnetic


class TestHollowCylLimits:
    """
    Verify that impedance_hollow_cyl recovers impedance_solid_cyl in
    limiting cases where the two geometries should be equivalent.
    """

    def test_solid_limit_when_inner_radius_is_zero(self):
        """
        A hollow cylinder whose inner radius a → 0 (i.e. wall thickness = b)
        is geometrically solid, so its impedance should equal impedance_solid_cyl
        regardless of the bore material.

        We use a = b * 1e-8 (effectively zero) and check that the hollow-cyl
        result matches the solid-cyl result to high relative accuracy.
        """
        b = 0.05           # outer radius [m]
        a = b * 1e-8       # inner radius ≈ 0

        # Bore material (arbitrary – should not matter when a ≈ 0)
        mu_bore    = mu_0
        sigma_bore = 1e-3  # S/m

        Z_solid  = impedance_solid_cyl(OMEGAS, b, MU_CAS, SIGMA_CAS)
        Z_hollow = impedance_hollow_cyl(
            OMEGAS, a, b,
            mu_bore, sigma_bore,   # bore (arbitrary, volume → 0)
            MU_CAS, SIGMA_CAS,     # shell material
        )

        np.testing.assert_allclose(
            Z_hollow, Z_solid,
            rtol=1e-6,
            err_msg="hollow_cyl with a→0 should equal solid_cyl",
        )

    def test_solid_limit_when_bore_has_same_material_as_shell(self):
        """
        When the bore medium has the same conductivity and permeability as the
        shell, the hollow cylinder is physically uniform and the impedance must
        equal that of a solid cylinder of radius b with the same material.

        Analytically: Z_L = η·I₀(γa)/I₁(γa) causes ρ = 0 exactly, giving
        Z = η·I₀(γb)/I₁(γb) = impedance_solid_cyl.
        """
        b = 0.05    # outer radius [m]
        a = 0.025   # inner radius [m]  (typical casing geometry)

        Z_solid  = impedance_solid_cyl(OMEGAS, b, MU_CAS, SIGMA_CAS)
        Z_hollow = impedance_hollow_cyl(
            OMEGAS, a, b,
            MU_CAS, SIGMA_CAS,   # bore  = shell material
            MU_CAS, SIGMA_CAS,   # shell
        )

        np.testing.assert_allclose(
            Z_hollow, Z_solid,
            rtol=1e-10,
            err_msg="hollow_cyl with bore == shell material should equal solid_cyl",
        )


class TestThinSheetLargeRadiusLimit:
    """
    For a hollow cylindrical shell with b >> t, the cylindrical geometry
    becomes locally planar and the impedance should converge to the flat
    thin-sheet impedance  Z = η·coth(γt).

    The bore is made nearly insulating (σ_bore → 0) to enforce the same
    open boundary condition (H_φ = 0 at the inner wall) that the thin-sheet
    formula assumes at its back face.

    The leading-order correction is O(t/b), so with t/b = 2e-3 we expect
    relative errors ≲ 1e-3.  Frequencies are capped to keep |γb| < 200
    and avoid Bessel function overflow.
    """

    # Shell material
    SIGMA = 1e6   # S/m
    MU    = mu_0

    # Geometry: t/b = 2e-3  →  O(t/b) error ~ 1e-3
    T = 0.01    # wall thickness [m]
    B = 5.0     # outer radius  [m]

    # Frequency range: skin depth crosses wall thickness in this window,
    # and |gamma * b|_max ≈ 140 (safe for Bessel evaluation)
    OMEGAS_THIN = 2 * np.pi * np.logspace(1, 2, 30)   # 10 – 100 Hz

    def test_hollow_cyl_large_radius_matches_thin_sheet(self):
        """
        impedance_hollow_cyl with b >> t and insulating bore should agree
        with impedance_sheet to within O(t/b) ≈ 2e-3 relative error.
        """
        a = self.B - self.T

        Z_sheet  = impedance_sheet(self.OMEGAS_THIN, self.T, self.MU, self.SIGMA)
        Z_hollow = impedance_hollow_cyl(
            self.OMEGAS_THIN, a, self.B,
            mu_0, 1e-6,             # bore: nearly insulating  → open BC
            self.MU, self.SIGMA,    # shell material
        )

        np.testing.assert_allclose(
            Z_hollow, Z_sheet,
            rtol=5e-3,   # 5× the nominal O(t/b) = 2e-3 correction
            err_msg="hollow_cyl with b >> t should approach the thin-sheet impedance",
        )


class TestInsulatingBoreEquivalence:
    """
    When the bore conductivity σ_bore → 0, the bore loading impedance Z_L → ∞
    and the inner boundary condition becomes H_φ(a) = 0 — identical to the
    assumption made by impedance_open_cyl.  The two functions must therefore
    agree in this limit.
    """

    def test_hollow_cyl_insulating_bore_matches_open_cyl(self):
        """
        impedance_hollow_cyl with σ_bore = 1e-10 S/m (≈ insulating)
        should equal impedance_open_cyl to high precision.
        """
        b = 0.05    # outer radius [m]
        a = 0.025   # inner radius [m]

        Z_open   = impedance_open_cyl(OMEGAS, a, b, MU_CAS, SIGMA_CAS)
        Z_hollow = impedance_hollow_cyl(
            OMEGAS, a, b,
            mu_0, 1e-10,         # bore: effectively insulating
            MU_CAS, SIGMA_CAS,   # shell
        )

        np.testing.assert_allclose(
            Z_hollow, Z_open,
            rtol=1e-6,
            err_msg="hollow_cyl with insulating bore should equal open_cyl",
        )


class TestPropagateImpedance:
    """
    Physical invariants for propagate_impedance_outward.
    """

    # Casing geometry and material
    B     = 0.05    # outer casing radius [m]
    A     = 0.025   # inner radius [m]

    # Background medium (fills the space outside the casing)
    MU_BG    = mu_0
    SIGMA_BG = 1e-3   # S/m

    def _Z_casing(self):
        """Helper: hollow-casing impedance at r = b."""
        return impedance_hollow_cyl(
            OMEGAS, self.A, self.B,
            self.MU_BG, self.SIGMA_BG,
            MU_CAS, SIGMA_CAS,
        )

    def test_identity_when_r_in_equals_r_out(self):
        """
        Propagating over zero distance must return the input impedance exactly.
        Analytically: the Wronskian identity I₀K₁ + I₁K₀ = 1/r causes the
        formula to reduce to Z_in when r_out = r_in.
        """
        Z_in  = self._Z_casing()
        Z_out = propagate_impedance_outward(
            OMEGAS, self.B, self.B, Z_in, self.MU_BG, self.SIGMA_BG
        )

        np.testing.assert_allclose(
            Z_out, Z_in,
            rtol=1e-10,
            err_msg="propagate with r_out == r_in should be identity",
        )

    def test_two_step_equals_one_step(self):
        """
        Propagating r0 → r1 → r2 through the same homogeneous medium must
        equal propagating r0 → r2 in a single step.  This is the chain rule
        (transitivity) of the impedance transformation.
        """
        r0, r1, r2 = self.B, 0.10, 0.20   # [m]

        Z0 = self._Z_casing()

        Z_via_r1 = propagate_impedance_outward(
            OMEGAS, r1, r2,
            propagate_impedance_outward(OMEGAS, r0, r1, Z0, self.MU_BG, self.SIGMA_BG),
            self.MU_BG, self.SIGMA_BG,
        )
        Z_direct = propagate_impedance_outward(
            OMEGAS, r0, r2, Z0, self.MU_BG, self.SIGMA_BG
        )

        np.testing.assert_allclose(
            Z_via_r1, Z_direct,
            rtol=1e-10,
            err_msg="two-step propagation should equal direct propagation",
        )

    def test_propagation_through_homogeneous_medium(self):
        """
        If the input impedance at r_in already equals the impedance of the
        background medium at that radius (i.e. no casing, just background
        everywhere), propagating to r_out should give the background impedance
        at r_out — not something else.

        This verifies that the background solid-cylinder impedance is a
        consistent fixed point of the propagation operator.
        """
        r_in, r_out = 0.05, 0.10   # [m]

        Z_bg_in  = impedance_solid_cyl(OMEGAS, r_in,  self.MU_BG, self.SIGMA_BG)
        Z_bg_out = impedance_solid_cyl(OMEGAS, r_out, self.MU_BG, self.SIGMA_BG)

        Z_propagated = propagate_impedance_outward(
            OMEGAS, r_in, r_out, Z_bg_in, self.MU_BG, self.SIGMA_BG
        )

        np.testing.assert_allclose(
            Z_propagated, Z_bg_out,
            rtol=1e-10,
            err_msg="propagating background impedance through homogeneous medium "
                    "should give background impedance at new radius",
        )


class TestAsymptoticLimits:
    """
    Physical limiting behaviour at extreme frequencies.
    """

    def test_dc_limit_of_thin_sheet_is_resistive(self):
        """
        At DC (ω → 0), γt → 0 and coth(γt) → 1/(γt), giving
            Z_sheet → η/(γt) = 1/(σt)
        which is the real-valued sheet resistance (Ω/□).

        We verify both that Re(Z) → 1/(σt) and that Im(Z)/Re(Z) → 0.
        """
        sigma = 1e6    # S/m
        t     = 0.01   # m

        omegas_dc = 2 * np.pi * np.array([1e-3, 1e-4, 1e-5])   # very low f
        Z = impedance_sheet(omegas_dc, t, mu_0, sigma)

        expected_R = 1.0 / (sigma * t)

        np.testing.assert_allclose(
            Z.real, expected_R,
            rtol=1e-6,
            err_msg="DC limit of Z_sheet should be the sheet resistance 1/(σt)",
        )
        np.testing.assert_allclose(
            np.abs(Z.imag) / expected_R, np.zeros(len(omegas_dc)),
            atol=1e-5,
            err_msg="DC limit of Z_sheet should be purely resistive",
        )

    def test_skin_effect_limit_phase_of_solid_cyl(self):
        """
        At high frequency (skin depth δ << b), the surface impedance of a
        solid cylinder approaches the plane-wave value

            Z → η = sqrt(iωμ/σ),   phase = π/4.

        We use b = 0.5 m so the skin-effect regime is reached at moderate
        frequencies without Bessel function overflow (|γb| ≈ 30–100).
        """
        b     = 0.5    # m  — large enough for δ << b at f ~ 1–10 kHz
        sigma = 1e6    # S/m
        mu    = mu_0

        # At these frequencies: δ = sqrt(2/(ωμσ)) ≈ 0.5–1.6 mm << b = 500 mm
        omegas_hf = 2 * np.pi * np.array([1e3, 5e3, 1e4])

        Z = impedance_solid_cyl(omegas_hf, b, mu, sigma)

        np.testing.assert_allclose(
            np.angle(Z), np.full(len(omegas_hf), np.pi / 4),
            atol=0.01,   # within 0.01 rad of π/4
            err_msg="high-frequency phase of Z_solid_cyl should approach π/4",
        )
