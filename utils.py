import numpy as np
from discretize import utils

from simpeg import maps, directives
from simpeg.electromagnetics import frequency_domain as fdem
from simpeg.utils.solver_utils import get_default_solver

import pickle

def pad_for_casing_and_data(
    casing_outer_radius,
    csx1=2.5e-3,
    csx2=25,
    pfx1=1.3,
    pfx2=1.5,
    domain_x=1000,
    npadx=10
):

    ncx1 = np.ceil(casing_outer_radius/csx1+2)
    npadx1 = np.floor(np.log(csx2/csx1) / np.log(pfx1))

    # finest uniform region
    hx1a = utils.unpack_widths([(csx1, ncx1)])

    # pad to second uniform region
    hx1b = utils.unpack_widths([(csx1, npadx1, pfx1)])

    # scale padding so it matches cell size properly
    dx1 = np.sum(hx1a)+np.sum(hx1b)
    dx1 = 3 #np.floor(dx1/self.csx2)
    hx1b *= (dx1*csx2 - np.sum(hx1a))/np.sum(hx1b)

    # second uniform chunk of mesh
    ncx2 = np.ceil((domain_x - dx1)/csx2)
    hx2a = utils.unpack_widths([(csx2, ncx2)])

    # pad to infinity
    hx2b = utils.unpack_widths([(csx2, npadx, pfx2)])

    return np.hstack([hx1a, hx1b, hx2a, hx2b])

def get_casing_model(
    mesh, casing_a, casing_b, casing_l, sigma_back, sigma_casing, sigma_air=1e-4, mur_casing=1
):

    model = np.ones(mesh.n_cells)*sigma_air
    model[mesh.cell_centers[:, 2] < 0] = sigma_back

    mur = np.ones(mesh.n_cells)

    inds_casing_x = (
        (mesh.cell_centers[:, 0] > casing_a) & 
        (mesh.cell_centers[:, 0] < casing_b)
    )
    inds_casing_z = (
        (mesh.cell_centers[:, 2] < 0) &
        (mesh.cell_centers[:, 2] > -casing_l)
    )
    inds_casing = inds_casing_x & inds_casing_z

    model[inds_casing] = sigma_casing
    mur[inds_casing] = mur_casing

    return model, mur

def create_maps(
    mesh, sigma, mur, 
    casing_a, casing_b, casing_t, casing_l, 
    true_inds=False
):
    indsx = (mesh.cell_centers[:, 0] < casing_b)
    if true_inds is True: 
        indsx = indsx & (mesh.cell_centers[:, 0] > casing_b - casing_t)
    
    inds_interior = (
        indsx &
        (mesh.cell_centers[:, 2] <= 0) &
        (mesh.cell_centers[:, 2] >= -casing_l)
    )

    active_inds_sigma = maps.InjectActiveCells(
        mesh, active_cells=inds_interior, 
        value_inactive=np.log(sigma[~inds_interior])
    )

    active_inds_mur = maps.InjectActiveCells(
        mesh, active_cells=inds_interior, 
        value_inactive=mur[~inds_interior]
    )

    projection_sigma = maps.SurjectUnits(indices=[np.ones(inds_interior.sum(), dtype=bool)])
    projection_mu = maps.SurjectUnits(indices=[np.ones(inds_interior.sum(), dtype=bool)])

    exp_map = maps.ExpMap(mesh)
    mur_map = maps.MuRelative(mesh)

    sigma_map = exp_map * active_inds_sigma * projection_sigma
    mu_map = mur_map * active_inds_mur * projection_mu

    return sigma_map, mu_map 

def create_receivers(rx_x, rx_z, components=["x", "z"]):
    rx_list = []
    if "x" in components: 
        rx_ex_re = fdem.receivers.PointElectricField(
            locations=utils.ndgrid(rx_x, np.r_[0], rx_z),
            orientation="x",
            component="real",
        )
        rx_ex_im = fdem.receivers.PointElectricField(
            locations=utils.ndgrid(rx_x, np.r_[0], rx_z),
            orientation="x",
            component="imag",
        )
        rx_list = np.r_[rx_list, rx_ex_re, rx_ex_im]
    if "y" in components: 
        rx_hy_re = fdem.receivers.PointMagneticField(
            locations=utils.ndgrid(rx_x, np.r_[0], rx_z),
            orientation="y",
            component="real",
        )
        rx_hy_im = fdem.receivers.PointMagneticField(
            locations=utils.ndgrid(rx_x, np.r_[0], rx_z),
            orientation="y",
            component="imag",
        )
        rx_list = np.r_[rx_list, rx_hy_re, rx_hy_im]
    if "z" in components: 
        rx_ez_re = fdem.receivers.PointElectricField(
            locations=utils.ndgrid(rx_x, np.r_[0], rx_z),
            orientation="z",
            component="real",
        )
        rx_ez_im = fdem.receivers.PointElectricField(
            locations=utils.ndgrid(rx_x, np.r_[0], rx_z),
            orientation="z",
            component="imag",
        )
        rx_list = np.r_[rx_list, rx_ez_re, rx_ez_im]
    return list(rx_list)

class SaveInversionProgress(directives.InversionDirective):
    """
    A custom directive to save items of interest during the course of an inversion
    """
    
    results_file = None
    
    def initialize(self):
        """
        This is called when we first start running an inversion
        """
        # initialize an empty dictionary for storing results 
        self.inversion_results = {
            "iteration":[],
            "phi_d":[],
            "dpred":[],
            "logsigma":[],
            "mur":[],
            "residual":[],
            "rms":[],
        }

    def endIter(self):
        """
        This is run at the end of every iteration. So here, we just append 
        the new values to our dictionary
        """
        
        # Save the data
        self.inversion_results["iteration"].append(self.opt.iter)
        self.inversion_results["phi_d"].append(self.invProb.phi_d)
        self.inversion_results["dpred"].append(self.invProb.dpred)
        self.inversion_results["logsigma"].append(self.invProb.model[0])
        self.inversion_results["mur"].append(self.invProb.model[1])

        dobs = self.invProb._dmisfit.objfcts[0].data.dobs
        
        self.inversion_results["residual"].append((dobs - self.invProb.dpred))
        self.inversion_results["rms"].append(np.sqrt(np.sum(self.inversion_results["residual"][-1]**2)/len(dobs)))
        
        if self.results_file is not None: 
            with open(f"{self.results_file}", "wb") as fp:
                pickle.dump(self.inversion_results, fp)
        
