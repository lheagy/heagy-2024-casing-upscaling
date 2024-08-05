import numpy as np
from discretize import utils

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

