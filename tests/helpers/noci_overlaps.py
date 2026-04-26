cisd_eqs = [
    [1.0, "->", ["", ""], ["T", "X"]],
    [1.0, "ji,ij->", ["ii", "ii", ""], ["T", "A", "X"]],
    [1.0, "ji,ij->", ["II", "II", ""], ["T", "B", "X"]],
    [0.5, "jlik,ij,kl->", ["iiii", "ii", "ii", ""], ["T", "A", "A", "X"]],
    [1.0, "jlik,ij,kl->", ["iIiI", "ii", "II", ""], ["T", "A", "B", "X"]],
    [0.5, "jlik,ij,kl->", ["IIII", "II", "II", ""], ["T", "B", "B", "X"]],
]

cisdt_eqs = [
    [1.0, "->", ["", ""], ["T", "X"]],
    [1.0, "ji,ij->", ["ii", "ii", ""], ["T", "A", "X"]],
    [1.0, "ji,ij->", ["II", "II", ""], ["T", "B", "X"]],
    [0.5, "jlik,ij,kl->", ["iiii", "ii", "ii", ""], ["T", "A", "A", "X"]],
    [1.0, "jlik,ij,kl->", ["iIiI", "ii", "II", ""], ["T", "A", "B", "X"]],
    [0.5, "jlik,ij,kl->", ["IIII", "II", "II", ""], ["T", "B", "B", "X"]],
    [
        0.16666666666666666,
        "jlnikm,ij,kl,mn->",
        ["iiiiii", "ii", "ii", "ii", ""],
        ["T", "A", "A", "A", "X"],
    ],
    [0.5, "jlnikm,ij,kl,mn->", ["iiIiiI", "ii", "ii", "II", ""], ["T", "A", "A", "B", "X"]],
    [0.5, "jlnikm,ij,kl,mn->", ["iIIiII", "ii", "II", "II", ""], ["T", "A", "B", "B", "X"]],
    [
        0.16666666666666666,
        "jlnikm,ij,kl,mn->",
        ["IIIIII", "II", "II", "II", ""],
        ["T", "B", "B", "B", "X"],
    ],
]

cisdtq_eqs = [
    [1.0, "->", ["", ""], ["T", "X"]],
    [1.0, "ji,ij->", ["ii", "ii", ""], ["T", "A", "X"]],
    [1.0, "ji,ij->", ["II", "II", ""], ["T", "B", "X"]],
    [0.5, "jlik,ij,kl->", ["iiii", "ii", "ii", ""], ["T", "A", "A", "X"]],
    [1.0, "jlik,ij,kl->", ["iIiI", "ii", "II", ""], ["T", "A", "B", "X"]],
    [0.5, "jlik,ij,kl->", ["IIII", "II", "II", ""], ["T", "B", "B", "X"]],
    [
        0.16666666666666666,
        "jlnikm,ij,kl,mn->",
        ["iiiiii", "ii", "ii", "ii", ""],
        ["T", "A", "A", "A", "X"],
    ],
    [0.5, "jlnikm,ij,kl,mn->", ["iiIiiI", "ii", "ii", "II", ""], ["T", "A", "A", "B", "X"]],
    [0.5, "jlnikm,ij,kl,mn->", ["iIIiII", "ii", "II", "II", ""], ["T", "A", "B", "B", "X"]],
    [
        0.16666666666666666,
        "jlnikm,ij,kl,mn->",
        ["IIIIII", "II", "II", "II", ""],
        ["T", "B", "B", "B", "X"],
    ],
    [
        0.041666666666666664,
        "jlnpikmo,ij,kl,mn,op->",
        ["iiiiiiii", "ii", "ii", "ii", "ii", ""],
        ["T", "A", "A", "A", "A", "X"],
    ],
    [
        0.16666666666666666,
        "jlnpikmo,ij,kl,mn,op->",
        ["iiiIiiiI", "ii", "ii", "ii", "II", ""],
        ["T", "A", "A", "A", "B", "X"],
    ],
    [
        0.25,
        "jlnpikmo,ij,kl,mn,op->",
        ["iiIIiiII", "ii", "ii", "II", "II", ""],
        ["T", "A", "A", "B", "B", "X"],
    ],
    [
        0.16666666666666666,
        "jlnpikmo,ij,kl,mn,op->",
        ["iIIIiIII", "ii", "II", "II", "II", ""],
        ["T", "A", "B", "B", "B", "X"],
    ],
    [
        0.041666666666666664,
        "jlnpikmo,ij,kl,mn,op->",
        ["IIIIIIII", "II", "II", "II", "II", ""],
        ["T", "B", "B", "B", "B", "X"],
    ],
]


def lowdin_pair(cw, cx, metric, thrd=1e-20):
    import numpy as np

    swx = cw.T @ metric @ cx
    if np.max(np.abs(swx - np.diag(np.diag(swx)))) > thrd:
        l, _, v = np.linalg.svd(swx)
        cw, cx = cw @ l, cx @ v.T
        cw[:, 0] *= np.linalg.det(l)
        cx[:, 0] *= np.linalg.det(v.T)
        swx = cw.T @ metric @ cx
    assert np.max(np.abs(swx - np.diag(np.diag(swx)))) < 1e-10
    return np.diag(swx), cw, cx


def reduced_overlap(sxx, thrd=1e-8):
    import numpy as np

    sxx_t = np.array(sxx)
    sxx_t[np.abs(sxx) <= thrd] = 1
    reduced_ovlp = np.prod(sxx[np.abs(sxx) > thrd])
    zeros = np.mgrid[: len(sxx)][np.abs(sxx) <= thrd]
    assert len(zeros) == 0
    return 1 / sxx_t, reduced_ovlp, zeros


# compute <ci|det>
def evaluate(mf, ci_amps, det_mo_coeff):
    import numpy as np

    t_ord = len(ci_amps) - 1
    tensor_eqs = [cisd_eqs, cisdt_eqs, cisdtq_eqs][t_ord - 2]
    metric = np.array(mf.get_ovlp())
    nocca, noccb = (mf.mol.nelectron + mf.mol.spin) // 2, (mf.mol.nelectron - mf.mol.spin) // 2
    cxs = mf.mo_coeff
    cws = det_mo_coeff
    cx_occs = np.array(cxs[0][:, :nocca]), np.array(cxs[1][:, :noccb])
    cw_occs = np.array(cws[0][:, :nocca]), np.array(cws[1][:, :noccb])
    rvlp, xmats, ymats = 1.0, [], []
    for ix, (cx_occ, cw_occ, cx) in enumerate(zip(cx_occs, cw_occs, cxs)):
        sxx, cx_occ, cw_occ = lowdin_pair(cx_occ, cw_occ, metric)
        inv_sxx, reduced_ovlp, zeros = reduced_overlap(sxx)
        mmat = cw_occ @ np.diag(inv_sxx) @ cx_occ.T
        mmat += cx_occ[:, zeros] @ cx_occ[:, zeros].T
        xmat_xx = cx.T @ metric @ mmat @ metric @ cx
        smat_xx = cx.T @ metric @ cx
        ymat_xx = smat_xx - xmat_xx
        rvlp *= reduced_ovlp
        xmats.append(xmat_xx)
        ymats.append(ymat_xx)

    nocca = (mf.mol.nelectron + mf.mol.spin) // 2
    noccb = (mf.mol.nelectron - mf.mol.spin) // 2
    nvira, nvirb = mf.mo_coeff[0].shape[1] - nocca, mf.mo_coeff[0].shape[1] - noccb
    n_occ, n_virt = (nocca, noccb), (nvira, nvirb)

    new_ci_amps = [list(x) for x in ci_amps]
    for ix, xcis in enumerate(ci_amps):
        for k, xci in enumerate(xcis):
            assert (
                xci.shape
                == (n_occ[0],) * (ix - k)
                + (n_occ[1],) * k
                + (n_virt[0],) * (ix - k)
                + (n_virt[1],) * k
            )
            new_ci_amps[ix][k] = np.zeros((n_occ[0] + n_virt[0],) * (ix + ix))
            xx = xci.transpose(*range(ix, ix + ix), *range(ix))
            new_ci_amps[ix][k][
                (slice(n_occ[0], n_occ[0] + n_virt[0]),) * (ix - k)
                + (slice(n_occ[1], n_occ[1] + n_virt[1]),) * k
                + (slice(n_occ[0]),) * (ix - k)
                + (slice(n_occ[1]),) * k
            ] = xx

    prx = lambda idx: (idx.count("I") + idx.count("E")) // 2

    result = 0
    for f, script, idx, nm in tensor_eqs:
        tensors = []
        for inm, iidx in zip(nm[:-1], idx[:-1]):
            if inm == "T":
                tensors.append(new_ci_amps[len(iidx) // 2][prx(iidx)])
            elif inm == "A":
                tensors.append({"ii": xmats[0].T, "ee": ymats[0]}[iidx])
            elif inm == "B":
                tensors.append({"II": xmats[1].T, "EE": ymats[1]}[iidx])
        result += f * np.einsum(script, *tensors, optimize=True)
    return result * rvlp
