import os
from typing import List, Tuple
import numpy as np

def write_interall(spec: List[Tuple[float, List[Tuple[int, int]]]], output: str):
    r"""Generates InterAll file for HPhi given Heisenberg Hamiltonian
    specification.

    :param spec: is a list of tuples of the form ``(J, edges)`` where ``J`` is a real
        coupling and ``edges`` is a list of edges.
    :param output: output filename.
    """

    def link(i, j, coupling=1.0):
        assert i < j
        return [
            (i, 0, i, 1, j, 1, j, 0, 0.5 * coupling, 0.0),
            (i, 1, i, 0, j, 0, j, 1, 0.5 * coupling, 0.0),
            (i, 0, i, 0, j, 0, j, 0, 0.25 * coupling, 0.0),
            (i, 0, i, 0, j, 1, j, 1, -0.25 * coupling, 0.0),
            (i, 1, i, 1, j, 0, j, 0, -0.25 * coupling, 0.0),
            (i, 1, i, 1, j, 1, j, 1, 0.25 * coupling, 0.0),
        ]

    interactions = sum(
        (link(i, j, coupling) for coupling, edges in spec for i, j in edges), []
    )
    with open(output, "w") as f:
        f.write(
            "======================\n"
            "NInterAll      {}\n"
            "======================\n"
            "========zInterAll=====\n"
            "======================\n"
            "".format(len(interactions))
        )
        for t in interactions:
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(*t))


def write_locspin(n: int, output: str):
    r"""Generates LocSpin file for HPhi.

    :param n: number of spins in the system.
    :param output: output filename.
    """
    assert n > 0
    with open(output, "w") as f:
        f.write(
            "================================\n"
            "NlocalSpin    {}\n"
            "================================\n"
            "========i_0LocSpn_1IteElc ======\n"
            "================================\n"
            "".format(n)
        )
        for i in range(n):
            f.write("{}\t1\n".format(i))


def write_calcmod(calc_type: str, output: str):
    calc_type = {"lanczos": 0, "cg": 3}[calc_type.lower()]
    with open(output, "w") as f:
        f.write(
            "CalcType\t{}\n"
            "CalcModel\t1\n"
            "CalcEigenVec\t0\n"
            "InitialVecType\t1\n"
            "OutputEigenVec\t1\n"
            "".format(calc_type)
        )


def write_modpara(n: int, output: str):
    assert n > 0
    with open(output, "w") as f:
        f.write(
            "--------------------\n"
            "Model_Parameters   0\n"
            "--------------------\n"
            "VMC_Cal_Parameters\n"
            "--------------------\n"
            "CDataFileHead  zvo\n"
            "CParaFileHead  zqp\n"
            "--------------------\n"
            "Nsite          {}\n"
            "2Sz            {}\n"
            "initial_iv     -1\n"
            "Lanczos_max    2000\n"
            "exct           1\n"
            "LanczosEps     14\n"
            "LanczosTarget  2\n"
            "".format(n, n % 2)
        )


def write_settings(cls, j2: float, calc_type: str = "cg", workdir: str = "workdir"):
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    n = len(cls.POSITIONS)
    calcmod = "calcmod.def"
    modpara = "modpara.def"
    locspin = "locspin.def"
    interall = "interall.def"
    write_calcmod(calc_type, output=os.path.join(workdir, calcmod))
    write_modpara(n, output=os.path.join(workdir, modpara))
    write_locspin(n, output=os.path.join(workdir, locspin))
    write_interall(
        [(4.0, cls.J1_EDGES), (4.0 * j2, cls.J2_EDGES)],
        output=os.path.join(workdir, interall),
    )
    with open(os.path.join(workdir, "namelist.def"), "w") as output:
        output.write(
            "CalcMod   {}\n"
            "ModPara   {}\n"
            "LocSpin   {}\n"
            "InterAll  {}\n"
            "".format(calcmod, modpara, locspin, interall)
        )


def load_eigenvector(filename):
    # fp = fopen("zvo_eigenvec_0_rank_0.dat", "wb");
    # fwrite(&number_of_interations, sizeof(int), 1,fp);
    # fwrite(&local_size, sizeof(unsigned long int),1,fp);
    # fwrite(&eigen_vector[0], sizeof(complex double),local_size+1, fp);
    # fclose(fp);
    from ctypes import sizeof, c_int, c_ulong, c_double

    offset = sizeof(c_int) + sizeof(c_ulong) + 2 * sizeof(c_double)
    with open(filename, "rb") as input:
        input.read(offset)
        xs = np.fromfile(input, dtype=np.complex128, count=-1, sep="")
        return xs.real.astype(np.float32)
