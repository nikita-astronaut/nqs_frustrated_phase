from ._core import System


def _chain(length, periodic=True):
    Chain = type("Chain{}".format(length), (System,), {})
    Chain.POSITIONS = [(i, 0) for i in range(length)]
    Chain.J1_EDGES = [(i, (i + 1) % length) for i in range(length - 1 + periodic)]
    Chain.J2_EDGES = []

    return Chain


Chain10 = _chain(10)
Chain12 = _chain(12)
Chain14 = _chain(14)
Chain16 = _chain(16)
