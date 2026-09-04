from collections.abc import Iterable, MutableSet, Sequence
from numbers import Number

import numpy as np

from pytrf import date
from pytrf.ts import fexp, flog, model, polynom


class MJDStore(MutableSet[float], Sequence[float]):
    def __init__(self, error: float = 0, iter: Iterable[float] | None = None):
        self._error = error
        self._set: set[float] = set(iter or [])
        self._list: list[float] = sorted(self._set)

    def add(self, value: float):
        if value in self:
            return

        self._set.add(value)
        self._list = sorted(self._set)

    def discard(self, value: float):
        self._set.discard(value)
        self._list = sorted(self._set)

    def __getitem__(self, index):
        return self._list.__getitem__(index)

    def __len__(self):
        return len(self._list)

    def __contains__(self, value):
        if not isinstance(value, Number):
            return False

        if len(self._list) == 0:
            return False

        start = 0
        end = len(self._list) - 1
        while start < end:
            i = start + (start + end) // 2
            v = self._list[i]
            if abs(value - v) <= self._error:
                return True
            if value > v:
                start = i + 1
            else:
                end = i

        return False

    def __iter__(self):
        return self._list.__iter__()

    def __str__(self):
        return self._list.__str__()

    def __eq__(self, other):
        if not isinstance(other, MJDStore):
            return False
        if len(other) != len(self):
            return False
        if self._error != other._error:
            return False

        for p in self:
            if p not in other:
                return False
        return True


def find_soln_station(solns: list, code: str, pt=None):
    for soln in solns:
        if pt is None:
            if code == soln.code:
                return soln
        elif code + pt == soln.code + soln.pt:
            return soln
    return None


def parse_soln_dates(
    soln, time_range: tuple[float, float] = (-np.inf, np.inf), error: float = 0
):
    tmin, tmax = time_range

    soln_x = set()
    soln_p = MJDStore(error)
    soln_v = MJDStore(error)

    for x in soln.X:
        # Get start and end MJDs
        start = -np.inf
        if x.start != "00:000:00000":
            start = date.from_tsnx(x.start).mjd

        end = np.inf
        if x.end != "00:000:00000":
            end = date.from_tsnx(x.end).mjd

        soln_x.add((start, end))

    # Full list of position discontinuities
    for p in soln.P:
        if p.end == "00:000:00000":
            continue
        t = date.from_tsnx(p.end).mjd
        if not tmin <= t <= tmax:
            continue
        soln_p.add(t)

    # Full list of velocity discontinuities
    for p in soln.V:
        if p.end == "00:000:00000":
            continue
        t = date.from_tsnx(p.end).mjd
        if not tmin <= t <= tmax:
            continue
        soln_v.add(t)

    return (soln_x, soln_p, soln_v)


def update_model_with_soln(
    model: model,
    solns: list,
    code: str,
    pt=None,
):
    soln = find_soln_station(solns, code, pt)
    if soln is None:
        return

    tmin = model.r.t[0]
    tmax = model.r.t[-1]

    soln_x, soln_p, soln_v = parse_soln_dates(soln, (tmin, tmax))

    for start, end in soln_x:
        # Delete observations within current period
        ind = np.nonzero((model.r.t >= start) * (model.r.t <= end))[0]
        model.del_points(ind)

    tp = list(soln_p)
    tv = list(soln_v)

    # Remove unobserved position discontinuities
    j = 0
    while j < len(tp) - 1:
        if np.sum((model.r.t >= tp[j]) * (model.r.t < tp[j + 1])) == 0:
            tp.pop(j)
        else:
            j += 1

    # Remove unobserved velocity discontinuities
    j = 0
    while j < len(tv) - 1:
        if np.sum((model.r.t >= tv[j]) * (model.r.t < tv[j + 1])) == 0:
            tv.pop(j)
        else:
            j += 1

    # Add discontinuities to model
    model.add_jumps(tp, deg=[0])
    model.add_jumps(tv, deg=[1])


def is_model_and_soln_in_sync(model: model, soln, error: float = 2 / 86_400) -> bool:
    tmin = model.r.t[0]
    tmax = model.r.t[-1]

    soln_parsed = parse_soln_dates(soln, (tmin, tmax), error)

    position = MJDStore(error)
    velocity = MJDStore(error)

    for sub_model in model:
        for fn in sub_model.f:
            if not isinstance(fn, polynom):
                continue

            jumps = {t for t in fn.t if tmin <= t <= tmax}
            if fn.deg == 0:
                position |= jumps
                continue
            if fn.deg == 1:
                velocity |= jumps

    return position == soln_parsed[1] and velocity == soln_parsed[2]


def is_model_and_psd_in_sync(
    model: model,
    psd,
    code: str,
    pt: str | None = None,
    dims: str = "ENU",
    error: float = 2 / 84_600,
) -> bool:
    assert len(dims) == model.nd
    count = {}

    for i in range(model.nd):
        count[i] = 0

    for i in range(psd.npar):
        param = psd.param[i]

        if param.type[0] != "A":
            continue
        if code != param.code:
            continue

        if pt is not None and pt != param.pt:
            continue

        if param.type[5] not in dims:
            continue
        model_index = dims.index(param.type[5])

        t = date.from_tsnx(param.tref).mjd
        search = None
        if param.type[1:4] == "EXP":
            search = fexp
        elif param.type[1:4] == "LOG":
            search = flog
        if search is None:
            continue

        for f in model[model_index].f:
            if not isinstance(f, search):
                continue
            if abs(f.t0 - t) > error:
                return False

            count[model_index] += 1

    for i in range(model.nd):
        sub_model = model[i]
        n = 0
        for f in sub_model.f:
            if isinstance(f, (fexp, flog)):
                n += 1
        if n != count[i]:
            return False

    return True


def remove_from_list_instance_of(l: list, t: type):
    assert isinstance(l, list)

    i = 0
    while i < len(l):
        if isinstance(l[i], t):
            l.pop(i)
        else:
            i += 1


def clear_all_discontinuities(model: model):
    for sub_model in model:
        for i in range(len(sub_model.f)):
            f = sub_model.f[i]
            if not isinstance(f, polynom):
                continue
            sub_model.f[i] = polynom(f.deg, [])
