"""
pytrf GUI State objects
"""

import datetime
import functools
import math
import os
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import (
    Hashable,
    Iterable,
    MutableMapping,
)
from dataclasses import dataclass
from enum import Enum, Flag, IntFlag, auto
from typing import Any, ClassVar, Final, overload

import numpy as np
import yaml

from pytrf import date, sinex
from pytrf.gui.utils import (
    clear_all_discontinuities,
    find_soln_station,
    is_model_and_psd_in_sync,
    is_model_and_soln_in_sync,
    remove_from_list_instance_of,
    update_model_with_soln,
)
from pytrf.io import (
    get_sitelog,
    read_sitelog,
    read_solns,
    read_yaml,
    sitelog_coord,
    update_solns,
)
from pytrf.math import cart2geo
from pytrf.ts import (
    ar1,
    fexp,
    flog,
    ggm,
    model,
    pl,
    pl_index,
    polynom,
    scale_param,
    sine,
    ts,
    vw,
    wn,
)
from pytrf.utils import record
from pytrf.webigs import (
    WebIGS,
    WebIGSCoseismicRequestStation,
    WebIGSCoseismicResponseCoseismicEntry,
)

TIME_ERROR = 2 / 86_400  # 2 sec for mjd


def call_time_prefix(prefix: str):
    def _call_time(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            result = f(*args, **kwargs)
            end_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            value = (end_ns - start_ns) / 10**6

            if prefix is None or len(prefix) == 0:
                print(f.__name__ + ":", value, "ms")
            else:
                print(
                    prefix + "." + f.__name__ + ":",
                    value,
                    "ms",
                )

            return result

        return wrapper

    return _call_time


call_time = call_time_prefix("")


class ModelModifier(ABC):
    """
    Protocol representing the abitility of an object to modify a pytrf model
    """

    @abstractmethod
    def apply(self, base_model: model) -> None:
        """
        Modify base_model
        """
        raise NotImplementedError()


class SinusoidalPeriod(Hashable, ModelModifier):
    """
    Object representing a sinusoidal period in a pytrf model (see: `pytrf.ts.sine`)
    """

    value: float
    """
    Period of the sinusoidal
    """

    def __init__(self, value: float):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, value):
        return isinstance(value, SinusoidalPeriod) and self.value == value.value

    def apply(self, base_model: model) -> None:
        assert isinstance(base_model, model)

        base_model.add_sine(self.value)

    def __str__(self):
        return f"SinusoidalPeriod({self.value})"


@dataclass
class GenericOption[T]:
    name: str
    """
    Option display name
    """
    key: str
    """
    Option key in the object
    """
    t: type
    """
    Type of the option value
    """
    default_value: Any = None
    """
    Default value of the option
    """

    @staticmethod
    def create_with_constructor[U](
        ctr: type[U], name: str, key: str, default_value=None
    ) -> "GenericOption[U]":
        """
        Create a `GenericOption` from the constructor ex:
        ```
        class Foo:
            bar: str

        opt = GenericOption.create_with_constructor(Foo, "Bar", "bar")
        ```
        """
        assert key in ctr.__annotations__

        t = ctr.__annotations__[key]
        return GenericOption(name=name, key=key, t=t, default_value=default_value)

    @staticmethod
    def create_with_instance[U](
        component: U, name: str, key: str, default_value=None
    ) -> "GenericOption[U]":
        """
        Create a `GenericOption` from the instance ex:
        ```
        class Foo:
            bar: str

        opt = GenericOption.create_with_instance(Foo(), "Bar", "bar")
        ```
        """
        return GenericOption.create_with_constructor(
            component.__class__, name, key, default_value
        )

    def validate(self, value):
        """
        Check that the value passed in argument is of type t.
        """
        if value is None and self.default_value is not None:
            return True

        return isinstance(value, self.t)

    def inject(self, value, instance: T):
        """
        Inject into instance the value. If the value is None the default value is used.
        """
        assert self.validate(value)

        if value is None:
            instance.__setattr__(self.key, self.default_value)
        else:
            instance.__setattr__(self.key, value)

    def get(self, instance: T):
        """
        Get the value injected from instance.
        """
        value = getattr(instance, self.key, None)

        if value is None:
            return self.default_value
        return value


class NoiseComponent(ModelModifier):
    """
    Base class for representing a noise component
    """

    name: str
    """
    Name of the component
    """
    OPTIONS: ClassVar[list[GenericOption["NoiseComponent"]]] = []
    """
    Options needed for the component
    """
    _enabled: bool

    @staticmethod
    def from_model(base_model: model) -> "NoiseComponent":
        """
        Get the component option from base_model
        """
        raise NotImplementedError()

    @staticmethod
    def clear(base_model: model):
        """
        Remove the component from base_model
        """
        raise NotImplementedError()

    def __init__(self, enabled: bool = False):
        self._enabled = enabled

    def __eq__(self, value):
        if not isinstance(value, NoiseComponent):
            return False
        return self._enabled == value._enabled

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def enabled(self) -> bool:
        return self._enabled

    def apply(self, base_model: model) -> None:
        if not self._enabled:
            return
        self.apply_impl(base_model)

    def apply_impl(self, base_model: model) -> None:
        """
        Must be implemented by component.
        """
        raise NotImplementedError()


class VariableWhiteNoiseComponent(NoiseComponent):
    """
    Noise component of type `pytrf.ts.vw`
    """

    name = "Variable white"

    def apply_impl(self, base_model):
        base_model.add_vw()

    def __eq__(self, value):
        return isinstance(value, VariableWhiteNoiseComponent) and super().__eq__(value)

    @staticmethod
    def from_model(base_model: model) -> "VariableWhiteNoiseComponent":
        for sub_model in base_model:
            for noise in sub_model.n:
                if isinstance(noise, vw):
                    return VariableWhiteNoiseComponent(True)

        return VariableWhiteNoiseComponent(False)

    @staticmethod
    def clear(base_model: model):
        for sub_model in base_model:
            remove_from_list_instance_of(sub_model.n, vw)


class WhiteNoiseComponent(NoiseComponent):
    """
    Noise component of type `pytrf.ts.wn`
    """

    name = "White noise"

    def __eq__(self, value):
        if not super().__eq__(value):
            return False
        return isinstance(value, WhiteNoiseComponent)

    def apply_impl(self, base_model):
        base_model.add_wn()

    @staticmethod
    def from_model(base_model: model):
        for sub_model in base_model:
            for noise in sub_model.n:
                if isinstance(noise, wn):
                    return WhiteNoiseComponent(True)
        return WhiteNoiseComponent(False)

    @staticmethod
    def clear(base_model: model):
        for sub_model in base_model:
            remove_from_list_instance_of(sub_model.n, wn)


class AROneNoiseComponent(NoiseComponent):
    """
    Noise component of type `pytrf.ts.ar1`
    """

    name = "AR(1)"
    OPTIONS: ClassVar = [
        GenericOption(name="Correlation time", key="correlation_time", t=int)
    ]

    correlation_time: int

    def __eq__(self, value):
        return super().__eq__(value) and isinstance(value, AROneNoiseComponent)

    def apply_impl(self, base_model):
        base_model.add_ar1(tau=self.correlation_time)

    @staticmethod
    def from_model(base_model: model):
        for sub_model in base_model:
            for noise in sub_model.n:
                if isinstance(noise, ar1):
                    result = AROneNoiseComponent(True)
                    for par in noise.par:
                        if (
                            isinstance(par, scale_param)
                            and par.type == "AR(1) correlation time"
                        ):
                            result.correlation_time = par.x
                    return result

        return AROneNoiseComponent(False)

    @staticmethod
    def clear(base_model: model):
        for sub_model in base_model:
            remove_from_list_instance_of(sub_model.n, ar1)


class PowerLawNoiseComponent(NoiseComponent):
    """
    Noise component of type `pytrf.ts.pl`

    """

    name = "Power law"
    OPTIONS: ClassVar = [
        GenericOption(name="Spectral index", key="spectral_index", t=int)
    ]

    spectral_index: int

    def apply_impl(self, base_model):
        assert isinstance(self.spectral_index, int)

        base_model.add_pl(a=self.spectral_index)

    @staticmethod
    def from_model(base_model: model):
        for sub_model in base_model:
            for noise in sub_model.n:
                if not isinstance(noise, pl):
                    continue
                result = PowerLawNoiseComponent(True)
                si = None
                for p in noise.par:
                    if isinstance(p, pl_index):
                        si = p.x
                        break

                result.spectral_index = si
                return result

        return PowerLawNoiseComponent(False)

    @staticmethod
    def clear(base_model: model):
        for sub_model in base_model:
            remove_from_list_instance_of(sub_model.n, pl)


class GeneralizedGMNoiseComponent(NoiseComponent):
    """
    Noise component of type `pytrf.ts.ggm`
    """

    name = "Generalized GM"

    OPTIONS: ClassVar = [
        GenericOption(name="Spectral index", key="spectral_index", t=int),
        GenericOption(name="Correlation time", key="correlation_time", t=int),
    ]

    spectral_index: int
    correlation_time: int

    def apply_impl(self, base_model):
        assert isinstance(self.spectral_index, int)
        assert isinstance(self.correlation_time, int)

        base_model.add_ggm(a=self.spectral_index, tau=self.correlation_time)

    @staticmethod
    def from_model(base_model: model):
        for sub_model in base_model:
            for noise in sub_model.n:
                if not isinstance(noise, ggm):
                    continue
                si = None
                ct = None
                for param in noise.par:
                    if param.type == "GGM spectral index":
                        si = param.x
                        continue
                    if param.type == "GGM correlation time":
                        ct = param.x

                result = GeneralizedGMNoiseComponent(True)
                result.spectral_index = si
                result.correlation_time = ct

        return GeneralizedGMNoiseComponent(False)

    @staticmethod
    def clear(base_model: model):
        for sub_model in base_model:
            remove_from_list_instance_of(sub_model.n, ggm)


class DiscontinuityElement(Enum):
    position = "position"
    velocity = "velocity"

    def degree(self):
        if self == DiscontinuityElement.position:
            return 0
        if self == DiscontinuityElement.velocity:
            return 1
        raise ValueError(
            f"Cannot determine degree of discontinuity element: {self.name} = {self.value}"
        )

    @staticmethod
    def from_degree(degree: int) -> "DiscontinuityElement | None":
        if degree == 0:
            return DiscontinuityElement.position
        if degree == 1:
            return DiscontinuityElement.velocity
        return None


class Direction(Enum):
    east = "east"
    north = "north"
    up = "up"

    @staticmethod
    def all() -> list["Direction"]:
        return [Direction.east, Direction.north, Direction.up]

    def direction_to_index(self):
        arr = Direction.all()
        result = arr.index(self)

        if result == -1:
            raise ValueError(
                f"Cannot determine the index of direction: {self.name} = {self.value}"
            )
        return result

    @staticmethod
    def index_to_direction(i: int) -> "Direction | None":
        arr = Direction.all()
        if i < 0 or i >= len(arr):
            return None
        return arr[i]


class DiscontinuityType(Enum):
    antenna_change = "antenna change"
    receiver_change = "receiver change"
    earthquake = "earthquake"
    unknown = "unknown"
    other = "other"

    def __str__(self):
        return self.value

    @staticmethod
    def all():
        return [
            DiscontinuityType.antenna_change,
            DiscontinuityType.receiver_change,
            DiscontinuityType.earthquake,
            DiscontinuityType.unknown,
            DiscontinuityType.other,
        ]

    @staticmethod
    def from_str(s: str):
        for t in DiscontinuityType.all():
            if str(t) == s:
                return t
        raise ValueError("Invalid type")

    @staticmethod
    def from_description(desc: str) -> tuple["DiscontinuityType", str]:
        desc_clean = desc.strip().lower()

        if desc_clean == "antenna change":
            return DiscontinuityType.antenna_change, ""
        if desc_clean == "receiver change":
            return DiscontinuityType.receiver_change, ""
        if desc_clean.startswith("eq"):
            return DiscontinuityType.earthquake, desc.removeprefix("EQ ")
        if desc_clean == "unknown":
            return DiscontinuityType.unknown, ""
        return DiscontinuityType.other, desc


class Discontinuity(ModelModifier):
    date: float
    """
    Date in mjd format
    """
    description: str
    """
    Description of discontinuity
    """
    type: DiscontinuityType
    """
    Type of discontinuity
    """

    elements: set[DiscontinuityElement]

    exp_components: set[Direction]
    log_components: set[Direction]

    container: "None | DiscontinuityMap" = None

    tooltip: str | None = None

    @overload
    def __init__(self, discontinuity: "Discontinuity", /): ...
    @overload
    def __init__(
        self,
        date: float,
        description: str,
        type: DiscontinuityType = DiscontinuityType.other,
        elements: set[DiscontinuityElement] | None = None,
        /,
    ): ...

    def __init__(
        self,
        discontinuity: "Discontinuity | float",
        description: str | None = None,
        type: DiscontinuityType = DiscontinuityType.other,
        elements: set[DiscontinuityElement] | None = None,
        /,
    ):
        if isinstance(discontinuity, Discontinuity):
            self.date = discontinuity.date
            self.description = discontinuity.description
            self.elements = set(discontinuity.elements)
            self.type = discontinuity.type
            return
        assert description is not None
        assert isinstance(discontinuity, (int, float))
        self.date = discontinuity
        self.description = description
        self.type = type
        self.elements = set() if elements is None else elements
        self.exp_components = set()
        self.log_components = set()

    def __eq__(self, value):
        if not isinstance(value, Discontinuity):
            return False

        if self.date != value.date:
            return False

        if self.elements != value.elements:
            return False

        if self.log_components != value.log_components:
            return False

        return self.exp_components == value.exp_components

    def apply(self, base_model: model) -> None:
        assert isinstance(base_model, model)

        deg = {element.degree() for element in self.elements}
        base_model.add_jumps([self.date], deg=deg)

        for exp_component in self.exp_components:
            base_model[exp_component.direction_to_index()].add_exp(self.date)

        for log_component in self.log_components:
            base_model[log_component.direction_to_index()].add_log(self.date)

    def is_empty(self) -> bool:
        """
        Return True if no components is present
        """
        return (
            len(self.elements) == 0
            and len(self.exp_components) == 0
            and len(self.log_components) == 0
        )

    def color_graph(self) -> str:
        if self.type == DiscontinuityType.antenna_change:
            return "blue"
        if self.type == DiscontinuityType.receiver_change:
            return "cyan"
        if self.type == DiscontinuityType.earthquake:
            return "orange"
        if self.type == DiscontinuityType.other:
            return "#bb00ff"

        return "red"

    def color_display(self) -> tuple[str, str]:
        if self.type == DiscontinuityType.antenna_change:
            return "#000000", "#3498db"
        if self.type == DiscontinuityType.receiver_change:
            return "#000000", "#1abc9c"
        if self.type == DiscontinuityType.earthquake:
            return "#000000", "#e67e22"
        if self.type == DiscontinuityType.other:
            return "#000000", "#d980fa"

        return "#000000", "#e74c3c"

    def __str__(self):
        return (
            f"Discontinuity({self.date}, {self.description}) "
            + "{\n  "
            + ",\n  ".join(str(e) for e in self.elements)
            + "\n}"
        )

    def set_date(self, value: float):
        if self.container is None:
            self.date = value
            return
        self.container.move(self.date, value)
        self.date = value

    def snx_cause(self) -> str:
        if DiscontinuityType.other:
            return self.description

        result = "EQ"
        if self.type != DiscontinuityType.earthquake:
            result = str(self.type)
        if len(self.description) > 0:
            result += " "
            result += self.description
        return result


class DiscontinuityMap(MutableMapping[float, Discontinuity]):
    def __init__(
        self,
        initial: Iterable[Discontinuity] | None = None,
    ):
        self._data: dict[float, Discontinuity] = {}

        if initial is not None:
            for discontinuity in initial:
                self[discontinuity.date] = discontinuity

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if key in self._data:
            raise KeyError("Cannot overwrite value")
        assert value.container is None
        value.container = self
        self._data[key] = value

    def __delitem__(self, key):
        v = self._data.pop(key)
        v.container = None

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return self._data.__len__()

    def remove(self, value: Discontinuity):
        self.discard(value)

    def discard(self, value: Discontinuity):
        del self[value.date]

    def move(self, old_key: float, new_key: float):
        self[new_key] = self[old_key]
        del self[old_key]

    def add(self, discontinuity: Discontinuity):
        self[discontinuity.date] = discontinuity

    def only_enabled_discontinuities(self):
        result = []
        for value in self.values():
            if value.is_empty():
                continue
            result.append(value)
        return result

    def get_discontinuity_for_date(
        self, date: float, error: float = 0
    ) -> Discontinuity | None:
        for d, discontinuity in self._data.items():
            if abs(d - date) <= error:
                return discontinuity
        return None

    def replace(self, t: float, value: Discontinuity):
        del self[t]
        self[t] = value

    def get_or_create(self, t: float, error: float = 0, description: str = ""):
        for d, discontinuity in self._data.items():
            if abs(d - t) <= error:
                return discontinuity
        self[t] = Discontinuity(t, description)
        return self[t]


class DeterministicModel(ModelModifier):
    """
    Object representing the deterministic part of a `pytrf.ts.model`
    """

    sinusoidal_periods: set[SinusoidalPeriod]
    discontinuities: DiscontinuityMap

    @call_time_prefix("DeterministicModel")
    def is_in_sync(self, base_model: model):
        return self == DeterministicModel.from_model(base_model)

    def __eq__(self, value):
        if not isinstance(value, DeterministicModel):
            return False

        if self.sinusoidal_periods != value.sinusoidal_periods:
            return False

        return (
            self.discontinuities.only_enabled_discontinuities()
            == value.discontinuities.only_enabled_discontinuities()
        )

    def add_sitelog_discontinuities(self, data, t_start: float, t_end: float):
        """
        Add discontinuities from sitelog in data.

        Parameters
        ----------
        data
            expect the output of the function `pytrf.io.read_sitelog`
        """
        if data is None:
            return

        receivers, antennas, _ = data

        for events, t in (
            (receivers, DiscontinuityType.receiver_change),
            (antennas, DiscontinuityType.antenna_change),
        ):
            for event in events:
                date_installed = date.from_tsnx(event.start).mjd
                if not t_start <= date_installed <= t_end:
                    continue

                discontinuity = self.discontinuities.get_or_create(
                    date_installed, TIME_ERROR
                )
                discontinuity.type = t

    @call_time_prefix("DeterministicModel")
    def apply(self, base_model: model):
        for period in self.sinusoidal_periods:
            period.apply(base_model)
        for discontinuity in self.discontinuities.values():
            discontinuity.apply(base_model)

    @staticmethod
    @call_time_prefix("DeterministicModel")
    def from_model(base_model: model):
        """
        Create an instance of DeterministicModel using the model data:

        - sine
        - polynom jumps
        - exp
        - log
        """
        dates = DiscontinuityMap()
        sines = set()

        t_start = base_model.r.t[0]
        t_end = base_model.r.t[-1]

        for i in range(len(base_model)):
            sub_model = base_model[i]
            for fn in sub_model.f:
                if isinstance(fn, sine):
                    sines.add(SinusoidalPeriod(fn.per))
                    continue
                if isinstance(fn, polynom):
                    degree = DiscontinuityElement.from_degree(fn.deg)
                    if degree is None:
                        continue
                    for t in fn.t:
                        if not t_start <= t <= t_end:
                            continue
                        dates.get_or_create(
                            t, TIME_ERROR, description="imported from model"
                        ).elements.add(degree)
                    continue
                if not isinstance(fn, (fexp, flog)):
                    continue

                direction = Direction.index_to_direction(i)
                if direction is None:
                    continue

                t = fn.t0
                if not t_start <= t <= t_end:
                    continue

                discontinuity = dates.get_or_create(
                    t, TIME_ERROR, description="imported from model"
                )

                if isinstance(fn, fexp):
                    discontinuity.exp_components.add(direction)
                else:
                    discontinuity.log_components.add(direction)

        result = DeterministicModel()
        result.discontinuities = dates
        result.sinusoidal_periods = sines
        return result

    def __str__(self):
        return (
            "DeterministicModel()\n"
            "discontinuities: {\n  "
            + ",\n".join(str(d) for d in self.discontinuities).replace("\n", "\n  ")
            + "\n}\n"
            + "sines: {\n  "
            + ",\n".join(str(s) for s in self.sinusoidal_periods).replace("\n", "\n  ")
            + "\n}"
        )

    @staticmethod
    @call_time
    def clear(base_model: model) -> None:
        """
        Clear what can be recovered from the method from_model.
        """
        for sub_model in base_model:
            i = 0
            while i < len(sub_model.f):
                fn = sub_model.f[i]
                if isinstance(fn, polynom):
                    sub_model.f[i] = polynom(
                        fn.deg,
                        [],
                    )
                if isinstance(fn, (sine, flog, fexp)):
                    sub_model.f.pop(i)
                else:
                    i += 1


class StochasticModel(ModelModifier):
    """
    The noise part of the `pytrf.ts.model`
    """

    NOISE_COMPONENTS: Final[list[type[NoiseComponent]]] = [
        VariableWhiteNoiseComponent,
        WhiteNoiseComponent,
        AROneNoiseComponent,
        PowerLawNoiseComponent,
        GeneralizedGMNoiseComponent,
    ]
    """
    Noise component available
    """
    components: list[NoiseComponent]

    def __init__(self, components: list[NoiseComponent] | None = None):
        if components is not None:
            assert len(components) == len(self.NOISE_COMPONENTS)

            for i in range(len(self.NOISE_COMPONENTS)):
                assert isinstance(components[i], self.NOISE_COMPONENTS[i])
            self.components = list(components)
            return

        self.components = []
        for ctr in self.NOISE_COMPONENTS:
            self.components.append(ctr())

    def apply(self, base_model):
        for component in self.components:
            component.apply(base_model)

    def is_in_sync(self, base_model: model) -> bool:
        """
        Check if base_model is in the same state as this object
        """
        return True  # TODO

    @staticmethod
    def from_model(base_model: model) -> "StochasticModel":
        result = []
        for ctr in StochasticModel.NOISE_COMPONENTS:
            result.append(ctr.from_model(base_model))
        return StochasticModel(result)

    @staticmethod
    def clear(base_model: model):
        for ctr in StochasticModel.NOISE_COMPONENTS:
            ctr.clear(base_model)


class ModelFitConfigOption:
    name: str
    """
    Name of the option
    """
    OPTIONS: ClassVar[list[GenericOption["ModelFitConfigOption"]]] = []
    _enabled = False

    def enabled(self):
        return self._enabled

    def set_enabled(self, v: bool):
        self._enabled = v

    def args(self) -> dict:
        if not self._enabled:
            return {}
        return self._args()

    def _args(self) -> dict:
        raise NotImplementedError()


class NormalizedResidualsModelFitOption(ModelFitConfigOption):
    name = "Normalized Residuals"

    OPTIONS: ClassVar = [GenericOption("Threshold", "threshold", float)]

    threshold: float

    def _args(self):
        return {"thr_norm": self.threshold}


class XTimesWRMSModelFitOption(ModelFitConfigOption):
    name = "X times WRMS"

    OPTIONS: ClassVar = [GenericOption("Threshold", "threshold", float)]

    threshold: float

    def _args(self):
        return {"thr_raw": self.threshold}


class MedianXTimesWRMSFitOption(ModelFitConfigOption):
    name = "median +- X times WRMS"

    OPTIONS: ClassVar = [
        GenericOption("Threshold", "threshold", float),
        GenericOption("Window length", "window_length", float),
    ]

    threshold: float
    window_length: float

    def _args(self):
        return {"thr_mad": self.threshold, "win_mad": self.window_length}


class RawResidualsFitOption(ModelFitConfigOption):
    name = "Raw Residuals"

    OPTIONS: ClassVar = [
        GenericOption("East (mm)", "east", float),
        GenericOption("North (mm)", "north", float),
        GenericOption("Up (mm)", "up", float),
    ]

    east: float
    north: float
    up: float

    def _args(self):
        return {"thr_abs": [self.east, self.north, self.up]}


class ModelFitConfig:
    OPTIONS: Final[list[type[ModelFitConfigOption]]] = [
        NormalizedResidualsModelFitOption,
        XTimesWRMSModelFitOption,
        MedianXTimesWRMSFitOption,
        RawResidualsFitOption,
    ]
    _options: list[ModelFitConfigOption]

    def __init__(self):
        self._options = []
        for option in self.OPTIONS:
            self._options.append(option())

    def args(self) -> dict:
        result: dict[str, Any] = {}
        for option in self._options:
            result |= option.args()
        return result

    def options(self) -> list[ModelFitConfigOption]:
        return list(self._options)


class StationCoseismic:
    _data: None | list[WebIGSCoseismicResponseCoseismicEntry] = None

    def __init__(
        self,
        name: str,
        span: tuple[datetime.date, datetime.date],
        position: tuple[float, float],
    ):
        self._name = name
        self._span = span
        self._position = position

    def data(self):
        if self._data is None:
            self.refresh()
        return self._data

    def refresh(self):
        self._data = []
        igs = WebIGS()
        result = igs.coseismic_request(
            self._span[0],
            self._span[1],
            [WebIGSCoseismicRequestStation(self._name, self._position, self._span[0])],
        )
        if len(result["result"]) == 0:
            return
        stations = result["result"][0]["stations"]
        if len(stations) == 0:
            return

        self._data = stations[0]["coseismic"]


class ModelAndSNXSyncStates(Flag):
    NONE = 0
    SOLN_OUT_OF_SYNC = auto()
    PSD_OUT_OF_SYNC = auto()


class Station:
    """
    Object representing a station
    """

    name: str
    """
    Code/Name of the station
    """

    _deterministic_model: DeterministicModel | None = None
    _stochastic_model: StochasticModel | None = None
    _fit_config: ModelFitConfig

    _active: bool = False
    _saved: bool = False
    _model: model | None = None
    _model_path: str | None
    _sitelogs_path: str | None
    _discontinuity_path: str | None
    _psd_path: str | None
    _coseismic: StationCoseismic | None = None

    _ts_path: str

    _notes_path: str | None = None
    _notes: str | None = None
    _notes_saved: bool = True

    @staticmethod
    def from_configuration(
        name: str, configuration: "ProjectConfiguration"
    ) -> "Station":
        """
        Create a station from the `ProjectConfiguration`
        """
        timeseries_path = configuration.find_timeseries_for_station(name)
        if timeseries_path is None:
            raise ValueError(f"Timeseries for station {name} not found.")

        model_path = configuration.find_model_for_station(name, True)
        sitelogs_path, _ = configuration.find_sitelog_for_station(name)

        notes_path = configuration.note_path_for_station(name)
        return Station(
            name,
            timeseries_path,
            model_path,
            sitelogs_path,
            configuration.discontinuity_path,
            configuration.psd_path,
            notes_path,
        )

    def __init__(
        self,
        name: str,
        timeseries_path: str,
        model_path: str | None = None,
        sitelogs_path: str | None = None,
        discontinuity_path: str | None = None,
        psd_path: str | None = None,
        notes_path: str | None = None,
    ):
        self.name = name
        self._ts_path = timeseries_path
        self._model_path = model_path
        self._sitelogs_path = sitelogs_path
        self._discontinuity_path = discontinuity_path
        self._psd_path = psd_path
        self._notes_path = notes_path
        self._fit_config = ModelFitConfig()

    def _read_ts(self) -> ts:
        assert self._ts_path is not None

        return ts.read(
            self._ts_path,
            usecols=(2, 4, 5, 6, 7, 8, 9, 10, 11, 12),
            format=("t", "x", "y", "z", "sx", "sy", "sz", "cxy", "cxz", "cyz"),
            dtrd=1,
            rotate=True,
        )

    def _create_base_model(self) -> model:
        result = model(self._read_ts())
        result.add_polynom(deg=0)
        result.add_polynom(deg=1)
        result.add_sine(per=365.25)
        result.add_sine(per=182.625)
        result.add_vw()
        result.fit_iter()
        return result

    def _create_model_from_soln(self) -> model:
        assert self._discontinuity_path is not None

        solns = read_solns(self._discontinuity_path)

        m = model.from_solns(
            self._read_ts(),
            solns,
            self.name,
            per=[365.25, 182.625],
            noise=["vw"],
            fix_tau=True,
            fix_amp=True,
        )
        m.fit_iter()
        return m

    def add_discontinuity(self, discontinuity: Discontinuity):
        assert self._active, "To add discontinuity the station must be active"
        assert self._deterministic_model is not None

        self._deterministic_model.discontinuities.add(discontinuity)

    def remove_discontinuity(self, discontinuity: Discontinuity):
        assert self._active, "To add discontinuity the station must be active"
        assert self._deterministic_model is not None

        self._deterministic_model.discontinuities.remove(discontinuity)

    def notes(self) -> str | None:
        assert self._active
        if self._notes is None:
            return None

        return self._notes

    def set_notes(self, notes: str):
        assert self._active
        assert self._notes_path is not None

        self._notes = notes
        self._notes_saved = False

    def write_notes(self):
        if self._notes_path is None or self._notes_saved:
            return
        assert self._notes is not None
        self._notes_saved = True

        if self._notes == "" and not os.path.exists(self._notes_path):
            return

        with open(self._notes_path, "w") as f:
            f.write(self._notes)

    def _read_notes(self):
        self._notes_saved = True
        if self._notes_path is None:
            self._notes = None
            return

        if not os.path.exists(self._notes_path):
            self._notes = ""
            return

        with open(self._notes_path, "r") as f:
            self._notes = f.read()

    @call_time_prefix("Station")
    def update_model(self):
        """
        Update the `pytrf.ts.model` and run `pytrf.ts.model.fit_iter` and `pytrf.ts.model.clean`
        """
        assert self._active, "To update the model the station must be active"
        assert self._model is not None
        assert self._deterministic_model is not None
        assert self._stochastic_model is not None

        DeterministicModel.clear(self._model)
        StochasticModel.clear(self._model)
        self._deterministic_model.apply(self._model)
        self._stochastic_model.apply(self._model)

        self._model.fit_iter(**self._fit_config.args())
        self._model.clean()
        self._saved = False

    def save_psd(self):
        assert self._active
        assert self._model is not None

        if self._psd_path is None:
            return

        self._model.write_psdsnx(
            self._psd_path, self.name, " A", metasnx=self._psd_path
        )

    @call_time_prefix("Station")
    def unfix_psd(self):
        assert self._active
        assert self._model is not None

        for sub_model in self._model:
            for f in sub_model.f:
                if isinstance(f, (fexp, flog)):
                    f.par[0].fixed = False
                    f.par[1].fixed = False
        self._model.fit_iter(**self._fit_config.args())

    def _load_coseismic_data(self):
        if self._coseismic is None:
            if self._ts_path is None:
                return

            start = date.from_mjd(self._model.r.t[0])
            end = date.from_mjd(self._model.r.t[-1])
            r = self.position_geographic(False)
            if r is None:
                return
            lat, lon, _ = r
            self._coseismic = StationCoseismic(
                self.name,
                (
                    datetime.date(int(start.yyyy), int(start.mm), int(start.dd)),
                    datetime.date(int(end.yyyy), int(end.mm), int(end.dd)),
                ),
                (lat, lon),
            )
        data = self._coseismic.data()
        if data is None:
            return
        for entry in data:
            t = date.from_tiso(entry["earthquake_date"]).mjd
            d = self._deterministic_model.discontinuities.get_or_create(t, TIME_ERROR)
            name = f"M{entry['earthquake_mw']} - {entry['earthquake_location']}"
            info = (
                f"Distance: {entry['dist[km]']}km\n"
                + f"Magnitude: {entry['earthquake_mw']}\n\n"
                + "Displacement:\n"
                + f"E: {entry['displ1_dE[mm]']}mm\n"
                + f"N: {entry['displ1_dN[mm]']}mm\n"
                + f"U: {entry['displ1_dU[mm]']}mm"
            )
            d.type = DiscontinuityType.earthquake
            d.description = name
            d.tooltip = info

    @call_time_prefix("Station")
    def set_active(self):
        """
        Load the sation data in memory
        """
        if self._active:
            return

        self._active = True
        self._read_notes()

        if self._model is not None:
            self._update_state()
            return

        if self._model_path is not None and os.path.exists(self._model_path):
            self._model = model.load(self._model_path)
            self._saved = True
        elif self._discontinuity_path is not None and os.path.exists(
            self._discontinuity_path
        ):
            self._model = self._create_model_from_soln()
            self._saved = False
        else:
            self._model = self._create_base_model()
            self._saved = False

        if self._psd_path is not None:
            self._model.add_psd(
                sinex.read(self._psd_path), self.name, fix_tau=True, fix_amp=True
            )

        self._model.clean()

        self._update_state()

    def model_and_snx_sync_state(self) -> ModelAndSNXSyncStates:
        assert self._active
        assert self._model is not None

        state = ModelAndSNXSyncStates.NONE

        if self._discontinuity_path is not None:
            solns = read_solns(self._discontinuity_path)

            if not is_model_and_soln_in_sync(
                self._model, find_soln_station(solns, self.name)
            ):
                state |= ModelAndSNXSyncStates.SOLN_OUT_OF_SYNC

        if self._psd_path is not None and not is_model_and_psd_in_sync(
            self._model, sinex.read(self._psd_path), self.name
        ):
            state |= ModelAndSNXSyncStates.PSD_OUT_OF_SYNC

        return state

    def sync_model_with_snx(self):
        assert self._active
        assert self._model is not None
        state = self.model_and_snx_sync_state()

        if ModelAndSNXSyncStates.SOLN_OUT_OF_SYNC in state:
            assert self._discontinuity_path is not None
            clear_all_discontinuities(self._model)
            update_model_with_soln(
                self._model, read_solns(self._discontinuity_path), self.name
            )
            self._saved = False

        if ModelAndSNXSyncStates.PSD_OUT_OF_SYNC in state:
            assert self._psd_path is not None
            self._model.add_psd(
                sinex.read(self._psd_path), self.name, fix_amp=True, fix_tau=True
            )
            self._saved = False

        if state:
            self._update_state(force=True)

    def clean_ts(self, sigmas: float):
        assert self._active
        assert self._model is not None

        self._model.r.clean_sigmas(sigmas)
        for i in range(len(self._model)):
            sub_model = self._model[i]
            sub_model.r = self._model.r[i]
            sub_model.set_oeq()

    def active(self) -> bool:
        return self._active

    def deterministic_model(self) -> DeterministicModel:
        assert self._active
        assert self._deterministic_model is not None

        return self._deterministic_model

    def stochastic_model(self) -> StochasticModel:
        assert self._active
        assert self._stochastic_model is not None

        return self._stochastic_model

    def _update_state(self, /, force: bool = False):
        assert self._model is not None

        if self._deterministic_model is None or force:
            self._deterministic_model = DeterministicModel.from_model(self._model)
            self._deterministic_model.add_sitelog_discontinuities(
                self.sitelog_parsed_content(), self._model.r.t[0], self._model.r.t[-1]
            )
            self._load_coseismic_data()

        if self._stochastic_model is None or force:
            self._stochastic_model = StochasticModel.from_model(self._model)

    @call_time_prefix("Station")
    def set_inactive(self):
        """
        Try to unload model data if the the model has been saved.
        """
        if not self._active:
            return
        assert self._model is not None

        self._active = False

        self.write_notes()
        self._notes = None

        if not self._saved:
            return

        if self._deterministic_model.is_in_sync(self._model):
            self._deterministic_model = None  # free mem
        if self._stochastic_model.is_in_sync(self._model):
            self._stochastic_model = None  # free mem
        self._model = None  # free mem

    def has_unsaved_changes(self) -> bool:
        if self._model is None:
            return (
                self._deterministic_model is not None
                or self._stochastic_model is not None
            )

        assert self._deterministic_model is not None
        assert self._stochastic_model is not None

        return (
            not self._deterministic_model.is_in_sync(self._model)
            or not self._stochastic_model.is_in_sync(self._model)
            or not self._saved
        )

    def set_model_path(self, path: str):
        self._model_path = path

    @call_time_prefix("Station")
    def save(self) -> bool:
        if self._model_path is None or self._model is None:
            return False

        print("Saving model to: ", self._model_path)

        self._model.dump(self._model_path)
        self._update_solns()

        self._saved = True
        return True

    def position(self):
        """
        Return the coordinates of the station.
        """
        if self._active:
            assert self._model is not None
            return np.dot((self._model.r.R).T, self._model.r.ctrd[0, :])

        # If model not loaded load temporarly the timeseries
        if self._ts_path is not None:
            temp = self._read_ts()
            return np.dot((temp.R).T, temp.ctrd[0, :])
        return None

    def position_geographic(self, rad=True):
        """
        Reutrn the coordinates of the station in geographic format. (from sitelog)
        """
        cart = self.position()
        if cart is None:
            return None

        lat, lon, h = cart2geo(cart)
        if rad:
            return lat, lon, h
        return lat / math.pi * 180, lon / math.pi * 180, h / math.pi * 180

    def sitelog_raw_content(self):
        """
        Return the content if the sitelog
        """
        if self._sitelogs_path is None:
            return None

        with open(self._sitelogs_path, "r") as f:
            return f.read()

    def sitelog_parsed_content(self):
        if self._sitelogs_path is None:
            return None

        return read_sitelog(self._sitelogs_path)

    def model(self) -> model:
        if self._model is None:
            raise RuntimeError("model is none")
        return self._model

    def fit_config(self) -> ModelFitConfig:
        return self._fit_config

    def _update_solns(self):
        assert self._model_path is not None
        assert self._model is not None
        assert self._deterministic_model is not None
        assert self._active

        if self._discontinuity_path is None:
            return
        solns = []
        try:
            solns = read_solns(self._discontinuity_path)
        except Exception as e:
            print("Failed to read solns:")
            traceback.print_exception(e)

        stations = [s for s in solns if s.code == self.name]

        if len(stations) == 0:
            station = record()
            station.pt = " A"
            station.code = self.name
            station.P = []
            station.V = []
            station.X = []
            solns.append(station)
        else:
            station = stations[0]

        tmin = self._model.r.t[0]
        tmax = self._model.r.t[-1]

        # Remove discontinuities that are in the scope of the model and UI

        for key, element in [
            ("P", DiscontinuityElement.position),
            ("V", DiscontinuityElement.velocity),
        ]:
            soln_discontinuities = getattr(station, key)

            p_before = [
                p
                for p in soln_discontinuities
                if p.end != "00:000:00000" and date.from_tsnx(p.end).mjd < tmin
            ]
            start_date = "00:000:00000"
            if len(p_before) > 0:
                start_date = p_before[-1].end

            middle = []

            for discontinuity in self._deterministic_model.discontinuities.values():
                if element not in discontinuity.elements:
                    continue
                t = date.from_mjd(discontinuity.date).tsnx()

                entry = record()
                entry.soln = ""
                entry.start = start_date
                entry.end = t
                entry.cause = discontinuity.snx_cause()
                middle.append(entry)
                start_date = t

            p_after = [
                p
                for p in soln_discontinuities
                if p.end == "00:000:00000" or date.from_tsnx(p.end).mjd > tmax
            ]

            if len(p_after) > 0:
                p_after[0].start = start_date
            else:
                entry = record()
                entry.soln = ""
                entry.start = start_date
                entry.end = "00:000:00000"
                entry.cause = ""
                p_after.append(entry)

            merged = p_before + middle + p_after

            for i, entry in zip(range(len(merged)), merged):
                entry.soln = "{:>4d}".format(i + 1)

            setattr(station, key, merged)

        update_solns(self._discontinuity_path, solns)


@dataclass
class ProjectConfiguration:
    file_path: str
    """
    Path to the project configuration (.yaml)
    """
    ts_path: str
    """
    Timeseries directory
    Look for .xyz files
    - Required
    """
    model_path: str
    """
    Model directory
    Look for .pkl files
    - Required
    """
    discontinuity_path: str | None
    """
    Discontinuity file (.snx)
    """
    psd_path: str | None
    """
    Post-seismic deformation models file (.snx)
    """
    sitelogs_path: str | None
    """
    Sitelogs OPT file (.opt)
    """
    update_model_with_snx_data: bool = False
    """
    Update the model with soln and psd files if needed.
    """
    notes_path: str | None = None
    """
    Notes directory for stations
    """

    @staticmethod
    @call_time_prefix("ProjectConfiguration")
    def load(path: str) -> "ProjectConfiguration":
        with open(path, "r") as f:
            result = yaml.load(f, Loader=yaml.FullLoader)

            keys = [
                "ts_path",
                "model_path",
                "discontinuity_path",
                "psd_path",
                "sitelogs_path",
                "update_model_with_snx_data",
                "notes_path",
            ]

            for key in keys:
                if key not in result:
                    result[key] = None

            return ProjectConfiguration(
                file_path=path,
                ts_path=result["ts_path"],
                model_path=result["model_path"],
                discontinuity_path=result["discontinuity_path"],
                psd_path=result["psd_path"],
                sitelogs_path=result["sitelogs_path"],
                update_model_with_snx_data=result["update_model_with_snx_data"]
                or False,
                notes_path=result["notes_path"],
            )

    def copy(self) -> "ProjectConfiguration":
        """
        Create a copy of the configuration.
        """
        return ProjectConfiguration(**self.__dict__)

    def validate(self) -> bool:
        """
        Check if the current configuration is valid.
        """
        return (
            self.ts_path is not None
            and isinstance(self.ts_path, str)
            and self.model_path is not None
            and isinstance(self.model_path, str)
        )

    def save(self) -> None:
        """
        Save the configuration
        """
        with open(self.file_path, "w") as f:
            yaml.dump(
                {
                    "ts_path": self.ts_path,
                    "model_path": self.model_path,
                    "discontinuity_path": self.discontinuity_path,
                    "psd_path": self.psd_path,
                    "sitelogs_path": self.sitelogs_path,
                    "update_model_with_snx_data": self.update_model_with_snx_data,
                    "notes_path": self.notes_path,
                },
                f,
            )

    def list_timeseries_codes(self) -> list[str]:
        """
        Return all .xyz filename in ts_path.
        """
        result = []
        for entry in sorted(os.listdir(self.ts_path)):
            if not entry.endswith(".xyz"):
                continue
            result.append(entry[:4].upper())
        return result

    def find_timeseries_for_station(self, station: str) -> str | None:
        """
        Return the .xyz file path for the station.
        """
        p = os.path.join(self.ts_path, f"{station}_igs.xyz")

        if os.path.exists(p):
            return p
        return None

    def find_model_for_station(self, station: str, no_check=False) -> str | None:
        """
        Return the .pkl model path of the station.

        if no_check is True then return the path without checking if the file exists.
        """
        p = os.path.join(self.model_path, f"{station}.pkl")

        if no_check:
            return p

        if os.path.exists(p):
            return p
        return None

    def sitelog_configuration(self):
        if self.sitelogs_path is None:
            return

        return read_yaml(self.sitelogs_path)

    def find_sitelog_for_station(self, station: str) -> tuple[str | None, str | None]:
        config = self.sitelog_configuration()
        if config is None:
            return None, None
        return get_sitelog(station, config)

    def note_path_for_station(self, station: str) -> str | None:
        if self.notes_path is None:
            return None
        return f"{self.notes_path}/{station}.txt"


class Project:
    config: ProjectConfiguration
    _stations: list[Station] | None = None

    def __init__(self, config: ProjectConfiguration):
        assert config.validate(), "Configuration must be valid"
        self.config = config

    def stations(self) -> list[Station]:
        if self._stations is None:
            return self._fetch_stations()
        return self._stations

    def _fetch_stations(self) -> list[Station]:
        station_codes = self.config.list_timeseries_codes()
        self._stations = []
        for code in station_codes:
            self._stations.append(Station.from_configuration(code, self.config))
        return self._stations
