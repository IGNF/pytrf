"""
WebIGS API client
"""

from dataclasses import dataclass
from datetime import date
from typing import Final, TypedDict

import requests


class WebIGSException(Exception):
    """
    Base exception for errors on the `WebIGS` API.
    """


@dataclass
class WebIGSCoseismicRequestStation:
    """
    Request type for route `WebIGS.coseismic_request`
    """

    name: str
    latlng: tuple[float, float]
    start_date: date | None = None


WebIGSCoseismicResponseCoseismicEntry = TypedDict(
    "WebIGSCoseismicResponseCoseismicEntry",
    {
        "displ1_dE[mm]": float,
        "displ1_dN[mm]": float,
        "displ1_dU[mm]": float,
        "displ2_dE[mm]": float,
        "displ2_dN[mm]": float,
        "displ2_dU[mm]": float,
        "earthquake_date": str,
        "earthquake_mw": float,
        "earthquake_location": str,
        "earthquake_lat[deg]": float,
        "earthquake_lon[deg]": float,
        "dist[km]": float,
    },
)


class WebIGSCoseismicResponseStation(TypedDict):
    """
    (part of) Response type for route `WebIGS.coseismic_request`
    """

    name: str
    latlng: tuple[float, float]
    coseismic: list[WebIGSCoseismicResponseCoseismicEntry]


class WebIGSCoseismicResponseResult(TypedDict):
    """
    (part of) Response type for route `WebIGS.coseismic_request`
    """

    stations: list[WebIGSCoseismicResponseStation]


class WebIGSCoseismicResponse(TypedDict):
    """
    Response type for route `WebIGS.coseismic_request`
    """

    info: list[str]
    result: list[WebIGSCoseismicResponseResult]


class WebIGSSolutionMetadataResponse(TypedDict):
    """
    Response type for route `WebIGS.igs_solutions_metadata`
    """

    ref_epoch: dict[str, str]


class StationsCoordsResponsePosition(TypedDict):
    """
    Coords type for `StationsCoordsResponse`
    """

    lat: float
    lng: float


class StationsCoordsResponse(TypedDict):
    """
    Response type for route `WebIGS.stations_coords`
    """

    id: str
    igscumul: bool
    position: StationsCoordsResponsePosition
    sitelog: str
    solutions: list[str]


class WebIGS:
    """
    WebIGS API Client

    Example:
    ```
    import datetime
    from pytrf.webigs import WebIGS

    client = WebIGS()
    result = client.coseismic_request(
        datetime.date(1980, 1, 1),
        datetime.date(2020, 1, 1),
        [
            WebIGSCoseismicRequestStation("BRAZ", [-15.947, -47.878], datetime.date(2003, 1, 1)),
            WebIGSCoseismicRequestStation("GOLD", [35.425, -116.889]),
        ]
    )
    print(result)
    ```
    """

    COMMON_HEADERS: Final = {"Accept": "application/json"}

    def __init__(self, root: str = "https://webigs-rf.ign.fr/api"):
        self._root: Final = root

    def doc(self):
        """
        Return doc page URL
        """
        return f"{self._root}/doc/"

    def stations(self) -> dict[str, bool]:
        """
        Provides IGS station list
        """
        result = requests.get(f"{self._root}/stations", headers=self.COMMON_HEADERS)

        if not result.ok:
            raise WebIGSException(result.text)

        return result.json()

    def igs_solutions_metadata(self) -> WebIGSSolutionMetadataResponse:
        """
        Provides IGS Solutions metadata (reference epoch...)
        """
        result = requests.get(
            f"{self._root}/igs_solutions_metadata", headers=self.COMMON_HEADERS
        )

        if not result.ok:
            raise WebIGSException(result.text)

        return result.json()

    def stations_coords(self, name_station: str) -> StationsCoordsResponse:
        """
        Provides station coordinates
        """
        result = requests.get(
            f"{self._root}/stations_coords/{name_station}", headers=self.COMMON_HEADERS
        )

        if not result.ok:
            raise WebIGSException(result.text)

        return result.json()

    def img(self, name_station: str) -> bytes:
        """
        Provides PNG coordinate plots
        """
        result = requests.get(
            f"{self._root}/stations_coords/{name_station}",
            headers={"Accept": "image/png"},
        )

        if not result.ok:
            raise WebIGSException(result.text)

        return result.content

    def coseismic_request(
        self,
        start_date: date,
        end_date: date,
        stations: list[WebIGSCoseismicRequestStation],
        dmin: int = 1,
    ) -> WebIGSCoseismicResponse:
        assert len(stations) > 0, "At least one station must be provided"
        assert end_date >= start_date, "end_date cannot be before start_date"
        assert dmin >= 1

        stations_ser = []
        for station in stations:
            entry = {"name": station.name, "latlng": station.latlng}
            if station.start_date is not None:
                entry["start_date"] = station.start_date.isoformat()
            stations_ser.append(entry)

        result = requests.post(
            f"{self._root}/coseismic_request",
            headers=self.COMMON_HEADERS,
            json={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "stations": stations_ser,
                "dmin": dmin,
            },
        )

        if not result.ok:
            raise WebIGSException(result.text)

        return result.json()
