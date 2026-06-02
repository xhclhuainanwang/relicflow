import math

from ..constants import MIN_NUMBER
from .base import ThermalChannel


class A4Model:
    name = "A4"

    def prepare_params(self, cards: dict) -> dict:
        p = dict(cards)
        p.setdefault("qphi", 2.0)
        p.setdefault("v", 246.0)
        p.setdefault("mh1", 125.1)
        p.setdefault("mW", 80.379)
        p.setdefault("mZ", 91.1876)
        p.setdefault("Gammah1", 4.07e-3)
        p.setdefault("chanNN", 1)
        p.setdefault("chanZpZp", 0)
        p.setdefault("chanWW", 1)
        p.setdefault("chanZZ", 1)

        m_dm = float(p["mDM"])
        p.setdefault("mS", (1.0 + float(p["delta"])) * m_dm)
        p.setdefault("mZp", float(p["rZ"]) * m_dm)
        p.setdefault("Lpar", float(p["mh2"]) ** 2 / (6.0 * float(p["vphi"]) ** 2))
        p.setdefault("kappa4", (m_dm**2 / float(p["vphi"]) ** 2 + 3.0 * float(p["Lpar"])) / 2.0)
        p.setdefault("lambda3", float(p["delta"]) * (2.0 + float(p["delta"])) * m_dm**2 / (2.0 * float(p["vphi"]) ** 2))
        p.setdefault("gX", float(p["mZp"]) / (abs(float(p["qphi"])) * float(p["vphi"])))
        p.setdefault("alphaMix", float(p["lamHphi"]) * float(p["v"]) * float(p["vphi"]) / (float(p["mh2"]) ** 2 - float(p["mh1"]) ** 2))
        p.setdefault("Gammah2", 1.0e-2 * float(p["mh2"]))

        mnlist = p.get("MNlist")
        if mnlist is None:
            mn = float(p["MN"])
            mnlist = [mn, mn, mn]
        p["MNlist"] = [float(x) for x in mnlist[:3]]
        return p

    def _sv_nn_i(self, s: float, idx: int, p: dict) -> float:
        if int(p["chanNN"]) == 0:
            return MIN_NUMBER
        m_dm = float(p["mDM"])
        m_n = p["MNlist"][idx]
        if m_dm <= m_n or s <= 4.0 * m_n**2:
            return MIN_NUMBER
        kin = 1.0 - 4.0 * m_n**2 / s
        if kin <= 0.0:
            return MIN_NUMBER
        val = (m_n**2 / (16.0 * math.pi * float(p["vphi"]) ** 4)) * (s / (s - 2.0 * m_dm**2)) * kin ** 1.5
        return max(float(val), MIN_NUMBER)

    def _ba4(self, s: float, p: dict) -> complex:
        alpha = float(p["alphaMix"])
        c1 = math.cos(alpha)
        c2 = math.sin(alpha)
        g1 = (float(p["lamHphi"]) * float(p["v"]) / 2.0) * c1 - 2.0 * float(p["kappa4"]) * float(p["vphi"]) * c2
        g2 = (float(p["lamHphi"]) * float(p["v"]) / 2.0) * c2 + 2.0 * float(p["kappa4"]) * float(p["vphi"]) * c1
        d1 = (s - float(p["mh1"]) ** 2) + 1j * float(p["mh1"]) * float(p["Gammah1"])
        d2 = (s - float(p["mh2"]) ** 2) + 1j * float(p["mh2"]) * float(p["Gammah2"])
        return (g1 * c1) / d1 + (g2 * c2) / d2

    def _sv_ww(self, s: float, p: dict) -> float:
        if int(p["chanWW"]) == 0:
            return 0.0
        m_dm = float(p["mDM"])
        m_w = float(p["mW"])
        if m_dm <= m_w or s <= 4.0 * m_w**2:
            return 0.0
        kin = 1.0 - 4.0 * m_w**2 / s
        if kin <= 0.0:
            return 0.0
        val = (
            float(p["chanWW"])
            * (s**2 / (16.0 * math.pi * float(p["v"]) ** 2))
            * (math.sqrt(kin) / (s - 2.0 * m_dm**2))
            * (1.0 - 4.0 * m_w**2 / s + 12.0 * m_w**4 / s**2)
            * abs(self._ba4(s, p)) ** 2
        )
        return max(float(val), 0.0)

    def _sv_zz(self, s: float, p: dict) -> float:
        if int(p["chanZZ"]) == 0:
            return 0.0
        m_dm = float(p["mDM"])
        m_z = float(p["mZ"])
        if m_dm <= m_z or s <= 4.0 * m_z**2:
            return 0.0
        kin = 1.0 - 4.0 * m_z**2 / s
        if kin <= 0.0:
            return 0.0
        val = (
            float(p["chanZZ"])
            * (s**2 / (32.0 * math.pi * float(p["v"]) ** 2))
            * (math.sqrt(kin) / (s - 2.0 * m_dm**2))
            * (1.0 - 4.0 * m_z**2 / s + 12.0 * m_z**4 / s**2)
            * abs(self._ba4(s, p)) ** 2
        )
        return max(float(val), 0.0)

    def build_channels(self, params: dict) -> list[ThermalChannel]:
        p = params
        if int(p.get("chanZpZp", 0)) != 0:
            raise NotImplementedError("A4 chanZpZp=1 is intentionally excluded in this nBE minimal framework.")

        m_dm = float(p["mDM"])
        out: list[ThermalChannel] = []
        if int(p.get("chanNN", 1)) != 0:
            out.append(ThermalChannel(
                name="NN",
                s_min=4.0 * m_dm**2,
                sigma_s=lambda s: self._sv_nn_i(s, 0, p) + self._sv_nn_i(s, 1, p) + self._sv_nn_i(s, 2, p),
            ))
        if int(p.get("chanWW", 1)) != 0:
            out.append(ThermalChannel(
                name="WW",
                s_min=max(4.0 * m_dm**2, 4.0 * float(p["mW"]) ** 2),
                sigma_s=lambda s: self._sv_ww(s, p),
            ))
        if int(p.get("chanZZ", 1)) != 0:
            out.append(ThermalChannel(
                name="ZZ",
                s_min=max(4.0 * m_dm**2, 4.0 * float(p["mZ"]) ** 2),
                sigma_s=lambda s: self._sv_zz(s, p),
            ))
        return out
