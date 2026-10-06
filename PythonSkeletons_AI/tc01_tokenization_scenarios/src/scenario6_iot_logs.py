"""
#  Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#  Author: Awase Khirni Syed
#  AlphaFactory Bootcamps: β ORI Inc. AI Foundations - NLP Techniques Part-I
#  Version: 1.2
#  Date: 2025-01-25
#  technical card 01: Tokenization
# scenario6_iot_logs.py
# Machine-generated text has a KNOWN grammar — so every token is emitted
# with a type, enabling raw logs -> structured records -> alerts, end to end.
# lets take sample raw IoT sensor text logs - say for example sycliq logs
"""
import re
from typing import Dict, List


class IoTLogTokenizer:
    """Rule-based tokenizer for sensor log lines; tokens carry a TYPE."""

    GRAMMAR = {
        "TIMESTAMP": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z",
        "DEVICE_ID": r"DEV-[A-Z]{2,6}-\d{3,6}",
        "GPS_PAIR":  r"pos=\(-?\d{1,3}\.\d+,-?\d{1,3}\.\d+\)",
        "METRIC":    r"[a-z_]+=-?\d+(?:\.\d+)?[A-Za-z/%]*",
        "STATUS":    r"status=[A-Z_]+",
        "EVENT":     r"\[[A-Z_]+\]",
        "WORD":      r"[A-Za-z]+",
    }

    def __init__(self):
        self._pattern = re.compile(
            "|".join(f"(?P<{n}>{p})" for n, p in self.GRAMMAR.items()))

    def tokenize_line(self, line: str) -> List[Dict]:
        return [{"token": m.group(), "type": m.lastgroup,
                 "span": (m.start(), m.end())}
                for m in self._pattern.finditer(line)]


class SensorLogParser:
    """Typed tokens -> structured records (the real payoff)."""

    METRIC_RE = re.compile(r"([a-z_]+)=(-?\d+(?:\.\d+)?)([A-Za-z/%]*)")

    def parse_line(self, line: str, tokenizer: IoTLogTokenizer) -> Dict:
        rec = {"timestamp": None, "device": None, "metrics": {},
               "status": None, "event": None, "raw": line}
        for tok in tokenizer.tokenize_line(line):
            t, typ = tok["token"], tok["type"]
            if typ == "TIMESTAMP":   rec["timestamp"] = t
            elif typ == "DEVICE_ID": rec["device"] = t
            elif typ == "EVENT":     rec["event"] = t.strip("[]")
            elif typ == "STATUS":    rec["status"] = t.split("=", 1)[1]
            elif typ == "GPS_PAIR":
                lat, lon = re.findall(r"-?\d+\.\d+", t)
                rec["metrics"].update({"lat": float(lat), "lon": float(lon)})
            elif typ == "METRIC":
                m = self.METRIC_RE.match(t)
                rec["metrics"][m.group(1)] = {"value": float(m.group(2)),
                                              "unit": m.group(3)}
        return rec

    def parse_stream(self, lines: List[str], tok: IoTLogTokenizer) -> List[Dict]:
        return [self.parse_line(ln, tok) for ln in lines]


class AnomalyDetector:
    """Business logic on top of structured records."""

    RULES = {"temp":    (">", 30.0, "HIGH_TEMPERATURE"),
             "bat":     ("<", 20.0, "LOW_BATTERY"),
             "voltage": (">", 240.0, "VOLTAGE_SPIKE")}

    def scan(self, records: List[Dict]) -> List[Dict]:
        alerts = []
        for r in records:
            for metric, (op, thr, label) in self.RULES.items():
                m = r["metrics"].get(metric)
                if isinstance(m, dict):
                    breached = m["value"] > thr if op == ">" else m["value"] < thr
                    if breached:
                        alerts.append({"alert": label, "device": r["device"],
                                       "timestamp": r["timestamp"],
                                       "value": f"{m['value']}{m['unit']}",
                                       "threshold": f"{op}{thr}"})
        return alerts


class GPSCoordinateTokenizer:
    """Tokenizes GPS strings in decimal or degrees-minutes-seconds form."""

    ATOM = re.compile(
        r"""
          (?P<DMS>  \d{1,3}°\d{1,2}'(?:\d{1,2}(?:\.\d+)?)?\"?[NSEW]? )
        | (?P<DEC>  -?\d{1,3}\.\d{1,8} )
        | (?P<HEMI> [NSEW] )
        | (?P<SEP>  [,;]|\s+ )
        """, re.VERBOSE)
    _DMS = re.compile(r"(\d{1,3})°(\d{1,2})'(?:\d{1,2}(?:\.\d+)?)?\"?([NSEW])?")

    def tokenize(self, s: str) -> List[Dict]:
        return [{"token": m.group(), "type": m.lastgroup} for m in self.ATOM.finditer(s)]

    def parse(self, s: str) -> Dict:
        values = []
        for atom in self.tokenize(s):
            tok, typ = atom["token"], atom["type"]
            if typ == "SEP":
                continue
            if typ == "DMS":
                m = self._DMS.match(tok)
                deg = float(m.group(1)) + float(m.group(2)) / 60
                if m.group(3): deg += float(m.group(3)) / 3600
                if m.group(4) in ("S", "W"): deg = -deg
                values.append(deg)
            elif typ == "DEC":
                values.append(float(tok))
            elif typ == "HEMI" and values and tok in ("S", "W"):
                values[-1] = -abs(values[-1])
        lat, lon, *_ = values + [None, None]
        return {"lat": lat, "lon": lon, "tokens": self.tokenize(s)}


# ----------------------------- SAMPLE DATA ---------------------------------
SAMPLE_STREAM = [
    "2024-06-01T12:00:03Z DEV-TH-0142 temp=23.4C hum=41% bat=87% status=OK",
    "2024-06-01T12:00:05Z DEV-GPS-0091 pos=(52.5200,13.4050) spd=4.2m/s status=MOVING",
    "2024-06-01T12:00:11Z DEV-PWR-0007 voltage=229.8V current=3.2A status=OK",
    "2024-06-01T12:00:13Z DEV-TH-0188 temp=34.7C hum=38% bat=17% [OVERHEAT]",
    "2024-06-01T12:00:20Z DEV-GPS-0091 pos=(52.5354,13.3898) spd=0.0m/s status=IDLE",
    "2024-06-01T12:00:26Z DEV-PWR-0007 voltage=243.1V current=3.5A [VOLTAGE_SPIKE]",
    "2024-06-01T12:00:29Z DEV-TH-0142 temp=22.9C hum=40% bat=86% status=OK",
]
GPS_SAMPLES = [
    "52.5200N, 013.4050E",            # Berlin — decimal + hemisphere
    "40°44'54.36\"N 73°59'8.36\"W",   # NYC — degrees/minutes/seconds
    "-33.865143, 151.209900",         # Sydney — signed decimal
]


# ----------------ASSIGNMENT Data source: embedded synthetic MQTT-style stream --------------
# -----------(for real-scale: search Kaggle for "IoT sensor logs", or Intel Berkeley lab sensor data----------



if __name__ == "__main__":
    tok = IoTLogTokenizer()

    print("--- Typed tokens for one log line ---")
    for t in tok.tokenize_line(SAMPLE_STREAM[3]):
        print(f"  {t['token']:26} {t['type']}")

    records = SensorLogParser().parse_stream(SAMPLE_STREAM, tok)
    print("\n--- Structured record (line 4) ---\n ", records[3])

    print("\n--- Alerts from anomaly rules ---")
    for a in AnomalyDetector().scan(records):
        print(" ", a)

    print("\n--- GPS coordinate tokenization ---")
    gps = GPSCoordinateTokenizer()
    for s in GPS_SAMPLES:
        p = gps.parse(s)
        print(f"  {s!r:34} -> lat={p['lat']:.6f}, lon={p['lon']:.6f}")
        print(f"      tokens: {[t['token'] for t in p['tokens']]}")
