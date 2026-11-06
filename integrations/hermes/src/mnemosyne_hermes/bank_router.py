"""Bank-Routing fuer den Hermes-Mnemosyne-Adapter.

Eine Vela, getrennte Erinnerungsbanken. Der Adapter bindet nativ eine Bank pro
Profil; hier wird daraus eine Bindung pro Gespraechspartner, abgeleitet aus der
**Plattform-Identitaet** des Turns (Session-Key / Chat-ID) — niemals aus dem
Gespraechsinhalt.

Regeln (fail-closed):

1. **Routing ist ein Schalter, kein Default.** Ohne Eintraege in
   `banks_map.json` ist die Funktion aus: es wird nichts umgebunden, keine Bank
   angelegt. Ein nie eingeschaltetes Feature darf keine Quarantaene erzeugen
   (Review P1-2).
2. Ein Eintrag in `banks_map.json` entscheidet: der laengste Praefix des
   Session-Keys gewinnt — **auf Komponentengrenze**. `…:dm:111` bindet
   `…:dm:1112` nicht (Review P1-5).
3. Kein Key vorhanden (CLI, Cron, interner Lauf ohne Chat) -> `_default`.
4. Key vorhanden, aber unbekannt (fremder Chat, Gruppe) -> `_unknown`
   (Default `quarantine`). Eine unbekannte Zuordnung darf niemals eine private
   Bank oeffnen.
5. Ist die Bank nicht lesbar/oeffenbar, wirft der Aufrufer — er faellt **nicht**
   auf die private Bank zurueck.
6. Banknamen sind eine Konfigurationsgrenze: ein Name mit Pfadtrennern oder
   `..` verlaesst das Banken-Verzeichnis nicht (Review P2).

Die Zuordnung ist damit Daten, nicht Code: eine neue Person ist ein Eintrag,
keine Aenderung am Adapter.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

MAP_FILENAME = "banks_map.json"
_ENV_MAP = "MNEMOSYNE_BANKS_MAP"
_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}
_meta_cache: dict[str, tuple[float, dict]] = {}


def data_dir(hermes_home: Optional[str] = None) -> Path:
    """Datenverzeichnis der Mnemosyne-Banken."""
    explicit = os.environ.get("MNEMOSYNE_DATA_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if hermes_home:
        return Path(hermes_home).expanduser() / "mnemosyne" / "data"
    return Path.home() / ".hermes" / "mnemosyne" / "data"


def map_path(hermes_home: Optional[str] = None) -> Path:
    explicit = os.environ.get(_ENV_MAP, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return data_dir(hermes_home) / MAP_FILENAME


_MAP_OK = "ok"
_MAP_ABSENT = "absent"
_MAP_FAILED = "failed"


def _read_map_state(hermes_home: Optional[str] = None) -> tuple[str, dict]:
    """(status, raw map) of the given home.

    ``ok``     — the file was read and parsed
    ``absent`` — there is no map file: the feature was never switched on
    ``failed`` — a map file exists but cannot be read or parsed

    The distinction matters. A broken map must not look like an absent one:
    "the feature is off" drops the provider back into the private profile
    storage, which is exactly the disclosure the follow-up review forbids
    (review follow-up F1). Only the *state* tells the two apart.
    """
    path = map_path(hermes_home)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        # The map file is not there: the feature was never switched on.
        return _MAP_ABSENT, {}
    except OSError:
        # The file may well exist — it only cannot be *inspected* (a
        # permission error, an I/O error, a path that is not a directory).
        # "Cannot be inspected" is not "does not exist": a missing map
        # switches the routing off, and that opens the private profile
        # storage (review round 2, R6).
        return _MAP_FAILED, {}
    key = str(path)
    with _lock:
        hit = _meta_cache.get(key)
        if hit and hit[0] == mtime:
            return _MAP_OK, hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A broken file is a failed configuration, never an empty map.
        return _MAP_FAILED, {}
    if not isinstance(data, dict):
        return _MAP_FAILED, {}
    with _lock:
        _meta_cache[key] = (mtime, data)
    return _MAP_OK, data


def _read_map(hermes_home: Optional[str] = None) -> dict:
    """Raw map of the given home (cached by path+mtime), or ``{}``."""
    return _read_map_state(hermes_home)[1]


def load_state(hermes_home: Optional[str] = None) -> tuple[str, dict]:
    """(status, entries) of the map — see ``_read_map_state`` for the states."""
    status, raw = _read_map_state(hermes_home)
    entries = {str(k): str(v) for k, v in raw.items() if not str(k).startswith("_")}
    with _lock:
        _cache[str(map_path(hermes_home))] = (0.0, entries)
    return status, entries


def load_map(hermes_home: Optional[str] = None) -> dict:
    """Zuordnung laden (mit mtime-Cache; Datei darf zur Laufzeit geaendert werden)."""
    return load_state(hermes_home)[1]


def routing_enabled(hermes_home: Optional[str] = None) -> bool:
    """Routing is on only when the map actually maps somebody.

    A missing file, an empty file, or a file with nothing but ``_default``/
    ``_unknown`` leaves the provider exactly as it is. Otherwise a deployment
    that never switched the feature on would silently start writing into a
    fresh quarantine database (review P1-2).

    A map that exists but cannot be read keeps routing **on**: the broken
    configuration is routed into the quarantine, never into the private
    profile storage (review follow-up F1).
    """
    status, entries = load_state(hermes_home)
    if status == _MAP_FAILED:
        return True
    return bool(entries)


def _is_valid_bank_name(name: str) -> bool:
    """A bank name must be a plain directory name inside the banks root.

    The router and the path builder must agree on this: the router decides
    *which* bank a turn belongs to, the builder decides *where* that bank
    lives. Two definitions would let a name pass the first and be rejected
    by the second (review round 2).
    """
    name = str(name or "").strip()
    if not name or name in {".", ".."}:
        return False
    return not any(sep in name for sep in ("/", "\\", "\x00"))


def _meta(hermes_home: Optional[str] = None) -> dict:
    """Metafelder (`_default`, `_unknown`) aus **derselben** Datei wie die Map.

    Der Pfad muss aus dem aufrufenden Home kommen; sonst liest ein fremdes
    Profil die Vorgaben des Standard-Homes (review P2).

    A meta value that names no usable bank falls back to ``quarantine``:
    an unusable name must never widen the reach of a turn (review round 2,
    R1). A blank value names no bank either.
    """
    raw = _read_map(hermes_home)

    def _meta_value(key: str) -> str:
        value = str(raw.get(key) or "").strip()
        return value if _is_valid_bank_name(value) else "quarantine"

    return {
        "_default": _meta_value("_default"),
        "_unknown": _meta_value("_unknown"),
    }


def is_chat_identity(key: str) -> bool:
    """Sieht dieser Key wie eine Chat-Identitaet aus dem Gateway aus?

    Format aus `gateway/session.py:build_session_key`:
    `<ns>:<platform>:<chat_type>[...]` mit `ns` = `agent:<profil>`.
    Eine reine Zustands-/Lauf-ID (Cron, Subagent, interner Lauf) ist **keine**
    Chat-Identitaet. Sie darf nicht als "unbekannter Chat" in Quarantaene fallen,
    denn sie gehoert keiner Person und keine fremde Person liest sie.
    """
    parts = str(key or "").split(":")
    # <ns> ist selbst zweiteilig ("agent:<profil>"), danach Plattform und Chat-Typ.
    if len(parts) < 4 or parts[0] != "agent" or not parts[1]:
        return False
    return parts[3] in {"dm", "group", "channel", "thread"}


def _is_component_prefix(key: str, pattern: str) -> bool:
    """Prefix match that stops at a ``:`` component boundary.

    ``dm:111`` must not select ``dm:1112``. A pattern that ends with ``:``
    (``…:group:``) is already a component prefix and matches as-is.
    """
    if not pattern or not key.startswith(pattern):
        return False
    if len(key) == len(pattern):
        return True
    return pattern.endswith(":") or key[len(pattern)] == ":"


def resolve_bank(session_key: str, hermes_home: Optional[str] = None) -> tuple[str, str]:
    """(bank, grund) fuer diesen Turn. Wirft nie; ``""`` heisst: nicht routen."""
    key = str(session_key or "").strip()
    status, entries = load_state(hermes_home)
    if status == _MAP_FAILED:
        # Fail closed: the map exists but is broken. Routing into the
        # quarantine keeps the private profile storage out of reach — a
        # broken file must not be read as "the feature is off"
        # (review follow-up F1).
        return "quarantine", "map:unreadable:fail-closed"
    if not entries:
        return "", "disabled:no-map"

    meta = _meta(hermes_home)

    if not key:
        return meta["_default"], "default:kein-session-key"

    if not is_chat_identity(key):
        return meta["_default"], "default:kein-chat-kontext"

    best = ""
    best_bank = ""
    for pattern, bank in entries.items():
        if _is_component_prefix(key, pattern) and len(pattern) > len(best):
            best, best_bank = pattern, bank
    if best:
        if not _is_valid_bank_name(best_bank):
            # A mapping that names no bank is a broken configuration, not
            # "routing is off". Reporting an empty bank here would switch
            # the provider back to the private profile storage and hand
            # the private memories of the profile owner to whoever the
            # entry names (review round 2, R1).
            return "quarantine", f"map:{best}:invalid-bank:fail-closed"
        return best_bank, f"map:{best}"

    return meta["_unknown"], "unknown:fail-closed"


def bank_db_path(bank: str, hermes_home: Optional[str] = None) -> Path:
    """Datenbankdatei einer Bank — nur innerhalb von ``<data>/banks``."""
    name = str(bank or "").strip()
    if not _is_valid_bank_name(name):
        raise ValueError(f"invalid bank name: {bank!r}")
    root = data_dir(hermes_home) / "banks"
    path = root / name / "mnemosyne.db"
    resolved_root = root.resolve()
    if resolved_root not in path.resolve().parents:
        raise ValueError(f"bank name escapes the banks root: {bank!r}")
    return path


def cache_key(bank: str, hermes_home: Optional[str] = None) -> str:
    """Cache-Schluessel einer Bank: die **Speicher-Identitaet**, nicht der Name.

    Zwei Homes mit gleichnamiger Bank sind zwei Datenbanken; ein Cache nach
    Banknamen liefert dem zweiten Home die Instanz des ersten (review P1-4).
    """
    return str(bank_db_path(bank, hermes_home).resolve())
