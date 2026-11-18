"""Real-store/render regressions from the sanitized #1263 fixture banks.

Keep admission, ordering and rendered inclusion separate: a ranking change can
preserve the admitted set while dropping a useful fact at the three-row limit.
"""

import pytest

import mnemosyne_hermes as mh
from mnemosyne.core.canonical import CanonicalStore
from mnemosyne.core.config import MnemosyneConfig


QUERY = "Welches Passwort muss ich für neue Passwörter verwenden?"
PASSWORD = "Festlegung: Passwort und Passwörter verwaltet Vaultwarden."
BOARD = "Regel: Kanban-Karten muss das Team für neue immer verifizieren."


@pytest.fixture
def provider(tmp_path, monkeypatch):
    for key in (
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE",
        "MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    MnemosyneConfig.reset_instance()
    instance = mh.MnemosyneMemoryProvider()
    instance.initialize(
        "reported-canonical-banks", hermes_home=str(tmp_path),
        agent_identity="default", profile_isolation=False,
    )
    try:
        assert instance._beam is not None
        instance._beam.canonical = CanonicalStore(
            db_path=instance._beam.db_path, conn=instance._beam.conn,
        )
        yield instance
    finally:
        instance.shutdown()
        MnemosyneConfig.reset_instance()


def _seed(provider, rows):
    for category, name, body in rows:
        provider._beam.canonical.remember("default", category, name, body)


def _names(provider, query, *, limit=3):
    return [row["canonical_name"] for row in mh._canonical_prefetch_rows(
        provider._beam.canonical, "default", query, limit=limit,
    )]


@pytest.mark.parametrize("prefix", ["aaa", "zzz"])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_function_word_ties_cannot_hide_the_fact_at_the_candidate_limit(provider, prefix, reverse):
    # Reverting the German admission/ranking repair makes the zzz fact rank
    # fourth and disappear from both the helper's top three and the prompt.
    name = prefix + "-password"
    rows = [
        ("preference", "homepage-link-rule", "Regel: Das muss gelten und neue Links duerfen nur von der Homepage kommen."),
        ("preference", "modellwahl-kontext", "Regel: Die Modellwahl muss fuer neue Projekte Kontext bekommen."),
        ("preference", "off-peak-nur-cloud", "Regel: Off-Peak laeuft nur in der Cloud, das muss beachtet werden."),
        ("preference", "kanban_dev_board_rule", "Regel: Kanban-Karten muss das Team neue immer verifizieren."),
        ("preference", name, PASSWORD),
    ]
    _seed(provider, rows[::-1] if reverse else rows)

    assert _names(provider, QUERY, limit=999) == [name]
    assert _names(provider, QUERY) == [name]
    block = provider.prefetch(QUERY)
    assert PASSWORD in block
    assert "[CANONICAL]" in block
    assert "[owner=default home=" in block
    assert "Kanban-Karten" not in block
    assert "Homepage" not in block


def test_function_word_inversion_does_not_render_the_board_rule(provider):
    # This twelve-slot shape still ranks the Board rule first under the
    # report's ranking-only R1 variant. German evidence must be discounted too.
    _seed(provider, [
        ("ops", "backup-window-rule", "Regel: Backup-Fenster liegen immer am Wochenende."),
        ("ops", "log-retention-rule", "Regel: Logs werden nach 30 Tagen geloescht."),
        ("ops", "tls-renewal-rule", "Regel: Zertifikate erneuern sich automatisch vor dem Ablaufdatum."),
        ("ops", "release-freeze-rule", "Regel: Releases frieren waehrend des Monatsabschlusses ein."),
        ("preference", "backup-note-a", "Notiz: Das musste ich beim letzten Backup beachten."),
        ("preference", "backup-note-b", "Notiz: Neue Regeln mussten fuer das Backup angepasst werden."),
        ("preference", "release-note", "Notiz: Fuer neue Releases gilt der Freeze."),
        ("preference", "log-note", "Notiz: Neue Logs muss das System separat puffern."),
        ("preference", "homepage-link-rule", "Regel: Das muss gelten und neue Links duerfen nur von der Homepage kommen."),
        ("preference", "kanban_dev_board_rule", BOARD),
        ("preference", "modellwahl-kontext", "Regel: Die Modellwahl muss fuer neue Projekte Kontext bekommen."),
        ("preference", "password", PASSWORD),
    ])

    assert _names(provider, QUERY, limit=999) == ["password"]
    block = provider.prefetch(QUERY)
    assert PASSWORD in block
    assert BOARD not in block


@pytest.mark.parametrize(
    ("login_body", "expected"),
    [
        ("Vaultwarden Login laeuft ueber das Haupt-Passwort.", ["backup", "totp"]),
        ("Vaultwarden Login schuetzt das Passwort.", ["backup", "login", "totp"]),
    ],
    ids=["compound-two-admitted", "separate-token-three-admitted"],
)
def test_frequent_domain_terms_keep_both_reported_bank_shapes(provider, login_body, expected):
    # A frequency-derived generic-token cutoff erases these useful positives.
    rows = [
        ("preference", "login", login_body),
        ("preference", "totp", "Vaultwarden TOTP-Secrets liegen neben dem Passwort."),
        ("preference", "backup", "Vaultwarden Backup enthaelt das Passwort und die Notfallcodes."),
    ]
    _seed(provider, rows[::-1])

    assert _names(provider, "Vaultwarden Passwort", limit=999) == expected
    assert _names(provider, "Vaultwarden Passwort") == expected
    block = provider.prefetch("Vaultwarden Passwort")
    for _, name, body in rows:
        assert (body in block) == (name in expected)


@pytest.mark.parametrize(
    ("query", "rows", "expected"),
    [
        ("which password must i use for new passwords?", [
            ("preference", "homepage", "Rule: links in chat may only come from the homepage."),
            ("preference", "model", "Rule: model choice always needs project context."),
            ("preference", "board", "Rule: kanban cards must always sit on the dev board."),
            ("preference", "password", "Rule: for new passwords the Vaultwarden password always applies."),
        ], ["password"]),
        ("quel mot de passe dois-je utiliser pour les nouveaux mots de passe?", [
            ("preference", "homepage", "Regle: les liens dans le chat viennent seulement de la page d'accueil."),
            ("preference", "model", "Regle: le choix du modele exige toujours le contexte du projet."),
            ("preference", "password", "Regle: pour les nouveaux mots de passe, le mot de passe Vaultwarden s'applique."),
        ], ["password"]),
        ("liga-voreinstellung", [
            ("task", "progress", "Kanban-Karten muessen vor dem Abschluss verifiziert werden."),
            ("preference", "cache", "Offline-Cache wird vor jedem Sync geleert."),
            ("identity", "rare", "liga-voreinstellung ist die Voreinstellung fuer Profilfragen."),
        ], ["rare"]),
        ("佐々野 の プロフィール", [
            ("identity", "sibling", "佐々木 の 名前 は プロフィール 記載 なし。"),
            ("identity", "target", "佐々野 の プロフィール は システム 設定 に ある。"),
            ("preference", "other", "別の ユーザー の 好み を 保持。"),
        ], ["target"]),
        ("Passwort", [
            ("preference", "compound", "Der Zugang laeuft ueber das Haupt-Passwort."),
            ("preference", "plain", "Der Zugang speichert nur ein Passwort."),
        ], ["plain"]),
        ("Passwörter", [
            ("preference", "inflected", "Notiz zu Passwörtern wird archiviert."),
        ], []),
        ("welcher Mond hat Ringe aus Eis?", [
            ("preference", "password", PASSWORD),
            ("preference", "board", BOARD),
        ], []),
    ],
    ids=["english", "french", "rare-single-token", "cjk-iteration", "compound", "inflection", "irrelevant"],
)
def test_other_evidence_keeps_its_admission_and_rendering(provider, query, rows, expected):
    # Catch overbroad exclusions, removed rarity/iteration guards or accidental
    # token splitting/stemming while changing German function-word evidence.
    _seed(provider, rows[::-1])

    assert _names(provider, query, limit=999) == expected
    block = provider.prefetch(query)
    for _, name, body in rows:
        assert (body in block) == (name in expected)
    assert bool(block) == bool(expected)
