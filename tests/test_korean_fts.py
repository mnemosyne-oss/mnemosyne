# 한국어 조사·2음절 토큰 회귀를 고정하는 FTS 검증 테스트
"""Korean recall regression suite for the FTS5 query layer.

Frozen benchmark: 60 memories (50 in-domain + 10 unrelated distractors) and
49 queries tagged ``exact`` / ``josa`` / ``paraphrase``. The ``josa`` bucket
is the largest because Korean particles are what the query layer used to
break on.

Vector search is deliberately excluded -- these tests call ``_fts_search``
directly so a regression in the lexical layer cannot be masked by embeddings.

Baseline before the Korean query fix, for reference:

    variant                       R@1     R@5     MRR    josa   fallback
    unpatched                    63.3%   71.4%   0.667   12/19   23/49
    josa prefix only             63.3%   87.8%   0.740   17/19    9/49
    two-syllable relaxation only 65.3%   77.6%   0.708   14/19   12/49
    both (this patch)            85.7%   98.0%   0.912   19/19    1/49

The two fixes are not additive -- either one alone leaves the other hole
open, which is why they ship together.
"""
import pytest

from mnemosyne.core import beam

MEMORIES = [
    ('m01', 'llama-server 8080 포트는 -c 131072 -np 1 설정으로 고정해서 서빙한다'),
    ('m02', '8082 포트는 hermes-honcho 별칭으로 Q4_K_XL 양자화 모델을 올린다'),
    ('m03', '8083 포트는 Qwen3-Embedding-0.6B 임베딩 전용 서버다'),
    ('m04', 'prompt cache 크기는 --cache-ram 플래그로 조절하며 기본값은 8192 MiB다'),
    ('m05', 'cram을 24576으로 올렸더니 캐시 축출이 33건에서 2건으로 줄었다'),
    ('m06', '컨텍스트 체크포인트는 최대 32개까지 저장되고 최소 간격은 256 토큰이다'),
    ('m07', 'M5 Max의 통합 메모리는 VRAM 110100 MiB와 CPU RAM 131072 MiB를 공유한다'),
    ('m08', 'ModelForge는 Electron 앱이고 llama-server 프로세스를 감시하며 재기동시킨다'),
    ('m09', 'ModelForge 실행 프로파일은 홈 디렉터리의 .modelforge.json에 저장된다'),
    ('m10', 'ModelForge 로그는 lifecycle.log에 ADOPT와 EXIT 이벤트를 남긴다'),
    ('m11', 'honcho는 Plastic Labs가 만든 에이전트 메모리 인프라 레이어다'),
    ('m12', 'honcho의 dialectic은 툴 루프를 돌기 때문에 LLM 호출이 여러 번 발생한다'),
    ('m13', 'dialectic 질의가 22건 모두 120초 타임아웃으로 실패했다'),
    ('m14', 'honcho의 전문검색이 영어 설정으로 하드코딩되어 한국어 조사를 못 자른다'),
    ('m15', 'honcho 결론 787건을 오염 때문에 전량 폐기했다'),
    ('m16', 'honcho 제거는 config.yaml의 provider 값을 비우는 것으로 시작한다'),
    ('m17', 'honcho 컨테이너는 api, deriver, database, redis 네 개로 구성된다'),
    ('m18', 'docker compose stop을 쓰면 볼륨이 남아서 롤백이 가능하다'),
    ('m19', 'hermes 게이트웨이는 기동 시점에만 설정 파일을 읽는다'),
    ('m20', 'compression과 skills_hub를 포함한 여섯 개 역할이 8082에 물려 있다'),
    ('m21', '압축 작업이 2분에서 17분까지 걸리며 원인이 아직 규명되지 않았다'),
    ('m22', '텔레그램 폴링 재연결 실패가 이틀간 44건 발생했다'),
    ('m23', 'state.db 파일 크기가 385 메가바이트까지 커졌다'),
    ('m24', '세션 자동 정리를 켜면 2만 건 넘는 행이 삭제되므로 백업이 먼저다'),
    ('m25', '에이전트 페르소나 이름은 유나로 설정되어 있다'),
    ('m26', '옵시디언 볼트는 SecB 이름으로 관리한다'),
    ('m27', '볼트의 AI 도구 스택 폴더에 메모리 프레임워크 비교 노트가 있다'),
    ('m28', 'Mem0는 사용자 정보와 선호도를 저장하는 데 쓴다'),
    ('m29', 'Letta는 중요한 사실만 모든 프롬프트에 노출시킨다'),
    ('m30', 'Graphiti는 시간에 따라 변하는 관계를 표현한다'),
    ('m31', '그래프 메모리는 일반 프로젝트에는 과도하다는 결론이 났다'),
    ('m32', '가장 큰 문제는 잘못 저장된 사실이 고쳐지지 않는 것이다'),
    ('m33', 'Hindsight는 LongMemEval에서 91.4퍼센트 정확도를 기록했다'),
    ('m34', 'Hindsight는 20B 오픈 모델로도 83.6퍼센트를 냈다'),
    ('m35', 'Hindsight는 텍스트 검색 백엔드를 다섯 가지 중에 고를 수 있다'),
    ('m36', 'pgroonga 백엔드는 한중일 문자를 기본으로 지원한다'),
    ('m37', 'Mnemosyne는 SQLite 하나만 쓰고 외부 서비스가 필요 없다'),
    ('m38', 'Mnemosyne는 384차원 임베딩을 48바이트로 이진 압축한다'),
    ('m39', 'InPhase 프로젝트 백업은 매일 새벽 세 시에 돌아간다'),
    ('m40', '백업 파일은 구글 드라이브에 30일간 보관된다'),
    ('m41', '구글 워크스페이스 인증 토큰은 7일마다 만료된다'),
    ('m42', '슈파베이스 프로젝트 소유 계정은 별도 계정으로 분리되어 있다'),
    ('m43', '커밋은 하나의 논리적 변경이 끝났을 때 바로 남긴다'),
    ('m44', '새 소스 파일 첫 줄에는 역할을 설명하는 한국어 주석을 넣는다'),
    ('m45', '테스트를 돌리기 전에는 작업이 끝났다고 말하지 않는다'),
    ('m46', '에러는 추측하지 말고 실제 로그 줄을 읽고 판단한다'),
    ('m47', '소스를 고칠 때는 위치와 소스를 먼저 제시하고 승인을 기다린다'),
    ('m48', '삭제 작업은 항목마다 개별 승인을 받는다'),
    ('m49', '체크리스트와 컨텍스트 노트를 먼저 만들고 코딩을 시작한다'),
    ('m50', '간단한 작업에는 판단을 써서 절차를 생략해도 된다'),
    ('m51', '겨울 등산은 해가 짧아서 하산 시각을 먼저 정해두는 게 안전하다'),
    ('m52', '김치찌개에는 묵은지를 쓰면 신맛이 깊어진다'),
    ('m53', '전세 계약 갱신 청구권은 한 번만 행사할 수 있다'),
    ('m54', '러닝화는 발볼이 넓으면 한 치수 크게 신는 편이 낫다'),
    ('m55', '커피 원두는 개봉 후 2주 안에 소진하는 것이 향이 좋다'),
    ('m56', '장마철에는 제습기를 옷장 근처에 두는 게 효과가 크다'),
    ('m57', '고양이는 사료를 갑자기 바꾸면 설사를 할 수 있다'),
    ('m58', '자전거 체인은 비 맞은 뒤에 반드시 기름칠을 해야 한다'),
    ('m59', '여권 갱신은 만료 6개월 전부터 신청할 수 있다'),
    ('m60', '실내 습도는 40에서 60 퍼센트 사이가 적정하다'),
]

QUERIES = [
    ('8080 포트 서빙 설정이 뭐였지', 'm01', 'exact'),
    ('8080은 컨텍스트를 얼마로 잡았나', 'm01', 'paraphrase'),
    ('8082에 올린 모델 양자화가 뭐야', 'm02', 'exact'),
    ('임베딩 서버는 몇 번 포트야', 'm03', 'exact'),
    ('프롬프트 캐시 기본 크기', 'm04', 'exact'),
    ('cache-ram 기본값이 얼마지', 'm04', 'josa'),
    ('cram을 올린 뒤 축출이 어떻게 됐나', 'm05', 'josa'),
    ('캐시 축출 건수 변화', 'm05', 'paraphrase'),
    ('체크포인트는 몇 개까지 저장되나', 'm06', 'exact'),
    ('통합 메모리 용량이 얼마나 되지', 'm07', 'paraphrase'),
    ('ModelForge가 하는 일이 뭐야', 'm08', 'josa'),
    ('실행 프로파일은 어디에 저장돼', 'm09', 'exact'),
    ('lifecycle 로그에 뭐가 남나', 'm10', 'josa'),
    ('honcho는 누가 만들었어', 'm11', 'exact'),
    ('dialectic이 왜 느린가', 'm12', 'josa'),
    ('dialectic 타임아웃 몇 건이었지', 'm13', 'paraphrase'),
    ('honcho가 한국어 검색을 못 하는 이유', 'm14', 'josa'),
    ('결론 몇 건을 버렸더라', 'm15', 'paraphrase'),
    ('honcho를 떼려면 뭐부터 고쳐', 'm16', 'josa'),
    ('혼초 컨테이너 구성이 어떻게 되지', 'm17', 'paraphrase'),
    ('볼륨을 남기면서 컨테이너 내리는 법', 'm18', 'paraphrase'),
    ('설정 파일은 언제 읽히나', 'm19', 'josa'),
    ('8082에 물려 있는 역할이 몇 개야', 'm20', 'josa'),
    ('압축이 오래 걸리는 문제', 'm21', 'josa'),
    ('텔레그램 재연결 실패 건수', 'm22', 'exact'),
    ('state.db 용량이 얼마나 커졌지', 'm23', 'josa'),
    ('세션 정리 전에 뭘 해야 해', 'm24', 'paraphrase'),
    ('에이전트 이름이 뭐지', 'm25', 'paraphrase'),
    ('옵시디언 볼트 이름', 'm26', 'exact'),
    ('메모리 프레임워크 비교 노트 위치', 'm27', 'exact'),
    ('Letta는 어떤 방식이야', 'm29', 'josa'),
    ('시간에 따른 관계 변화를 다루는 건', 'm30', 'paraphrase'),
    ('그래프 메모리에 대한 결론', 'm31', 'josa'),
    ('메모리 시스템의 최대 문제가 뭐였지', 'm32', 'paraphrase'),
    ('Hindsight 벤치마크 점수', 'm33', 'paraphrase'),
    ('오픈 모델로 낸 정확도', 'm34', 'josa'),
    ('한중일 문자를 지원하는 백엔드', 'm36', 'exact'),
    ('Mnemosyne는 외부 의존성이 있나', 'm37', 'paraphrase'),
    ('임베딩을 몇 바이트로 줄이지', 'm38', 'josa'),
    ('백업은 몇 시에 도는가', 'm39', 'paraphrase'),
    ('백업 보관 기간이 며칠이야', 'm40', 'josa'),
    ('인증 토큰 만료 주기', 'm41', 'exact'),
    ('새 파일 첫 줄에 뭘 쓰지', 'm44', 'paraphrase'),
    ('완료라고 말하기 전에 할 일', 'm45', 'paraphrase'),
    ('에러가 났을 때 원칙', 'm46', 'josa'),
    ('소스 수정할 때 절차가 뭐야', 'm47', 'josa'),
    ('고양이 사료 바꿀 때 주의점', 'm57', 'paraphrase'),
    ('실내 적정 습도', 'm60', 'exact'),
    ('여권은 언제부터 갱신 가능해', 'm59', 'josa'),
]


# Function-scoped on purpose. conftest's autouse `_reset_thread_local_connections`
# closes every cached connection around each test, so a module-scoped connection
# would already be dead by the second test. 60 rows is cheap to rebuild.
@pytest.fixture
def corpus(tmp_path):
    db = tmp_path / "ko.db"
    beam.init_beam(db)
    conn = beam._get_connection(db)
    rowid_of = {}
    for mid, content in MEMORIES:
        rowid_of[mid] = conn.execute(
            "INSERT INTO episodic_memory"
            " (id, content, source, session_id, importance)"
            " VALUES (?, ?, 'test', 'default', 0.5)",
            (mid, content),
        ).lastrowid
    conn.commit()
    return conn, rowid_of


def _ranks(conn, rowid_of, k=5):
    out = []
    for query, gold, kind in QUERIES:
        got = [r["rowid"] for r in beam._fts_search(conn, query, k=k)]
        target = rowid_of[gold]
        out.append((kind, got.index(target) + 1 if target in got else 0))
    return out


def test_hangul_query_terms_use_stem_prefix():
    """조사 is stripped and the stem becomes a prefix term, not a phrase."""
    assert beam._fts_query_terms("여권은 언제 갱신하지") == ["여권*", "언제*", "갱신하지*"]


def test_single_syllable_particles_can_over_trim():
    """Known limitation: 나/야/여 are particles but also verb endings.

    `_strip_ko_josa` is suffix trimming, not morphology, so "갱신하나" is
    trimmed to "갱신하". Because the result is used as a prefix term the
    over-trimmed form still matches the intended documents, and the 49-query
    benchmark shows no recall loss from this. Frozen here so the behaviour is
    a documented decision rather than an accident.
    """
    assert beam._strip_ko_josa("갱신하나") == "갱신하"
    assert beam._fts_query_terms("여권은 언제 갱신하나") == ["여권*", "언제*", "갱신하*"]


def test_two_syllable_hangul_tokens_survive():
    """Two-syllable Korean words are whole words and must not be filtered."""
    assert "캐시" in beam._ko_relaxed_recall_tokens("프롬프트 캐시 기본 크기")
    assert beam._fts_query_terms("백업 주기") == ["백업*", "주기*"]


def test_non_hangul_queries_are_unchanged():
    """Every other language keeps the previous quoted-phrase behaviour."""
    assert beam._fts_query_terms("vault backup policy") == [
        '"vault"', '"backup"', '"policy"'
    ]


def test_strip_ko_josa_is_suffix_trimming_only():
    assert beam._strip_ko_josa("갱신은") == "갱신"
    assert beam._strip_ko_josa("회사에서") == "회사"
    assert beam._strip_ko_josa("백업") == "백업"      # nothing to strip
    assert beam._strip_ko_josa("나가") == "나가"      # too short to trim


def test_korean_recall_at_5(corpus):
    conn, rowid_of = corpus
    ranks = _ranks(conn, rowid_of)
    hit = sum(1 for _, r in ranks if r)
    assert hit / len(ranks) >= 0.95, f"R@5 regressed to {hit}/{len(ranks)}"


def test_korean_recall_at_1(corpus):
    conn, rowid_of = corpus
    ranks = _ranks(conn, rowid_of)
    top1 = sum(1 for _, r in ranks if r == 1)
    assert top1 / len(ranks) >= 0.80, f"R@1 regressed to {top1}/{len(ranks)}"


def test_particle_inflected_queries_all_recall(corpus):
    """The bucket the old query layer failed on must be perfect."""
    conn, rowid_of = corpus
    josa = [r for kind, r in _ranks(conn, rowid_of) if kind == "josa"]
    assert all(josa), f"josa recall regressed to {sum(1 for r in josa if r)}/{len(josa)}"


def test_like_fallback_is_rarely_needed(corpus):
    """FTS should answer nearly everything; LIKE is a safety net, not a path."""
    conn, rowid_of = corpus
    calls = {"n": 0}
    original = beam._cjk_like_search

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    beam._cjk_like_search = counting
    try:
        _ranks(conn, rowid_of)
    finally:
        beam._cjk_like_search = original
    assert calls["n"] <= 3, f"LIKE fallback used {calls['n']}/{len(QUERIES)} times"


def test_search_path_makes_no_network_calls(corpus, monkeypatch):
    """Structural proof that lexical recall never reaches an LLM."""
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("search path opened a socket")

    monkeypatch.setattr(socket, "socket", blocked)
    conn, rowid_of = corpus
    ranks = _ranks(conn, rowid_of)
    assert sum(1 for _, r in ranks if r) > 0
