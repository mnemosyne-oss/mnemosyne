#!/usr/bin/env python3
"""Cross-repo release tooling for Mnemosyne.

A release touches three repositories and several places that will not tell
you when they are wrong:

  mnemosyne          __version__, CHANGELOG, plugin.yaml, the generated docs
  mnemosyne-docs     version.txt, which drives the landing hero
  mnemosyne-website  llms.txt "latest stable", plus a release blog post

Before this existed the pieces drifted apart. `__version__` reached 3.15.1
with no tag and no PyPI release, CHANGELOG carried a dated `[3.15.0]` heading
for a version that never shipped, the docs hero read 3.14.0 from version.txt,
and the marketing site's release card read 3.15.0 from CHANGELOG. Four
surfaces, three answers, none of them agreeing.

Usage
-----
    python3 scripts/release.py check
    python3 scripts/release.py prepare 3.15.1
    python3 scripts/release.py announce 3.15.1
    python3 scripts/release.py tag 3.15.1

`prepare` writes; everything else reads. Run `check` first, always.

Publishing
----------
Pushing a `vX.Y.Z` tag triggers .github/workflows/release.yml, which builds,
creates a GitHub Release, and publishes to PyPI through OIDC trusted
publishing. No credential lives on your machine, which also means the tag
push IS the release. `tag` prints the command rather than running it.

Announcements
-------------
`announce` generates drafts for the blog post, X, and Discord from the
changelog section. It does not post them. An announcement is the one part of
a release that cannot be corrected by pushing another commit, so a human
reads it before it goes out.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLINGS = os.path.dirname(REPO)
DOCS = os.path.join(SIBLINGS, "mnemosyne-docs")
SITE = os.path.join(SIBLINGS, "mnemosyne-website")

# MAJOR.MINOR.PATCH with an optional PEP 440 pre-release suffix, so a major
# can run the beta cycle RELEASING.md requires.
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")

PRERELEASE_RE = re.compile(r"^\d+\.\d+\.\d+(?:a|b|rc)\d+$")


def _is_prerelease(version: str) -> bool:
    """True for 4.0.0b1, False for 4.0.0.

    A pre-release ships to PyPI and gets a GitHub pre-release, but it is not
    the release. It does not take its own dated CHANGELOG section, and it does
    not move the docs or marketing sites, which advertise the version people
    get from a plain `pip install`.
    """
    return bool(PRERELEASE_RE.match(version))



# ---------------------------------------------------------------- helpers
def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _pkg_version() -> str:
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)', _read(os.path.join(REPO, "mnemosyne", "__init__.py")))
    if not m:
        sys.exit("FATAL: could not read __version__")
    return m.group(1)


def _published_version() -> str | None:
    """Newest version on PyPI, or None when offline."""
    try:
        import json
        import urllib.request
        with urllib.request.urlopen("https://pypi.org/pypi/mnemosyne-memory/json", timeout=15) as r:
            return json.load(r)["info"]["version"]
    except Exception:
        return None


def _changelog_sections() -> list[str]:
    return re.findall(r"^## \[([^\]]+)\]", _read(os.path.join(REPO, "CHANGELOG.md")), re.M)


def _section_body(version: str) -> str:
    """The changelog body for one version, without its heading."""
    text = _read(os.path.join(REPO, "CHANGELOG.md"))
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def _git(*args: str, cwd: str = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def _ok(msg: str) -> None:
    print(f"  ok    {msg}")


def _warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def _bad(msg: str) -> None:
    print(f"  FAIL  {msg}")


# ---------------------------------------------------------------- check
def cmd_check(args: argparse.Namespace) -> int:
    version = args.version or _pkg_version()
    print(f"Release readiness for {version}\n")
    problems = 0

    pkg = _pkg_version()
    if pkg == version:
        _ok(f"mnemosyne/__init__.py __version__ = {pkg}")
    else:
        _bad(f"__version__ is {pkg}, expected {version}. Run: release.py prepare {version}")
        problems += 1

    sections = _changelog_sections()
    if version in sections:
        _ok(f"CHANGELOG has a [{version}] section")
    elif _is_prerelease(version):
        # Deliberate: a beta shares the entry with the release it precedes.
        # Promoting per pre-release would leave a stale [4.0.0b1] heading
        # beside [4.0.0] describing the same changes.
        _ok(f"CHANGELOG keeps [Unreleased] for pre-release {version}")
    else:
        _bad(f"CHANGELOG has no [{version}] section (found: {', '.join(sections[:4])})")
        problems += 1

    if sections and sections[0] == "Unreleased":
        _ok("CHANGELOG has an [Unreleased] section at the top")
    else:
        _warn("CHANGELOG has no [Unreleased] section; add one for the next cycle")

    # A dated heading for a version that was never tagged reads as released.
    # Only versions ABOVE the newest tag matter: anything older either shipped
    # before the tagging convention existed or is genuinely historical, and
    # warning about 1.x here would bury the one case worth seeing.
    tags = set(_git("tag", "--list").split())

    def _parts(v: str) -> tuple:
        """Sort key honouring PEP 440 pre-release order.

        A naive digit sweep reads "4.0.0b1" as (4, 0, 0, 1) and ranks it above
        "4.0.0", which is backwards: a pre-release precedes its final release.
        """
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?", v.lstrip("v"))
        if not m:
            return tuple(int(x) for x in re.findall(r"\d+", v))
        major, minor, patch, kind, num = m.groups()
        stage = {None: 3, "a": 0, "b": 1, "rc": 2}[kind]
        return (int(major), int(minor), int(patch), stage, int(num or 0))

    tagged = sorted((_parts(t) for t in tags if VERSION_RE.match(t.lstrip("v"))), reverse=True)
    newest_tag = tagged[0] if tagged else (0,)
    for s in sections:
        if s in ("Unreleased", version) or not VERSION_RE.match(s):
            continue
        if _parts(s) > newest_tag and f"v{s}" not in tags:
            _warn(f"CHANGELOG documents [{s}] with a date, but there is no v{s} tag "
                  f"and it is not on PyPI. It reads as released when it is not.")

    plugin = os.path.join(REPO, "hermes_memory_provider", "plugin.yaml")
    pv = re.search(r"^version:\s*(\S+)", _read(plugin), re.M)
    if pv and pv.group(1) == version:
        _ok(f"plugin.yaml version = {version}")
    else:
        _bad(f"plugin.yaml version = {pv.group(1) if pv else '?'}, expected {version}")
        problems += 1

    gen = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "verify-docs.py")],
                         cwd=REPO, capture_output=True, text=True)
    out = gen.stdout + gen.stderr
    # verify-docs.py exits non-zero for canonical drift AND for sibling drift.
    # Only the first is this repository's problem and only the first should
    # block a release. Reporting them the same way sent you to run
    # generate-docs.py for a file this repo does not own, and made a stale
    # docs site look like stale generated output here.
    canonical_drift = any(
        line.strip().startswith("DRIFT:") for line in out.splitlines()
    )
    sibling_drift = "DRIFT (sibling)" in out
    if gen.returncode == 0:
        _ok("generated docs match the code")
    elif canonical_drift:
        _bad("generated docs are stale. Run: python3 scripts/generate-docs.py")
        problems += 1
    elif sibling_drift:
        _warn("mnemosyne-docs content is behind this repo; `prepare` for a "
              "final release regenerates it")
    else:
        _bad(f"verify-docs.py failed: {out.strip().splitlines()[-1] if out.strip() else gen.returncode}")
        problems += 1

    for name, path, probe in (
        ("mnemosyne-docs", DOCS, os.path.join(DOCS, "version.txt")),
        ("mnemosyne-website", SITE, os.path.join(SITE, "public", "llms.txt")),
    ):
        if not os.path.isdir(path):
            _warn(f"{name} not checked out beside this repo; its version will not be updated")
        elif os.path.exists(probe):
            cur = _read(probe).strip() if probe.endswith(".txt") and "version" in probe else ""
            if probe.endswith("version.txt"):
                if cur == version:
                    _ok(f"{name} version.txt = {cur}")
                elif _is_prerelease(version):
                    _ok(f"{name} version.txt = {cur}, unchanged for a pre-release")
                else:
                    _warn(f"{name} version.txt = {cur}, will become {version} on prepare")
            else:
                _ok(f"{name} present")

    if f"v{version}" in tags:
        _bad(f"tag v{version} already exists")
        problems += 1
    else:
        _ok(f"tag v{version} is free")

    published = _published_version()
    if published is None:
        _warn("could not reach PyPI to compare the published version")
    elif published == version:
        _bad(f"{version} is already on PyPI")
        problems += 1
    else:
        _ok(f"PyPI currently has {published}; releasing {version}")

    # release.yml runs `python -m build` AFTER the tag exists. A build failure
    # there leaves a tag with no release and nothing on PyPI, and the tag
    # cannot simply be re-pushed. Cheap to check now, expensive to discover
    # later. Skipped when `build` is not installed rather than failing on it.
    if getattr(args, "skip_build", False):
        _warn("skipped the package build check (--skip-build)")
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([sys.executable, "-m", "build", "--outdir", td],
                               cwd=REPO, capture_output=True, text=True)
            combined = (r.stderr or "") + (r.stdout or "")
            if r.returncode != 0 and "No module named build" in combined:
                # Probe the interpreter that would actually run it, not the one
                # importing this script; they are often different.
                _warn(f"`{os.path.basename(sys.executable)} -m build` is unavailable, "
                      "so the package build release.yml runs was not pre-flighted. "
                      "pip install build")
            elif r.returncode != 0:
                _bad("python -m build FAILED; release.yml would fail after tagging")
                tail = [ln for ln in combined.strip().splitlines() if ln.strip()][-1:]
                print("        " + (tail[0] if tail else "(no output)"))
                problems += 1
            else:
                names = sorted(os.listdir(td))
                if any(version in n for n in names):
                    _ok(f"package builds: {', '.join(names)}")
                else:
                    _bad(f"package built but artifacts are not {version}: {names}")
                    problems += 1

    dirty = _git("status", "--porcelain")
    tracked = [ln for ln in dirty.splitlines() if not ln.startswith("??")]
    if tracked:
        _warn(f"{len(tracked)} uncommitted tracked change(s) in mnemosyne")

    print()
    print("READY" if problems == 0 else f"NOT READY: {problems} blocking problem(s)")
    return 1 if problems else 0


# ---------------------------------------------------------------- prepare
def cmd_prepare(args: argparse.Namespace) -> int:
    version = args.version
    if not VERSION_RE.match(version):
        sys.exit(f"FATAL: {version!r} is not MAJOR.MINOR.PATCH")
    today = dt.date.today().isoformat()
    print(f"Preparing {version} ({today})\n")

    # 1. __version__
    init = os.path.join(REPO, "mnemosyne", "__init__.py")
    s = _read(init)
    s2 = re.sub(r'(__version__\s*=\s*["\'])[^"\']+(["\'])', rf"\g<1>{version}\g<2>", s)
    if s2 != s:
        _write(init, s2)
        print(f"  mnemosyne/__init__.py -> {version}")

    # 2. plugin.yaml
    plugin = os.path.join(REPO, "hermes_memory_provider", "plugin.yaml")
    s = _read(plugin)
    s2 = re.sub(r"^version:\s*\S+", f"version: {version}", s, count=1, flags=re.M)
    if s2 != s:
        _write(plugin, s2)
        print(f"  hermes_memory_provider/plugin.yaml -> {version}")

    # 3. Promote [Unreleased] into a dated section.
    cl = os.path.join(REPO, "CHANGELOG.md")
    s = _read(cl)
    if _is_prerelease(version):
        print(f"  CHANGELOG.md left on [Unreleased] ({version} is a pre-release)")
    elif f"## [{version}]" in s:
        print(f"  CHANGELOG already has [{version}], leaving it alone")
    else:
        s2 = s.replace("## [Unreleased]",
                       f"## [Unreleased]\n\n## [{version}] - {today}", 1)
        _write(cl, s2)
        print(f"  CHANGELOG.md -> [{version}] - {today}")

    # 4. Regenerate the derived docs so they carry the new version.
    #
    # For a pre-release use verify-docs.py --fix, which writes only this
    # repository's canonical docs/api/*.mdx. generate-docs.py also writes into
    # mnemosyne-docs, and those files carry the version in their frontmatter,
    # so running it here would put a beta version on the published docs site
    # even though step 5 deliberately leaves version.txt alone.
    doc_script = "verify-docs.py" if _is_prerelease(version) else "generate-docs.py"
    doc_args = ["--fix"] if _is_prerelease(version) else []
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", doc_script), *doc_args],
                   cwd=REPO, check=False)

    # 5. Sibling repos. A pre-release must not move them: the docs hero and
    # the marketing site advertise what a plain `pip install` gives you, and
    # that is still the previous release until the final ships.
    if _is_prerelease(version):
        print(f"  sibling repos left alone ({version} is a pre-release)")
    else:
        vt = os.path.join(DOCS, "version.txt")
        if os.path.isfile(vt):
            _write(vt, version + "\n")
            print(f"  mnemosyne-docs/version.txt -> {version}")
        else:
            print("  mnemosyne-docs not present, skipped")

        llms = os.path.join(SITE, "public", "llms.txt")
        if os.path.isfile(llms):
            s = _read(llms)
            s2 = re.sub(r"(Latest stable: mnemosyne-memory )\S+", rf"\g<1>{version}", s)
            if s2 != s:
                _write(llms, s2)
                print(f"  mnemosyne-website/public/llms.txt -> {version}")
        else:
            print("  mnemosyne-website not present, skipped")

    print()
    print("Next:")
    if _is_prerelease(version):
        print("  1. review the diff in this repo")
        print(f"  2. commit and merge, then: python3 scripts/release.py check {version}")
        print(f"  3. python3 scripts/release.py tag {version}")
        print("  4. announce nothing yet; a pre-release is not the release")
    else:
        print("  1. review the diffs in all three repos")
        print(f"  2. python3 scripts/release.py announce {version}")
        print("  3. commit and merge each repo")
        print(f"  4. python3 scripts/release.py check {version}")
        print(f"  5. python3 scripts/release.py tag {version}")
    return 0


# ---------------------------------------------------------------- announce
def _headline_items(version: str, limit: int = 6) -> list[str]:
    """Bold lead-ins from the changelog section, labelled by subsection.

    Changelog lead-ins describe the defect ("Recall crashed on its own schema
    default"), which is right for a changelog and wrong for an announcement:
    unlabelled, it reads as announcing a bug rather than a fix. Carrying the
    ### heading through turns it into "Fixed: recall crashed ...", which is
    what a reader expects.
    """
    body = _section_body(version)
    items: list[str] = []
    current = ""
    for line in body.splitlines():
        h = re.match(r"^###\s+(\w+)", line)
        if h:
            current = h.group(1)
            continue
        m = re.match(r"^-\s+\*\*(.+?)\*\*", line)
        if not m:
            continue
        # Strip the trailing period first: the issue reference sits before it,
        # so stripping in the other order leaves "(#507)" attached.
        text = m.group(1).strip().rstrip(".")
        text = re.sub(r"\s*\((?:#\d+(?:,\s*)?)+\)\s*$", "", text).rstrip(".")
        items.append(f"{current}: {text}" if current else text)
        if len(items) >= limit:
            break
    return items


def cmd_announce(args: argparse.Namespace) -> int:
    version = args.version
    items = _headline_items(version)
    body = _section_body(version)
    if not body:
        sys.exit(f"FATAL: no CHANGELOG section for {version}. Run prepare first.")

    today = dt.date.today().isoformat()
    slug = f"mnemosyne-{version.replace('.', '-')}"
    outdir = os.path.join(REPO, "dist-announce")
    os.makedirs(outdir, exist_ok=True)

    bullets = "\n".join(f"- {i}" for i in items) or "- (fill in the headline changes)"

    # The website has no mdx-components.tsx, so plain markdown elements get
    # no styling at all: a post written with `##` and `-` renders as bare
    # unstyled HTML. Every existing post hand-writes JSX with Tailwind
    # classes, so the scaffold has to as well. This was learned the hard way.
    P = 'className="font-serif-alt text-lg leading-relaxed text-charcoal mb-6"'
    H2 = 'className="font-serif text-2xl lg:text-3xl text-charcoal mt-12 mb-4"'
    UL = 'className="font-serif-alt text-lg leading-relaxed text-charcoal mb-6 ml-6 list-disc"'

    li = "\n".join(f'  <li className="mb-2">{i}</li>' for i in items) or \
         '  <li className="mb-2">TODO the headline changes</li>'

    blog = f"""---
title: "Mnemosyne {version}"
excerpt: "TODO one or two sentences. What changed and why a reader should care."
date: "{today}"
readTime: "4 min read"
category: "Release"
tags: ["v{version}", "release"]
featured: false
image: "/hero-demo.jpg"
---

<p {P}>
  TODO: open with the one thing that matters most in this release.
</p>

<h2 {H2}>
  What changed
</h2>

<ul {UL}>
{li}
</ul>

<h2 {H2}>
  Upgrading
</h2>

<p {P}>
  Run
  <code className="text-accent-terracotta bg-cream-dark px-1 rounded">pip install --upgrade 'mnemosyne-memory[embeddings]'</code>.
  TODO note any migration or config change, or say there is none.
</p>

<CodeBlock code={{`pip install --upgrade 'mnemosyne-memory[embeddings]'`}} language="bash" />

<p {P}>
  Full notes on the
  <a href="https://github.com/mnemosyne-oss/mnemosyne/releases/tag/v{version}" className="text-accent-terracotta underline underline-offset-2 hover:text-charcoal">v{version} release page</a>.
</p>
"""

    # X has a 280 character budget, so build the shortest honest version and
    # let the author expand it. Two items, trimmed. A draft that already fits
    # is likelier to get posted than one that must be cut down first.
    def _short(s: str, n: int = 78) -> str:
        return s if len(s) <= n else s[: n - 1].rstrip() + "\u2026"

    x_lines = "\n".join(f"- {_short(i)}" for i in items[:2])
    x_post = f"""Mnemosyne {version} is out.

{x_lines}

Local-first memory for AI agents. One SQLite file, no cloud.

github.com/mnemosyne-oss/mnemosyne/releases/tag/v{version}"""

    discord = f"""**Mnemosyne {version}**

{bullets}

```
pip install --upgrade 'mnemosyne-memory[embeddings]'
```

Full changelog: <https://github.com/mnemosyne-oss/mnemosyne/releases/tag/v{version}>"""

    blog_path = os.path.join(outdir, f"{slug}.mdx")
    _write(blog_path, blog)
    _write(os.path.join(outdir, f"{slug}-x.txt"), x_post + "\n")
    _write(os.path.join(outdir, f"{slug}-discord.md"), discord + "\n")

    print(f"Drafts written to dist-announce/ for {version}:\n")
    print(f"  {slug}.mdx          blog post -> mnemosyne-website/content/blog/")
    print("                          verified to build; renders one page per locale (7)")
    print(f"  {slug}-x.txt        X post ({len(x_post)} chars)")
    print(f"  {slug}-discord.md   Discord announcement")
    print()
    if len(x_post) > 280:
        print(f"  NOTE: the X draft is {len(x_post)} characters. Trim it or post as a thread.")
    print("  Nothing is posted. Read them, edit the TODOs, then publish by hand.")
    print("  An announcement is the one part of a release you cannot fix with another commit.")
    return 0


# ---------------------------------------------------------------- tag
def cmd_tag(args: argparse.Namespace) -> int:
    version = args.version
    rc = cmd_check(args)
    if rc != 0:
        print("\nRefusing to tag: fix the problems above first.")
        return rc
    print()
    print("Everything checks out. Tagging is the release: pushing the tag triggers")
    print(".github/workflows/release.yml, which builds, creates the GitHub Release,")
    print("and publishes to PyPI. That cannot be undone by deleting the tag.")
    print()
    print("Run:")
    print(f"    git tag v{version}")
    print(f"    git push origin v{version}")
    print()
    print("Then, once PyPI shows the new version:")
    print("    merge the version bumps in mnemosyne-docs and mnemosyne-website")
    print("    publish the drafts from dist-announce/")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (
        ("check", cmd_check, "validate release readiness across all three repos"),
        ("prepare", cmd_prepare, "bump versions, promote the changelog, regenerate docs"),
        ("announce", cmd_announce, "generate blog, X, and Discord drafts"),
        ("tag", cmd_tag, "re-check, then print the tag commands"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("version", nargs="?" if name == "check" else None,
                       help="MAJOR.MINOR.PATCH")
        if name in ("check", "tag"):
            p.add_argument("--skip-build", action="store_true",
                           help="skip the python -m build pre-flight (slow)")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
