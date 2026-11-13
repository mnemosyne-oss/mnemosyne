# Native Hermes integration platform boundary (#1230)

The native Mnemosyne Hermes integration excludes `sys.platform == "android"`
and `sys.platform == "emscripten"`. Runtime guards also recognize Android's
documented pre-Python-3.13 identity: `sys.platform == "linux"` with
`sys.getandroidapilevel` present, without calling that API. Ordinary Linux
without that Android feature remains supported. This is the bounded support policy agreed in
[#1230](https://github.com/mnemosyne-oss/mnemosyne/issues/1230#issuecomment-6367263068),
not a new Hermes host policy or a change to Mnemosyne Core's public API.
Emscripten is an agreed exclusion; its independent necessity for the historical
resolver failure has **not** been established.

Both catalog dependency edges use the identical exclusion marker. The metadata
markers cannot inspect `sys.getandroidapilevel`; the older Android identity is a
runtime refusal guarantee, not an older-Android dependency resolver guarantee.
No Linux exclusion marker is added. Runtime entry points refuse excluded
platforms even if dependencies are already installed:

- catalog directory provider and early file-loaded CLI sibling;
- standalone package/plugin/provider entry points, direct CLI and installer;
- core-free `mnemosyne-hermes` console bootstrap, before cached installer reuse;
- source-checkout legacy provider and CLI;
- newly generated persistent external-wrapper bootstrap, before site activation.

Static manifest/source discovery is allowed; it does not mean the provider is
available. Runtime module loading raises a clear unsupported-platform error;
the console bootstrap prints that diagnostic and returns a nonzero status.
No excluded-path provider/tool/CLI registration, model/store activity,
configuration rewrite, installation, automatic provider switch, or fallback is
performed. Existing configuration is left alone. The repair does not change
supported-desktop entry paths or their missing-dependency diagnostics.

## Release and acceptance gates

The guards first shipped in published `mnemosyne-hermes` `0.7.7`; `0.7.6` and
earlier do not contain them. The catalog bounds are
`mnemosyne-hermes>=0.7.7,<0.8` and `mnemosyne-memory[embeddings]>=4.0.0`; the
Core ONNX Runtime ceiling is unchanged.

1. Done: `0.7.7` is published, and its PyPI wheel carries the package, CLI,
   installer and bootstrap guards. Installed from PyPI with `sys.platform` set to
   `android` or `emscripten`, importing the package raises the unsupported-platform
   error, and the `mnemosyne-hermes` console bootstrap prints it and returns 1.
2. Done: the catalog requires that published guard-bearing version.
3. Open: native acceptance on a real Android or Emscripten host. Resolver
   success and the Python-boundary checks above are not runtime acceptance.

The catalog's local guards protect its own source paths independently of the
installed dependency version. They do not retrofit old standalone distributions,
legacy symlinks or previously generated wrappers. The existing PM-adoption path
preserves the explicitly selected prior Core/provider pair; this patch does not
silently upgrade that pair, rewrite existing candidates, or change its ownership
contract. Adoption of a pre-guard version does not acquire these guards.
Existing installations need a maintainer-approved release/refresh path; this
patch does not perform it.

Regression tests run the actual entry files and installed entry-point metadata
in fresh desktop subprocesses with excluded `sys.platform` values, dependency
absence/presence controls, import/delegation/registration spies and side-effect
traps. They establish the Python boundary, **not native Android/Emscripten
execution or new desktop/ABI support**. In particular they do not claim Intel
macOS CPython 3.14 support. No host update/eviction behavior, production install,
configuration, package publication, deployment or merge is authorized by #1230's
bounded patch-preparation agreement.
