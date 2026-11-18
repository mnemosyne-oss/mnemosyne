"""Validate an immutable core release tag without importing release code."""
import argparse
import ast
import json
from pathlib import Path
import re


def validate_tag(tag: str, version_file: Path) -> dict:
    if not re.fullmatch(r'v[1-9][0-9]*\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?', tag):
        raise ValueError('Expected a core release tag vMAJOR.MINOR.PATCH, optionally a/b/rcN')
    tree=ast.parse(version_file.read_text())
    values=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='__version__' for t in node.targets)]
    if len(values)!=1 or values[0]!=tag[1:]:
        raise ValueError('Release tag does not match the literal source version')
    return {'version':values[0],'stable':bool(re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+',tag))}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    parser.add_argument('--version-file',type=Path,default=Path('mnemosyne/__init__.py'))
    args=parser.parse_args()
    try:
        result=validate_tag(args.tag,args.version_file)
    except (ValueError,SyntaxError,OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
