"""Verify supplement integrity with Python's standard library; writes nothing."""
from pathlib import Path, PurePosixPath
import hashlib, json
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text())
expected = set()
for item in manifest['files']:
    name = item['path']; p = PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and '\\' not in name, name
    target = root / name
    assert target.is_file() and not target.is_symlink(), name
    assert target.stat().st_size == item['bytes'], name
    assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256'], name
    expected.add(name)
actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
assert actual == expected | {'MANIFEST.json', 'MANIFEST.sha256'}, sorted(actual ^ expected)
lines = (root / 'MANIFEST.sha256').read_text().splitlines()
assert lines == [item['sha256'] + '  ' + item['path'] for item in manifest['files']]
print(json.dumps({'verified_files': len(expected), 'all_sha256_match': True, 'unexpected_files': 0}))
