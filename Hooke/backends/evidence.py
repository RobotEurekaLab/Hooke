"""Audit saved inputs or trajectories without importing either physics runtime."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compare_archives(left: Path, right: Path) -> dict:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        shared = sorted(set(a.files) & set(b.files))
        fields = {}
        for key in shared:
            left_array, right_array = a[key], b[key]
            fields[key] = bool(left_array.dtype == right_array.dtype and
                               np.array_equal(left_array, right_array, equal_nan=True))
        missing = {'left': sorted(set(b.files)-set(a.files)),
                   'right': sorted(set(a.files)-set(b.files))}
    return dict(equal=all(fields.values()) and not any(missing.values()),
                equal_fields=fields, missing_fields=missing,
                archives_sha256=[file_sha256(left), file_sha256(right)])


def compare_sources(left: Path, right: Path) -> dict:
    arrays = compare_archives(left/'model.npz', right/'model.npz')
    metadata = [json.loads((folder/'scene.json').read_text()) for folder in (left, right)]
    assets = [{key: value['sha256'] for key, value in item['source_assets'].items()}
              for item in metadata]
    invalid = {}
    for side, folder, hashes in zip(('left', 'right'), (left, right), assets):
        invalid[side] = [name for name, expected in sorted(hashes.items())
                         if not (folder/name).is_file() or file_sha256(folder/name) != expected]
    names_equal = metadata[0]['names'] == metadata[1]['names']
    xml_equal = (left/'scene.xml').read_bytes() == (right/'scene.xml').read_bytes()
    assets_equal = assets[0] == assets[1]
    return dict(equal=arrays['equal'] and names_equal and xml_equal and assets_equal and not any(invalid.values()),
                numeric_field_count=len(arrays['equal_fields']), arrays=arrays,
                asset_file_count=len(assets[0]), asset_manifests_equal=assets_equal,
                invalid_asset_files=invalid, names_equal=names_equal, canonical_xml_equal=xml_equal,
                parity_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('source', 'trajectory'))
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()
    compare = compare_sources if args.kind == 'source' else compare_archives
    result = compare(args.left, args.right)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['equal'] else 1)


if __name__ == '__main__':
    main()
