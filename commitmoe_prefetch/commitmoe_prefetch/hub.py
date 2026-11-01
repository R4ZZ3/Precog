"""Find a published predictor on disk or on the Hugging Face Hub and check its files.

Only data files are read (config.json, model.safetensors, manifest.json). The Python files
that the model repository also ships are for readers and other runtimes; this package
never imports or runs them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA_FILES = ('config.json', 'model.safetensors')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(source: str) -> Path:
    """A local folder as is, otherwise ``repo_id`` or ``repo_id@revision`` from the Hub."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    if path.exists():
        raise ValueError(f'--expert-predictor {source}: expected a folder or a Hugging Face repo id, got a file')
    from huggingface_hub import snapshot_download

    repo_id, _, revision = source.partition('@')
    return Path(snapshot_download(repo_id, revision=revision or None,
                                  allow_patterns=[*DATA_FILES, 'manifest.json']))


def verify(folder: Path) -> None:
    """Raise unless the release gate passed and every data file matches its manifest sha256.

    A folder without a manifest is refused: the manifest is what ties the weights to the
    release gate that checked them against the measured run.
    """
    manifest_path = folder / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'{folder}: no manifest.json; this does not look like a published predictor')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('gate', {}).get('passed') is not True:
        raise ValueError(f'{folder}: manifest.json does not record a passed release gate')
    files = manifest.get('files', {})
    for name in DATA_FILES:
        want = files.get(name)
        if not isinstance(want, str):
            raise ValueError(f'{folder}: manifest.json has no sha256 for {name}')
        got = _sha256(folder / name)
        if got != want:
            raise ValueError(f'{folder / name}: sha256 {got} differs from the manifest {want}')
