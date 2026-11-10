"""Import compact CPU bundles from the completed Kaggle ZIP without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def install(archive, destination, model="all", expected_sha256=None):
    if model not in ('all', 'bpr', 'text-bpr', 'text_bpr'):
        raise ValueError('Unknown model selection')
    if expected_sha256 is not None and sha256_file(archive) != expected_sha256:
        raise ValueError('Archive checksum mismatch: do not install this download')
    destination = Path(destination).resolve()
    requested = ("bpr", "text_bpr") if model == "all" else (model.replace("-", "_"),)
    with zipfile.ZipFile(archive) as bundle:
        chosen = {}
        # Only explicit members are read; archive paths are never used as local paths.
        for name in requested:
            matches = [p for p in bundle.namelist() if p.endswith(f"/{name}/cpu/model_info.json")]
            if len(matches) != 1:
                raise ValueError(f"Expected one CPU bundle for {name}; found {len(matches)}")
            info_path = matches[0]
            weights_path = info_path.removesuffix("model_info.json") + "weights.npz"
            info = json.loads(bundle.read(info_path))
            expected_type = "BPRMatrixFactorization" if name == "bpr" else "TextBPRMatrixFactorization"
            if (info.get("format_version") != 3 or info.get("representation") != "compact"
                    or info.get("model_type") != expected_type):
                raise ValueError(f"Unsupported CPU bundle: {name}")
            bundle.getinfo(weights_path)
            if (destination / name).exists():
                raise FileExistsError(f"Refusing to overwrite {destination / name}")
            chosen[name] = (info_path, weights_path, info)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".model_import_", dir=destination) as temporary:
            staging = Path(temporary)
            assert staging.resolve().parent == destination
            for name, (info_path, weights_path, info) in chosen.items():
                cpu = staging / name / "cpu"
                cpu.mkdir(parents=True)
                (cpu / "model_info.json").write_bytes(bundle.read(info_path))
                checksum = hashlib.sha256()
                with bundle.open(weights_path) as source, (cpu / "weights.npz").open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        checksum.update(chunk)
                        target.write(chunk)
                if checksum.hexdigest() != info.get("weights_sha256"):
                    raise ValueError(f"Incomplete or corrupt weights for {name}")
                print(f"Verified {name}", flush=True)
            # All requested checksums pass before any bundle becomes visible.
            for name in chosen:
                shutil.move(str(staging / name), str(destination / name))
                print(f"Installed {destination / name / 'cpu'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--download", action='store_true', help='Download a private release using authenticated GitHub CLI')
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "artifacts/models")
    parser.add_argument("--model", choices=("all", "bpr", "text-bpr"), default="all")
    parser.add_argument('--manifest', type=Path, default=PROJECT_ROOT / 'configs/pretrained_models.json')
    parser.add_argument('--sha256', help='Expected archive SHA256 for a separately supplied legacy archive')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8')) if args.manifest.exists() else None
    if args.download:
        if args.model == 'all' or manifest is None:
            parser.error('Choose --model bpr or text-bpr and provide a release manifest')
        if shutil.which('gh') is None:
            parser.error('Install/sign in to GitHub CLI, or download the ZIP in your signed-in browser and use --archive')
        record = manifest['models'][args.model]
        with tempfile.TemporaryDirectory(prefix='model_download_') as temporary:
            download_dir = Path(temporary).resolve()
            assert download_dir.parent == Path(tempfile.gettempdir()).resolve()
            subprocess.run(['gh', 'release', 'download', manifest['tag'], '--repo', manifest['repository'],
                            '--pattern', record['asset'], '--dir', str(download_dir)], check=True)
            install(download_dir / record['asset'], args.destination, args.model, record['sha256'])
    else:
        checksum = args.sha256
        if checksum is None and manifest is not None:
            checksum = next((r['sha256'] for r in manifest['models'].values()
                             if r['asset'] == args.archive.name), None)
        if checksum is None:
            parser.error('Archive is not in the release manifest; supply its trusted --sha256')
        install(args.archive, args.destination, args.model, checksum)


if __name__ == "__main__":
    main()
