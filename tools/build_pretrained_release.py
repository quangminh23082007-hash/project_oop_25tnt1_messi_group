"""Prepare compact CPU release assets locally; never uploads or retrains."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    output = ROOT / 'dist/pretrained-release'
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(repository='quangminh23082007-hash/project_oop_25tnt1_messi_group',
                    tag='pretrained-legacy-v1', protocol='legacy-kaggle-70-15-15', models={})
    for model in ('bpr', 'text-bpr'):
        folder = model.replace('-', '_')
        source = ROOT / 'artifacts/models' / folder / 'cpu'
        metadata = json.loads((source / 'model_info.json').read_text(encoding='utf-8'))
        if metadata.get('format_version') != 3 or metadata.get('representation') != 'compact':
            raise ValueError('Only original format-3 compact bundles can be packaged')
        if digest(source / 'weights.npz') != metadata['weights_sha256']:
            raise ValueError(f'Corrupt source weights: {model}')
        asset = output / f'{model}-cpu-legacy-v1.zip'
        if asset.exists():
            raise FileExistsError(f'Refusing to overwrite release asset: {asset}')
        with zipfile.ZipFile(asset, 'x') as archive:
            archive.write(source / 'weights.npz', f'bundle/{folder}/cpu/weights.npz', compress_type=zipfile.ZIP_STORED)
            archive.write(source / 'model_info.json', f'bundle/{folder}/cpu/model_info.json', compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
        manifest['models'][model] = dict(asset=asset.name, sha256=digest(asset), bytes=asset.stat().st_size,
                                        num_users=len(metadata['user_ids']), num_items=len(metadata['item_ids']),
                                        weights_sha256=metadata['weights_sha256'])
        print(f'Prepared {asset.name}: {asset.stat().st_size:,} bytes', flush=True)
    (ROOT / 'configs/pretrained_models.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (output / 'SHA256SUMS.txt').write_text(''.join(f"{v['sha256']}  {v['asset']}\n" for v in manifest['models'].values()), encoding='utf-8')


if __name__ == '__main__':
    main()
