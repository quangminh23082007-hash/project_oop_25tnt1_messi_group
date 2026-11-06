# Distributing and loading the trained CPU models

The current [Contracts v1.1](Contracts_v1_1_Pretrained_2026-11-06.html) accepts these
existing bundles as the application model. Keep their original provenance;
the Data interface reference is GitHub's `data` branch at `c70e4f4f`.

## Published deliverables

The [private release `pretrained-legacy-v1`](https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/releases/tag/pretrained-legacy-v1)
was published on 2026-11-06 from models commit `b1ed736c`. It contains:

- `bpr-cpu-legacy-v1.zip` — original BPR compact CPU weights and model_info.json.
- `text-bpr-cpu-legacy-v1.zip` — original TextBPR compact CPU weights and model_info.json.
- `SHA256SUMS.txt` — archive digests, also committed in `configs/pretrained_models.json`.

Each bundle has 1,394,194 mapped users and 23,089 games, with 64 factors. Each ZIP
is about 376 MB. Keep both files inside each bundle together; metadata includes
the original user/item order, catalog and training provenance. The archives
contain no raw review text, prepared train data, large encoder or GPU checkpoint.
They do retain source user identifiers and observed training pairs, so distribute
them within the existing private repository and do not change repository privacy.

Both uploaded archive sizes and SHA256 hashes match the committed manifest;
authenticated downloads were checked. Both original ZIPs were installed and
used for CPU recommendations without retraining before publication.

The owner's local ZIPs remain in `dist/pretrained-release/` (ignored by Git).
`python tools/build_pretrained_release.py` can
reproduce packaging from installed compact bundles in a fresh output directory;
it refuses to overwrite existing release files and never trains a model.

## Release maintenance

Initial publication is complete. Source changes were pushed only to `models`;
`main` and other members' branches were not modified. The release is marked as a
prerelease because it delivers trained models, not the integrated application.

Keep the published tag, ZIPs and checksums together. If models change later,
publish a new tag and update the manifest rather than replacing these files
under the same version. Keep repository visibility private.

A clone does not fetch release assets automatically. Teammates with repository
read access can download them while signed in; an anonymous link grants no access.
Choose the named model ZIPs, not GitHub's automatic Source code archives.

## Teammate installation

Clone/check out `models` (or a future integrated branch containing these files),
create a Python 3.10+ environment and run `python -m pip install -e .`.
Download a ZIP while signed in, then:

```powershell
python scripts/install_model.py --archive "C:\Users\HP\Downloads\bpr-cpu-legacy-v1.zip" --model bpr
python scripts/recommend.py --model bpr --user "!" --k 3
```

Or, with GitHub CLI installed and authenticated (`gh auth login`):

```powershell
python scripts/install_model.py --download --model bpr
python scripts/install_model.py --download --model text-bpr
```

The local manifest supplies the exact expected archive SHA256. Downloads go to a
temporary directory, checks run before installation, and existing models are not
overwritten. For the older large both_models_and_reports.zip, provide its trusted
`--sha256` explicitly; its checksum is different from these smaller release ZIPs.
Do not download new weights at every application start: install once, load once
per backend process, then reuse. Start with a single backend worker because every
worker loads its own large arrays and ID mappings. Startup can take tens of
seconds or longer; RAM use exceeds the compressed ZIP size.

## Backend integration

Import `PretrainedRecommender` from `src.models.pretrained` and call `.load(path)`
at startup. Use `.recommend_games(raw_user_id, ...)` for requests and convert the
returned shared RecommendationResult objects with dataclasses.asdict. No training,
PyTorch, CUDA, transformer model or raw dataset is needed for inference.

The inference API deliberately does not pretend to be a newly fitted v1
BaseRecommender. Compact legacy bundles lack v1 split/review fingerprints and the
full labelled train Dataset. Do not force them through Service.train(), fabricate
fingerprints or evaluate them as v1 experiments. The Service owner can add a
separate load_pretrained factory that delegates inference to this loader.

`raw_user_id` refers to the original source mapping. New application accounts must
not be linked by display-name similarity. Default unknown-user behavior raises
UnknownUserError; opt into `allow_cold_start_fallback=True` for global popularity.
Fallback results set is_fallback=True and model_name=Popularity. Include the app's
current reviewed/owned game IDs in `extra_seen`; training history is already
excluded. `exclude_played=False` explicitly disables both history filters.

The bundle catalog supplies titles, image_url and categories. Categories are not
silently relabelled as genres, and unknown prices remain None. For reliable
filters, pass a callable `catalog_lookup(game_id)` returning the matching shared
GameItem from the team's current catalog. Join by original game IDs, never new
row positions. `get_catalog_metadata(game_id)` exposes the original image metadata.

## What this does and does not complete

This gives the team a working CPU inference path and distributable trained
artifacts without retraining. It does not implement a missing UI, HTTP endpoint,
new-user personalization, or a new training/evaluation experiment.
The older v1 training design is optional future work; the v1.1 delivery gate is
verified bundle installation, backend loading and actual UI recommendations.

Unit checks:

```powershell
python -m unittest discover -s tests -p test_pretrained.py -v
python -m unittest discover -s tests -p test_team_models.py -v
python -m unittest discover -s tests -p test_numerical_core.py -v
```
