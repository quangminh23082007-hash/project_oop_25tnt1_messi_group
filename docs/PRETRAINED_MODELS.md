# Distributing and loading the trained CPU models

The current [Contracts v1.1](Contracts_v1_1_Pretrained_2026-11-06.html) accepts these
existing bundles as the application model. Keep their original provenance;
the Data interface reference is GitHub's `data` branch at `c70e4f4f`.

## Prepared deliverables

Two locally generated ZIPs live in `dist/pretrained-release/`:

- `bpr-cpu-legacy-v1.zip` — original BPR compact CPU weights and model_info.json.
- `text-bpr-cpu-legacy-v1.zip` — original TextBPR compact CPU weights and model_info.json.
- `SHA256SUMS.txt` — archive digests, also committed in `configs/pretrained_models.json`.

Each bundle has 1,394,194 mapped users and 23,089 games, with 64 factors. Each ZIP
is about 376 MB. Keep both files inside each bundle together; metadata includes
the original user/item order, catalog and training provenance. The archives
contain no raw review text, prepared train data, large encoder or GPU checkpoint.
They do retain source user identifiers and observed training pairs, so distribute
them within the existing private repository and do not change repository privacy.

The ZIPs are already prepared. `python tools/build_pretrained_release.py` can
reproduce packaging from installed compact bundles in a fresh output directory;
it refuses to overwrite existing release files and never trains a model.

## Publish once (repository owner)

No release has been created by this change. The connected GitHub tools used for
this task can inspect releases but do not upload binary release assets; GitHub CLI
is not installed on the inspected laptop. The files and instructions below are
ready for an upload through GitHub's signed-in website.

Stage only these source changes from the models checkout (leave other local
prototypes untracked):

```powershell
Set-Location 'E:\project_OOP\game-recsys-oop'
git add -- README.md src/models/pretrained.py configs/pretrained_models.json
git add -- scripts/install_model.py scripts/recommend.py tools/build_pretrained_release.py tests/test_pretrained.py
git add -- docs/PRETRAINED_MODELS.md docs/PRETRAINED_RELEASE_NOTES.md docs/TEAM_COMPATIBILITY.md
git add -- docs/Contracts_v1_1_Pretrained_2026-11-06.html docs/TRAINED_RUN_REFERENCE.json docs/MODEL_HANDOFF.md docs/Contracts_v1_Game_Recommender_2026-11-05.html
git diff --cached --stat
git diff --cached --check
git commit -m "Add verified pretrained CPU bundles workflow and serving loader"
git push --no-follow-tags origin refs/heads/models:refs/heads/models
```

1. Commit/push the loader, installer, tests, manifest and docs to `models` first.
   Keep other members' branches and `main` unchanged. The release must target that
   new commit, not the previous commit that lacked the installer/loader.
2. Open https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/releases/new
3. Create tag `pretrained-legacy-v1`, target `models`, title `Pretrained CPU models (legacy Kaggle experiment)`.
   Mark it a prerelease so it is not confused with a completed v1 application release.
4. Attach the two ZIPs and SHA256SUMS.txt from `dist/pretrained-release/`. Wait for
   both uploads to finish. Copy the description in `docs/PRETRAINED_RELEASE_NOTES.md`.
5. Publish the release. Keep repository visibility private. Open the asset links
   while signed in and verify their sizes; download/install one as a teammate.
6. Update README's release-status sentence only after verifying publication.

GitHub Releases support binary assets below 2 GiB each, so both assets fit:
https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
This avoids storing hundreds of MB in normal Git history. A clone does not fetch
release assets automatically. Teammates with read access can obtain them from
that private release; an anonymous link alone will not grant access.

With GitHub CLI installed and authenticated, the equivalent publishing command is:

```powershell
gh release create pretrained-legacy-v1 dist/pretrained-release/bpr-cpu-legacy-v1.zip dist/pretrained-release/text-bpr-cpu-legacy-v1.zip dist/pretrained-release/SHA256SUMS.txt --repo quangminh23082007-hash/project_oop_25tnt1_messi_group --target models --title "Pretrained CPU models (legacy Kaggle experiment)" --notes-file docs/PRETRAINED_RELEASE_NOTES.md --prerelease
```

This command creates a release/tag and uploads assets, not a merge into main.

## Teammate installation

Clone/check out `models` (or a future integrated branch containing these files),
create a Python 3.10+ environment and run `python -m pip install -e .`.
Download a ZIP while signed in, then:

```powershell
python scripts/install_model.py --archive "C:\Users\HP\Downloads\bpr-cpu-legacy-v1.zip" --model bpr
python scripts/recommend.py --model bpr --user "!" --k 3
```

Or, after publication and `gh auth login`:

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
