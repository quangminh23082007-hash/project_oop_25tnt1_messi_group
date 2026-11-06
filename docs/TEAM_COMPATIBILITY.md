# Cross-branch integration findings — 2026-11-06

**Scope update:** [Contracts v1.1](Contracts_v1_1_Pretrained_2026-11-06.html) now
accepts the existing trained models for the current application delivery. The
findings below remain valid comparisons against the older v1 training design,
but not every mismatch is a blocker for the revised pretrained path.

Required now: publish/install the bundles; connect the inference loader to the
backend/UI; preserve original model mappings; adapt catalog/result fields; handle
history, fallback, missing metadata and errors correctly; and verify a clean-clone
CPU startup. Data may keep GitHub branch `c70e4f4f`'s Dataset/Game classes and files.

Deferred unless a new training/benchmark workflow is requested: a full
InteractionDataset/DatasetSplits rewrite, v1 split/review artifacts, new mapping
fingerprints, and replacing the evaluator's entire public API. Use original
artifacts/reports for the accepted experiment. Do not pass incompatible record
types across interfaces or call the existing Service.train path to load weights.

The original finalization checklist below is historical v1 work, not a new
requirement to retrain or rewrite those layers for the current delivery.

Checked live GitHub branch heads, then inspected their source trees and interfaces.
This is a source/interface review plus targeted smoke checks, not a full test of
the future application. No teammate branch or main was changed.

## Versions inspected

- models: `8acf26f986f71b20cb79374b1cb1c788fc7ebf11`
- data: `c70e4f4ff5fed932c14adfeea1ad58ca7d29907c`
- feat/service: `1a7ecf9cc5e59a10323c90fb33cba0cd680e6893`
- feature/evaluation: `613b715c26075ac20bfa9ea11be78d7160c26411`
- main: `c4bf891088545430a80be3d1d00dc6972af9b9c7`

Also inspected the trees of master, feature/test-pr and feature/add-contracts-and-services.
No committed web UI, desktop UI or HTTP endpoint was found in those branch trees.
If the UI exists elsewhere or only on someone's computer, it still needs to be
provided/integrated before its actual client API can be verified.

## 1. Shared definitions still differ — integration blocker

[Service contracts](https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/blob/1a7ecf9cc5e59a10323c90fb33cba0cd680e6893/src/contracts/dto.py)
still lack shared Interaction, ReviewRecord, ItemFeatures, GameItem and
EvaluationResult. InteractionRecord has no text and accepts ratings up to five;
ModelConfig lacks patience/text_reg. Service's BaseRecommender still uses the old
fit/_fit_internal template. The models branch imports the new definitions.

Use one reviewed v1 src/contracts across all branches. Do not resolve merges by
keeping several independent copies or blindly accepting either entire branch.
Review the shared contract change together before integrating member modules.

## 2. Data is not yet a v1 Dataset provider — integration blocker

[Data implementation](https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/blob/c70e4f4ff5fed932c14adfeea1ad58ca7d29907c/src/data_layer.py#L195)
exposes Dataset.num_games, user2id/game2id and a private interaction store. Models
and current Service require num_items, shared mapped Interaction records, mappers,
split_name and mapping/train/review fingerprints, plus the history/review methods.
Passing this Dataset directly fails before fitting; changing just a class name
or num_games to num_items will not supply the missing semantics.

Its train_test_split returns two lists; get_interaction_matrix_data defaults to
rating_weighted (1–5), while v1 training requires explicit binary 0/1 labels and
three split datasets with mappings frozen from all train interactions. Review CSV
output is user_id,game_id,review, rather than the agreed labelled/timestamped/split
handoff plus per-split mapped JSONL files. Data should implement the v1 public
adapter/pipeline and artifact report before official retraining.

## 3. Evaluation's call signature and types differ — integration blocker

[Evaluator.evaluate](https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/blob/613b715c26075ac20bfa9ea11be78d7160c26411/src/evaluation/evaluator.py#L39)
expects train_interactions, test_interactions, num_users, num_items and
additional_seen_interactions. Service calls evaluate(model, train_dataset,
evaluation_dataset, additional_seen_dataset=...). These calls do not match.

[Evaluation protocol](https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/blob/613b715c26075ac20bfa9ea11be78d7160c26411/src/evaluation/protocol.py)
defines its own Interaction and rejects instances of the shared v1 Interaction
with TypeError. This failure was reproduced using the fetched source. Its result
type also uses skipped_reasons counts instead of the shared skipped_users records.
Default K is (3,5), not the contract's (5,10), so NDCG@10 is not available by default.

Evaluation should reuse/re-export shared DTOs and wrap its existing metric core
with BaseEvaluator's dataset API, fingerprints, split checks and agreed K values.
The formulas do not need to be rewritten solely to change the interface.

## 4. Service improved, but two fallback issues remain

The latest Service code, when imported with MODELS' shared contracts and a small
v1-shaped dataset, successfully trained BPR and returned recommendations in a
targeted smoke check. This does not mean its own branch is independently runnable.

In [recommend_games](https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/blob/1a7ecf9cc5e59a10323c90fb33cba0cd680e6893/src/service/recommendation_service.py#L312),
a known user with no positive profile becomes user_idx=None, then the code calls
_fallback_scores(None). That drops their negative review history, allowing those
games to be recommended again. Preserve the known user index for history filtering
even when using global popularity scores; use ColdStartError when fallback is off.

In _to_result (line 219), model_name always names the primary model, even for
popularity fallback. Use the actual fallback model name together with is_fallback.
Neither bug was changed in the teammate's branch during this task.

## 5. No committed trained-model startup path — addressed locally

The current remote models branch has no release assets and no committed model
installer/startup loader. The latest Service has a train method, but no factory
for attaching the existing format-3 bundles without retraining.

Local additions now provide PretrainedRecommender.load(), recommendation DTOs,
explicit global fallback, live-history exclusions, optional catalog lookup,
checksum-verified installation, CLI smoke checks, release packaging and a manifest.
See PRETRAINED_MODELS.md. These changes are not published merely by writing this
report: commit/push the selected source files and publish the prepared assets.

The legacy inference loader is deliberately separate from v1 training. Existing
weights preserve their original 70/15/15 mapping and provenance. No fabricated v1
fingerprints, fake training Dataset or automatic retraining is involved.

## 6. Application packaging and catalog handoff remain

The models pyproject currently packages only src/contracts and src/models. When
the team integrates Data/Evaluation/Service, expand package discovery and declare
their dependencies. Source-folder imports alone are not proof an installed wheel
contains the whole application.

Old model catalogs contain title/image_url/categories, but no reliable price or
genre fields. The UI/backend should join the team's current catalog by raw game
ID; unknown prices must not be treated as free. Application account IDs cannot
be associated with unverified source usernames by display-name matching.

## Recommended integration sequence

1. Adopt the same shared v1 contracts across member branches.
2. Data supplies the agreed datasets, mappings, reviews and fingerprints.
3. Evaluation adapts its public interface and shared record/result types.
4. Service fixes fallback history/name handling and connects the real UI endpoint.
5. For an immediate demonstration, install the legacy CPU bundle and delegate
   backend inference to PretrainedRecommender; no retraining is required.
6. Run a separate official v1 training/evaluation experiment when the pipeline is
   integrated; keep its results distinct from the existing Kaggle experiment.

## Checks performed for the local pretrained changes

- Actual BPR and TextBPR ZIPs were created from checksum-verified source bundles.
- Both archives were installed into a fresh temporary directory, then loaded on
  CPU to produce real recommendations. No fit/train method or GPU was used.
- Tiny-bundle tests cover archive/weight corruption, overwrite refusal, history
  filtering, raw ID order, explicit fallback and missing-price behavior.
- Existing 28 numerical/model tests remain part of the proposed source checks.

Do not merge every local untracked file: the earlier src/data, src/evaluation,
src/service and demo prototypes in the local folder still target older contracts.

## Owner checklist for finalization (branch heads rechecked 2026-11-06)

This checklist records confirmed interface/behavior mismatches against the supplied
contract, plus required delivery work. It is not a claim that every possible bug
has been found. The existing 35 local tests cover Models and pretrained delivery,
not the complete team application.

### Team lead / shared-contract owner

- Agree on one reviewed src/contracts version and adopt it in every branch.
  Model's v1 definitions are the current implementation candidate, not a reason
  to blindly overwrite the other members' work. Use shared DTOs and exceptions.
- Freeze the official split and selection protocol. The supplied contract remains
  80/10/10 per-user; 70/15/15 was discussed, not adopted as a revised contract.
  If changed, update the document, config and tests together. A matching ratio
  alone does not make the old weights a new-contract experiment.
- Integrate in a dedicated branch: contracts, Data, Models, Evaluation, Service/UI.
  Resolve source conflicts explicitly; preserve original member branches.
- Expand package discovery beyond Models/Contracts, declare optional dependencies,
  provide one documented startup command and add a clean-clone integration test.

### Data owner

- Implement the complete v1 Dataset interface: mapped shared interactions,
  num_items, split_name, frozen user/item mappers, all three fingerprints,
  history/positive-history methods and iter_reviews. Preserve working parsing code
  internally instead of merely renaming num_games.
- Produce binary labels directly from recommended. Keep Wilson/playtime ranking
  separate; current rating_weighted default is not a valid v1 training label.
- Clean/deduplicate raw records first; split into three datasets; construct mappings
  from all train interactions (including negatives), then freeze them. The current
  build_mapping uses all loaded records before the two-way global shuffle.
- Apply the agreed per-user split, small-user handling and unknown-positive policy.
  Record removed rows and excluded users separately and never shrink a user's
  held-out positive truth silently to inflate Recall.
- Fix duplicate semantics: get_interaction_matrix_data overwrites duplicate pair
  scores (last wins), and add_reviews_batch overwrites review text independently.
  Keep label, timestamp and text from the same first valid source record. Raw CSV
  preparation computes users.duplicated before invalid-label filtering, so a later
  valid row can be lost when an earlier invalid row has the same user in a chunk.
  Filter invalid rows first, then deduplicate in deterministic file/row order.
- Require each review's exact user-game pair to exist in its dataset, not just
  that the user and game separately exist (current add_reviews_batch checks only
  membership). Preserve source Unicode/text and timestamp provenance.
- Output the agreed reviews.csv and train/validation/test_reviews.jsonl schemas,
  mapped IDs, split labels, checksums and missing/rejected-text counts. When the
  optional second user-frequency filtering pass removes users, filter reviews too;
  current reviews.csv is written in pass 1 and not filtered again.
- Supply catalog titles, real genres, prices/currency and images keyed by raw game
  ID. Unknown price stays None. Verify source account IDs rather than assuming
  display names identify one account across games.

Data is ready when a tiny fixture proves: disjoint splits; train-only frozen
mapping; binary labels; negative history retained; duplicate record/text alignment;
no held-out reviews in train.iter_reviews; and changed text changes its fingerprint.

### Model owner

- Keep the tested v1 adapters and shared configuration API. Coordinate the official
  X features with Data; use only train reviews and a pinned frozen encoder revision.
- Commit the locally prepared pretrained loader, installer, release manifest,
  regression tests and docs, then publish the two verified ZIPs as private Release
  assets. These additions are still local as of this review.
- Keep the original format-3 weights with their metadata and ID order. Use the
  inference-only loader for the current demonstration, without pretending it has
  the labelled train Dataset or new-contract fingerprints.
- For an official new-contract experiment, adapt the Kaggle GPU preparation,
  validation and export flow to the final Data handoff. Current preserved GPU
  tools are legacy; the new public adapters are CPU implementations. Do not claim
  the GPU notebook is v1-compatible just because the adapter tests pass.
- When that integration is ready, run validation-based selection and final test;
  retrain only if the required protocol/data actually differs. Export artifact
  provenance and a bundle the serving loader can validate.

Models are ready for the demonstration when a teammate can download, verify,
install and load weights once, then receive CPU recommendations without any train
call. Both prepared ZIPs already passed that workflow locally.

### Evaluation owner

- Replace the public raw-list/dimensions API with BaseEvaluator's dataset API,
  while retaining the existing metric calculations internally.
- Import/re-export shared Interaction, EvaluationResult, UserEvaluation and
  SkippedUser. Return skipped-user records, not only aggregate reason counts.
- Validate roles, dimensions, mapping fingerprint, model train fingerprint and
  disjoint user-item pairs. Allow additional_seen_dataset only as validation when
  evaluating test; never let it alter training or positive-profile eligibility.
- Require at least one positive train interaction for evaluated users. Current
  evaluator.py adds every train user to train_users, including negative-only users.
- Default to Recall/NDCG at K=5,10 and select checkpoints using validation NDCG@10;
  retain K=3,5 only when explicitly labelled as a demo configuration.
- Compare BPR, TextBPR, Popularity and optional Cornac with identical users,
  candidates, split and history rules; report Data exclusions separately from
  evaluator skips. Never tune on test or silently refit train+validation in v1.
- Export summary.csv, per_user.csv and run_config.json with model parameters,
  seed, split/mapping/review/feature provenance, timings and package versions.

Evaluation is ready when the shared fixture can be passed directly from Data
through Models to Evaluator, both Service and a validation callback can call it,
and held-out test data cannot influence epoch selection.

### Service / backend owner

- Adopt the same contracts. The latest Service implementation already passes a
  basic train/recommend smoke check when combined with Model's shared contracts;
  its branch's old shared definitions must still be replaced coherently.
- Add a supported load_pretrained startup path that delegates to the prepared
  inference loader. Load once per backend process and do not call train on startup.
- Fix known-user/no-positive fallback: retain the known user's ID to exclude both
  negative and positive history; use ColdStartError when fallback is disabled.
- In fallback DTOs set model_name to the actual global model, alongside
  is_fallback=True. _to_result currently always labels the primary model.
- Protect model/dataset state: reject a pre-fitted replacement model with another
  mapping/train fingerprint. Current swap_model(..., retrain_immediately=False)
  retains the old dataset and is_ready tests only is_fitted. For TextBPR retraining,
  forward valid ItemFeatures and the intended validation callback.
- Align rejected training with the failure lifecycle: train() currently validates
  its dataset and callback before clearing readiness. After a successful train,
  passing a test dataset to train() raises but leaves is_ready=True. Invalidate at
  the beginning of the attempted training operation if following v1's stated
  failed-training rule, and cover both preflight and backend failures in tests.
- In evaluate_model enforce validation-history-only-for-test before forwarding;
  current code accepts additional validation history while evaluating validation.
- Join games through raw IDs, include the app's current reviewed/owned history,
  and obtain correct price/genre metadata. New app accounts must use a deliberate
  cold-start policy, not a source-user display-name match.
- Make the API return serializable shared result fields and distinguish missing
  weights/unknown users/invalid filters with useful client errors.

Service is ready when known users exclude all reviewed games; new and negative-only
users follow the configured fallback policy; filters fill K when enough games
exist; mismatched bundles/datasets are rejected; and failed operations do not
leave readiness pointing at an incompatible model/dataset pair.

### UI owner / integrator

- Supply/commit the actual UI and HTTP endpoint code: none was present in the
  inspected branch trees. Its behavior cannot yet be tested from this repository.
- Agree on endpoint inputs (authenticated app account, count, filters) and result
  fields (game_id, title, predicted_score, price, genres, model_name, is_fallback;
  images joined by game_id). Scores are ranking values, not stars/probabilities.
- Implement loading, empty results, cold-start fallback and backend error states.
  Avoid trusting client-supplied dataset usernames to select another person's profile.
- Prove the complete fresh-clone flow on a CPU-only machine: install source,
  download/verify model, start backend, start/open UI, request recommendations and
  confirm displayed game IDs/titles match the backend response. No training should
  happen in this serving flow.

### Final acceptance gate

Do not call the whole project complete based only on one member's unit tests.
Use a clean integrated checkout and one shared tiny dataset to check Data → Model
→ Evaluation → Service, then a real release bundle to check installer → startup
loader → API → UI. Check malformed inputs, changed fingerprints, corrupt downloads,
negative history, unknown users, filters, failed callbacks and no GPU dependency.
Keep the working legacy demonstration and the official new-contract experiment
clearly labelled until both have passed their respective checks.
