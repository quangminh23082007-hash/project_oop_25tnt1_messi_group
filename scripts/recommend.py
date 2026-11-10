"""Print recommendations from downloaded CPU weights without training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.pretrained import PretrainedRecommender


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('bpr', 'text-bpr'), default='bpr')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--user', required=True, help='Exact original source ID, not an app display name')
    parser.add_argument('--k', type=int, default=10)
    fallback = parser.add_mutually_exclusive_group()
    fallback.add_argument('--allow-fallback', dest='allow_fallback', action='store_true',
                          help='Use popularity for unknown users (default)')
    fallback.add_argument('--no-fallback', dest='allow_fallback', action='store_false',
                          help='Raise an error for unknown users instead of using popularity')
    parser.set_defaults(allow_fallback=True)
    parser.add_argument('--extra-seen', nargs='*', default=[])
    args = parser.parse_args()
    folder = args.model_dir or ROOT / 'artifacts/models' / args.model.replace('-', '_') / 'cpu'
    print('Loading weights once; this can take a few minutes...', file=sys.stderr, flush=True)
    start = time.perf_counter()
    model = PretrainedRecommender.load(folder, allow_cold_start_fallback=args.allow_fallback)
    print(f'Loaded in {time.perf_counter()-start:.1f}s', file=sys.stderr, flush=True)
    rows = model.recommend_games(args.user, top_k=args.k, extra_seen=args.extra_seen)
    print(json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
