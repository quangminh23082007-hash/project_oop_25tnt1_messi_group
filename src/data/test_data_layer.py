import sys, json, tempfile, warnings
from unittest.mock import MagicMock
from pathlib import Path
import numpy as np, pandas as pd
sys.modules["pyarrow"] = MagicMock(); sys.modules["pyarrow.parquet"] = MagicMock()
sys.path.insert(0, "/home/claude/proj")
warnings.simplefilter("ignore")
from src.contracts import dto, exceptions as exc
from src.contracts.interfaces import BaseRecommender
from src.data import InteractionDataset, GameItem
from src.data.parsing import to_timestamp, to_price, label_from_rating
from src.data.entities import make_game
from src.data.loader import ParquetDataLoader
from src.data.ingest import _timestamps

ok = lambda m: print("OK ", m)

# --- parsing
assert to_timestamp("-5") == 0 and to_timestamp(-3) == 0 and to_timestamp(1.7e9) == 1700000000
assert to_timestamp("2020-01-01") == 1577836800
assert to_timestamp("May 14, 2020") == 1589414400
assert to_timestamp("garbage") == 0 and to_timestamp(None) == 0
assert to_timestamp(pd.Timestamp("2020-01-01")) == 1577836800
assert to_price(None) is None and to_price(-1) is None and to_price(0) == 0.0 and to_price("19.99") == 19.99
assert label_from_rating("1") == 1.0 and label_from_rating(0) == 0.0
try: label_from_rating(3.5); raise SystemExit("should fail")
except exc.DataValidationError: pass
ok("parsing: timestamp >= 0, price None, rating nhị phân")

# --- entities / GameItem là của contract
g, st = make_game("10.0", "", None, '["A","B"]', 90, 10, '["t"]', ["Dev"])
assert type(g) is dto.GameItem and g is not None
assert g.item_id == "10" and g.title == "10" and g.price is None and g.genres == ("A", "B") and g.developer == "Dev"
assert st.wilson_rating > 4
assert GameItem is dto.GameItem
ok("entities: dùng dto.GameItem, title fallback, price None")

# --- build raw files
tmp = Path(tempfile.mkdtemp())
games_json = {str(i): {"name": f"Game {i}", "price": i * 1.5, "genres": ["Action"], "positive": 10, "negative": 2,
                       "tags": {"x": 1}, "developers": ["Dev%d" % i]} for i in range(1, 9)}
games_json["99"] = {"name": "", "price": -1}
(tmp / "games.json").write_text(json.dumps(games_json))
rng = np.random.default_rng(0)
rows = []
for u in range(40):
    n = 1 if u < 3 else (2 if u < 6 else int(rng.integers(4, 8)))
    for it in rng.choice(np.arange(1, 9), size=n, replace=False):
        d = pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 700)))
        rows.append((f"u{u}", int(it), rng.random() < 0.7, d.strftime("%Y-%m-%d"), f"review {u}-{it}" if rng.random() < .5 else ""))
rows.append(("u10", 1, False, "2021-01-01", "dup later"))  # trùng cặp, mới hơn
rows.append(("u11", 777, True, "2020-01-01", "unknown game"))  # game ngoài catalog
df = pd.DataFrame(rows, columns=["user_id", "game_id", "recommended", "post_date", "review"])
df.drop(columns="review").to_csv(tmp / "interactions.csv", index=False)
df[df.review != ""][["user_id", "game_id", "review"]].to_csv(tmp / "reviews.csv", index=False)

loader = ParquetDataLoader()
games = loader.load_games(str(tmp / "games.json"))
assert all(type(v) is dto.GameItem for v in games.values())
assert games["99"].title == "99" and games["99"].price is None
recs = loader.load_interactions(str(tmp / "interactions.csv"))
assert all(isinstance(r, dto.InteractionRecord) and r.rating in (0.0, 1.0) for r in recs)
ds = loader.load_dataset(str(tmp / "games.json"), str(tmp / "interactions.csv"), str(tmp / "reviews.csv"))
ok(f"loader: {len(recs)} records, dataset users={ds.num_users} items={ds.num_items} inter={ds.num_interactions} reviews={ds.num_reviews}")

# --- Protocol InteractionDataset
for attr in ["split_name", "interactions", "num_interactions", "num_users", "num_items", "user_mapper", "item_mapper",
             "mapping_fingerprint", "fingerprint", "review_fingerprint", "get_game_by_raw_id", "get_user_seen_items",
             "get_user_positive_items", "iter_reviews"]:
    assert hasattr(ds, attr), attr
assert ds.user_mapper.size == ds.num_users and ds.item_mapper.size == ds.num_items
assert all(isinstance(i, dto.Interaction) for i in ds.interactions) and len(ds.interactions) == ds.num_interactions
assert isinstance(ds.get_user_seen_items(0), frozenset) and isinstance(ds.get_user_positive_items(0), frozenset)
assert ds.get_user_positive_items(0) <= ds.get_user_seen_items(0)
assert ds.num_duplicates_dropped == 1 and ds.num_synthetic_items == 1
# dòng trùng: giữ bản mới nhất (Not Recommended, 2021-01-01)
u10, i1 = ds.user_mapper.to_idx("u10"), ds.item_mapper.to_idx("1")
dup = [x for x in ds.interactions if x.user_id == u10 and x.item_id == i1]
assert len(dup) == 1 and dup[0].rating == 0.0 and dup[0].timestamp == to_timestamp("2021-01-01")
revs = list(ds.iter_reviews())
assert len(revs) == ds.num_reviews and all(isinstance(r, dto.ReviewRecord) for r in revs)
assert ds.get_review("u10", "1") == "dup later"
ok("dataset: đủ thành viên Protocol, dedup giữ bản mới nhất, ReviewRecord hợp lệ")

# --- lỗi contract
for bad, e in [(lambda: ds.get_user_seen_items(10**6), exc.UnknownUserError),
               (lambda: ds.get_user_seen_items(True), exc.UnknownUserError),
               (lambda: ds.get_game_by_raw_id("nope"), exc.UnknownItemError),
               (lambda: ds.user_mapper.to_idx("zzz"), exc.UnknownUserError),
               (lambda: ds.split(test_ratio=0), exc.InvalidParameterError),
               (lambda: ds.split(val_ratio=0.6, test_ratio=0.5), exc.InvalidParameterError),
               (lambda: InteractionDataset([("a", "b")], {}), exc.DataValidationError)]:
    try: bad(); raise SystemExit("không ném lỗi")
    except e: pass
ok("exceptions đúng loại")

# --- split
for strategy in ("random", "temporal"):
    sp = ds.split(strategy=strategy)
    assert isinstance(sp, dto.DatasetSplits)
    tr, va, te = sp.train, sp.validation, sp.test
    assert (tr.split_name, va.split_name, te.split_name) == ("train", "validation", "test")
    assert len({tr.mapping_fingerprint, va.mapping_fingerprint, te.mapping_fingerprint, ds.mapping_fingerprint}) == 1
    assert tr.num_users == va.num_users == te.num_users == ds.num_users and tr.num_items == ds.num_items
    key = lambda d: {(x.user_id, x.item_id) for x in d.interactions}
    assert not (key(tr) & key(va)) and not (key(tr) & key(te)) and not (key(va) & key(te))
    assert key(tr) | key(va) | key(te) == key(ds)
    assert len({tr.fingerprint, va.fingerprint, te.fingerprint}) == 3
    for u in range(ds.num_users):  # mọi user có tương tác đều còn trong train
        if ds.get_user_seen_items(u): assert tr.get_user_seen_items(u)
    for d in (tr, va, te):
        assert d.num_reviews == sum(1 for _ in d.iter_reviews())
    assert tr.num_reviews + va.num_reviews + te.num_reviews == ds.num_reviews
    if strategy == "temporal":
        for u in range(ds.num_users):
            t_tr = [x.timestamp for x in tr.interactions if x.user_id == u]
            t_te = [x.timestamp for x in te.interactions if x.user_id == u]
            if t_tr and t_te: assert max(t_tr) <= min(t_te)
    again = ds.split(strategy=strategy)
    assert again.train.fingerprint == tr.fingerprint and again.test.fingerprint == te.fingerprint  # tái lập
    print("   ", strategy, {k: v for k, v in sp.report.items() if k.endswith("interactions") or k.endswith("only") or "cold" in k})
ok("split: chung mapper, rời nhau, hợp = full, train đủ user, tái lập theo seed, review đi theo split")

# --- BaseRecommender thật chạy trên split của mình
class Pop(BaseRecommender):
    def fit(self, train_dataset, **kw):
        self._num_users, self._num_items = train_dataset.num_users, train_dataset.num_items
        self._mapping_fingerprint, self._train_fingerprint = train_dataset.mapping_fingerprint, train_dataset.fingerprint
        c = np.zeros(self._num_items)
        for x in train_dataset.interactions:
            if x.rating == 1.0: c[x.item_id] += 1
        self.c = c; self._is_fitted = True
    def score_items(self, user_id, item_ids): return self.c[item_ids]
    def config(self): return {}
sp = ds.split()
m = Pop("pop"); m.fit(sp.train)
out = m.recommend(5, sp.train, top_k=3)
assert len(out) == 3 and all(o.item_idx not in sp.train.get_user_seen_items(5) for o in out)
try: m.recommend(5, sp.test); raise SystemExit("phải từ chối test split")
except exc.DataValidationError: pass
ok("BaseRecommender.recommend chạy được với train split (fingerprint/split_name khớp)")

# --- sparse
mat = sp.train.get_sparse_matrix()
assert mat.shape == (ds.num_users, ds.num_items) and set(mat.data) <= {1.0}
ok("get_sparse_matrix positive-only")

# --- ingest timestamps
s = pd.Series(["2020-01-01", "May 14, 2020", "1577836800", "", "bad", "2020-03-01 10:00:00", "-100"])
print("   ", _timestamps(s).tolist())
r = _timestamps(s).tolist()
assert r[0] == 1577836800 and r[1] == 1589414400 and r[2] == 1577836800 and r[3] == 0 and r[4] == 0 and r[6] == 0 and r[5] > r[0]
assert _timestamps(pd.Series(["2020-01-01"] * 3)).dtype == "int64"
ok("ingest._timestamps")
print("ALL PASSED")
