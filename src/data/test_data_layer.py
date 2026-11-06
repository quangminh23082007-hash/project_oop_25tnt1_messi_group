"""
Kiểm tra Data layer theo Contracts v1 (HTML 2026-11-05) và các quyết định của nhóm.
Chạy: python -m pytest test_data_layer.py   hoặc   python test_data_layer.py
(Cần đặt dưới thư mục chứa `src/`; không cần pyarrow nếu chỉ chạy qua CSV.)
"""
import csv, hashlib, json, random, sys, tempfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:  # sandbox không có pyarrow: stub để import được loader/ingest, chỉ đường CSV được kiểm tra
    import pyarrow  # noqa: F401
except Exception:
    from unittest.mock import MagicMock
    sys.modules["pyarrow"] = MagicMock(); sys.modules["pyarrow.parquet"] = MagicMock()
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/claude/proj")
warnings.simplefilter("ignore")

from src.contracts import dto, exceptions as exc
from src.contracts.interfaces import BaseRecommender
from src.data import DataPipeline, GameItem, SplitConfig
from src.data.dataset import InteractionDataset
from src.data.ingest import _timestamps
from src.data.loader import ParquetDataLoader
from src.data.parsing import clean_id, label_from_rating, text_or_none, to_price, to_timestamp
from src.data.pipeline import _split_sizes

MS_2020 = 1577836800000


def ok(message):
    print("OK ", message)


def write_case(tmp, rows, n_games=12, with_text=True):
    """rows: (user, game, recommended(bool), post_date, review)."""
    games = {str(i): {"name": f"Game {i}", "price": i * 1.5, "genres": ["Action"], "positive": 10, "negative": 2,
                      "tags": {"x": 1}, "developers": [f"Dev{i}"]} for i in range(1, n_games + 1)}
    (tmp / "games.json").write_text(json.dumps(games))
    with open(tmp / "interactions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["user_id", "game_id", "recommended", "post_date"] + (["review"] if with_text else [])
        writer.writerow(header)
        for user, game, rec, date, review in rows:
            writer.writerow([user, game, "Recommended" if rec else "Not Recommended", date] + ([review] if with_text else []))
    return str(tmp / "games.json"), str(tmp / "interactions.csv")


def random_rows(seed=0, users=60, games=12):
    rng = random.Random(seed)
    rows = []
    for u in range(users):
        n = 1 if u < 4 else (2 if u < 8 else rng.randint(4, games))
        for g in rng.sample(range(1, games + 1), n):
            day = rng.randint(0, 700)
            date = (pd.Timestamp("2019-01-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            text = f"review {u}-{g}, \"quoted\"\nline2 Chơi vui" if rng.random() < 0.5 else ""
            rows.append((f"user {u}", str(g), rng.random() < 0.7, date, text))
    return rows


# ---------------------------------------------------------------- parsing
assert to_timestamp("2020-01-01") == MS_2020                       # mili giây
assert to_timestamp("20-Jul-15") == to_timestamp("7/20/2015") == to_timestamp("2015-07-20") == 1437350400000
assert to_timestamp(1577836800) == MS_2020 and to_timestamp(MS_2020) == MS_2020   # giây hoặc mili giây
assert to_timestamp("-5") == 0 and to_timestamp(-3) == 0 and to_timestamp("garbage") == 0 and to_timestamp(None) == 0
assert to_price(None) is None and to_price(-1) is None and to_price(0) == 0.0
assert clean_id("  Alice Smith ") == "Alice Smith" and clean_id("Bob") != clean_id("bob")   # không đổi hoa/thường
assert clean_id("570.0") == "570.0" and clean_id(570.0) == "570" and clean_id(570) == "570"   # không sửa nội dung ID
assert text_or_none("  giữ nguyên  ") == "  giữ nguyên  " and text_or_none("   ") is None and text_or_none(None) is None
try: label_from_rating(3.5); raise SystemExit("phải lỗi")
except exc.DataValidationError: pass
s = _timestamps(pd.Series(["2020-01-01", "May 14, 2020", "1577836800", "", "bad", "May 14", "20-Jul-15", "-100"]))
assert s.tolist() == [MS_2020, 1589414400000, MS_2020, 0, 0, 0, 1437350400000, 0], s.tolist()
ok("parsing: timestamp mili giây, ngày không có năm = 0, ID chỉ chuẩn hóa khoảng trắng, text giữ nguyên")

# ---------------------------------------------------------------- cỡ split 70/10/20
cfg = SplitConfig()
assert (cfg.val_ratio, cfg.test_ratio, cfg.random_state, cfg.strategy) == (0.1, 0.2, 42, "random")
assert [_split_sizes(n, cfg) for n in (3, 4, 10, 20, 100)] == [(1, 1), (1, 1), (1, 2), (2, 4), (10, 20)]
a = DataPipeline._assign_splits(np.repeat([0, 1, 2], [10, 3, 2]).astype(np.int64), np.zeros(15, dtype=np.int64), 3, cfg)
assert [int((a[:10] == c).sum()) for c in (0, 1, 2)] == [7, 1, 2]          # n=10 -> 7/1/2
assert [int((a[10:13] == c).sum()) for c in (0, 1, 2)] == [1, 1, 1]        # n=3
assert (a[13:] == 0).all()                                                # n=2 -> toàn bộ train
for bad in (dict(val_ratio=0), dict(test_ratio=1), dict(val_ratio=.6, test_ratio=.5), dict(random_state=True), dict(strategy="x")):
    try: SplitConfig(**bad); raise SystemExit(f"phải lỗi {bad}")
    except exc.InvalidParameterError: pass
ok("split mặc định 70/10/20: val=max(1,floor(.1n)), test=max(1,floor(.2n)); user <3 tương tác chỉ có train")

tmp = Path(tempfile.mkdtemp())
games_path, inter_path = write_case(tmp, random_rows())
loader = ParquetDataLoader()
records = loader.load_interactions(inter_path)
assert all(isinstance(r, dto.InteractionRecord) and r.rating in (0.0, 1.0) for r in records)
assert loader.load_stats["rows_read"] == len(records) and loader.load_stats["text_column_present"] == 1
assert any(r.text for r in records) and all(r.text is None or r.text.strip() for r in records)
assert all(type(g) is dto.GameItem for g in loader.load_games(games_path).values())
ok("loader: đọc cột text từ nguồn interactions, nhãn 0/1, đếm dòng đọc")

pipeline = DataPipeline()
splits = pipeline.prepare(games_path, inter_path)
tr, va, te = splits.train, splits.validation, splits.test
assert isinstance(splits, dto.DatasetSplits) and (tr.split_name, va.split_name, te.split_name) == ("train", "validation", "test")

# ---------------------------------------------------------------- mapping fit từ train
train_items = {tr.item_mapper.to_raw(x.item_id) for x in tr.interactions}
train_users = {tr.user_mapper.to_raw(x.user_id) for x in tr.interactions}
assert set(tr.item_mapper.raw_ids) == train_items and set(tr.user_mapper.raw_ids) == train_users
assert tr.num_items == len(train_items) and tr.item_mapper.size == tr.num_items
assert tr.item_mapper is va.item_mapper is te.item_mapper and tr.user_mapper is va.user_mapper
assert len({tr.mapping_fingerprint, va.mapping_fingerprint, te.mapping_fingerprint}) == 1
assert tr.num_users == va.num_users == te.num_users and tr.num_items == va.num_items == te.num_items
for ds in (tr, va, te):
    assert all(0 <= x.user_id < ds.num_users and 0 <= x.item_id < ds.num_items for x in ds.interactions)
key = lambda d: {(x.user_id, x.item_id) for x in d.interactions}
assert not (key(tr) & key(va)) and not (key(tr) & key(te)) and not (key(va) & key(te))
assert len({tr.fingerprint, va.fingerprint, te.fingerprint}) == 3
ok("ba split không trùng cặp, dùng chung mapping fit từ train (đối chiếu raw/mapped)")

# ---------------------------------------------------------------- Protocol InteractionDataset
for ds in (tr, va, te):
    for attr in ["split_name", "interactions", "num_interactions", "num_users", "num_items", "user_mapper", "item_mapper",
                 "mapping_fingerprint", "fingerprint", "review_fingerprint", "get_game_by_raw_id", "get_user_seen_items",
                 "get_user_positive_items", "iter_reviews"]:
        assert hasattr(ds, attr), attr
    assert all(isinstance(i, dto.Interaction) for i in ds.interactions) and len(ds.interactions) == ds.num_interactions
    assert isinstance(ds.get_user_seen_items(0), frozenset) and isinstance(ds.get_user_positive_items(0), frozenset)
neg_seen = [u for u in range(tr.num_users) if tr.get_user_seen_items(u) - tr.get_user_positive_items(u)]
assert neg_seen, "seen phải gồm cả negative"
assert tr.get_game_by_raw_id("3").title == "Game 3" and tr.get_game_by_raw_id("3").price == 4.5
ok("Protocol đủ thành viên; seen gồm cả positive lẫn negative; GameItem từ catalog")

# ---------------------------------------------------------------- review text
for ds in (tr, va, te):
    reviews = list(ds.iter_reviews())
    assert len(reviews) == ds.num_reviews and all(isinstance(r, dto.ReviewRecord) for r in reviews)
    pairs = {(x.user_id, x.item_id): x for x in ds.interactions}
    for r in reviews:  # đúng cặp, nhãn, timestamp của interaction trong CÙNG split
        i = pairs[(r.user_id, r.item_id)]
        assert (i.rating, i.timestamp) == (r.rating, r.timestamp)
assert {(r.user_id, r.item_id) for r in tr.iter_reviews()} <= key(tr)
assert not ({(r.user_id, r.item_id) for r in tr.iter_reviews()} & (key(va) | key(te)))
assert tr.num_reviews + va.num_reviews + te.num_reviews > 0
raw_text = {(r.user_id, r.item_id): r.text for r in records if r.text}
for r in te.iter_reviews():   # text nguyên văn kể cả dấu phẩy, nháy, xuống dòng, Unicode
    assert raw_text[(te.user_mapper.to_raw(r.user_id), te.item_mapper.to_raw(r.item_id))] == r.text
ok("train.iter_reviews() không chứa cặp val/test; text nguyên văn; nhãn/timestamp khớp interaction")

# ---------------------------------------------------------------- review_fingerprint nhạy với text, nhãn, timestamp
g = {str(i): GameItem(item_id=str(i), title=f"G{i}") for i in range(1, 4)}
base = [dto.InteractionRecord("x", "1", 1.0, 100, "great")]
fp = lambda rec: InteractionDataset(rec, g).review_fingerprint
assert fp(base) != fp([dto.InteractionRecord("x", "1", 1.0, 100, "gre4t")])    # đổi text
assert fp(base) != fp([dto.InteractionRecord("x", "1", 0.0, 100, "great")])    # đổi nhãn
assert fp(base) != fp([dto.InteractionRecord("x", "1", 1.0, 101, "great")])    # đổi timestamp
assert fp(base) == fp([dto.InteractionRecord("x", "1", 1.0, 100, "great")])    # ổn định
ok("review_fingerprint đổi khi đổi text, nhãn hoặc timestamp")

# ---------------------------------------------------------------- dedup giữ dòng đầu, text của dòng được giữ
rows = random_rows(seed=3)
rows += [("user 10", "1", True, "2019-02-01", "FIRST text"), ("user 10", "1", False, "2021-05-05", "SECOND text")]
rows = [r for r in rows if not (r[0] == "user 10" and r[1] == "1" and r[4] not in ("FIRST text", "SECOND text"))]
t2 = Path(tempfile.mkdtemp()); gp, ip = write_case(t2, rows)
sp2 = DataPipeline().prepare(gp, ip)
assert sp2.report["rows"]["duplicate_rows_dropped"] == 1 and sp2.report["rows"]["duplicate_pairs_with_conflicting_label"] == 1
found = None
for ds in (sp2.train, sp2.validation, sp2.test):
    if ds.user_mapper._to_idx.get("user 10") is not None and ds.item_mapper._to_idx.get("1") is not None:
        u, i = ds.user_mapper.to_idx("user 10"), ds.item_mapper.to_idx("1")
        for x in ds.interactions:
            if (x.user_id, x.item_id) == (u, i):
                found = (x.rating, ds.get_review("user 10", "1"))
assert found == (1.0, "FIRST text"), found
ok("dedup giữ dòng hợp lệ đầu tiên; nhãn, timestamp, text cùng một bản ghi; đếm trùng và nhãn mâu thuẫn")

# ---------------------------------------------------------------- ID ngoài mapping (kịch bản xác định, dùng temporal)
scenario = []
day = lambda d: f"2020-01-{d:02d}"
for u in range(6):   # 6 user chuẩn, xoay thứ tự item để mọi item 1..10 đều có trong train
    for k in range(10):
        scenario.append((f"base{u}", str((u + k) % 10 + 1), True, day(k + 1), ""))
for g_ in range(1, 8): scenario.append(("A", str(g_), True, day(g_), ""))
scenario += [("A", "8", True, day(8), ""), ("A", "9", True, day(9), ""), ("A", "99", True, day(10), "cold positive")]
for g_ in range(1, 8): scenario.append(("B", str(g_), True, day(g_), ""))
scenario += [("B", "98", False, day(8), ""), ("B", "9", True, day(9), ""), ("B", "10", True, day(10), "")]
t3 = Path(tempfile.mkdtemp()); gp, ip = write_case(t3, scenario, n_games=99)
sp3 = DataPipeline().prepare(gp, ip, split_config=SplitConfig(strategy="temporal"))
rep = sp3.report
assert "99" not in sp3.train.item_mapper.raw_ids and "98" not in sp3.train.item_mapper.raw_ids
for ds in (sp3.validation, sp3.test):
    assert all(0 <= x.item_id < ds.num_items for x in ds.interactions)   # không có -1, không mở rộng mapping
assert sp3.test.item_mapper.size == sp3.train.item_mapper.size
ex_test, ex_val = rep["exclusions_by_mapping"]["test"], rep["exclusions_by_mapping"]["validation"]
assert ex_test["users_excluded_for_positive_outside_mapping"] == 1
assert ex_test["positive_rows_outside_mapping"] == 1 and ex_test["rows_removed_with_excluded_users"] == 2
a_idx = sp3.train.user_mapper.to_idx("A")
assert not sp3.test.get_user_seen_items(a_idx)                                   # A bị loại khỏi phép chấm test
assert ex_val["negative_rows_outside_mapping_dropped"] == 1                      # negative ngoài mapping: bỏ và đếm riêng
b_idx = sp3.train.user_mapper.to_idx("B")
assert sp3.validation.get_user_seen_items(b_idx) == frozenset() or "98" not in {sp3.validation.item_mapper.to_raw(i) for i in sp3.validation.get_user_seen_items(b_idx)}
assert sp3.test.get_user_positive_items(b_idx)                                   # B vẫn được chấm test
assert rep["rows"]["unique_user_item_pairs"] == sum(v["rows_before_filter"] for v in rep["exclusions_by_mapping"].values())
assert rep["exclusions_by_mapping"]["train"]["rows_kept"] == rep["exclusions_by_mapping"]["train"]["rows_before_filter"] == sp3.train.num_interactions
ok("ID ngoài mapping: user có positive ngoài mapping bị loại khỏi split và ghi nhận; negative ngoài mapping bỏ và đếm riêng")

# ---------------------------------------------------------------- xuất artifact review
out = Path(tempfile.mkdtemp()); manifest = pipeline.export_artifacts(splits, out)
assert {"reviews.csv", "train_reviews.jsonl", "validation_reviews.jsonl", "test_reviews.jsonl", "mapping.json"} <= set(manifest)
assert (out / "data_report.json").is_file()
with open(out / "reviews.csv", encoding="utf-8", newline="") as handle:
    table = list(csv.reader(handle))
assert table[0] == ["user_id", "item_id", "rating", "timestamp", "text", "split"]
total = tr.num_interactions + va.num_interactions + te.num_interactions
assert len(table) - 1 == total == manifest["reviews.csv"]["records"]
assert {r[5] for r in table[1:]} == {"train", "validation", "test"}
texts_csv = {(r[0], r[1]): r[4] for r in table[1:] if r[4]}
assert texts_csv == {k: v for k, v in raw_text.items() if k in texts_csv} and len(texts_csv) == sum(d.num_reviews for d in (tr, va, te))
for name, ds in (("train", tr), ("validation", va), ("test", te)):
    lines = (out / f"{name}_reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == ds.num_reviews == manifest[f"{name}_reviews.jsonl"]["records"]
    for line in lines:
        o = json.loads(line)
        assert set(o) == {"user_id", "item_id", "rating", "timestamp", "text"} and isinstance(o["user_id"], int) and o["text"].strip()
for name, info in manifest.items():
    assert hashlib.sha256((out / name).read_bytes()).hexdigest() == info["sha256"]
mp = json.loads((out / "mapping.json").read_text(encoding="utf-8"))
assert mp["user_raw_ids"] == list(tr.user_mapper.raw_ids) and mp["item_raw_ids"] == list(tr.item_mapper.raw_ids)
assert splits.report["artifacts"] == manifest and json.loads((out / "data_report.json").read_text(encoding="utf-8"))["artifacts"]
ok("artifact: reviews.csv (có split, quote đúng), 3 JSONL id nội bộ, mapping.json, report + checksum khớp")

# ---------------------------------------------------------------- report
r = splits.report
for k in ("protocol", "rows", "mapping", "splits", "exclusions_by_mapping", "text", "timestamp", "catalog", "user_id_source", "fingerprints"):
    assert k in r, k
assert r["protocol"]["random_state"] == 42 and r["timestamp"]["unit"] == "milliseconds_utc"
assert r["text"]["available"] and r["text"]["train_items_with_review"] > 0
assert r["catalog"]["price_currency"] == "unspecified" and "tên" in r["user_id_source"]["note"]
assert r["fingerprints"]["mapping"] == tr.mapping_fingerprint
json.dumps(r, default=str)
ok("report đủ mục: rows/users/items tách riêng, text, timestamp, fingerprints, giới hạn user ID")

# ---------------------------------------------------------------- tái lập, không phụ thuộc thứ tự đầu vào
again = DataPipeline().prepare(games_path, inter_path)
assert (again.train.fingerprint, again.test.fingerprint) == (tr.fingerprint, te.fingerprint)
shuffled = list(random_rows()); random.Random(9).shuffle(shuffled)
t4 = Path(tempfile.mkdtemp()); gp, ip = write_case(t4, shuffled)
sp4 = DataPipeline().prepare(gp, ip)
assert (sp4.train.fingerprint, sp4.validation.fingerprint, sp4.test.fingerprint) == (tr.fingerprint, va.fingerprint, te.fingerprint)
other = DataPipeline().prepare(games_path, inter_path, split_config=SplitConfig(random_state=7))
assert other.train.fingerprint != tr.fingerprint
ok("tái lập theo seed; kết quả không phụ thuộc thứ tự dòng đầu vào")

# ---------------------------------------------------------------- chỉ đọc
try: tr._users[0] = 5; raise SystemExit("mảng phải chỉ đọc")
except ValueError: pass
assert not hasattr(tr, "add_reviews_batch") and not hasattr(tr, "split")
assert isinstance(tr.item_mapper.raw_ids, tuple)
ok("bản ghi và mapping chỉ đọc; không còn API sửa review sau khi giao")

# ---------------------------------------------------------------- BaseRecommender thật
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
m = Pop("pop"); m.fit(tr)
out_rec = m.recommend(5, tr, top_k=3)
assert len(out_rec) == 3 and all(o.item_idx not in tr.get_user_seen_items(5) for o in out_rec)
try: m.recommend(5, te); raise SystemExit("phải từ chối test split")
except exc.DataValidationError: pass
mat = tr.get_sparse_matrix(); assert mat.shape == (tr.num_users, tr.num_items) and set(mat.data) <= {1.0}
ok("BaseRecommender.recommend chạy với train split; ma trận thưa chỉ tương tác dương")

# ---------------------------------------------------------------- lỗi, thiếu text, reviews_path
for bad, e in [(lambda: tr.get_user_seen_items(10**6), exc.UnknownUserError),
               (lambda: tr.get_user_seen_items(True), exc.UnknownUserError),
               (lambda: tr.get_game_by_raw_id("nope"), exc.UnknownItemError),
               (lambda: tr.user_mapper.to_idx("zzz"), exc.UnknownUserError),
               (lambda: tr.item_mapper.to_raw(10**6), exc.UnknownItemError)]:
    try: bad(); raise SystemExit("không ném lỗi")
    except e: pass
t5 = Path(tempfile.mkdtemp()); gp, ip = write_case(t5, [(f"u{u}", str(g_), True, "2020-01-01", "") for u in range(3) for g_ in range(1, 3)])
try: DataPipeline().prepare(gp, ip); raise SystemExit("không user nào đủ 3 tương tác phải báo lỗi")
except exc.EmptyDatasetError: pass
t6 = Path(tempfile.mkdtemp()); gp, ip = write_case(t6, random_rows(), with_text=False)
no_text = DataPipeline().prepare(gp, ip)
assert not no_text.report["text"]["available"] and "TextBPR" in no_text.report["text"]["warning"]
side = t6 / "reviews_side.csv"
with open(side, "w", encoding="utf-8", newline="") as handle:
    w = csv.writer(handle); w.writerow(["user_id", "game_id", "review"])
    for row in random_rows():
        if row[4]: w.writerow([row[0], row[1], row[4]])
    w.writerow(["user 0", "1", "   "])
with_side = DataPipeline().prepare(gp, ip, reviews_path=str(side))
assert with_side.report["text"]["available"] and with_side.report["text"]["rows_with_text_from_reviews_path"] > 0
assert with_side.report["text"]["reviews_path_stats"]["rows_blank_text"] >= 1
try: DataPipeline().prepare(gp, ip, split_config=SplitConfig(strategy="temporal")) if False else None
except Exception: pass
try:
    DataPipeline().prepare(games_path, inter_path, split_config=SplitConfig(strategy="temporal")); print("   (temporal chạy được vì có timestamp thật)")
except exc.InvalidParameterError:
    raise SystemExit("temporal phải chạy với timestamp thật")
ok("lỗi đúng loại; thiếu text -> cảnh báo TextBPR; reviews_path gắn text và đếm dòng trống")

print("ALL PASSED")
