# Bàn giao Data Layer – Game Recommendation System

Người bàn giao: Member 1 (Data) · Ngày: 06/11/2026 · Áp dụng: Contracts v1 (2026-11-05)

Tài liệu dành cho Member 2 (Model), Member 3 (Evaluation) và nhóm Service. Phần cuối hướng dẫn chạy data layer trên Kaggle và trên máy local, kèm lưu ý về RAM.

## 1. Tổng quan

Data layer biến dữ liệu thô (`games.json` và thư mục `Games_Reviews/*.csv`) thành đối tượng `DatasetSplits` gồm 3 tập `train`, `validation`, `test` (kiểu `InteractionDataset` chỉ đọc) đã gắn chỉ số nguyên cùng báo cáo `report`, để mọi model và evaluator dùng chung một bộ dữ liệu chuẩn theo Contracts v1.

Luồng xử lý:

1. `ingest.py` đọc dữ liệu thô và ghi ra `games.parquet`, `interactions.parquet` (đã tích hợp sẵn cột `text` và `timestamp` mili giây), cùng `ingest_report.json`. Chỉ chạy khi chưa có Parquet hoặc khi `rebuild=True`.

2. `loader.py` (`ParquetDataLoader`) đọc Parquet/CSV thành `InteractionRecord` (có trường `text`) và `GameItem` của contract, đồng thời thống kê các dòng bị loại do thiếu ID hoặc sai nhãn.

3. `pipeline.py` (`DataPipeline`) là cổng chuẩn bị dữ liệu chính: gắn review text, khử trùng lặp (dedup), chia tập theo từng user (`70/10/20`), **fit `IdMapper` duy nhất từ `train` rồi đóng băng**, lọc dữ liệu ngoài mapping ở `validation`/`test`, dựng 3 `InteractionDataset` bất biến và xuất bộ artifacts bàn giao (`export_artifacts`).

4. `dataset.py` (`IdMapper`, `Catalog`, `InteractionDataset`) quản lý cấu trúc dữ liệu chỉ đọc, tính toán `mapping_fingerprint`, `fingerprint` và `review_fingerprint`.

5. `data_layer.py` là điểm vào nhanh: hàm `main()` tự động kiểm tra/build Parquet, gọi `DataPipeline.prepare()` và trả về trực tiếp `DatasetSplits` (hàm `make_splits()` cũ đã được thay thế hoàn toàn bởi `DataPipeline`).

Đã chạy thành công trên Kaggle với bộ dữ liệu game Steam và vượt qua toàn bộ kiểm thử tự động trong `test_data_layer.py`.

## 2. Cây thư mục và vai trò file

```
src/
  __init__.py
  contracts/            # do cả nhóm chung, KHÔNG sửa riêng
    __init__.py
    dto.py              # InteractionRecord, Interaction, ReviewRecord, GameItem, DatasetSplits, ...
    exceptions.py       # RecSysError và các lỗi con
    interfaces.py       # BaseRecommender, BaseEvaluator, BaseMetric, BaseDataLoader
  data/                 # phần của Member 1
    __init__.py         # re-export DataPipeline, SplitConfig, InteractionDataset, IdMapper, GameItem, GameStats
    parsing.py          # làm sạch ID, đọc ngày sang mili giây UTC, nhãn 0/1, giữ nguyên text, Wilson
    entities.py         # GameItem (re-export từ contract), GameStats, make_game()
    ingest.py           # dữ liệu thô -> games.parquet, interactions.parquet, ingest_report.json
    loader.py           # Parquet/CSV -> DTO của contract, đếm thống kê load_stats
    dataset.py          # IdMapper, Catalog, InteractionDataset (chỉ đọc)
    pipeline.py         # DataPipeline.prepare(), DataPipeline.export_artifacts(), SplitConfig
    data_layer.py       # main(), CLI
```

## 3. Những gì data layer đảm bảo

| Hạng mục | Cam kết |
| :--- | :--- |
| Nhãn rating | Nhị phân: `Recommended = 1.0`, `Not Recommended = 0.0`. Không còn thang 1–5, không chứa điểm Wilson hay playtime. |
| Trùng cặp (user, item) | Loại dòng lỗi trước khi dedup. Mỗi cặp giữ **bản ghi hợp lệ đầu tiên** theo thư tự file/dòng xác định (`argsort(kind="stable")`). Nhãn, timestamp và text cam kết lấy từ cùng một bản ghi được giữ. |
| Timestamp | `int >= 0`, **Unix mili giây UTC**, lấy từ cột `post_date`/`timestamp`. Ngày thiếu năm (như `"May 14"`) hoặc không đọc được thì gán `0`. |
| Chuẩn hóa ID & Text | ID gốc chỉ `.strip()` khoảng trắng, không đổi hoa/thường hay sửa nội dung. Review text giữ nguyên văn chuỗi gốc và Unicode; thiếu hoặc toàn khoảng trắng thì là `None`. |
| Chỉ số id & Mapping | `user_mapper` và `item_mapper` được **fit duy nhất từ toàn bộ tương tác `train`** (cả `1.0` và `0.0`) rồi đóng băng (`0..N-1`). `validation` và `test` dùng chung đúng bộ mapper này, không tự mở rộng mapping và không gán `-1`. |
| Xử lý ID ngoài mapping | Ở `validation`/`test`: nếu một user có bất kỳ tương tác **positive (`1.0`)** nào nằm ngoài mapping `train`, **toàn bộ user đó bị loại khỏi phép chấm của split đó** và được ghi nhận; các dòng **negative (`0.0`)** ngoài mapping thì chỉ bỏ dòng đó và đếm riêng. |
| Cách chia (`SplitConfig`) | Mặc định đã thống nhất: **Per-user random split, seed `42`, tỷ lệ `70% / 10% / 20%`** (`val_ratio=0.1`, `test_ratio=0.2`). User có $n < 3$ tương tác: đưa toàn bộ vào `train`. User có $n \ge 3$ tương tác: $\text{val} = \max(1, \lfloor 0.1n \rfloor)$, $\text{test} = \max(1, \lfloor 0.2n \rfloor)$, còn lại vào `train`. Đầu vào được sắp xếp xác định trước khi chia nên kết quả không phụ thuộc thứ tự dòng hay `chunksize`. |
| Chiến lược chia | `"random"` (mặc định chuẩn của Contracts v1) hoặc `"temporal"` (tùy chọn mở rộng ngoài v1: cũ nhất vào `train`, mới nhất vào `test`, yêu cầu dữ liệu có timestamp thật $> 0$). |
| Tính bất biến (Read-only) | Mọi mảng numpy bên trong `InteractionDataset` đều khóa `writeable = False`. Đã bỏ các hàm sửa đổi sau khởi tạo (`split()`, `add_reviews_batch()`). |
| GameItem & GameStats | `GameItem` là `src.contracts.dto.GameItem` (`title` rỗng lấy `item_id`, `price` thiếu/âm là `None`). Thông tin ngoài contract (`positive`, `negative`, `tags`, `wilson_rating`) tách sang `GameStats`. Game chỉ có trong interactions được tạo `GameItem` tổng hợp, đếm bằng `num_synthetic_items`. |
| Fingerprint | `mapping_fingerprint` (chung cho mọi split), `fingerprint` (nội dung tương tác + mapping), `review_fingerprint` (băm cả cặp `user-item`, `rating`, `timestamp` và nội dung `text` của split). |
| Bàn giao Review Text | Text đi theo đúng cặp `(user, item)` và split chứa cặp đó. `train.iter_reviews()` chỉ trả review không rỗng thuộc `train`, tuyệt đối không lẫn `validation`/`test`. Hỗ trợ xuất đủ bộ 6 file artifacts ra đĩa. |

### Cách lấy dữ liệu (API nhanh)

```python
from pathlib import Path
from src.data import DataPipeline, SplitConfig
from src.data.data_layer import main

# Cách 1: Gọi qua hàm main() (tự nạp Parquet, chia tập và tùy chọn xuất artifacts)
splits = main(
    data_dir=Path("data/processed"),
    export_dir=Path("data/artifacts"),  # tùy chọn: ghi reviews.csv, *_reviews.jsonl, mapping.json, data_report.json
    price_currency="USD",
)
train, val, test = splits.train, splits.validation, splits.test

# Cách 2: Gọi trực tiếp cổng chuẩn bị dữ liệu DataPipeline.prepare()
pipeline = DataPipeline(price_currency="USD")
splits = pipeline.prepare(
    games_path="data/processed/games.parquet",
    interactions_path="data/processed/interactions.parquet",
    split_config=SplitConfig(val_ratio=0.1, test_ratio=0.2, random_state=42, strategy="random"),
)
```

| Cần gì | Gọi gì |
| :--- | :--- |
| Số lượng | `num_users`, `num_items`, `num_interactions`, `num_positive`, `num_negative`, `num_active_users`, `num_reviews` |
| Tương tác (id int) | `interactions` (tuple `Interaction`), `indexed_interactions()` (list `(u, i, rating)`) |
| Ma trận thưa | `get_sparse_matrix(positive_only=True)`, CSR `(num_users, num_items)`, mặc định chỉ giữ tương tác dương |
| Item đã thấy | `get_user_seen_items(user_id)` trả `frozenset[int]` (gồm cả tương tác dương `1.0` và âm `0.0` trong split) |
| Item tích cực | `get_user_positive_items(user_id)` trả `frozenset[int]` (chỉ `rating == 1.0`) |
| Review text | `iter_reviews()` trả `Iterator[ReviewRecord]` (chỉ review có text không rỗng của split đó); `get_review(user_raw, item_raw)` trả chuỗi |
| Metadata game | `get_game_by_raw_id(raw)` trả `GameItem`; `game_stats[raw]` trả `GameStats` (`positive`, `negative`, `tags`, `wilson_rating`) |
| Id thô <-> chỉ số | `user_mapper.contains_raw(raw)`, `user_mapper.to_idx(raw)`, `item_mapper.to_raw(idx)`, `item_mapper.size`, `item_mapper.raw_ids` |
| Báo cáo & Artifacts | `mapping_report()`, `splits.report`, `pipeline.export_artifacts(splits, output_dir)` |

### Bộ Artifacts bàn giao trên đĩa (`export_artifacts`)

Khi gọi `pipeline.export_artifacts(splits, output_dir)` hoặc truyền `--export-dir`, hệ thống ghi 6 file sau kèm mã kiểm tra `sha256` trong `splits.report["artifacts"]`:

1. `reviews.csv`: `user_id,item_id,rating,timestamp,text,split` (ID gốc; 1 dòng cho mỗi interaction sau lọc/dedup; thiếu text để ô rỗng `""`).
2. `train_reviews.jsonl`: `{"user_id": int, "item_id": int, "rating": int, "timestamp": int, "text": str}` (ID nội bộ; chỉ review có text không rỗng của tập `train` — **nguồn chuẩn để tạo features cho TextBPR**).
3. `validation_reviews.jsonl`: Cùng schema ID nội bộ, chỉ chứa review không rỗng của `validation`.
4. `test_reviews.jsonl`: Cùng schema ID nội bộ, chỉ chứa review không rỗng của `test`.
5. `mapping.json`: Lưu `mapping_fingerprint`, `user_raw_ids` và `item_raw_ids` để đối chiếu hai chiều.
6. `data_report.json`: Toàn bộ nội dung `splits.report` kèm thống kê làm sạch, loại trừ mapping, thống kê text và checksum của từng file artifact.

## 4. Dành cho Model (Member 2)

* **Chỉ fit trên `train`.** Không đưa `validation` hay `test` vào mảng huấn luyện. Validation chỉ dùng qua callback `validation_score` để chọn checkpoint / early stopping.
* **Phải gán fingerprint khi fit.** `BaseRecommender.recommend()` kiểm tra `split_name == "train"` và so khớp `num_users`, `num_items`, `mapping_fingerprint`, `fingerprint`. Trong `fit()` cần đặt:

```python
self._num_users = train_dataset.num_users
self._num_items = train_dataset.num_items
self._mapping_fingerprint = train_dataset.mapping_fingerprint
self._train_fingerprint = train_dataset.fingerprint
self._is_fitted = True
```

* **Nhãn là `0.0` / `1.0`.** Rating `0.0` nghĩa là Not Recommended (phản hồi tiêu cực thật, không phải "chưa xem").
  * `Popularity` đếm tương tác dương (`rating == 1.0`) trong `train`.
  * `BPR` / `TextBPR` học từ các cặp dương (`1.0`), nhưng dùng toàn bộ tương tác đã quan sát trong `train` (cả `1.0` và `0.0`, lấy qua `get_user_seen_items`) để chặn khi lấy mẫu âm (negative sampling).
* **Text và `ItemFeatures` cho TextBPR.**
  * Kiểm tra `splits.report["text"]["available"]` trước khi chạy TextBPR. Nếu nguồn không có review text, cờ này sẽ là `False` kèm cảnh báo.
  * Chỉ đọc review từ `train.iter_reviews()` hoặc file `train_reviews.jsonl`. Tuyệt đối không nối cả 3 file JSONL để học encoder/vocabulary/IDF.
  * Khi tạo `ItemFeatures`, dùng `train.item_mapper.raw_ids` làm `item_ids` để hàng `i` ứng với item nội bộ `i`; gán `mapping_fingerprint = train.mapping_fingerprint`, `train_fingerprint = train.fingerprint`, và `text_fingerprint = train.review_fingerprint`.
* **Không còn item ngoài mapping ở Validation/Test.** Do mapping được đóng băng từ `train` và Data layer đã lọc các dòng ngoài mapping theo đúng quy tắc của Contract, mọi `item_id` xuất hiện trong `validation` và `test` đều nằm trong khoảng `0 .. num_items - 1`.
* **User ít tương tác.** User có dưới 3 tương tác được đưa toàn bộ vào `train` và không xuất hiện trong `validation` hay `test`.
* **Lỗi nên ném** (đúng contract): `ModelNotFittedError` khi gọi trước `fit`, `ConfigurationError` khi siêu tham số/features sai, `InvalidParameterError` khi `top_k` hoặc `user_id` sai, `TrainingError` khi huấn luyện hoặc callback thất bại.

## 5. Dành cho Evaluation (Member 3)

* **Giao diện nhận trực tiếp `DatasetSplits` / `InteractionDataset`.** Evaluator nhận `train_dataset` và `evaluation_dataset` (`validation` hoặc `test`), kiểm tra khớp `num_users`, `num_items`, `mapping_fingerprint` và `model.train_fingerprint == train_dataset.fingerprint` (so sánh bằng giá trị chuỗi `==`, không dùng `is`).
* **Ground truth** của một user là `evaluation_dataset.get_user_positive_items(user_id)`. User không có positive trong `train` hoặc không có positive ground truth trong split đánh giá được đưa vào `SkippedUser` (kèm lý do), không tính vào mẫu số metric.
* **Loại item đã thấy (Seen Filtering).** Mặc định loại mọi item trong `train_dataset.get_user_seen_items(user_id)` (gồm cả `1.0` và `0.0`). Khi chấm trên `test` và có truyền `additional_seen_dataset=splits.validation`, hợp thêm `validation.get_user_seen_items(user_id)` vào tập lịch sử cần loại trừ.
* **Tách biệt số liệu loại trừ của Data và Skips của Evaluator.**
  * Các dòng/user bị loại ở bước Data do rơi ra ngoài mapping `train` đã được thống kê sẵn trong `splits.report["exclusions_by_mapping"][split_name]` (gồm `users_excluded_for_positive_outside_mapping` và `negative_rows_outside_mapping_dropped`).
  * Evaluator chỉ đếm riêng những user bị skip trong lúc chấm điểm (`EvaluationResult.skipped_users`), không cộng trộn hai con số này làm một.
* **Metadata cho Benchmark (`run_config.json`).** Ghi lại `splits.report` (chứa đầy đủ `protocol`, `rows`, `mapping`, `splits`, `exclusions_by_mapping`, `text`, `fingerprints` và `artifacts`).

## 6. Dành cho Service

**Khởi động:** nhận `splits = main(...)` (hoặc từ `DataPipeline.prepare()`), gọi `service.train(splits.train, ...)` và lưu giữ `splits.train` làm nguồn tra cứu catalog, lịch sử và mapping.

**Luồng gợi ý (`recommend_games`):**

```python
if not train_dataset.user_mapper.contains_raw(raw_user_id):
    # Xử lý UnknownUserError hoặc chuyển sang Fallback nếu bật
user_idx = train_dataset.user_mapper.to_idx(raw_user_id)
if not train_dataset.get_user_positive_items(user_idx):
    # User chỉ có review tiêu cực trong train -> ColdStartError hoặc chuyển sang Fallback

scored = model.recommend(user_idx, train_dataset, top_k=top_k, exclude_seen=exclude_played)
for rank, s in enumerate(scored, 1):
    raw_item_id = train_dataset.item_mapper.to_raw(s.item_idx)
    game = train_dataset.get_game_by_raw_id(raw_item_id)   # GameItem
    # Tạo RecommendationResult(rank, game.item_id, game.title, game.genres,
    #                          s.score, game.price, game.developer, model.name, is_fallback=False)
```

* **Chính sách Fallback & Cold-start.**
  * Mặc định không tự bật fallback; nếu user không thuộc `user_mapper` thì ném `UnknownUserError`, nếu user thuộc mapping nhưng chưa có tương tác dương (`1.0`) trong `train` thì ném `ColdStartError`.
  * Khi bật fallback, chỉ dùng model toàn cục (như `Popularity`) đã train trên cùng `splits.train` và đặt `is_fallback=True`.
* **Lọc theo giá và thể loại.**
  * Theo Contract, `game.price` có thể là `None` (không rõ giá). Khi người dùng lọc theo `max_price`, **các game có `price is None` phải bị loại khỏi kết quả lọc** (không được tự coi `None` là `0.0`/miễn phí).
  * `genres` là `tuple[str, ...]`, `developer` mặc định là `"Unknown"`.
* **Giới hạn định danh User từ nguồn.** Dữ liệu gốc chỉ có tên hiển thị (`display_name`), không có Steam ID duy nhất; thông tin này đã được ghi rõ tại `splits.report["user_id_source"]`.

## 7. Chạy trên Kaggle

**Chuẩn bị:** thêm hai dataset vào notebook bằng Add Input: dataset chứa `src` (`synnix/source`) và dataset dữ liệu gốc (`synnix/gamedataset`, gồm `games.json` và `Games_Reviews`).

**Cell 1 – chép `src` sang chỗ ghi được và import** (`/kaggle/input` chỉ đọc):

```python
import sys, shutil
from pathlib import Path

for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[name]

WORK = Path("/kaggle/working")
shutil.copytree("/kaggle/input/datasets/synnix/source/src", WORK / "src", dirs_exist_ok=True)

sys.path.insert(0, str(WORK))
from src.data import DataPipeline, SplitConfig
from src.data.data_layer import main
```

**Cell 2 – build Parquet và chia `DatasetSplits` (chạy lần đầu, mất vài phút đến vài chục phút):**

```python
splits = main(
    raw_dir=Path("/kaggle/input/datasets/synnix/gamedataset"),
    output_dir=Path("/kaggle/working/processed"),
    export_dir=Path("/kaggle/working/artifacts"),  # xuất luôn bộ review text bàn giao
    rebuild=True,
    min_reviews_per_game=5,
    min_interactions_per_user=1,
)
train, val, test = splits.train, splits.validation, splits.test
```

File Parquet mới nằm ở `/kaggle/working/processed` (`games.parquet`, `interactions.parquet`, `ingest_report.json`) và bộ file bàn giao nằm ở `/kaggle/working/artifacts`.

**Cell 3 – kiểm tra nhanh báo cáo và timestamp:**

```python
print("Tỉ lệ timestamp = 0:", splits.report["timestamp"]["zero_ratio"])
print("Cờ review text:", splits.report["text"]["available"])
print("Fingerprints:", splits.report["fingerprints"])
```

**Các lần sau**, nạp thẳng Parquet đã build thay vì rebuild:

```python
splits = main(
    data_dir=Path("/kaggle/working/processed"),
    export_dir=Path("/kaggle/working/artifacts"),
)
```

| Lỗi | Cách xử lý |
| :--- | :--- |
| `NameError: main` | Chưa chạy Cell 1 hoặc kernel đã khởi động lại. |
| `FileNotFoundError ... games.json` | `raw_dir` sai; trỏ vào thư mục gốc của dataset chứa `games.json`. |
| `rating phải là nhãn nhị phân 0/1` | Parquet cũ có rating thang 1–5; chạy lại với `rebuild=True`. |
| `strategy='temporal' cần timestamp thật` | Parquet cũ chưa có cột `timestamp`; chạy lại với `rebuild=True`. |
| Hết RAM | Xem mục 9. |

## 8. Chạy trên máy local

**Yêu cầu:** Python 3.10 trở lên.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas pyarrow numpy scipy pytest
```

**Cách khuyến nghị (nhẹ cho máy local):** build Parquet trên Kaggle, tải `games.parquet`, `interactions.parquet` (và `ingest_report.json`) về thư mục `data/processed/`, rồi chạy ở local:

```
project/
  src/...
  data/processed/games.parquet
  data/processed/interactions.parquet
  data/processed/ingest_report.json
```

```bash
# Chạy từ thư mục gốc của project (đã sửa _ROOT = parents[2] chuẩn xác)
python -m src.data.data_layer --data-dir ./data/processed --export-dir ./data/artifacts
```

Hoặc trong Python:

```python
from pathlib import Path
from src.data.data_layer import main

splits = main(data_dir=Path("data/processed"), export_dir=Path("data/artifacts"))
```

**Nếu muốn build từ dữ liệu thô ở local:**

```bash
python -m src.data.data_layer --raw-dir ./data/raw --output-dir ./data/processed \
    --rebuild --min-reviews-per-game 5 --min-interactions-per-user 1 \
    --export-dir ./data/artifacts
```

**Các cờ dòng lệnh hỗ trợ:**
* `--data-dir`: Thư mục chứa file Parquet.
* `--raw-dir`: Thư mục chứa `games.json` và `Games_Reviews`.
* `--output-dir`: Nơi lưu Parquet mới tạo.
* `--reviews-path`: File CSV review rời (chỉ cần khi dùng Parquet cũ chưa có cột `text`).
* `--rebuild`: Tạo lại Parquet từ dữ liệu gốc.
* `--min-reviews-per-game` (mặc định `5`), `--min-interactions-per-user` (mặc định `1`).
* `--strategy {random,temporal}` (mặc định `random`), `--val-ratio` (mặc định `0.1`), `--test-ratio` (mặc định `0.2`), `--seed` (mặc định `42`).
* `--export-dir`: Thư mục xuất bộ artifacts review text và report.
* `--price-currency`: Đơn vị tiền tệ ghi vào report (mặc định `"unspecified"`).

**Chạy bộ kiểm thử tự động:**

```bash
python test_data_layer.py
# hoặc: python -m pytest test_data_layer.py
```

## 9. Lưu ý về RAM

**Dữ liệu lớn cần nhiều RAM.** Đây là điểm cần cân nhắc trước khi chạy trên toàn bộ tập dữ liệu Steam.

**Vì sao tốn RAM:**
* Bước nạp (`loader.py`) tạo danh sách `InteractionRecord` trước khi `DataPipeline` chuyển sang các mảng numpy đóng băng.
* Review chữ được giữ trong bộ nhớ theo từng cặp `(user, item)` có text để phục vụ `iter_reviews()` và tính `review_fingerprint`. Với hàng triệu review dài, chuỗi văn bản là phần chiếm nhiều RAM nhất.
* Thuộc tính `dataset.interactions` và `dataset.records` được khởi tạo lười (lazy) ở lần truy cập đầu tiên rồi cache lại thành tuple các đối tượng Python.
* Bước `ingest.py` và `export_artifacts()` ghi theo từng lô/luồng nên rất nhẹ RAM.

**Ước tính thô cho đỉnh RAM lúc nạp (chưa tính review chữ):**

| Số tương tác | RAM ước tính |
| :--- | :--- |
| 1 triệu | khoảng 0,5 GB |
| 10 triệu | khoảng 5 GB |
| 30 triệu | khoảng 15 GB |

**Cách giảm RAM:**
1. Tăng `min_reviews_per_game` và `min_interactions_per_user` khi build để thu nhỏ dữ liệu ngay từ bước tạo Parquet.
2. Nếu chỉ thử nghiệm `Popularity` hoặc `BPR` trên máy ít RAM và chưa cần chạy `TextBPR`, có thể build Parquet với `include_reviews=False` (lúc này report sẽ tự động ghi nhận `text.available = False` và cảnh báo không đủ điều kiện chạy `TextBPR`).
3. Tránh gọi `dataset.records` trên tập dữ liệu hàng chục triệu dòng; khi xuất file hoặc duyệt dữ liệu thô, hãy dùng `dataset.iter_raw_rows()` hoặc `dataset.iter_reviews()` (duyệt theo generator, không tạo tuple khổng lồ).
4. Build Parquet và xuất artifacts trên Kaggle (30 GB RAM), sau đó tải bộ file đã xử lý về máy local.

## 10. Hạn chế của Data Layer và Dữ liệu gốc

Trong quá trình xử lý và thiết kế theo Contracts v1, data layer có một số hạn chế khách quan cần lưu ý:

1. **Định danh User dựa trên Tên hiển thị (`display_name`):**
   * Dữ liệu gốc từ Steam Reviews không cung cấp Steam ID 64-bit duy nhất cho từng tài khoản, mà chỉ lưu tên hiển thị (`user`). Do đó, các tài khoản có cùng tên hiển thị sẽ bị hệ thống gộp chung thành một `user_id`. Giới hạn này đã được ghi nhận tại `splits.report["user_id_source"]`.
2. **Sự thiếu hụt và thưa thớt của Review Text:**
   * Không phải mọi tương tác đều có nội dung review (nhiều user chỉ bấm nút Recommended/Not Recommended mà không viết chữ). Một số lượng lớn các cặp `(user, item)` có text `None`, đòi hỏi mô hình TextBPR phải xử lý khéo léo các trường hợp vector zero (như được quy định trong contract).
3. **Chất lượng và Định dạng Ngày tháng (`post_date`):**
   * Một số dòng dữ liệu có định dạng ngày tháng không chuẩn hoặc khuyết thiếu năm (ví dụ chỉ có `"May 14"` mà không có năm). Theo quy tắc an toàn, data layer sẽ chuyển đổi các trường hợp này về giá trị `0` thay vì cố gắng đoán năm, dẫn đến một tỷ lệ timestamp bằng `0` (có thể kiểm tra qua `splits.report["timestamp"]["zero_ratio"]`).
4. **Cold-start cực đoan cho Item và User:**
   * Do chính sách đóng băng mapping từ `train` và loại bỏ user có positive ngoài mapping, các user mới xuất hiện ở tập `validation`/`test` không thuộc mapping train sẽ bị loại khỏi phép chấm điểm metric của Evaluator. Tương tự, một số item hoàn toàn mới xuất hiện ở test (cold items) đòi hỏi thuật toán đề xuất phải có chiến lược fallback hợp lý.

## 11. Lưu ý chung và cập nhật so với bản cũ

* **Parquet cũ phải rebuild.** Parquet tạo ở phiên bản trước lưu `timestamp` theo giây và chưa tích hợp sẵn cột `text` trong `interactions.parquet`; cần chạy lại với `rebuild=True` để đồng bộ sang **Unix mili giây UTC** và giữ review text đi kèm từng dòng tương tác.
* **Những thay đổi API quan trọng đã áp dụng theo Contracts v1:**
  * Thay `make_splits()` và `dataset.split()` bằng `DataPipeline.prepare()` (hoặc gọi thẳng `main()` trả về `DatasetSplits`).
  * Thay quy tắc dedup từ "giữ dòng có timestamp mới nhất" sang **"giữ dòng hợp lệ đầu tiên theo thứ tự file/dòng xác định"** và đếm số cặp có nhãn mâu thuẫn.
  * Thay việc fit mapping trên toàn bộ dataset bằng **fit mapping chỉ trên tập `train` rồi đóng băng**, kèm cơ chế lọc user có positive ngoài mapping ở `validation`/`test`.
  * Bổ sung hàm `pipeline.export_artifacts(splits, output_dir)` để xuất đủ 6 file bàn giao chuẩn kèm mã băm `SHA-256`.
  * Đã sửa `_ROOT = Path(__file__).resolve().parents[2]` và đồng bộ mặc định `--min-reviews-per-game = 5` trong `data_layer.py`.
* **Định dạng ngày.** `post_date` dạng `20-Jul-15`, `7/20/2015`, `2015-07-20`, `May 14, 2020` đều đọc được; ngày không có năm (ví dụ `"May 14"`) sẽ trả về `0` thay vì tự đoán năm.
* **Liên hệ.** Gặp lỗi liên quan dữ liệu hoặc cần đối chiếu `data_report.json`, gửi nguyên thông báo lỗi và đường dẫn file cho Member 1.