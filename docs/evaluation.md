# Evaluation Layer và Benchmark — bộ mã tự học

Bộ mã này gồm đúng 14 file để bạn đọc và tự gõ lại vào đồ án. Đây là một bản triển khai độc lập của phần Evaluation và Benchmark, không import `contracts`, Data Layer, Model Layer hoặc Service Layer của nhóm. Các file Python không có comment hoặc docstring; phần giải thích được đặt trong tài liệu này.

## Chạy lần đầu

Giải nén toàn bộ thư mục. Mở Terminal tại thư mục có `benchmarks`, `configs`, `src`, `tests`, `docs`. Không chạy từ bên trong `src/evaluation`.

Bản demo và phần lớn kiểm thử chỉ cần Python 3.10 trở lên, không cần cài NumPy, pytest hoặc Cornac:

```powershell
python -B -m unittest discover -s tests/evaluation -v
python -B -m benchmarks.run_benchmark --config configs/evaluation.json
```

Kiểm thử kết nối Cornac thật sẽ được đánh dấu skipped nếu môi trường chưa có Cornac. Những kiểm thử adapter còn lại vẫn chạy bằng backend giả được kiểm soát. Backend giả chỉ dùng trong kiểm thử, không được dùng thay Cornac khi chạy benchmark.

Sau khi chạy demo, chương trình in đường dẫn đến ba file kết quả. Mỗi lần chạy có thư mục riêng dưới `results/`; thư mục này chưa tồn tại trong gói tải về và được tạo khi chạy.

## Chạy với Cornac thật

Bộ mã đã được kiểm tra với CPython 3.12 và Cornac 3.1.0 trên Windows. Bạn cần một Python có thể cài các gói tương thích; bản Python MSYS và Python chính thức có thể có khả năng cài wheel khác nhau.

Nếu máy có Python 3.12 và Python Launcher, tạo môi trường riêng tại thư mục bộ mã:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install cornac==3.1.0
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/evaluation -v
.\.venv\Scripts\python.exe -B -m benchmarks.run_benchmark --config configs/evaluation.json --with-cornac
```

Nếu không có lệnh `py`, thay `py -3.12` bằng đường dẫn đến Python 3.12 đã cài. Demo tiêu chuẩn vẫn chạy được mà không cần thực hiện phần cài Cornac.

`--with-cornac` chạy `DemoPopularity` và `CornacMostPop` thật. Nếu Cornac thiếu hoặc lỗi import, lệnh báo lỗi và dừng; không thay thế bằng kết quả mô phỏng. Cài thư viện vào đúng môi trường Python dùng để chạy lệnh.

Hai model mặc định đều đếm số tương tác tích cực trong train. Vì vậy, điểm và thứ hạng bằng nhau là kết quả dự kiến, giúp kiểm tra adapter và cách chấm có nhất quán. Đây chưa phải thực nghiệm chứng minh model của nhóm tốt hơn baseline. Khi model của nhóm hoàn thiện, truyền nó vào `run_benchmark` để so sánh thật.

## Danh sách 14 file và thứ tự đọc

| Thứ tự | File | Trách nhiệm |
|---:|---|---|
| 1 | `src/evaluation/metrics.py` | Lớp cha metric, Recall@K, NDCG@K, kiểm tra đầu vào metric. |
| 2 | `src/evaluation/protocol.py` | Interaction, cấu hình quy tắc đánh giá, giao diện model và kiểm tra interaction. |
| 3 | `src/evaluation/result.py` | Kết quả từng user, kết quả tổng hợp và xuất CSV/JSON. |
| 4 | `tests/evaluation/conftest.py` | Dữ liệu nhỏ và model cố định để kiểm tra vòng chấm. |
| 5 | `tests/evaluation/test_metrics.py` | Kiểm tra metric, protocol và lưu kết quả bằng unittest. |
| 6 | `src/evaluation/evaluator.py` | Chuẩn bị ứng viên, lấy điểm model, xếp hạng, chấm từng user và tính trung bình. |
| 7 | `tests/evaluation/test_evaluator.py` | Kiểm tra Evaluator, CLI, dữ liệu CSV và bộ chạy benchmark. |
| 8 | `src/evaluation/__init__.py` | Công khai các thành phần cần import từ package evaluation. |
| 9 | `configs/evaluation.json` | Cấu hình thực nghiệm. |
| 10 | `benchmarks/cornac_adapter.py` | Kết nối với MostPop của Cornac và chuyển đổi ánh xạ ID. |
| 11 | `tests/evaluation/test_cornac_adapter.py` | Kiểm tra mapping, dữ liệu, lỗi và kết nối Cornac thật khi có thư viện. |
| 12 | `benchmarks/__init__.py` | Công khai CornacMostPopAdapter; không import bộ chạy. |
| 13 | `benchmarks/run_benchmark.py` | Model demo, đọc cấu hình/CSV, huấn luyện, gọi Evaluator và lưu kết quả. |
| 14 | `docs/evaluation.md` | Hướng dẫn này; nên đọc phần chạy và giao ước trước khi tự gõ code. |

`src` là namespace package của Python trong bộ mã độc lập này, nên không cần thêm `src/__init__.py`. Khi đưa code vào repo nhóm, dùng `src/__init__.py` chung nếu repo đã có.

## Giao ước dữ liệu

Một interaction là:

```python
Interaction(user_id=0, item_id=3, rating=1.0, timestamp=100)
```

- `user_id`, `item_id`: số nguyên không âm, là ID nội bộ. Với `num_users=5`, user hợp lệ từ 0 đến 4; với `num_items=8`, game hợp lệ từ 0 đến 7.
- `rating`: số thực hữu hạn. Với đánh giá nhị phân, 1 là tích cực và 0 là tiêu cực. Giá trị thiếu không được biến thành review tiêu cực.
- `timestamp`: số nguyên không âm, mặc định 0. Có thể dùng Unix timestamp do Data Layer cung cấp; cả ba split phải dùng cùng đơn vị nếu cần đối chiếu thời gian.
- Trong một split, một cặp user–game chỉ xuất hiện một lần. Cặp đó cũng không được xuất hiện đồng thời trong train, validation hoặc test. Nếu dữ liệu có nhiều lần cập nhật review, Data Layer cần chốt cách giữ bản ghi trước khi đưa vào benchmark.

Chuỗi như `U01`, ID Steam hoặc ID thưa như 123456 không tự được ánh xạ. Data Layer hoặc adapter tích hợp cần ánh xạ chúng sang một không gian chỉ số chung, rồi truyền kích thước catalog tương ứng. `num_items` mô tả toàn bộ catalog ứng viên đã chốt, bao gồm game có thể chưa có positive train; nó không phải số game còn lại sau khi lọc lịch sử.

`Interaction` là dataclass bất biến, giúp tránh sửa nhầm bản ghi khi trao đổi giữa các phần.

## Quy tắc đánh giá

Cấu hình mặc định:

```json
{
  "ks": [3, 5],
  "positive_threshold": 1.0,
  "exclude_seen": true,
  "empty_ground_truth_policy": "skip",
  "cold_start_policy": "skip"
}
```

- K phải là số nguyên dương, không trùng; protocol sắp xếp K tăng dần.
- Relevant item là game có `rating >= positive_threshold` trong split đang chấm.
- Với dữ liệu 5 sao, có thể đặt `positive_threshold` thành 4.0 để chọn review từ 4 sao trở lên. Cần chốt trước khi đánh giá cuối cùng.
- Seen item gồm toàn bộ tương tác train của user, bao gồm review tiêu cực. Nếu `exclude_seen=true`, loại các game đó khỏi ứng viên.
- Ground truth test được giữ riêng, không đưa vào lịch sử cần loại. Các game được đánh giá tiêu cực trong test vẫn có thể là ứng viên, nhưng không được tính là relevant.
- `empty_ground_truth_policy="skip"` bỏ qua user không có relevant item; `"zero"` đưa user đó vào trung bình với điểm 0.
- Cold start là user không có bất kỳ tương tác train nào. `"skip"` bỏ qua, `"error"` dừng và báo lỗi. Bộ mã chưa triển khai đánh giá riêng cho recommender cold-start có thông tin phụ.
- Nếu user vừa cold start vừa không có relevant item, lý do `cold_start` được tính trước; không đếm user hai lần.
- Mọi metric và mọi model trong một lần chạy dùng cùng nhóm user đủ điều kiện, cùng ứng viên và cùng quy tắc.
- Khi điểm model bằng nhau, ưu tiên item ID nhỏ hơn. Không dùng thứ tự của set hoặc mapping để phá hòa.
- Không lấy mẫu negative ngẫu nhiên; chấm trên toàn bộ catalog đã chốt sau khi lọc seen.
- Game chưa có positive train vẫn có thể nằm trong catalog ứng viên. Hai baseline mặc định cho game đó điểm 0; model cá nhân hóa khác cần định nghĩa cách cho điểm phù hợp.
- Nếu không còn user đủ điều kiện, Evaluator báo lỗi thay vì xuất một điểm trung bình gây hiểu nhầm.

Evaluator còn hỗ trợ `additional_seen_interactions` khi bên gọi muốn dùng lịch sử đã biết như validation để lọc. Các interaction này phải tách khỏi train và split đang chấm. Runner mặc định chỉ dùng train làm lịch sử; nó không đưa validation vào lịch sử hay train một cách ngầm định.

## Công thức metric

Với một user, gọi R là tập game tích cực trong split đánh giá và L là danh sách đề xuất đã xếp hạng.

`Recall@K = số game thuộc cả R và L[:K] / |R|`.

Recall chia cho tổng số game cần tìm, không chia cho K. Với `R={1,3,5}`, `L=[3,8,1,9,5]`, K=3 thì Recall bằng 2/3.

NDCG dùng relevance nhị phân. Game đúng ở hạng r đóng góp `1/log2(r+1)` vào DCG; game không thuộc R đóng góp 0. IDCG là điểm khi đặt `min(K, |R|)` game đúng ở những vị trí đầu. `NDCG@K = DCG@K / IDCG@K`.

Với ví dụ trên, DCG@3=1.5, IDCG@3 xấp xỉ 2.13093 và NDCG@3 xấp xỉ 0.703918. Nếu danh sách thực tế ngắn hơn K, IDCG vẫn dựa trên K và số đáp án để không tăng điểm giả.

`RankingMetric.compute` kiểm tra ID và lấy top-K trước khi gọi `_compute_score` của lớp con. Kết quả một user là số thực trong [0,1]. Ground truth rỗng trả 0 ở cấp metric; protocol của Evaluator quyết định user đó được bỏ qua hay tính 0 trong trung bình.

Metric kiểm tra toàn bộ ID trong hai đầu vào, nhưng chỉ yêu cầu không có ID trùng trong đoạn top-K đang chấm. Thứ tự đề xuất không bị metric thay đổi. `num_total_items` là tham số tùy chọn để kiểm tra phạm vi ID; hai công thức không phụ thuộc trực tiếp vào kích thước catalog.

Điểm cuối cùng là macro average: tính điểm riêng từng user đủ điều kiện, rồi lấy trung bình đều giữa các user.

## Giao diện model để tích hợp

Evaluator nhận model đã fit, có các thuộc tính:

```text
name: str
is_fitted: bool
num_users: int
num_items: int
```

Và phương thức:

```python
score_items(user_id, item_ids)
```

Đầu ra phải là dãy điểm số thực hữu hạn, mỗi điểm tương ứng đúng vị trí của một ID trong `item_ids`. Không trả dict hoặc set. NumPy array một chiều cũng có thể làm đầu ra nếu các phần tử là số thực hợp lệ; riêng đầu vào `item_ids` do Evaluator truyền là tuple ID.

Evaluator tự tạo tập ứng viên và tự xếp hạng. Model cần cho điểm đúng các ứng viên được truyền vào; không trả danh sách đã hydrate thành tiêu đề game hoặc đối tượng hiển thị.

Runner còn yêu cầu:

```python
fit(train_interactions, num_users, num_items)
```

Có thể truyền các model của nhóm qua tham số `models`:

```python
paths = run_benchmark(
    "configs/evaluation.json",
    models=[your_model],
    include_cornac=True,
)
```

Tên `your_model` ở đây minh họa đối tượng do nhóm cung cấp. Model cần tuân theo giao diện trên hoặc được bọc bằng adapter. Khi ghép repo, đối chiếu tên lớp/phương thức, ID, ngoại lệ và kiểu báo cáo; các lớp trong bộ mã này chưa tự kế thừa giao diện `BaseMetric` hoặc `BaseRecommender` của nhóm.

Runner luôn gọi fit bằng train. Nếu gọi Evaluator trực tiếp, bên gọi chịu trách nhiệm bảo đảm model chỉ học từ dữ liệu được phép; Evaluator không thể kiểm tra nội dung mà model từng học. Với model thêm vào, lưu cả cấu hình siêu tham số của model cùng báo cáo thực nghiệm; metadata hiện tại chỉ tự ghi tên, lớp, threshold và seed nếu model có các thuộc tính tương ứng.

## Dữ liệu demo và ý nghĩa kết quả

Dữ liệu dựng sẵn có 5 user, 8 game, 12 interaction train, 5 validation và 9 test. User 0–3 có train; user 4 là cold start và bị bỏ qua theo mặc định. Train có cả review tích cực và tiêu cực. Bộ dữ liệu được viết trong `_demo_data` của bộ chạy, nên không cần file dữ liệu thứ 15.

Với cấu hình mặc định và split test, cả DemoPopularity và CornacMostPop thật cho kết quả dự kiến:

| Metric | Điểm |
|---|---:|
| Recall@3 | 0.75 |
| Recall@5 | 1.0 |
| NDCG@3 | 0.4404648768 |
| NDCG@5 | 0.5481340163 |

Số user được chấm là 4; số cold-start bị bỏ qua là 1. Những con số này chỉ dành cho dữ liệu demo có sẵn. Chương trình tự tính lại khi chạy; không có bảng điểm được hard-code vào Evaluator.

## Chạy validation và test

```powershell
python -B -m benchmarks.run_benchmark --config configs/evaluation.json --split validation
python -B -m benchmarks.run_benchmark --config configs/evaluation.json --split test --with-cornac
```

Dùng validation để thử/chọn cấu hình. Sau khi chốt cấu hình, dùng test cho đánh giá cuối cùng. Bộ mã không tự tìm siêu tham số và không tự chia dữ liệu. Nó cũng không kiểm tra rằng timestamp của mọi split đã tuân theo thứ tự thời gian mong muốn; đây là quy tắc chuẩn bị split mà nhóm phải chốt và bảo đảm.

Model chỉ fit trên train ở cả hai lệnh. Bản này không tự refit trên train+validation sau khi chọn cấu hình.

## Dùng CSV đã chia sẵn

CSV dùng đúng các cột `user_id,item_id,rating` và có thể thêm `timestamp`:

```csv
user_id,item_id,rating,timestamp
0,0,1,100
0,1,0,101
1,0,1,102
```

Tạo các file train/validation/test riêng ngoài 14 file nguồn. Chỉnh phần `data` trong cấu hình, ví dụ:

```json
{
  "source": "csv",
  "train_csv": "data/train.csv",
  "validation_csv": "data/validation.csv",
  "test_csv": "data/test.csv",
  "num_users": 5,
  "num_items": 8
}
```

Giữ nguyên các phần cấu hình khác. `validation_csv` có thể là null nếu chỉ chạy test; train và test vẫn cần đường dẫn hợp lệ. Với dữ liệu demo, các trường đường dẫn và kích thước phải để null.

Các đường dẫn tương đối trong cấu hình được tính từ `config_file.parent.parent`, tương ứng thư mục gốc bộ mã nếu file cấu hình nằm trong `configs/`. Nên giữ cấu hình ở thư mục đó hoặc dùng đường dẫn tuyệt đối khi chuyển vị trí. Đường dẫn `--output-dir` trên CLI được tính từ thư mục Terminal hiện tại.

Reader từ chối cột lạ, tên cột trùng, dòng sai số trường, ID không phải số nguyên, NaN/Infinity, ID vượt kích thước và cặp user–game bị trùng. CSV chưa ánh xạ ID phải được xử lý trước.

## Kết quả xuất ra

- `summary.csv`: một dòng cho mỗi model, chứa điểm trung bình, số user chấm/bỏ qua, lý do bỏ qua, thời gian fit và thời gian đánh giá.
- `per_user.csv`: điểm từng user, số ứng viên, danh sách đề xuất và relevant item. Các danh sách được mã hóa JSON trong ô CSV.
- `run_config.json`: cấu hình thực tế sau các tùy chọn CLI, seed, phiên bản Python/Cornac, split, tên model, số interaction và SHA-256 của dữ liệu từng split.

Mỗi lần chạy tạo thư mục có tên riêng. Hàm lưu kết quả từ chối ghi đè file kết quả đã có. Thời gian chạy và tên thư mục có thể thay đổi; với cùng dữ liệu/cấu hình, hai baseline mặc định cho điểm và thứ hạng xác định.

## Phạm vi kiểm thử

Các bài unittest kiểm tra công thức bằng kết quả tính tay, K/ID không hợp lệ, danh sách rỗng/ngắn, không sửa dữ liệu đầu vào, protocol, xuất báo cáo, lọc lịch sử gồm review tiêu cực, giữ đáp án test trong ứng viên, cold start, macro average, tie-break, điểm NaN/Infinity, đầu ra model sai thứ tự/định dạng, trùng train/test, CLI, CSV và mapping Cornac.

`conftest.py` ở đây là module dữ liệu hỗ trợ unittest, không yêu cầu pytest. Chạy đúng lệnh discover ở đầu tài liệu để Python tìm thấy module hỗ trợ này.

## Nguồn API Cornac

- Dataset và ánh xạ ID: https://github.com/PreferredAI/cornac/blob/master/cornac/data/dataset.py
- MostPop: https://github.com/PreferredAI/cornac/blob/master/cornac/models/most_pop/recom_most_pop.py
- Metric tham chiếu: https://github.com/PreferredAI/cornac/blob/master/cornac/metrics/ranking.py

MostPop được train bằng các interaction đã lọc tích cực và chuyển rating thành 1.0. Các review tiêu cực vẫn được Evaluator dùng làm lịch sử cần loại. Việc dùng user nội bộ 0 của Cornac cho user không xuất hiện trong positive train chỉ hợp lệ ở adapter này vì MostPop không cá nhân hóa; không áp dụng quy tắc đó cho các thuật toán khác.
