# Evaluation và benchmark

## 1. Hai đường chạy

| Mode | Việc thực hiện | Metric xếp hạng |
| --- | --- | --- |
| `recommend` | Nạp một bundle, chạy đề xuất, kiểm tra đầu ra và nhập reference gốc | Không tính lại Recall/NDCG |
| `benchmark` | Nạp model nhóm, fit Cornac MostPop từ train gốc và chấm với ground truth gốc | Tính lại Recall/NDCG trên cùng protocol |

Mode mặc định là `recommend` để giữ cách gọi đang dùng. Hai mode dùng cấu hình và báo cáo khác nhau. Không đổi tên original validation thành kết quả benchmark vừa chạy.

Phần Evaluation tính metric và kiểm tra ranking. Adapter nối artifact/model với giao diện đánh giá. Runner điều phối và ghi báo cáo. Không định nghĩa bản sao DTO, không sửa Cornac và không train lại BPR/TextBPR trong đường benchmark pretrained.

## 2. Chuẩn bị source và môi trường

Cần repo đã có toàn bộ `src/models/`, `src/contracts/` thống nhất và các module Evaluation hiện hữu như `metrics.py`, `result.py`. Package Model còn import public modules khác; chỉ chép riêng loader sẽ không đủ.

Các lệnh PowerShell sau chạy từ gốc repo bằng môi trường CPython `.venv` của dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests/evaluation -p "test_*.py" -v
```

Không dùng Python MSYS hoặc `pip` của môi trường khác. Không cần cài editable khi repo chưa có `pyproject.toml` tương ứng. Cornac phải được cài thật để thực hiện benchmark ngoài; production runner không âm thầm bỏ baseline khi thư viện thiếu.

Các bundle compact chạy bằng CPU/NumPy. Không cần Qwen encoder, GPU, raw reviews hay chạy lại feature builder để inference. TextBPR compact đã lưu effective item factors từ lần train trước. Cornac MostPop dùng thư viện Cornac và các dependency của thư viện.

## 3. Artifact gốc bắt buộc cho benchmark

`prepared_dir` phải chứa đúng bốn file từ lần train đã tạo ra weights:

```text
prepared_dir/
    dataset.npz
    mappings.json
    report.json
    catalog.json
```

`dataset.npz` chứa các mảng chỉ số nguyên `(N, 2)`:

| Mảng | Nội dung |
| --- | --- |
| `train` | Các cặp train tích cực |
| `train_seen` | Toàn bộ cặp train đã quan sát, gồm tích cực và tiêu cực |
| `validation` | Các cặp validation tích cực làm ground truth |
| `validation_seen` | Hợp của lịch sử train và mọi observation validation |
| `test` | Các cặp test tích cực làm ground truth |

`mappings.json` giữ danh sách `user_ids`/`item_ids` theo đúng thứ tự hàng weights. `report.json` giữ fingerprint gốc và thông tin preprocessing. `catalog.json` là metadata, không được dùng tổng số review để thay nhãn tương tác.

Fingerprint được kiểm tra bằng thuật toán của prepared format gốc: JSON mapping với `sort_keys=True`, rồi tên, shape và bytes canonical `<i8` của từng mảng theo thứ tự tên. Runner không tự thay fingerprint để hợp thức hóa dữ liệu mới.

Loader kiểm tra shape, bounds, cặp trùng, train positives thuộc train history, quan hệ lịch sử và target không lẫn history. Khi nạp saved model, fingerprint và toàn bộ thứ tự raw IDs phải khớp prepared data. Checksum weights do public loader Model kiểm tra.

**Nếu chưa nhận được bốn artifact prepared gốc thì chưa thể chạy benchmark thực với weights hiện có.** Không tạo ground truth ngẫu nhiên, dùng fixture test hoặc chia lại Data mới rồi gắn kết quả vào pretrained bundle. Source trên GitHub và ZIP Source code không tự chứa prepared/weights.

Data hiện tại có thể phục vụ một lần train mới. Tỷ lệ 70/15/15, seed hoặc dimensions giống nhau vẫn chưa đủ để tái hiện lần train legacy. Bộ `mapping.json` và review CSV/JSONL của Data mới cũng khác bộ prepared `mappings.json`/`dataset.npz` nói trên.

## 4. Protocol benchmark cố định

| Thành phần | Validation | Test |
| --- | --- | --- |
| Ground truth | `validation` positives | `test` positives |
| Lịch sử bị loại | `train_seen` | `validation_seen` |
| Tập item trước lọc | Toàn bộ item trong mapping train gốc | Cùng mapping train gốc |
| User | User có target thuộc split, theo cohort đã chọn | Tương tự trên test |

Giữ cả negative trong lịch sử: user đã đánh giá tiêu cực một game thì game đó cũng đã được quan sát. Ground truth của split đang chấm không được đưa vào lịch sử loại trừ. Không dùng negative sampling cho tập ứng viên đánh giá.

User không có positive train, ground truth rỗng hoặc không có ứng viên hợp lệ được xử lý và ghi nhận theo protocol Evaluation. Các model phải có cùng cohort/candidate signature; không cho baseline được chấm thêm user mà model nhóm bỏ qua.

`max_users=0` nghĩa là tất cả user eligible. Giá trị dương lấy mẫu không hoàn lại một lần bằng `benchmark.seed` rồi giữ cùng danh sách cho mọi model. Đây chỉ là lấy mẫu user để đánh giá; train và catalog không bị lấy mẫu lại.

Ranking dùng score giảm dần, hòa điểm theo internal item index tăng dần trong mapping gốc. `Recall@K` bằng số relevant item xuất hiện trong top-K chia tổng relevant item. `NDCG@K` dùng discount `1/log2(rank+1)` và ideal ranking của cùng ground truth. Kết quả tổng hợp là trung bình theo user được chấm.

Nếu ít ứng viên hơn K, trả đủ số ứng viên thực tế; không lặp item hoặc bù item đã xem. Metric không biến predicted score thành sao hoặc xác suất. Lần chạy không có user đủ điều kiện phải thể hiện trạng thái/lý do, không bịa metric bằng 0.

## 5. Model nhóm và Cornac

`SavedModelRanker` dùng public `PretrainedRecommender` của nhóm, giữ mapping/provenance đã xác minh và không bật cold-start fallback. BPR/TextBPR được nạp từ compact bundle; không gọi `fit`, không chọn epoch mới và không tạo lại features.

`CornacMostPopAdapter` là lớp của dự án bao một `cornac.models.MostPop` nguyên bản. Cornac được fit **chỉ bằng `train` positives**. MostPop đếm observation nên không đưa negative như một observation để tăng độ phổ biến. Lịch sử loại ứng viên vẫn lấy từ `train_seen` hoặc `validation_seen` của prepared gốc.

Không sửa file trong package Cornac, không kế thừa rồi thay thuật toán MostPop, không dùng bộ đếm tự viết thay kết quả Cornac. Adapter chỉ chuẩn hóa dữ liệu, mapping và giao diện ranking. [API MostPop](https://cornac.readthedocs.io/en/stable/_modules/cornac/models/most_pop/recom_most_pop.html), [API Dataset](https://cornac.readthedocs.io/en/stable/api_ref/data.html).

Một run benchmark cần ít nhất một saved model nhóm và Cornac MostPop. BPR cùng TextBPR có thể được so sánh trong cùng run nếu đã có cả hai bundle khớp prepared data. Cornac BPR không phải điều kiện của đường chạy tối thiểu này.

## 6. Cấu hình, CLI và API benchmark

Sửa `configs/benchmark.json` để trỏ tới artifact thật. Các trường gồm `run_name`, `prepared_dir`, `models`, `protocol`, `benchmark`, `output_dir`.

- `models`: danh sách `{kind, name, model_dir}` với saved `bpr`/`text_bpr`; Cornac dùng `{kind: "cornac_most_pop", name}`.
- `protocol.ks`: các K dương, ví dụ `[5, 10]`.
- `benchmark`: `split` là `validation`/`test`, `max_users` và `seed`.

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --mode benchmark --config configs/benchmark.json
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --mode benchmark --config configs/benchmark.json --split validation --max-users 2000
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --mode benchmark --config configs/benchmark.json --split test --max-users 2000 --output-dir results/benchmark_test
```

```python
from benchmarks.run_pretrained import run_benchmark

paths = run_benchmark(
    "configs/benchmark.json",
    split="validation",
    max_users=2000,
    output_dir="results/benchmark_validation",
)
```

Overrides thay các giá trị tương ứng của config. Không dùng `--user` ở benchmark mode. Tất cả đường dẫn tương đối lấy gốc là thư mục repo chứa `benchmarks/`, không lấy cwd hoặc thư mục của config làm gốc.

## 7. Đầu ra benchmark và cách diễn giải

| File | Nội dung |
| --- | --- |
| `summary.csv` | Metric mới theo model/split, số user và thời gian |
| `per_user.csv` | Ranking, ground truth, candidate count và metric từng user |
| `skipped_users.csv` | User bị bỏ qua cùng lý do |
| `run_config.json` | Protocol, cohort, fingerprint, checksum artifact, phiên bản và thông tin model |

Số eligible_user_count trong metadata là số user có positive target trong prepared trước các quy tắc loại của Evaluator. evaluated_user_count/skipped_user_count mô tả cohort đã chọn; summary.csv ghi số được chấm của từng model.

Mỗi run ghi vào thư mục mới để giữ kết quả cũ. Metadata có `mode="benchmark"` và `recomputed_metrics=true`; các số metric đến từ ground truth được đọc và chấm trong chính run đó. Không nhập original validation vào bảng summary mới.

Thời gian load saved model khác thời gian fit Cornac và thời gian evaluation. Không gộp chúng thành một latency request hoặc kết luận model nhanh hơn trên mọi máy. Full-catalog evaluation phụ thuộc RAM, CPU, thư viện số và số user được chấm. Khi trình bày, ghi rõ split, K, sample size, seed, history policy và phiên bản dependency.

Không hứa metric mới sẽ trùng số reference cũ chỉ vì cùng K. Trước khi so sánh còn phải xác nhận đúng artifact, cohort và protocol. User nguồn có thể là tên hiển thị chưa được xác thực; tên trùng có thể làm giảm tính đúng đắn của personalized metrics. Random split không tự chứng minh dự đoán tương lai theo thời gian.

## 8. Mode đề xuất vẫn giữ nguyên

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --config configs/evaluation_pretrained.json
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --mode recommend --config configs/evaluation_pretrained.json --user "!"
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --mode recommend --config configs/evaluation_pretrained.json --user "!" --user "ID_NGUON_KHAC" --output-dir results/pretrained
```

Mode `recommend` dùng bundle và `TRAINED_RUN_REFERENCE.json` tương ứng, không cần prepared data gốc. `top_k`, `extra_seen`, `exclude_played` và fallback lấy từ config inference. `--user` lặp lại thay danh sách raw user IDs trong config. User mới của app không tự trở thành profile nguồn chỉ vì có tên giống nhau.

Mặc định từ chối user ngoài mapping. Chỉ khi chủ động bật fallback mới dùng Popularity và gắn `is_fallback=true`. `predicted_score` là điểm xếp hạng; giữ score âm/lớn hơn 1. Catalog thiếu price không có nghĩa game miễn phí; categories không tự biến thành genres.

Mode này tạo `recommendations.csv`, `inference_checks.json`, `original_validation.csv`, `run_config.json`. Original validation là số đã ghi từ lần train trước. Inference checks kiểm tra vận hành và hợp đồng, không phải Recall/NDCG. Mỗi user được gọi recommend hai lần để kiểm tra lặp lại, nên thời gian vòng lặp không phải latency một request.

Xuất top-K cho toàn bộ user là một tác vụ inference riêng có thể tốn nhiều thời gian/đĩa. Nó không cần để benchmark một cohort đã chọn và không tự tạo ground truth hoặc metric chất lượng.

## 9. Kiểm thử

- `conftest.py` giữ các fixture v1 và thêm prepared/bundle compact tổng hợp nhỏ.
- `test_evaluator.py` kiểm tra metrics/cohort/ranking qua Evaluation.
- `test_cornac_adapter.py` kiểm tra adapter và gọi Cornac thật khi dependency có sẵn.
- `test_model_integration.py` giữ ba ca tích hợp native model đã chuyển khỏi file bị đặt nhầm tên.
- `test_benchmark_integration.py` chạy saved loaders cùng Cornac trên cùng fixture, đối chiếu metric tính tay, lọc negative/history validation, kiểm tra provenance sai và ghi báo cáo mới.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/evaluation -p "test_*.py" -v
```

Các ca benchmark/pretrained nạp bundle nhỏ, không train, không chạy encoder và không tải weights lớn. Ba ca native v1 trong `test_model_integration.py` vẫn fit trên fixture tổng hợp nhỏ để giữ kiểm tra API training cũ; ca TextBPR dùng hàm encode test nhỏ. Những ca dùng Model/Cornac thật phải báo skip rõ nếu thiếu điều kiện; không tính skip thành integration pass. Bộ test không thay cho việc nhận artifact gốc và chạy benchmark thực. Không đưa các metric của fixture tổng hợp vào kết quả đồ án như số đo trên Steam/Kaggle.
