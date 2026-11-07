# Evaluation và benchmark với model đã train

Gói cập nhật này giữ đường đề xuất pretrained v1.1 và bổ sung benchmark tính lại Recall/NDCG từ prepared data gốc. Benchmark nạp BPR/TextBPR đã train, fit Cornac MostPop trên đúng train positives gốc, rồi chấm các model trên cùng user, ground truth, lịch sử và tập ứng viên.

Không train lại BPR/TextBPR, không chạy encoder và không sửa thư viện Cornac.

## Các file cập nhật

```text
README.md
requirements-evaluation.txt
benchmarks/
    run_pretrained.py
    legacy_adapter.py
    cornac_adapter.py
configs/
    benchmark.json
docs/
    evaluation.md
src/evaluation/
    evaluator.py
    protocol.py
tests/evaluation/
    conftest.py
    test_evaluator.py
    test_cornac_adapter.py
    test_model_integration.py
    test_benchmark_integration.py
```

Chép vào đúng đường dẫn trong repo đã tích hợp. Giữ những file đang có như `metrics.py`, `result.py`, `pretrained_report.py`, cấu hình inference và các bộ kiểm thử khác. Gói cập nhật không thay thế toàn bộ repo, source Model hay weights.

## Chuẩn bị

Chạy từ thư mục gốc dự án, dùng CPython trong `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
```

Cần toàn bộ `src/models/` và một bộ `src/contracts/` thống nhất. Không chỉ chép riêng `pretrained.py`: package Model còn dùng các public modules và `_array/`. Không bắt buộc `pip install -e .` khi repo chưa có `pyproject.toml` phù hợp.

Benchmark cần hai loại artifact:

1. Bundle model đã train, gồm `weights.npz` và `model_info.json`, ví dụ `artifacts/models/bpr/cpu` và `artifacts/models/text_bpr/cpu`.
2. Prepared data **đúng lần train đó**, gồm `dataset.npz`, `mappings.json`, `report.json`, `catalog.json` trong một thư mục.

Clone source hoặc tải ZIP Source code của GitHub không cung cấp các artifact này. Nếu còn thiếu prepared data gốc, cần nhận lại từ người train Model trước khi chạy benchmark thực; không có cách suy ra ground truth từ weights.

Prepared data mới do Data hiện tại tạo, hoặc bộ `mapping.json`/review CSV/JSONL mới xuất, không tự thay thế được bộ prepared gốc. Giống tỷ lệ 70/15/15 và seed vẫn chưa chứng minh cùng mapping và cùng split. Runner kiểm tra fingerprint và thứ tự raw IDs trước khi chấm.

Nếu dùng installer của Model để cài ZIP BPR đã tải:

```powershell
.\.venv\Scripts\python.exe -B scripts/install_model.py --archive "$env:USERPROFILE\Downloads\bpr-cpu-legacy-v1.zip" --model bpr
```

Cài TextBPR theo tài liệu/manifest tương ứng của Model. Mỗi bundle phải giữ đúng cặp metadata và weights, không đổi mapping để ép khớp Data mới.

## Benchmark tính metric mới

Sửa `configs/benchmark.json`: đặt `prepared_dir`, đường dẫn các bundle và những model muốn so sánh. Phải giữ ít nhất một model nhóm và Cornac MostPop; không được tắt Cornac rồi báo đã hoàn thành benchmark ngoài.

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --mode benchmark --config configs/benchmark.json
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --mode benchmark --config configs/benchmark.json --split validation --max-users 2000
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --mode benchmark --config configs/benchmark.json --split test --max-users 2000 --output-dir results/benchmark_test
```

`max_users=0` chấm toàn bộ user eligible; giá trị dương chọn một cohort cố định bằng seed. Việc lấy mẫu chỉ áp dụng user đánh giá, không giảm train của Cornac hoặc đổi weights. Mọi model dùng cùng cohort và thứ tự item gốc. Không truyền `--user` trong benchmark mode.

Mỗi run thành công tạo thư mục riêng với:

- `summary.csv`: Recall/NDCG mới, số user được chấm/bỏ qua và thời gian.
- `per_user.csv`: metric, ground truth, đề xuất và số ứng viên của từng user/model.
- `skipped_users.csv`: lý do bỏ qua, nếu có.
- `run_config.json`: cấu hình, phiên bản thư viện, checksum nguồn, provenance và `recomputed_metrics=true`.

Cornac là thư viện thực được gọi qua adapter của nhóm. Thiếu Cornac hoặc artifact sai phải gây lỗi rõ ràng, không dùng bộ đếm tự viết thay thư viện và không điền metric giả.

## Đề xuất và kiểm tra inference

Mode mặc định vẫn là `recommend`:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --config configs/evaluation_pretrained.json
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --mode recommend --config configs/evaluation_pretrained.json --user "!"
```

Mode này cần bundle và `docs/TRAINED_RUN_REFERENCE.json` tương ứng. Nó tạo `recommendations.csv`, `inference_checks.json`, `original_validation.csv`, `run_config.json`. `original_validation.csv` nhập lại chỉ số cũ; inference checks không phải accuracy. `!` là ví dụ raw ID nguồn của bundle đã bàn giao, không phải user mới của ứng dụng.

Mọi đường dẫn tương đối lấy gốc là thư mục repo chứa `benchmarks/`, kể cả config, prepared, bundle và output. Chọn đúng `.venv`, tránh dùng nhầm Python MSYS hoặc môi trường khác.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/evaluation -p "test_*.py" -v
```

Các test dùng fixture nhỏ, bao gồm loader Model thật và Cornac thật khi dependency có sẵn. Fixture tổng hợp không phải dữ liệu hoặc kết quả Kaggle. Ca cần Model/Cornac báo skip rõ nếu chưa tích hợp; skip không được tính là đã kiểm tra thành công đường chạy thực.

Ba test tích hợp native model cũ đã chuyển sang `test_model_integration.py`; `test_cornac_adapter.py` trở lại đúng trách nhiệm kiểm thử Cornac. Đọc [docs/evaluation.md](docs/evaluation.md) để xem protocol, API và cách diễn giải báo cáo.
