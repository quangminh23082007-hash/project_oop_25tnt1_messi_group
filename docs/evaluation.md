# Evaluation & Benchmark — Contracts v1

Phần này đánh giá thứ hạng game bằng Recall@K và NDCG@K.
Luồng thực nghiệm: fit trên train, chọn checkpoint bằng validation,
chấm split được chọn và xuất báo cáo.

Runner sử dụng API huấn luyện v1; việc nạp bundle pretrained thuộc luồng khác.

## Môi trường và chạy demo

Cần CPython 3.10 trở lên và NumPy.
unittest có sẵn trong Python.
Public Models cần được bàn giao đầy đủ cùng dependency của phần Model.
Cornac là dependency tùy chọn.

Các lệnh PowerShell sau chạy tại thư mục repo,
với môi trường .venv đã được tạo:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy
.\.venv\Scripts\python.exe -B -m benchmarks.run_benchmark --config configs/evaluation.json
```

Để chạy demo cùng Cornac thật:

```powershell
.\.venv\Scripts\python.exe -m pip install cornac
.\.venv\Scripts\python.exe -B -m benchmarks.run_benchmark --config configs/evaluation.json --with-cornac
```

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/evaluation -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/evaluation -p test_model_integration.py -v
```

| File | Phạm vi |
|---|---|
| conftest.py | Dataset, mapper, reviews và model điểm cố định cho tests |
| test_metrics.py | Công thức, protocol, shared DTO và báo cáo |
| test_evaluator.py | Dataset API, cohort, fingerprints, ranking và runner |
| test_cornac_adapter.py | Mapping, positive counts, trạng thái fit và Cornac |
| test_model_integration.py | Public Models đến Evaluation và báo cáo |

Khi chưa có src.models, ba bài integration được skipped.
Khi chưa cài Cornac, một bài Cornac thật được skipped.
Package có mặt nhưng import hỏng sẽ làm tests lỗi.

Fixtures có cả review tích cực và tiêu cực.
Encoder hai chiều là đầu vào kiểm thử và không tải model ngoài.
Tests xuất báo cáo vào TemporaryDirectory rồi tự dọn.
Điểm demo/fixtures chỉ phục vụ kiểm tra kỹ thuật.

## API và đầu vào

```python
from src.evaluation import EvaluationProtocol, Evaluator, summarize_result

evaluator = Evaluator(EvaluationProtocol(ks=(5, 10)))
result = evaluator.evaluate(model, train_dataset, test_dataset)
summary = summarize_result(result)
```

Model phải đã fit.
Callback validation nhận public candidate có thể scoring ở epoch hiện tại.
Data bàn giao DatasetSplits gồm train, validation, test và preparation report.

Mỗi Dataset cần:

- interactions chứa shared Interaction.
- Dimensions và số interaction nhất quán.
- Mappers có size phù hợp.
- mapping_fingerprint, fingerprint và review_fingerprint không rỗng.

IDs là chỉ số nội bộ bắt đầu từ 0.
Rating đúng 0 hoặc 1.
Timestamp là UTC milliseconds hoặc 0 nếu không biết.
Interaction không tồn tại là ý kiến chưa biết, không tự biến thành rating 0.

Mappings được fit từ toàn bộ train interactions rồi giữ cố định.
Evaluator kiểm tra range, duplicates, cặp giao nhau, dimensions và mapping.
Model.train_fingerprint phải khớp train.fingerprint.
Fingerprints được so sánh bằng giá trị thay vì identity object.

Data/Model chịu trách nhiệm tạo fingerprints đúng.
Data chịu trách nhiệm clean, chia split và xử lý raw IDs ngoài mapping.
Evaluator không kiểm tra chronology giữa các split.

## Protocol

1. Relevant là game có rating 1 trong split đang đánh giá.
2. User cần có ít nhất một positive train interaction.
3. Seen chứa mọi train interaction, cả positive và negative.
4. Candidates là toàn bộ game trong mapping train trừ seen.
5. Target interactions không được dùng làm seen history.
6. Score giảm dần, hòa điểm theo item ID tăng dần.
7. Điểm tổng hợp là trung bình trên user thực sự được chấm.

Recall chia cho toàn bộ số relevant.
IDCG của NDCG dùng min(K, số relevant), kể cả khi ranking ngắn hơn K.

Additional seen mặc định None.
Chỉ được truyền validation khi chấm test:

```python
result = evaluator.evaluate(
    model,
    train_dataset,
    test_dataset,
    additional_seen_dataset=validation_dataset,
)
```

Additional history không bổ sung positive train profile.

Skips dùng SkippedUser với reason:

- no_positive_train.
- empty_ground_truth.
- no_candidates.

Data exclusions được giữ riêng trong preparation report.

Evaluator có thể trả metrics={} khi không có user đủ điều kiện.
Exporter ghi no_eligible_users và để ô metric trống.
Runner dừng nếu split được chọn không có user được chấm.
Callback báo lỗi nếu thiếu selection metric, không thay bằng điểm 0.

## Cấu hình thật và TextBPR

configs/evaluation.json dùng synthetic demo, K = 3, 5.
configs/evaluation_real.json dùng public Popularity/BPR,
K = 5, 10 và mục tiêu NDCG@10.

Tham số BPR là cấu hình khởi đầu,
chưa được tối ưu bằng thực nghiệm.

Cấu hình real dùng source=provided,
cần truyền DatasetSplits qua Python:

```python
from benchmarks.run_benchmark import run_benchmark

paths = run_benchmark(
    "configs/evaluation_real.json",
    dataset_splits=splits,
)
```

splits là đầu vào được Data bàn giao.
CLI đơn lẻ chưa tạo được đầu vào provided.

Khi có provider v1 thật, có thể dùng source=provider
với module:function và kwargs.
Đường dẫn trong kwargs do provider quản lý.

Seed cấu hình là mặc định để tạo model.
Runner không chia lại splits hoặc đổi seed của model truyền sẵn.
Tên cấu hình là nhãn thực nghiệm.
Metadata giữ cả public_model_name.

Cornac mặc định tắt.
Sau khi cài, API có thể nhận include_cornac=True.

Runner nhận ItemFeatures hoặc (ItemFeatures, report) cho TextBPR.
Provider chỉ nhận train.

Builder chính thức src.models.text_features.build_item_features
dùng train.iter_reviews(), gồm positive và negative reviews,
rồi trả features/report.

Encoder thực hiện inference do bên gọi cung cấp
và cần revision cố định.
Feature rows phải đúng thứ tự train item mapper,
khớp ba fingerprints, hữu hạn và có norm 1 hoặc zero.

Cấu hình real chưa chọn TextBPR sẵn.
Để thêm, cần model kind=text_bpr và features theo đúng nhãn
qua provider hoặc item_features.

Metadata lưu report có thật, encoder, fingerprints, shape,
ordered item-ID checksum và matrix checksum.
Checksum ma trận dùng float64 little-endian
theo thứ tự C cùng header.

Không suy ra thiếu text chỉ từ việc vector bằng zero.

## Báo cáo

| File | Nội dung |
|---|---|
| summary.csv | Metric trung bình, status, số user, lý do skip và thời gian |
| per_user.csv | Metric từng user, candidate count, recommendation và relevant IDs |
| skipped_users.csv | ID từng user bị bỏ qua và reason |
| run_config.json | Cấu hình, phiên bản, Data report, fingerprints và model/features metadata |

Danh sách trong CSV được mã hóa JSON.
Recommendation giữ thứ tự; relevant IDs được sort.

Số user/lý do được suy ra từ records.
DTO frozen; dùng dataclasses.replace để bổ sung timing.

CSV của runner chứa split được chọn.
Validation summary nằm trong run_config.json.

Mỗi run có thư mục riêng.
Exporter từ chối ghi đè và kiểm tra dữ liệu trước khi ghi.

## Chẩn đoán

- src.contracts.evaluator không tồn tại:
  kiểm tra có dán nhầm Evaluation exports vào contracts/__init__.py không.
- Unexpected keyword train_interactions:
  runner hoặc tests vẫn dùng API cũ.
- Thiếu src.models:
  cần bàn giao package Model trước khi chạy model thật.
- source=provided thiếu dataset_splits:
  truyền DatasetSplits qua Python API.
- FrozenInstanceError:
  dùng dataclasses.replace thay vì gán field.
- Không tìm thấy NumPy:
  chọn interpreter .venv và cài thư viện vào đúng môi trường.