# Steam Game Recommender

Đồ án xây dựng hệ thống đề xuất game với các tầng Data, Model, Service và Evaluation. Mọi tầng dùng chung DTO, giao diện và ngoại lệ trong `src/contracts/`.

## Mã nguồn và tài liệu

| Phần | Vị trí |
| --- | --- |
| Data | [`src/data/`](src/data/) |
| Model | [`src/models/`](src/models/), [hướng dẫn bàn giao](docs/MODEL_HANDOFF.md) |
| Service | [`src/service/`](src/service/) |
| Evaluation | [`src/evaluation/`](src/evaluation/), [`benchmarks/`](benchmarks/), [giao thức benchmark](docs/evaluation.md) |
| Hợp đồng | [`src/contracts/`](src/contracts/), [Contracts v1.1 cho model đã train](docs/Contracts_v1_1_Pretrained_2026-11-06.html) |

## Môi trường

Chạy các lệnh từ thư mục gốc của repo. Trên Windows, dùng CPython 3.10 trở lên.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
```

Trên Linux hoặc macOS, dùng trình thông dịch `.venv/bin/python` tương ứng. Để chạy Data với Parquet, cài thêm nhóm phụ thuộc `data` của [`pyproject.toml`](pyproject.toml). Cấu hình đóng gói hiện tại bao gồm Model và contracts. Các lệnh benchmark bên dưới chạy từ mã nguồn trong repo.

## Dùng BPR và TextBPR đã train

Hai bundle CPU được phát hành riêng với mã nguồn. Tải [BPR và TextBPR từ release `pretrained-legacy-v1`](https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/releases/tag/pretrained-legacy-v1) bằng tài khoản có quyền đọc repo. Script cài đặt kiểm tra checksum của gói tải về và trọng số. Xem [hướng dẫn đầy đủ](docs/PRETRAINED_MODELS.md) và [metadata lần train gốc](docs/TRAINED_RUN_REFERENCE.json).

```powershell
.\.venv\Scripts\python.exe -B scripts/install_model.py --archive "$env:USERPROFILE\Downloads\bpr-cpu-legacy-v1.zip" --model bpr
.\.venv\Scripts\python.exe -B scripts/install_model.py --archive "$env:USERPROFILE\Downloads\text-bpr-cpu-legacy-v1.zip" --model text-bpr
.\.venv\Scripts\python.exe -B scripts/recommend.py --model bpr --user "!" --k 10
```

`!` là một ID trong dữ liệu huấn luyện. Người dùng chưa có hồ sơ nhận đề xuất theo độ phổ biến mặc định. Dùng `--no-fallback` khi cần kiểm tra hành vi báo lỗi. Khi tích hợp ứng dụng, nạp `PretrainedRecommender` một lần lúc khởi động và chuyển lịch sử của ứng dụng qua `extra_seen`. Dùng ID ứng dụng riêng và giữ ánh xạ của bundle cho người dùng nguồn.

## Chạy benchmark

Benchmark nạp BPR và TextBPR đã train, fit `cornac.models.MostPop` trên các cặp tích cực của tập train gốc và chấm cả ba mô hình trên cùng người dùng, lịch sử và tập ứng viên. Các độ đo là Recall@5, Recall@10, NDCG@5 và NDCG@10.

Đặt hai bundle vào `artifacts/models/bpr/cpu/` và `artifacts/models/text_bpr/cpu/`. Đặt bốn file prepared gốc (`dataset.npz`, `mappings.json`, `report.json`, `catalog.json`) vào `artifacts/prepared_original/`. Runner đối chiếu fingerprint và thứ tự ID của dữ liệu với bundle. Có thể đổi các đường dẫn trong [`configs/benchmark.json`](configs/benchmark.json).

Chạy thử 100 người dùng của tập test.

```powershell
.\.venv\Scripts\python.exe -B -m benchmarks.run_pretrained --mode benchmark --config configs/benchmark.json --split test --max-users 100
```

Để chấm toàn bộ nhóm người dùng đủ điều kiện trong tập test, đổi `--max-users 100` thành `--max-users 0`. Terminal hiện phần trăm và thời gian theo từng mô hình. Báo cáo trong `results/benchmark/` gồm `summary.csv`, `per_user.csv`, `skipped_users.csv` và `run_config.json`. Chi tiết cách đọc kết quả nằm trong [tài liệu Evaluation](docs/evaluation.md).

## Kiểm tra Model và Evaluation

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_team_models.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_numerical_core.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/evaluation -v
```

Kết quả benchmark đầy đủ được lưu riêng cùng cấu hình lần chạy để dùng cho báo cáo. Mã nguồn trên GitHub tham chiếu tới dữ liệu gốc và release model khi tái lập thực nghiệm.
