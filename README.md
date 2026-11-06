# Evaluation/Benchmark — bàn giao pretrained v1.1

Gói này bổ sung một đường chạy CPU để kiểm tra model đã train và xuất báo cáo.
Runner nạp bundle hiện có, gọi inference và nhập lại validation của lần train gốc.
Nó không train lại, không tạo features mới và không tính Recall/NDCG mới từ các đề xuất.

## Sáu file được bàn giao

```text
benchmarks/run_pretrained.py
benchmarks/pretrained_report.py
configs/evaluation_pretrained.json
tests/evaluation/test_pretrained.py
docs/evaluation.md
README.md
```

Đặt các file vào đúng đường dẫn trong dự án sau khi đọc phần chuẩn bị bên dưới.
Đây là một phần bàn giao, không phải toàn bộ source Model hoặc một bộ weights.
Các DTO chung tiếp tục lấy từ `src/contracts`; không tạo bản sao DTO trong runner.

## Chuẩn bị source và model

Cần nhánh đã tích hợp phần Model cung cấp:

- Toàn bộ `src/models/`, gồm `pretrained.py`, `_array/` và các public modules.
- Bộ `src/contracts/` thống nhất với Model.
- `scripts/install_model.py`, `configs/pretrained_models.json`.
- `docs/TRAINED_RUN_REFERENCE.json` từ cùng phần bàn giao Model.
- Môi trường `.venv` của dự án bằng CPython chuẩn, cùng NumPy.

Giữ nguyên toàn bộ `src/models/`: package init còn import các public modules và loader dùng `_array`.
Không chỉ chép riêng `pretrained.py`, không gán weights vào một Dataset/mapping mới.
Runner không dựng Dataset giả để đi qua `Service.train()` hay `BaseRecommender.fit()`.

Các ví dụ sau chạy từ thư mục gốc dự án, dùng đúng Python trong `.venv` đã chuẩn bị:

```powershell
.\.venv\Scripts\python.exe -m pip install "numpy>=1.24,<3"
.\.venv\Scripts\python.exe scripts/install_model.py --download --model bpr
```

Chỉ dùng `pip install -e .` nếu đã tích hợp `pyproject.toml` của Model; dự án hiện tại không mặc định có file đó.
Lệnh download cần GitHub CLI đã đăng nhập tài khoản có quyền đọc private release.
Có thể tải đúng ZIP model khi đã đăng nhập rồi cài bằng `--archive` theo tài liệu Model.
[Release pretrained-legacy-v1](https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/releases/tag/pretrained-legacy-v1)
chứa asset model; ZIP `Source code` tự động của GitHub không chứa weights.


Cách cài từ ZIP đã tải trong Downloads, không cần GitHub CLI:

```powershell
.\.venv\Scripts\python.exe -B scripts/install_model.py --archive "$env:USERPROFILE\Downloads\bpr-cpu-legacy-v1.zip" --model bpr
```

Thư mục mặc định được chọn là `artifacts/models/bpr/cpu`.
Installer của Model kiểm tra checksum trước khi cài; giữ `weights.npz` và `model_info.json` cùng bundle.
Inference không cần GPU, encoder lớn hoặc toàn bộ dataset raw.
Mỗi tiến trình nạp model riêng, nên bắt đầu bằng một backend worker và đủ RAM.

## Chạy kiểm tra inference

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --config configs/evaluation_pretrained.json
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --user "!"
```

CLI chỉ hỗ trợ các tùy chọn đã công bố trong runner: `--config`, `--user`, `--output-dir`.
Muốn đổi `top_k`, sửa giá trị trong config. Chọn thư mục xuất hoặc nhiều user:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --user "!" --output-dir results/pretrained
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --user "!" --user "ID_NGUON_KHAC"
```

`--user` lặp lại để truyền nhiều raw user IDs và thay danh sách trong config.
Mọi đường dẫn tương đối, kể cả `--config` và `--output-dir`, được tính từ gốc dự án chứa `benchmarks/`.
`!` là ví dụ ID nguồn dùng với bundle hiện tại, không phải tài khoản mới của ứng dụng.
Mặc định không fallback: user ngoài mapping gây lỗi rõ ràng, không bị thay bằng user 0.
Muốn dùng Popularity cho user lạ, bật `allow_cold_start_fallback` trong config.

## Bốn đầu ra

Một run thành công tạo một thư mục kết quả riêng, gồm:

- `recommendations.csv`: đề xuất thực từ loader và các cờ fallback.
- `inference_checks.json`: kết quả kiểm tra đầu ra của các request đã chạy.
- `original_validation.csv`: validation đã ghi trước đó, nhập từ reference tương ứng.
- `run_config.json`: thông tin run và nguồn model/reference để đối chiếu.

`original_validation.csv` không phải phép chấm mới; `inference_checks.json` không phải accuracy.
Mỗi user được gọi recommend hai lần để kiểm tra tính lặp lại; thời gian ghi nhận không phải latency trung bình một request.
Thiếu reference hoặc metadata không khớp phải được xử lý rõ, không tự điền metric bằng 0.
Tham khảo [docs/evaluation.md](docs/evaluation.md) để đọc protocol và giới hạn.

## Kiểm thử và phạm vi

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/evaluation -p test_pretrained.py -v
```

Test report dùng thư viện chuẩn; test tích hợp tạo bundle nhỏ rồi gọi public loader Model thật khi Model/NumPy có sẵn.
Các ca tích hợp báo skip rõ nếu thiếu điều kiện; chúng không thay việc chạy asset Kaggle đã phân phối.
Việc chạy bundle CPU được bàn giao là một bước kiểm tra riêng sau khi cài model.
API training/evaluation v1 cũ có thể giữ lại cho thử nghiệm sau; nó không phải điều kiện để dùng bundle hiện tại.
Gói này không bổ sung UI/HTTP endpoint và không khôi phục bộ regression test Cornac cũ.
