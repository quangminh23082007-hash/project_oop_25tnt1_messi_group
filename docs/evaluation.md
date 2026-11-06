# Evaluation và báo cáo cho đường pretrained v1.1

## 1. Mục đích và ranh giới

Đường chạy này dùng model đã train để đề xuất game bằng CPU và kiểm tra việc tích hợp inference.
Runner không gọi `fit`, không chọn lại epoch và không chạy encoder để tạo feature matrix.
Validation được xuất từ hồ sơ lần train gốc, không được suy ra từ danh sách top-K vừa chạy.

Phân biệt ba kết quả:

| Nội dung | Nguồn | Ý nghĩa |
| --- | --- | --- |
| Đề xuất | `PretrainedRecommender` chạy với bundle được chọn | Đầu ra inference của run hiện tại |
| Inference checks | Runner kiểm tra các request và đầu ra hiện tại | Kiểm tra vận hành/hợp đồng, không phải metric xếp hạng |
| Original validation | `TRAINED_RUN_REFERENCE.json` | Chỉ số đã ghi ở lần train gốc, được nhập lại |

Chỉ được gọi là “Recall/NDCG tính lại” khi thực sự có ground truth và chạy lại protocol đánh giá phù hợp.
Nhiều request inference thành công không tự chứng minh chất lượng đề xuất cao hơn baseline.

## 2. Các file và trách nhiệm

- `benchmarks/run_pretrained.py`: đọc config, nạp một model, chạy request và điều phối báo cáo.
- `benchmarks/pretrained_report.py`: kiểm tra/định dạng kết quả và ghi các artifact báo cáo.
- `configs/evaluation_pretrained.json`: model, bundle, reference và request mẫu.
- `tests/evaluation/test_pretrained.py`: kiểm tra runner/report bằng dữ liệu nhỏ và mock có kiểm soát.
- `docs/evaluation.md`: quy tắc sử dụng và diễn giải kết quả.
- `README.md`: hướng dẫn bàn giao và chạy nhanh.

Không định nghĩa lại `GameItem`, `RecommendationResult` hoặc các exception chung trong gói này.
Không chuyển legacy bundle thành một v1 Dataset bằng metadata/fingerprints tự đặt.

## 3. Phần phải nhận từ Model

Giữ toàn bộ `src/models/` và `src/contracts/` thống nhất, cùng NumPy trong môi trường `.venv`.
Không chỉ lấy `pretrained.py` và `_array`: `src/models/__init__.py` còn import các public modules khác.
Cần installer/manifest từ Model cùng `docs/TRAINED_RUN_REFERENCE.json` đúng phiên bản.
Sáu file Evaluation này không thay thế các thành phần đó và không chứa weights.

Release model là private; tài khoản tải cần quyền đọc repository/release.
Clone source không tự tải các asset ZIP của release.
Chọn asset `bpr-cpu-legacy-v1.zip` hoặc model tương ứng, không chọn ZIP Source code tự động.
Installer kiểm tra checksum; không đổi tên model_info/weights hoặc ghép hai bundle khác nhau.

```powershell
.\.venv\Scripts\python.exe -m pip install "numpy>=1.24,<3"
.\.venv\Scripts\python.exe scripts/install_model.py --download --model bpr
```

Lệnh download dùng GitHub CLI đã đăng nhập; cách cài ZIP local xem `docs/PRETRAINED_MODELS.md` của Model.
Các lệnh dùng CPython trong `.venv` của dự án, tránh chạy nhầm Python MSYS hoặc môi trường khác.
Chỉ cài editable bằng `pip install -e .` khi đã có `pyproject.toml` tương ứng; đây không phải bước mặc định.

Cách cài từ ZIP đã tải trong Downloads, không cần GitHub CLI:

```powershell
.\.venv\Scripts\python.exe -B scripts/install_model.py --archive "$env:USERPROFILE\Downloads\bpr-cpu-legacy-v1.zip" --model bpr
```

Inference không cần train lại, PyTorch/CUDA, Qwen encoder hoặc toàn bộ raw reviews.
Nạp bundle một lần cho mỗi lần chạy runner; backend ứng dụng nên giữ object đã nạp cho nhiều request.
Nhiều backend worker sẽ nạp các bản model riêng, nên cần kiểm tra RAM thực tế.

## 4. Cấu hình và API

```python
from benchmarks.run_pretrained import run_pretrained

paths = run_pretrained(
    "configs/evaluation_pretrained.json",
    user_ids=["!"],
    output_dir="results/pretrained",
)
```

Đây là API chạy một lượt và xuất báo cáo; gọi lại API sẽ thực hiện một run mới.
`user_ids=None` dùng danh sách trong config; đối số được truyền vào thay danh sách đó.
`top_k`, fallback và lịch sử bổ sung lấy từ config.

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --config configs/evaluation_pretrained.json
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --user "!" --user "ID_NGUON_KHAC"
.\.venv\Scripts\python.exe -m benchmarks.run_pretrained --user "!" --output-dir results/pretrained
```

CLI có `--config`, `--user` lặp lại và `--output-dir`; không có tùy chọn `--top-k`.
Mọi đường dẫn tương đối được resolve từ gốc dự án chứa `benchmarks/`, kể cả config, bundle, reference và output.
Quy tắc này cũng áp dụng cho `--output-dir`; nó không lấy thư mục chứa config hoặc cwd tùy ý làm gốc.
Các ví dụ chạy từ thư mục gốc dự án. Sửa config khi muốn đổi model hoặc K.
Mặc định là BPR tại `artifacts/models/bpr/cpu` và K=10.
Khi chọn TextBPR, đổi đồng thời model và model_dir để trỏ đúng bundle; không chỉ đổi nhãn báo cáo.
Runner nhận `text_bpr` hoặc alias `text-bpr`; reference dùng key `text_bpr`.

## 5. ID, mapping, lịch sử và metadata

`user_ids` là ID nguồn nằm trong mapping của bundle, không phải chỉ số hàng hoặc tên tài khoản mới của app.
`!` là request mẫu của bundle đã bàn giao; bundle khác có thể cần một ID nguồn khác.
Không thay mapping trong bundle bằng mapping của Data mới, kể cả khi dimensions trùng nhau.

`exclude_played=true` loại lịch sử đã quan sát lưu cùng bundle.
`extra_seen` chứa các raw game IDs ứng dụng muốn loại thêm, chẳng hạn game vừa review/đã sở hữu.
`exclude_played=false` tắt cả lịch sử trong bundle và lịch sử bổ sung theo API loader.
User ngoài mapping mặc định gây lỗi; không tự mượn user 0 và không sinh kết quả rỗng giả thành công.

Nếu chủ động bật fallback, loader dùng Popularity toàn cục và kết quả phải thể hiện `is_fallback=true`.
Fallback là một chính sách serving, không phải bằng chứng cá nhân hóa user mới.
Không nối user mới với profile nguồn chỉ vì tên hiển thị giống nhau.

`game_id` trong kết quả là raw ID; `predicted_score` là điểm xếp hạng, không phải số sao hoặc xác suất.
Giữ cả score âm hoặc score lớn hơn 1; không ép về thang 0–1 để trình bày.
Catalog trong bundle có thể thiếu genres/price; categories không tự biến thành genres và giá thiếu không là miễn phí.
Muốn bổ sung catalog cho app, dùng adapter theo raw game ID trả shared `GameItem`; không đổi thứ tự hàng weights.
Đường config tối thiểu này không tự nối một catalog Data tùy ý hoặc bổ sung UI/endpoint.

## 6. Bốn artifact đầu ra

Một run thành công tạo một thư mục kết quả riêng:

| File | Cách đọc |
| --- | --- |
| `recommendations.csv` | Các đề xuất thực, kèm user, thứ hạng, score và fallback |
| `inference_checks.json` | Những kiểm tra vận hành đã thực hiện cho run hiện tại |
| `original_validation.csv` | Các metric validation nhập từ reference; không phải test mới |
| `run_config.json` | Cấu hình hiệu lực và thông tin nguồn để truy vết |

Đọc nhãn nguồn/provenance trước khi đưa số vào báo cáo thuyết trình.
Không đổi nhãn original validation thành test, accuracy hoặc “benchmark vừa chạy”.
Không tự chèn 0 khi thiếu metric, thiếu reference hoặc không đủ dữ liệu đánh giá.
Không coi kết quả của mock loader là kết quả đo bằng model thật.

Runner gọi recommend hai lần cho mỗi user và so sánh kết quả để kiểm tra tính lặp lại.
`inference_seconds` gồm đọc lịch sử, cả hai lần recommend, kiểm tra và chuyển đổi kết quả trong vòng lặp.
`load_seconds` đo riêng lời gọi loader, không bao gồm mọi bước đọc config/reference trước đó.
Không diễn giải các số này thành latency trung bình một request hoặc phép benchmark thông lượng.

## 7. Protocol gốc và giới hạn

Bundle đang được phân phối thuộc lần chạy legacy random 70/15/15 và giữ mapping/provenance gốc.
Không đổi lời mô tả thành v1 80/10/10 chỉ vì source hiện tại đã có contracts mới.
Reference hiện ghi validation trên mẫu 2.000 user; số user eligible và toàn bộ user mapping là các đại lượng khác.
Random split không tự chứng minh khả năng dự đoán tương lai theo thời gian.
Danh tính user nguồn chưa được xác thực và có thể là tên hiển thị; đọc metric cùng hạn chế này.

Original validation nhập cả BPR, TextBPR và Popularity của cùng hồ sơ gốc; không có nghĩa runner đã nạp cả hai model.
Việc nhập lại những số đó không huấn luyện/chạy Cornac trong run hiện tại.
Muốn benchmark lại phải có artifacts/ground truth gốc và tái hiện cùng cohort, candidate history và K.
Không thay protocol hoặc tạo một bộ ground truth ngẫu nhiên rồi gắn metric vào weights cũ.

## 8. Kiểm thử và phần v1 giữ lại

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/evaluation -p test_pretrained.py -v
```

Bộ test mới gồm các ca report dùng thư viện chuẩn và các ca tích hợp với public loader thật trên bundle NumPy nhỏ.
Nếu thiếu Model hoặc NumPy, các ca tích hợp phải ghi lý do skip; không báo chúng là đã chạy thành công.
Bundle nhỏ trong test không phải asset Kaggle. Sau khi cài asset, chạy thêm một lượt CPU với user thật trong mapping.
Không khẳng định toàn bộ bộ test v1 cũ đã tương thích chỉ vì bộ test này pass.
File cũ `test_cornac_adapter.py` đang chứa integration tests khác; gói này không khôi phục Cornac regression suite.
Các module metric/Evaluator và runner training v1 có thể giữ làm công việc tùy chọn sau.
Chúng không phải bước bắt buộc để nạp bundle, và không được tự train lại ở startup ứng dụng.
