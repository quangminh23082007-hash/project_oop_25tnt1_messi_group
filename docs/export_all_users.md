# Xuất top-K cho toàn bộ user trong mapping

Gói này xuất đề xuất từ model đã train. Không train lại và không tính Recall/NDCG.
Nó dùng đúng user_ids, item_ids, weights và lịch sử train lưu trong bundle.
Phạm vi "toàn bộ" ở đây là toàn bộ user trong mapping của bundle; không phải mọi dòng raw reviews.

## Các file mới

- benchmarks/export_all_users.py
- tests/evaluation/test_export_all_users.py
- docs/export_all_users.md

Đặt ba file đúng đường dẫn trong repo hiện có. Không thay thế các file pretrained trước đó.
Cần toàn bộ src/models, src/contracts, NumPy và bundle CPU đã cài.
Mặc định dùng BPR tại artifacts/models/bpr/cpu.

## Kiểm thử

Chạy tại thư mục gốc repo trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/evaluation -p test_export_all_users.py -v
```

Các tests dùng bundle nhỏ với public Model loader thật, không train hoặc tải weights.
Chúng kiểm tra xuất đầy đủ, user không còn candidates, tiếp tục khi gián đoạn,
part đã ghi nhưng checkpoint chưa cập nhật, dữ liệu thay đổi và tránh ghi trùng.

## Chạy toàn bộ BPR

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"

.\.venv\Scripts\python.exe -B -m benchmarks.export_all_users --model-dir artifacts/models/bpr/cpu --output-dir results/bpr_all_users_top10 --top-k 10 --users-per-part 10000
```

Không truyền --limit-users: mặc định 0, nghĩa là toàn bộ mapping.
Model được load một lần mỗi tiến trình; mỗi user được gọi recommend một lần.
Không đưa 1,4 triệu user IDs vào configs/evaluation_pretrained.json.
Đường dẫn tương đối được tính từ thư mục repo chứa benchmarks.

Muốn chạy thử riêng 1.000 user, dùng output folder khác:

```powershell
.\.venv\Scripts\python.exe -B -m benchmarks.export_all_users --output-dir results/bpr_pilot_1000 --top-k 10 --users-per-part 500 --limit-users 1000
```

Không đổi limit/K/chunk/bundle/runtime rồi tiếp tục vào thư mục kết quả cũ.
Giới hạn 1.000 user là pilot, không được ghi là đã xuất toàn bộ.

## Đầu ra

```text
results/bpr_all_users_top10/
    manifest.json
    summary.json
    catalog.csv.gz
    export.lock
    part_000000000_000010000.csv.gz
    part_000010000_000020000.csv.gz
    ...
```

Mỗi part tối đa 10.000 user, tương ứng tối đa 100.000 dòng khi K=10.
Các cột: user_index, user_id, rank, game_id, predicted_score.
Tên game và metadata được xuất riêng một lần trong catalog.csv.gz; nối bằng game_id.
Genres/price có thể thiếu trong bundle. Categories giữ riêng, không tự đổi thành genres.
CSV dùng UTF-8. Các cột dạng danh sách trong catalog được mã hóa JSON.

User đã quan sát hết catalog có thể không có đề xuất; manifest vẫn đếm user đã xử lý.
Vì vậy số dòng có thể nhỏ hơn mapped_users nhân K.
Gzip giúp giảm dung lượng; không giữ toàn bộ danh sách đề xuất trong RAM.
Mỗi part có SHA-256, số dòng, range user và số user có/không có đề xuất.
Chỉ đọc các part được liệt kê trong manifest khi tổng hợp.

## Dừng và chạy tiếp

Có thể dừng bằng Ctrl+C. Chạy lại nguyên khối lệnh full, gồm các biến môi trường,
để tiếp tục. Các part đã commit được kiểm tra hash rồi giữ nguyên.
Phần đang viết có thể phải tính lại; dữ liệu không bị nối thêm trùng.
Nếu part đã hoàn tất nhưng checkpoint chưa commit, chương trình tính lại và đối chiếu
nội dung trước khi nhận lại part đó. File đã commit bị thay đổi sẽ làm chương trình dừng.
File export.lock dùng khóa của hệ điều hành; chỉ một tiến trình được dùng một output folder.
Khi tiến trình kết thúc, khóa được giải phóng dù file export.lock vẫn còn.

Giữ máy hoạt động trong lúc chạy. Có thể chạy lại sau khi ngắt hoặc khởi động lại máy.
Đừng xóa manifest/parts để "tiếp tục"; dùng nguyên output folder và cùng tham số.

## Xác nhận hoàn thành toàn bộ

Mở summary.json và kiểm tra:

- status là complete
- processed_users bằng mapped_users
- all_mapped_users_exported là true
- tổng recommendation_rows là số đề xuất thực đã xuất
- recomputed_metrics là false

BPR hiện tại có 1.394.194 user và 23.089 game; K=10 cho tối đa 13.941.940 đề xuất.
Không mở toàn bộ dữ liệu nhiều triệu dòng như một sheet Excel; có thể mở từng part sau giải nén.

## Bằng chứng thử trên máy hiện tại

Bản đã được thử với 1.000 user thật và tạo 10.000 dòng trong 2 part.
Bộ 8 tests đã đạt, gồm kiểm tra recovery. Đây không phải bằng chứng job toàn bộ đã hoàn tất.
Thử 100 user với một BLAS thread đạt khoảng 281 user/giây sau load; ngoại suy khoảng 1,4 giờ.
Tốc độ khác nhau theo profile user, CPU, trạng thái máy và ổ đĩa; không coi ngoại suy là cam kết thời gian.
Elapsed_seconds gồm load và công việc đã ghi nhận, không phải phép đo latency chuẩn.

Original validation 2.000 user ở các file trước vẫn là báo cáo gốc.
Để chấm lại metric cần workflow ground truth riêng; xuất top-K toàn bộ không tự tạo Recall/NDCG mới.
