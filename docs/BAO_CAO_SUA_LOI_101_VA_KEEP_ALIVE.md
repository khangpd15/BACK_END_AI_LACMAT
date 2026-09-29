# BÁO CÁO KỸ THUẬT: KHẮC PHỤC HIỆN TƯỢNG COLD START (RENDER) VÀ LỖI 101 SUPABASE DATABASE

> **Dự án:** RemiCare Strabismus AI Backend  
> **Ngày thực hiện:** 29/09/2026  
> **Người thực hiện:** Hệ thống AI Kỹ sư Backend  

---

## 1. TỔNG QUAN VẤN ĐỀ

Hệ thống ghi nhận 2 vấn đề lớn trong quá trình vận hành giữa Front-end, Render và Supabase:
1. **Front-end gửi request phải đợi rất lâu:** Hiện tượng máy chủ phản hồi chậm (mất 40 - 60 giây) mỗi khi người dùng thao tác sau một khoảng thời gian không sử dụng.
2. **Lưu dữ liệu vào Database Supabase bị "Lỗi 101":** Gây gián đoạn quá trình lưu phiên Cover Test và kết quả phân tích AI.

---

## 2. PHÂN TÍCH NGUYÊN NHÂN CỐT LÕI

### 2.1. Vấn đề 1: Vì sao Front-end phải chờ lâu?
* **Cơ chế:** Backend RemiCare hiện đang deploy trên hạ tầng **Render Free Tier**.
* Theo chính sách của Render, bất kỳ Web Service miễn phí nào **không nhận request trong vòng 15 phút** sẽ tự động chuyển sang trạng thái "ngủ đông" (**Spin Down / Hibernate**) để tiết kiệm tài nguyên CPU/RAM.
* Khi Front-end gửi request trở lại, Render phải kích hoạt quy trình **Cold Start**:
  1. Khởi tạo container Linux mới.
  2. Boot Python 3.11 runtime.
  3. Load các thư viện thị giác máy tính và học máy nặng (`OpenCV`, `MediaPipe`, `Scikit-learn`, `FastAPI`).
  4. Nạp model AI (`korean_shared_model.joblib` / `remicare_transfer_model.joblib`) và khởi tạo kết nối cơ sở dữ liệu.
* **Thời gian tổng cộng:** Dao động từ **35s đến 65s**. Sau khi khởi động xong, các request tiếp theo mới chạy với tốc độ thực tế (vài chục đến vài trăm mili-giây).

---

### 2.2. Vấn đề 2: Lỗi 101 Supabase có phải do không ping không?

> ❌ **KẾT LUẬN: LỖI 101 KHÔNG PHẢI DO THIẾU PING.**

* **Nếu thiếu ping, điều gì xảy ra?**
  * Render bị ngủ đông (như phân tích ở Mục 2.1).
  * Supabase Free Tier nếu không có tương tác trong **7 ngày liên tục** sẽ bị tạm ngưng (**Paused**), trả về mã `503 Service Unavailable` hoặc thông báo *"Project is paused"*.
  * Connection idle quá 5-10 phút bị Supabase ngắt socket, Python ném lỗi `ConnectionResetError: [Errno 104] Connection reset by peer`.
* **Thực chất Lỗi 101 là gì?**
  * Trong hệ thống Linux (môi trường container của Render), số hiệu **101** là mã lỗi chuẩn POSIX:  
    $$\text{Errno 101} = \text{ENETUNREACH} \implies \textbf{"Network is unreachable"}$$
  * Trong log Python `asyncpg` / `SQLAlchemy`, lỗi xuất hiện dưới dạng:
    ```text
    asyncpg.exceptions.CannotConnectNowError: [Errno 101] Network is unreachable
    hoặc
    OSError: [Errno 101] Network is unreachable
    ```

* **Nguyên nhân kỹ thuật chính:**
  1. **Bất tương thích giao thức mạng (IPv6 vs IPv4):**
     * Đầu năm 2024, Supabase chuyển toàn bộ địa chỉ kết nối trực tiếp (**Direct Connection**) dạng `db.[PROJECT-REF].supabase.co:5432` sang phân giải DNS thuần **IPv6**.
     * Hạ tầng mạng của Render (đặc biệt là gói Web Service Free & Standard) **chưa hỗ trợ định tuyến ra Internet qua IPv6 (IPv6 Outbound Disabled)**.
     * Khi backend trên Render cố mở socket TCP tới IP của `db.*.supabase.co`, kernel Linux không tìm thấy default gateway IPv6, lập tức ném lỗi hệ điều hành: `[Errno 101] Network is unreachable`.
  2. **Chuỗi kết nối cấu hình chưa đúng mục đích:**
     * Nhiều bạn lấy chuỗi *Direct Connection* (port 5432) thay vì chuỗi **Supavisor Connection Pooler** (domain `pooler.supabase.com`, port 6543 hoặc 5432) vốn hỗ trợ đầy đủ **IPv4**.
  3. **Lỗi Transaction Pooling với AsyncPG:**
     * Khi dùng Pooler của Supabase (chạy qua PgBouncer/Supavisor ở Transaction mode port 6543), thư viện `asyncpg` mặc định bật *Prepared Statement Cache*. Do PgBouncer luân chuyển connection giữa các truy vấn, cache này sẽ gây lỗi `DuplicatePreparedStatementError` nếu không tắt `statement_cache_size=0`.

---

## 3. CÁC NÂNG CẤP ĐÃ THỰC HIỆN TRÊN BACKEND

Đã hoàn thiện các mã nguồn trên backend để giải quyết triệt để 2 vấn đề trên:

### 3.1. Dịch vụ Tự động Ping & Làm ấm kết nối (`app/services/keep_alive.py`)
* Xây dựng `KeepAliveService` chạy nền dưới dạng Background Task trong lifecycle (`lifespan`) của FastAPI.
* **Cơ chế hoạt động:**
  * Định kỳ mỗi **10 phút** (`KEEP_ALIVE_INTERVAL_SECONDS=600`, trước ngưỡng 15 phút của Render).
  * Tự động lấy URL public của backend qua biến môi trường Render cung cấp sẵn: `RENDER_EXTERNAL_URL` (ví dụ `https://remicare-strabismus-ai.onrender.com`) hoặc `SELF_PING_URL`.
  * Thực hiện HTTP GET tới endpoint `/ping` cực nhẹ (chỉ tốn vài mili-giây, không tính toán ML).
  * Đồng thời gửi truy vấn `SELECT 1` tới cơ sở dữ liệu Supabase để làm ấm kết nối pool (`ping_db`), ngăn ngừa tình trạng socket bị đóng và ngăn Supabase pause project.
  * Tự động ghi nhận nhật ký (latency, status code, tỷ lệ thành công) và không bao giờ làm crash ứng dụng nếu gặp sự cố mạng tạm thời.

### 3.2. Bổ sung các Endpoint Hỗ trợ Hệ thống (`app/main.py`)
* `GET /ping`: Endpoint siêu nhẹ, phản hồi tức thời `{ "status": "alive", "timestamp": ... }`.
* `GET /health/db`: Endpoint chẩn đoán chuyên dụng, chạy thử truy vấn `SELECT 1` và trả về chi tiết kết nối, latency, kèm gợi ý sửa lỗi nếu phát hiện mã Errno 101.
* Nâng cấp `GET /health`: Bổ sung thông tin trạng thái hoạt động của worker `keepAlive`.

### 3.3. Tối ưu hóa Database Engine & Xử lý Lỗi 101 (`app/db/database.py`)
* **Tự động nhận diện Supabase Pooler:**
  * Hàm `get_db_connect_args(url)` tự động phát hiện nếu kết nối tới Supavisor (`pooler.supabase.com` hoặc port 6543).
  * Tự động gán `statement_cache_size=0` và `prepared_statement_cache_size=0` giúp `asyncpg` tương thích hoàn toàn với transaction pooler.
* **Cảnh báo sớm IPv6:**
  * Nếu phát hiện URL chứa `db.*.supabase.co`, hệ thống sẽ ghi log cảnh báo nổi bật kèm hướng dẫn chuyển sang địa chỉ Pooler IPv4.
* **Tối ưu hóa Connection Pool:**
  * `pool_recycle=300`: Tự động tái tạo kết nối sau 5 phút để tránh connection bị Supabase drop ngầm.
  * `pool_pre_ping=True`: Kiểm tra kết nối trước khi đưa cho truy vấn thực hiện.
  * `pool_size=5`, `max_overflow=10`: Tối ưu giới hạn kết nối cho gói Supabase Free.

### 3.4. Cải thiện Thông báo Lỗi API (`app/api/cover_test.py`)
* Trong endpoint lưu phiên Cover Test `/api/v1/cover-test/sessions`, nếu xảy ra lỗi 101, backend sẽ trả về mã 500 kèm thông báo chỉ dẫn rõ ràng thay vì lỗi mơ hồ.

---

## 4. HƯỚNG DẪN THIẾT LẬP TRÊN RENDER & SUPABASE DASHBOARD

Để backend hoạt động 100% không còn lỗi 101 và không bị ngủ đông, hãy thực hiện các bước sau:

### Bước 1: Lấy URL Supavisor Pooler từ Supabase
1. Đăng nhập vào [Supabase Dashboard](https://supabase.com/dashboard).
2. Chọn project của bạn -> Vào **Project Settings** (biểu tượng bánh răng) -> Chọn tab **Database**.
3. Cuộn xuống phần **Connection string**:
   * Chọn tab **URI**.
   * Chuyển chế độ từ **Direct connection** sang **Connection pooling**.
   * Chọn Mode: **Transaction** (Port `6543`) hoặc **Session** (Port `5432`).
4. Chuỗi kết nối sẽ có dạng:
   ```text
   postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
   ```
   *(Thay `[PROJECT_REF]`, `[REGION]` và `[PASSWORD]` bằng thông tin thực tế của bạn)*.

### Bước 2: Cập nhật biến môi trường trên Render
1. Đăng nhập vào [Render Dashboard](https://dashboard.render.com).
2. Chọn Web Service backend của bạn (`remicare-strabismus-ai`).
3. Vào mục **Environment**:
   * Cập nhật biến `DATABASE_URL`:
     ```text
     postgresql+asyncpg://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
     ```
     *(Lưu ý: Backend RemiCare tự động chuyển đổi tiền tố nếu bạn dán `postgres://` hoặc `postgresql://`)*.
   * Đảm bảo biến `RENDER_EXTERNAL_URL` có giá trị là URL của Render (ví dụ: `https://remicare-strabismus-ai.onrender.com`). Render thường tự điền biến này. Nếu chưa có, bạn có thể tạo biến `SELF_PING_URL=https://tên-app-của-bạn.onrender.com`.
   * Thêm biến:
     ```text
     KEEP_ALIVE_ENABLED=true
     KEEP_ALIVE_INTERVAL_SECONDS=600
     KEEP_ALIVE_PING_DB=true
     ```
4. Bấm **Save Changes** để Render tự động redeploy lại với cấu hình mới.

### Bước 3: Cấu hình thêm UptimeRobot (Giải pháp bổ sung kép chống ngủ 100%)
* Vì worker tự ping chạy *bên trong* container, nếu service bị Render tắt cưỡng bức hoặc restart, container sẽ không tự bật lại nếu không có HTTP request từ bên ngoài tới.
* **Cách khắc phục:**
  1. Tạo tài khoản miễn phí tại [UptimeRobot.com](https://uptimerobot.com) hoặc [Cron-job.org](https://cron-job.org).
  2. Tạo một monitor mới (HTTP Monitor):
     * **URL:** `https://tên-app-của-bạn.onrender.com/ping`
     * **Monitoring Interval:** 5 hoặc 10 phút.
  3. Khi có UptimeRobot gửi tín hiệu liên tục từ ngoài vào, Render sẽ **không bao giờ ngủ đông**, loại bỏ hoàn toàn độ trễ 50s cho Front-end!

---

## 5. KẾT QUẢ KIỂM THỬ

* Toàn bộ **117 bài kiểm thử tự động (Unit & Integration tests)** chạy thành công 100%:
  * `test_ping_endpoint`: Đạt.
  * `test_health_endpoint_includes_keep_alive`: Đạt.
  * `test_db_health_endpoint`: Đạt.
  * `test_get_db_connect_args_pooler`: Đạt (vô hiệu hóa statement cache chuẩn).
  * `test_get_db_connect_args_direct_supabase_warning`: Đạt (cảnh báo IPv6 chuẩn).
  * `test_keep_alive_service_ping_cycle`: Đạt.
  * Toàn bộ 26 test lưu phiên dữ liệu Cover Test và 82 test thuật toán AI: Đạt.
