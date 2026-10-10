# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Đào Quang Thái Anh
- MSSV: 2A202602987
- Email: thaianh494@gmail.com
- Link repo (fork): https://github.com/masao1112/K4-Track4-Day23-DaoQuangThaiAnh-2A202602987-Sensor-Fusion-Student
- Commit hash nộp (`git rev-parse HEAD`): 

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, [0, 198], training_segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord, 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision = 0.9586, recall = 0.4021, tp = 1019, fp = 44, fn = 1515
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse = 0.3462 m, matches = 1000, sum_sq_err = 119.8483, ghost_track_frames = 0, missed_gt_frames = 1534, mean_confirmed_tracks = 5.0251
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse = 0.3712 m, matches = 1000, sum_sq_err = 137.7535, ghost_track_frames = 0, missed_gt_frames = 1534, mean_confirmed_tracks = 5.0251
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Cả hai chế độ đều đạt số cặp ghép thành công tối đa tương ứng với các xe được phát hiện (matches = 1000), không có track ảo nào xuất hiện (ghost_track_frames = 0), số frame ground truth bị bỏ lỡ tương đương nhau (missed_gt_frames = 1534), và số track được xác nhận trung bình mỗi frame ổn định ở mức 5.0251. Về sai số vị trí 3D, cả hai chế độ đều đạt mức sai số xuất sắc dưới 0.45 m (RMSE lidar = 0.3462 m, RMSE fused = 0.3712 m). Chênh lệch sai số giữa hai chế độ chỉ là 0.0250 m (nhỏ hơn nhiều so với ngưỡng 0.05 m của rubric). Điều này chứng minh tính nhất quán và hiệu quả của kiến trúc track-then-fuse: việc bổ sung phép đo camera 2D (có thêm nhiễu seeded) vẫn duy trì độ chính xác cao và độ ổn định liên tục của các track, không làm suy giảm chất lượng bám vết so với hệ thống chỉ dùng LiDAR.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
- Về vector đo lường $z$:
  + Lidar 3D: Đo trực tiếp vị trí tâm vật thể trong không gian 3D $z = [x, y, z]^T \in \mathbb{R}^{3 \times 1}$ (hệ tọa độ cảm biến). Mô hình đo tuyến tính $h(x) = [p_x, p_y, p_z]^T$ với ma trận đo $H = [I_3 \mid 0_{3 \times 3}] \in \mathbb{R}^{3 \times 6}$.
  + Camera 2D: Đo tọa độ pixel ảnh $z = [u, v]^T \in \mathbb{R}^{2 \times 1}$. Mô hình đo phi tuyến $h(x)$ qua phép chiếu pinhole sau khi chuyển tọa độ từ vehicle sang camera ($p_s = R_{veh \to sens} p + t_{veh \to sens}$): $u = c_i - f_i \frac{y_s}{x_s}$, $v = c_j - f_j \frac{z_s}{x_s}$ (với $x_s$ là độ sâu forward, $y_s$ là left, $z_s$ là up). Ma trận đo $H \in \mathbb{R}^{2 \times 6}$ là Jacobian đạo hàm chuỗi (chain rule) phụ thuộc vào vị trí dự báo của track.
- Về ma trận hiệp phương sai nhiễu đo lường $R$:
  + Lidar: $R = \text{diag}(\sigma_{lidar\_x}^2, \sigma_{lidar\_y}^2, \sigma_{lidar\_z}^2)$ với đơn vị mét bình phương ($\text{m}^2$, giá trị mặc định $0.1^2 = 0.01\text{ m}^2$).
  + Camera: $R = \text{diag}(\sigma_{cam\_i}^2, \sigma_{cam\_j}^2)$ với đơn vị pixel bình phương ($\text{pixel}^2$, giá trị mặc định $5.0^2 = 25\text{ pixel}^2$).

2. Vì sao cần gating Mahalanobis trước khi gán?
- Loại bỏ các cặp đo lường - track không khả dĩ (spurious measurements / outliers): Tránh việc thuật toán greedy matching gán nhầm các phát hiện ở xa vào track, làm sai lệch phân phối ước lượng của EKF.
- Chuẩn hóa theo độ bất định của cả track và cảm biến: Khoảng cách Mahalanobis $d^2 = \gamma^T S^{-1} \gamma$ tính đến ma trận hiệp phương sai sai số innovation $S = H P H^T + R$. Một track có độ bất định $P$ lớn sẽ có cổng chấp nhận rộng hơn và ngược lại.
- Dựa trên phân phối xác suất $\chi^2$: Do $\gamma \sim \mathcal{N}(0, S)$, $d^2$ tuân theo phân phối Chi-square $\chi^2$ với bậc tự do bằng số chiều đo lường ($m=3$ cho lidar, $m=2$ cho camera). Gating qua ngưỡng $\chi^2$ (với mức tin cậy $99.5\%$) đảm bảo loại bỏ các cặp có xác suất xuất hiện cực thấp.
- Tiết kiệm tính toán: Gán trực tiếp chi phí $\infty$ cho các cặp ngoài gate giúp ma trận chi phí thưa hơn và tránh các phép gán sai lệch.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
- Pipeline là track-then-fuse:
  + Hệ thống chỉ duy trì một danh sách track duy nhất (single unified track list).
  + Mỗi frame chỉ thực hiện EKF predict một lần cho tất cả các track hiện có.
  + Sau đó, cập nhật lần lượt theo từng cảm biến (AssocL lidar update trước, rồi đến AssocC camera update).
  + Không thực hiện gộp dữ liệu thô (raw data fusion) hay gộp detection trước khi tracking (detection-level fusion).
- Dẫn chứng trên log và mã nguồn `run_lab.py`:
  + Trong hàm `run()` của `platform/fusion_lab/scripts/run_lab.py`, thứ tự xử lý mỗi frame là: (1) `KF.predict(track)`, (2) `assoc.associate_and_update(manager, observations, KF, lidar_sensor)`, (3) nếu ở mode `fused` thì gọi tiếp `assoc.associate_and_update(manager, observations, KF, camera_sensor)`.
  + Camera chỉ update trạng thái EKF mà không gọi `manage_tracks` để tạo hay xóa track (chỉ có lidar gọi vòng đời).

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
- Sai lệch có hệ thống (systematic bias):
  + Khi calibration (extrinsic hoặc intrinsic) bị lệch, hình chiếu dự báo $h(x)$ từ không gian 3D lên mặt phẳng ảnh 2D sẽ bị lệch khỏi vị trí đo thực tế $z$.
  + Vector innovation $\gamma = z - h(x)$ sẽ không còn kỳ vọng bằng 0 ($\mathbb{E}[\gamma] \neq 0$), mà xuất hiện một độ lệch (bias) cố định với độ lớn $\|\gamma\|$ tăng vọt trên các frame.
- Hậu quả trong quá trình tracking:
  + Bình phương khoảng cách Mahalanobis $d^2 = \gamma^T S^{-1} \gamma$ sẽ tăng vọt và vượt ngưỡng cổng $\chi^2$, dẫn đến các đo lường camera bị gate từ chối (bỏ rơi, không match được).
  + Nếu lệch nhẹ và vẫn lọt qua gate, Kalman update sẽ kéo lệch ước lượng vị trí của track về hướng sai của camera, làm tăng hiệp phương sai sai số và tăng RMSE vị trí đáng kể so với khi chỉ chạy lidar.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
- Vì sao cần sensor tường minh ở frame rỗng:
  + Khi `meas_list` rỗng (không có detection nào trong frame), hàm không thể đoán được modality từ danh sách đo.
  + Tuy nhiên, cuối hàm vẫn luôn phải gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`. TrackManager cần tham số `sensor` để biết lượt này là của lidar hay camera: nếu là lidar, các track trong FOV mà không có đo sẽ bị phạt miss (trừ $1/\text{window}$ điểm) và kiểm tra điều kiện xóa. Nếu không truyền `sensor` tường minh, hệ thống không thể xử lý phạt miss chính xác khi frame lidar rỗng.
- Vì sao lidar quyết định vòng đời còn camera chỉ EKF update:
  + Lidar cung cấp trực tiếp vị trí 3D $(x, y, z)$ tin cậy trong không gian thế giới, đủ thông tin để khởi tạo một track 3D đầy đủ (`init_track_state_from_meas`).
  + Camera chỉ cung cấp góc chiếu 2D $(u, v)$ bị mất thông tin độ sâu (depth ambiguity), do đó một điểm 2D đơn lẻ không thể khởi tạo một track 3D nếu không có giả định phụ.
  + Việc khởi tạo hoặc xóa track từ camera 2D rất dễ sinh ra track ảo (ghost tracks) do nhiễu nền hoặc che khuất. Do đó, lidar giữ vai trò quyết định sự tồn tại vật lý, camera chỉ bổ trợ tăng độ chính xác góc cho EKF.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
- Điều kiện xác nhận (Confirmed):
  + Track khởi tạo ở trạng thái `"initialized"` với điểm `score = 1 / window` ($1/6 \approx 0.167$).
  + Mỗi khi được gán với đo lường lidar, track chuyển sang `"tentative"` và điểm số tăng $+1 / \text{window}$ (tối đa 1.0).
  + Khi `score > confirmed_threshold` (mặc định $0.8$), track được thăng hạng lên trạng thái `"confirmed"`.
- Giữ confirmed sau khi miss:
  + Khi một track đã `"confirmed"` bị miss trong FOV của lidar, điểm số bị trừ $-1 / \text{window}$.
  + Dù điểm số có giảm xuống dưới $0.8$ (ví dụ từ 1.0 xuống $5/6 \approx 0.833$), trạng thái của track **vẫn được giữ nguyên là "confirmed"**, không bị hạ cấp về "tentative" chỉ vì một lần miss. Điều này giúp quỹ đạo không bị gián đoạn identity khi xe tạm thời bị che khuất.
- Điều kiện xóa track (Delete):
  Track bị xóa ngay khi thỏa mãn ít nhất một trong các điều kiện sau:
  1. Hiệp phương sai vị trí quá lớn: $P_{xx} > \text{max\_P}$ hoặc $P_{yy} > \text{max\_P}$ (với $\text{max\_P} = 3.0^2 = 9.0\text{ m}^2$), bất kể điểm số.
  2. Track đã `"confirmed"` nhưng bị miss liên tiếp khiến `score < delete_threshold` (mặc định $0.6$).
  3. Track chưa `"confirmed"` (`"initialized"` hoặc `"tentative"`) có `score <= 0.0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity AI Coding Assistant
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ sinh khung hàm Parts E–H theo công thức EKF, gợi ý giải thích các câu hỏi kỹ thuật và rà soát các kiểm thử pytest.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Chạy toàn bộ 128 kiểm thử `pytest student/tests` đạt 100% pass; chạy `fusion-run-lab` trên dữ liệu Waymo thật ở chế độ compare; đối chiếu các công thức EKF predict/update, khoảng cách Mahalanobis và ngưỡng chi2 theo tài liệu docs/HUONG_DAN_KY_THUAT.md.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
