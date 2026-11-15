# JobRec GNN — triển khai nghiên cứu từ đầu

Pipeline Python cho hệ gợi ý Top-20 việc làm từ JTH cục bộ. Kế hoạch chi tiết: [KE_HOACH_TRIEN_KHAI_TU_DAU.md](KE_HOACH_TRIEN_KHAI_TU_DAU.md).

## Đã triển khai

- Đọc đúng `data/jobs (1).csv`, giữ raw bất biến, audit khóa/FK/missing/date, SHA-256 và dataset manifest.
- Trích 78.195 stage events, gộp thành 41.308 first-pair positives; không lấy outcome tương lai làm feature.
- Temporal split toàn cục, không tách sự kiện cùng ngày; explicit availability và báo số mẫu bị loại.
- Feature hashing ổn định, chuẩn hóa địa bàn `75.0`/`75`, salary bins/vocabulary fit từ profile có sẵn trước train cutoff; mask thiếu/nguồn LLM.
- Graph G0/G1/G2, reverse edges, train-only interaction messages và disjoint supervision để tránh lộ cạnh đích.
- Baseline Random, MostPopular, content cosine, Logistic Regression; MF-BPR và LightGCN qua trainer chung.
- **Heterogeneous GraphSAGE, R-GCN, HGT**; BPR, early stopping, checkpoint, số tham số/runtime/memory.
- Gate uniform/global/profile/history; history pooling chỉ từ quá khứ. Không suy salary preference từ current salary.
- Full-catalog Recall/Precision/NDCG/MRR@5/10/20, fill/coverage, warm/cold users, per-user artifacts và paired bootstrap.
- Policy `pass/fail/unknown`, lọc rồi xếp theo score; predictor hỗ trợ thay trọng số và tương tác phiên; Streamlit demo.
- Notebook end-to-end cho Kaggle/Colab, cấu hình smoke test, script nhiều seed; EDA SVG/HTML không cần matplotlib.

Đây là **bản chạy được đầu tiên**, không phải nghiên cứu đã hoàn tất: chưa tuning ba seed, chưa chứng minh cải tiến, chưa có graph-level neighbor minibatching, temporal rolling-query protocol, learned/diversity reranker, funnel-horizon labels, VietJobs/MEDAJob hay RAG. Trainer hiện full-batch; giảm cấu hình khi thiếu RAM/VRAM, không gọi đây là neighbor sampling. `edge_retention` là random pruning graph train, không phải parameter pruning; graph đánh giá giữ đầy đủ thông tin quá khứ.

## 1. Cài đặt

Python CPython native 3.10+ (đã kiểm thử Python 3.11). Không dùng MSYS/MinGW Python để cài wheel PyTorch Windows. Cài Torch phù hợp GPU/runtime theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/), rồi:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[train,baseline,dev,demo]"
```

Linux/Kaggle: dùng Python runtime của notebook; không tạo Windows venv. Kiểm tra CUDA thực tế, không mặc định mọi Kaggle session có GPU. Dependency declarations là khoảng tương thích, không phải lockfile; lưu môi trường mỗi benchmark bằng `python -m pip freeze > artifacts/environment.txt` sau khi cài thành công.

Chỉ chạy data/EDA và ba baseline cơ bản thì không cần thư viện bên thứ ba:

```powershell
$env:PYTHONPATH = "src"
python -m jobrec --help
python -m jobrec prepare
python -m jobrec graph
python -m jobrec baseline --model all --partition validation
```

Trên máy Windows hiện tại có Python native ở `%LOCALAPPDATA%\Programs\Python\Python311\python.exe`. Launcher sau tự chuyển từ MSYS sang Python native này; không cần thay PowerShell execution policy:

```powershell
python scripts/jobrec.py prepare
python scripts/jobrec.py train --model sage --config configs/smoke.json --max-users 10 --max-pairs 100 --output-dir artifacts/smoke/sage
```

`baseline --model all` chỉ chạy Random/Popular/Content, vì chúng không cần dependency. Chạy riêng Logistic Regression bằng `--model logreg` sau khi cài extra `baseline`.

## 2. Pipeline và artifacts

`configs/default.json` ánh xạ dữ liệu và hyperparameters; `prepare --data-dir PATH --jobs-file NAME --output-dir PATH` ghi đè đường dẫn khi chạy Kaggle. Cấu hình smoke giảm chiều/lớp, chỉ giữ 5% cạnh train; **không dùng kết quả smoke làm benchmark**.

Sau `prepare`:

```text
artifacts/prepared/
  manifest.json          nguồn, checksum, cảnh báo snapshot/version
  audit.json             schema, missing, FK, sparsity, phân phối tương tác
  split.json             ngày cutoff, train/validation/test pairs, split hash
  coverage.json          mẫu đủ/không đủ điều kiện và lý do
  preprocessors.json     train-only vocabulary/salary bins
  bundle.json.gz         profile/event/pair/split dùng chung
  eda/index.html         mở trong trình duyệt để xem các SVG
```

Graph command ghi summary/mappings schema thống kê, không lưu bản tensor GPU vào raw data. Mỗi lần train fit lại preprocessors theo cấu hình trên **cùng train cutoff**, rồi đóng gói trong checkpoint; không fit từ validation/test.

### Giới hạn dữ liệu quan trọng

- Local dataset card ghi v1.0, nguồn chính thức ghi v1.1: manifest dùng `local_unverified`, không tự nhận version đã xác minh. JTH là dữ liệu **Pháp**, giấy phép nghiên cứu phi thương mại; kiểm tra release trước phân phối.
- Cutoff thực tế theo cấu hình mặc định: **02/07/2023**, **18/05/2024**. Có 28.914/6.196/6.198 first-pair positives ở train/validation/test.
- Với catalog/profile đóng băng tại đầu mỗi kỳ, validation chỉ có **370 positive pairs / 350 user**, test **386 / 354** đủ điều kiện. Đây là khoảng **6% positive coverage**, không đại diện toàn bộ 37.554 ứng viên. Luôn đọc `coverage.json`; không giấu phần bị loại hoặc gọi đây là kết quả tổng quát toàn JTH.
- New user/item xuất hiện sau cutoff không được đưa ngược vào graph quá khứ. Muốn đánh giá phần lớn các ca này cần rolling-query hoặc snapshot protocol khác, kèm kiểm thử trước benchmark tiếp theo.
- Profile tĩnh, timestamp nhiễu và LLM features có giới hạn lịch sử; temporal edge split không bảo đảm mọi thuộc tính thực sự có sẵn tại quá khứ.
- `zipcode` không phải tọa độ/khoảng cách, `actual_salary` không phải lương mong muốn, trường `llm_*` không phải xác minh hard constraints.

## 3. Huấn luyện và so sánh

```powershell
python -m jobrec train --model sage --output-dir artifacts/sage
python -m jobrec train --model rgcn --output-dir artifacts/rgcn
python -m jobrec train --model hgt --output-dir artifacts/hgt
python -m jobrec train --model mf --output-dir artifacts/mf
python -m jobrec train --model lightgcn --output-dir artifacts/lightgcn
python -m jobrec baseline --model logreg --partition validation
```

Thêm `--device cuda` khi có GPU; mặc định CPU. Cấu hình default có thể nặng; chạy smoke trước:

```powershell
python -m jobrec train --model hgt --config configs/smoke.json --max-users 10 --max-pairs 100 --output-dir artifacts/smoke/hgt
```

`--max-users` lấy cohort ngẫu nhiên có seed cố định để debug, không thay candidate catalog; `--max-pairs` giới hạn supervision mỗi epoch. Artifact ghi hai giới hạn này để không nhầm với final benchmark. Test set không được truy cập trong `train`.

Để công bằng và tránh cạnh đích lọt vào message passing, trainer chia cố định train positives thành khoảng **80% message edges / 20% supervision**, cùng seed cho mọi model. Những cạnh supervision và reverse của chúng không được đưa vào graph train hoặc history gate train. Đây là giao thức khởi đầu minh bạch, không có nghĩa mọi positive đều được dùng làm supervision. Validation graph được phép dùng toàn bộ train interactions đã biết. MF/LightGCN tự chuyển sang G0, dùng cùng supervision split; MF không dùng side features, ID chưa train có embedding 0.

G0/G1/G2 thay đổi **cấu trúc graph**; vector profile đầu vào vẫn giống nhau để cô lập tác dụng thêm node/edge. Thử nghiệm loại bỏ hẳn một nhóm feature phải có cấu hình riêng sau, không coi graph ablation này là feature ablation hoàn chỉnh.

```powershell
python scripts/run_experiments.py --models sage rgcn hgt --seeds 42 2026 3407 --device cuda
python -m jobrec train --model sage --gate history --output-dir artifacts/sage_history
```

Gate nằm ở scorer, **không phải đã sửa convolution**. Các chế độ `uniform/global/profile/history` dùng cùng sáu relation groups; salary-preference bị mask vì JTH không có sở thích lương xác thực. History hiện là mean pooled **unique past jobs**, không dùng final stage để giả hành động của ứng viên. Demo có thể đổi state bằng save-event mà không retrain GNN. Không gọi gate này là sở thích thật đã được chứng minh.

Mỗi run ghi `best.pt`, `history.json`, `profile.json`. Checkpoint chứa model arguments, weights, preprocessors, schema metadata, message split, config và hashes. Chưa hỗ trợ resume optimizer giữa epoch. Không load checkpoint không rõ nguồn; `weights_only=True` giảm nhưng không loại mọi rủi ro file không tin cậy.

## 4. Đánh giá

```powershell
python -m jobrec evaluate --checkpoint artifacts/sage/best.pt --partition validation --output-dir artifacts/evaluation/sage_val
python -m jobrec evaluate --checkpoint artifacts/sage/best.pt --partition test --output-dir artifacts/evaluation/sage_test
python -m jobrec compare --left artifacts/baseline/content_validation/per_user.json.gz --right artifacts/evaluation/sage_val/per_user.json.gz
```

Chỉ chạy final test sau khi chốt thiết kế/config. Test graph đóng băng ở cutoff thứ hai, được dùng train+validation interactions **đã xảy ra trước cutoff**, nhưng weights và vocab không refit. Đây là cập nhật context quá khứ, không phải huấn luyện bằng test hay rolling updates trong test. Ground truth và candidate catalog chung cho mọi model trong cùng partition. Negative train chỉ lấy job tạo trước thời điểm positive, loại known train positives; không nhìn test labels để chọn âm.

Candidate catalog là **job đã biết**, không khẳng định mọi job còn đang tuyển vì chưa có close date. Ties theo job ID; macro theo user có positive đủ điều kiện; precision/fill giữ mẫu số K kể cả trả ít. Bootstrap chỉ chấp nhận hai cohort giống hệt nhau; CI theo user của hai runs không thay thế độ lệch qua nhiều seed. Chưa có funnel-graded evaluation hay violation metric đủ nhãn xác thực; không ghi 0 cho “chưa đo được”.

## 5. Demo

```powershell
python -m streamlit run app/streamlit_app.py
```

Nhập checkpoint đã train; mặc định `artifacts/sage/best.pt`. Chọn ID ứng viên ẩn danh, xem Top-20, gate weights; lưu job trong phiên để thử history update. Muốn trọng số đổi theo history phải dùng checkpoint `--gate history`. Slider là mô phỏng ưu tiên, không phải nhãn JTH thật. Demo hiện dùng hồ sơ có sẵn, chưa hỗ trợ đăng ký CV mới hoặc production authentication.

`constraint_policy` nhận **verified job fields** từ phía gọi; không tự biến các trường LLM hay lương đơn lẻ JTH thành điều kiện đã xác minh. `unknown` được giữ lại và hiển thị, không tự loại. Reranker MVP hiện lọc hard-fail rồi sắp theo hybrid score; learned reranking/MMR là bước sau. Không dùng demo công khai với toàn bộ raw profile; không có log cá nhân lưu ra đĩa mặc định.

## 6. Kiểm thử

```powershell
python -m pytest -q
```

Test data pipeline chạy không cần Torch; model tests được skip rõ nếu thiếu Torch/PyG. Có test forward/backward ba GNN, gate toàn missing, gradient history, train/save/load/predict cho ba GNN + MF/LightGCN trên toy graph, chống leakage/cạnh đảo, availability, negative sampling, metrics tính tay và policy unknown.

Đã kiểm tra tại máy hiện tại: **26 tests pass** trên CPython 3.11.9, Torch 2.10.0 CPU, PyG 2.7.0; đã chạy smoke một epoch trên CSV thật cho ba GNN, MF, LightGCN và SAGE history gate. Các cell notebook dữ liệu/baseline chạy được; training notebook mặc định tắt. Chưa kiểm thử giao diện Streamlit hoặc CUDA; smoke không phải kết quả final.

Notebook chính: `notebooks/01_kaggle_end_to_end.ipynb`. Upload code và CSV dưới quyền truy cập phù hợp; chỉ rõ project/input paths theo notebook. Notebook mặc định chưa chạy training hàng loạt để tránh dùng quota ngoài ý muốn.

## 7. Các bước nghiên cứu tiếp theo

1. Nghiệm thu protocol với thầy: đặc biệt coverage 6%, version JTH, GraphSafe và xác nhận GNN thay vì RNN.
2. Thử rolling-query availability và sensitivity snapshot/no-LLM, không đổi protocol sau khi nhìn test score.
3. Huấn luyện/tuning nhiều seed, report charts/CI, ablation theo kế hoạch; thêm neighbor sampling khi full-batch quá nặng.
4. Bổ sung reranker đa tiêu chí, metric tính khả thi trên nhãn xác thực và profiling latency p95.
5. VietJobs/MEDAJob/RECSYS và RAG sau khi kiểm tra nguồn/license/interaction labels; không tạo nhãn giả làm benchmark thật.

Tham khảo triển khai: [PyG heterogeneous learning](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html), [JTH nguồn chính thức](https://github.com/Aunsiels/JTH). Không coi các kết quả smoke hoặc baseline sơ bộ là bằng chứng mô hình đề xuất tốt hơn.
#   G N N - f o r - R C M - v 2  
 