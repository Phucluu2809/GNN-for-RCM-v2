# Kế hoạch triển khai từ đầu: hệ gợi ý việc làm bằng GNN

**Nhóm:** Huy + Phúc.  
**Ngày lập:** 09/11/2026.  
**Trạng thái:** đặc tả để bắt đầu lập trình, chưa phải báo cáo kết quả.  
**Nguồn dữ liệu chính:** các CSV hiện có trong `data/`.  
**Môi trường chính:** Kaggle Notebook GPU; Colab là phương án thay thế.  
**Tài liệu kế hoạch hiện hành:** file này; thay thế các quyết định triển khai không còn phù hợp trong `phuong_an_du_an_gnn_goi_y_viec_lam.md`.

**Tiến độ code:** đã có bản triển khai đầu tiên trong `src/jobrec/`, notebook Kaggle, baseline, ba encoder GNN, gate và demo. Xem `README.md` để chạy và đọc giới hạn. Đây chưa phải benchmark nghiên cứu hoàn tất: đang dùng full-batch, fixed-cutoff availability chỉ bao phủ khoảng 6% positive validation/test; cần nghiệm thu protocol trước khi kết luận hiệu quả.

> Người dùng ghi “3 model RNN”, nhưng đề tài và yêu cầu giảng viên tập trung vào GNN. Kế hoạch tạm chốt **ba mô hình GNN: Heterogeneous GraphSAGE, R-GCN, HGT**. RNN/GRU chỉ là hướng bổ sung để mã hóa lịch sử, không đánh tráo RNN thành GNN. Nếu thầy yêu cầu đúng ba RNN, phải xác nhận lại phạm vi trước khi code mô hình.

## 1. Tên đề tài, vấn đề và mục tiêu

**Tên đề xuất:** Nghiên cứu hệ gợi ý việc làm trên đồ thị không đồng nhất với trọng số quan hệ theo ứng viên, lịch sử tương tác và tái xếp hạng có ràng buộc.

**Bài toán:** tại thời điểm gợi ý `t`, từ hồ sơ ứng viên và lịch sử đã quan sát trước `t`, xếp hạng các việc làm trong tập ứng viên tuyển dụng hợp lệ để trả về Top-20. Không dùng thông tin tuyển dụng xảy ra sau `t` làm đầu vào.

**Vấn đề cần kiểm chứng:** mô hình chỉ dùng kỹ năng–ngành có thể thiếu thông tin kinh nghiệm, địa điểm, lương, học vấn. Điểm phân loại cao không bảo đảm Top-K tốt. Đây là giả thuyết nghiên cứu, không kết luận sẵn rằng các yếu tố bị thiếu là nguyên nhân chính.

**Mục tiêu đo được:**

1. Xây pipeline tái lập từ CSV → EDA → dữ liệu chuẩn → đồ thị → baseline → ba GNN → đánh giá.
2. So sánh ba GNN công bằng trên cùng dữ liệu và giao thức, chọn một encoder bằng validation.
3. Phát triển hybrid: encoder GNN + mô-đun ưu tiên quan hệ + đặc trưng phù hợp cặp + bộ lọc/re-ranking.
4. Đánh giá chất lượng xếp hạng, mức vi phạm, độ phủ, thời gian, bộ nhớ và tổng tham số.
5. Làm web demo có điều chỉnh ưu tiên và lưu tương tác; tách dữ liệu demo khỏi benchmark.
6. Thử dữ liệu Việt Nam nếu hoàn thành lõi; không tuyên bố hiệu quả tại Việt Nam chỉ từ benchmark Pháp.

**Không cam kết trước:** “tăng mạnh”, “giảm đáng kể”, “hài lòng cao hơn”. Chỉ kết luận sau thí nghiệm và phân tích bất định; hài lòng cần khảo sát hoặc thử nghiệm người dùng riêng.

## 2. Phân loại toàn bộ góp ý của giảng viên

Các góp ý được gửi chung cho nhiều nhóm. Cần bao phủ bằng quyết định rõ ràng, không ghép tất cả lĩnh vực vào một đề tài.

| Góp ý/yêu cầu | Quyết định cho nhóm việc làm | Bằng chứng bàn giao |
|---|---|---|
| Gợi ý Top-K việc làm, mục tiêu thực tiễn | Bắt buộc, K chính là 20 | Đặc tả đầu vào/đầu ra và demo |
| Chọn nhiều GNN, lý do chọn | Ba encoder SAGE, R-GCN, HGT | Bảng lý thuyết + benchmark |
| HeteroGNN, HG + T | Định nghĩa hetero graph; HGT là Heterogeneous Graph Transformer; không tự suy diễn “T” thành temporal | Sơ đồ kiến trúc và giải thích thuật ngữ |
| GraphSafe | Chưa có tài liệu xác định thuật toán mà thầy muốn nói; không đồng nhất với GraphSAGE | Xin tên/paper; tạm dùng tên mô-đun `ConstraintPolicy` |
| Trọng số động, mỗi user khác nhau | Gate theo ứng viên và lịch sử quá khứ, có mask thiếu dữ liệu | Công thức, log trọng số, ablation |
| Điều chỉnh theo tương tác NSD | Cập nhật state khi có sự kiện; không bắt buộc retrain GNN mỗi click | Thí nghiệm replay và demo |
| GNN cải tiến thành hybrid | GNN + content/pair features + dynamic gate + reranking | Ablation từng thành phần |
| Bổ sung kinh nghiệm, địa bàn, lương, học vấn | Bổ sung node/quan hệ khi dữ liệu hỗ trợ; không bịa trường thiếu | Schema, đồ thị G2, báo cáo missing |
| MEDAJob/medajon, RECSYS | Nguồn đối chứng có điều kiện; phải xác định đúng bộ/version/license | Dataset manifest hoặc ghi chưa có quyền truy cập |
| VietJobs, dữ liệu Việt Nam | Nhánh mở rộng nội dung và demo; cần log thật nếu muốn đánh giá cá nhân hóa | Báo cáo domain shift và nguồn nhãn |
| Dữ liệu Mỹ | JTH đang có là Pháp, không phải Mỹ; nếu bắt buộc Mỹ thì đổi/bổ sung dataset sau kiểm tra quyền truy cập | Quyết định dataset có xác nhận |
| Amazon | Benchmark miền phụ nếu thầy yêu cầu, không thay nhãn việc làm | Kết quả tách riêng, không suy diễn sang tuyển dụng |
| Nhà hàng, rating, thu thập Đà Nẵng | Thuộc nhóm nhà hàng, ngoài phạm vi nhóm này | Ghi rõ loại trừ; không bỏ sót góp ý |
| Ba loại đồ thị/đồ thị nhị thể | Chuyển thành ba mức graph ablation G0/G1/G2; không gọi heterograph nhiều loại nút là nhị thể | Sơ đồ và thống kê từng graph |
| EDA, đặc trưng, tiền xử lý, trực quan graph | Bắt buộc trước huấn luyện | Notebook và các hình EDA |
| ML baseline, chỉ số/biểu đồ | Bắt buộc | Bảng kết quả, đường học, biểu đồ Top-K |
| Cắt tỉa và hiệu suất | Neighbor/edge pruning trên train, có đối chứng chi phí–chất lượng | Pareto plot và đo peak memory |
| RAG có nút | Tùy chọn sau lõi; mô tả rõ graph retrieval, không cần tạo “nút RAG” vô nghĩa | Giải thích có dẫn nguồn thuộc tính |
| Kaggle/Colab, notebook | Kaggle chính; dùng code module chung để chạy Colab | Notebook Run All và environment lock |
| Độ phức tạp, số tham số, đánh giá triển khai | Bắt buộc, đo cả encoder và toàn hệ | Complexity note + profiling |

## 3. Kiểm kê dữ liệu thực tế trong thư mục

### 3.1. Kết quả đọc trực tiếp CSV ngày 09/11/2026

| File hiện có | Số dòng | Kết quả kiểm tra |
|---|---:|---|
| `data/candidates.csv` | 37.554 | 37.554 `candidate_id` duy nhất |
| `data/jobs (1).csv` | 6.011 | 6.011 `job_id` duy nhất; giữ nguyên tên raw |
| `data/history.csv` | 42.288 | 42.288 `application_id` duy nhất; 26.101 ứng viên và 4.116 việc có lịch sử |
| `data/dataset_card.md` | — | Ghi JTH v1.0, July 2026 |
| `data/README (1).md` | — | Chỉ ghi `JTH_DATASET`, không đủ chứng minh phiên bản |

Đã kiểm tra: không có dòng history tham chiếu candidate/job ngoài hai bảng hồ sơ. Có **41.308 cặp Candidate–Job khác nhau**, nghĩa là **980 dòng vượt số cặp duy nhất**. Đây có thể là tiến trình tuyển dụng lặp lại, không tự động coi là bản ghi lỗi. Có **18.126 ứng viên chỉ có một việc duy nhất trong lịch sử**, chỉ **3.384 ứng viên có ít nhất ba việc khác nhau**. Khoảng ngày sự kiện đầu tiên của từng tiến trình: **09/03/2018–21/04/2025**.

Hệ quả: dữ liệu tương tác rất thưa, không thích hợp mặc định chia leave-one-out cho mọi ứng viên hoặc dùng RNN dài làm trọng tâm. Số người đủ train/validation/test phải được báo cáo sau split, không lấy 37.554 làm mẫu số đánh giá mặc định.

### 3.2. Thiếu dữ liệu đã đo

Quy ước kiểm tra sơ bộ: chuỗi rỗng hoặc `nan/null/none` sau trim và lowercase là thiếu; bước EDA sẽ kiểm tra thêm sentinel riêng từng cột.

| Trường | Ứng viên: số thiếu / tổng | Việc làm: số thiếu / tổng |
|---|---:|---:|
| `skills` | 4.270 / 37.554 | 132 / 6.011 |
| `years_experience` | 31.356 / 37.554 | 5.342 / 6.011 |
| `zipcode` | 15.605 / 37.554 | 156 / 6.011 |
| Lương thủ công | `actual_salary`: 35.257 / 37.554 | `salary`: 2.054 / 6.011 |
| `create_date` | 2.788 / 37.554 | 0 / 6.011 |

Hai trường bằng cấp `llm_highest_diploma` và `llm_required_lowest_diploma` không thiếu theo quy ước trên, nhưng vẫn là suy luận LLM, không phải xác minh thực tế. Missing rate thực đo khác một số con số trong dataset card; pipeline phải theo CSV và lưu audit.

### 3.3. Phiên bản và tính toàn vẹn

Nguồn chính thức hiện mô tả JTH v1.1, trong khi card cục bộ ghi v1.0. **Chưa xác nhận bản CSV cục bộ là v1.0 hay v1.1** chỉ bằng số dòng. Không tự đổi nhãn thành v1.1. Đối chiếu checksum/metadata phát hành khi có thể, ghi `local_unverified` cho tới lúc xác minh. Nguồn: [JTH chính thức](https://github.com/Aunsiels/JTH).

SHA-256 hiện tại để đóng băng dữ liệu thí nghiệm:

| File | SHA-256 |
|---|---|
| `candidates.csv` | `4328ceb1678757ce19b62105e97cca70759f38c332bef13cf9746badbbf3f692` |
| `jobs (1).csv` | `5bcd9e4379e558b98418e2242d321757612a96887b934e0d3ece6eb4a9952628` |
| `history.csv` | `0348bb0017156a91fe40fdd4346846982587940a0796dae3941da55312ade2a3` |

Không sửa raw, không đổi tên file người dùng. Config ánh xạ `jobs_path: data/jobs (1).csv`; file chuẩn hóa ghi sang `data/processed/`.

### 3.4. Giới hạn phải tuân thủ

- `actual_salary` là lương hiện tại, không phải lương mong muốn; không tạo điều kiện “lương tối thiểu của user” từ trường này.
- Tách lương năm và daily rate, tiền tệ, đơn vị; không tự coi `salary` là cận trên của một khoảng lương.
- `zipcode` là mã địa bàn Pháp đã làm thô, đọc dạng chuỗi; cùng mã không có nghĩa là gần nhà. Không tính kilomet từ mã này.
- Thiếu điều kiện mong muốn về địa điểm, lương, mức bắt buộc; các slider do người dùng nhập trong demo là dữ liệu bổ sung, không phải sở thích thật quan sát trong JTH.
- JTH có diễn tiến tuyển dụng do recruiter ghi nhận; không coi mọi stage là hành động tự nguyện của ứng viên hoặc đại diện hoàn toàn cho mức hài lòng.
- Không có raw CV/job description trong các CSV đang dùng. Không hứa dùng BERT trên văn bản gốc chưa có; content ban đầu là kỹ năng và thuộc tính có cấu trúc.
- Không có dữ liệu phúc lợi đủ xác thực; để mục này là extension, không bịa node/cạnh.
- Không dùng giới tính, quốc tịch, nhóm tuổi, trường top làm đầu vào xếp hạng mặc định; không thử tái định danh. Audit công bằng chỉ khi có cơ sở đạo đức/pháp lý và cỡ mẫu phù hợp.
- Chỉ có profile snapshot, chưa có lịch sử chỉnh sửa hồ sơ: temporal split vẫn không bảo đảm tái dựng được mọi thuộc tính ở quá khứ. Báo cáo rõ hạn chế và chạy sensitivity chỉ dùng thuộc tính ít biến động/không dùng LLM.

## 4. Chọn dữ liệu và các cổng quyết định

| Nguồn | Vai trò | Điều kiện trước khi đưa vào thí nghiệm |
|---|---|---|
| JTH cục bộ | Benchmark chính, bắt đầu ngay | Manifest, kiểm tra version/license, schema và split |
| MEDAJob | Đối chứng việc làm nếu được cung cấp | Có link/file chính xác, schema user/job/interaction/timestamp, giấy phép; chưa xác minh được nguồn chính thức qua tìm kiếm lần này |
| RECSYS do thầy cung cấp | Đối chứng theo notebook/dataset cụ thể | Không mặc định chữ RECSYS là tên một dataset; lấy notebook và nguồn gốc |
| XING RecSys Challenge 2016 | Đối chứng job recommendation khác | Kiểm tra khả năng tải và điều khoản; không cam kết tải được ngay |
| VietJobs | Job content Việt Nam và demo cold-start | Kiểm tra schema/license dữ liệu thực tải; không mặc định có user–job log |
| Amazon Reviews 2023 | Kiểm chứng phụ về cá nhân hóa nếu cần | Chọn một category phù hợp tài nguyên, timestamp và metadata; kết quả thuộc miền sản phẩm |

Nguồn kiểm tra: [XING Challenge](https://2016.recsyschallenge.com/), [schema XING](https://github.com/recsyschallenge/2016/blob/master/TrainingDataset.md), [VietJobs](https://github.com/VinNLP/VietJobs), [Amazon Reviews](https://mcauleylab.ucsd.edu/public_datasets/amazon_demo/).

**Quyết định hiện tại:** không tải thêm dữ liệu lớn vì JTH đã đủ khởi động pipeline. VietJobs mô tả corpus tin tuyển dụng, chưa cung cấp bằng chứng về log ứng viên trong nguồn đã kiểm tra; vì vậy chưa đủ làm benchmark collaborative cá nhân hóa. Không ghép ID ứng viên Pháp với tin Việt Nam để tạo tương tác giả. Nhãn sinh bằng quy tắc chỉ phục vụ smoke test hoặc thí nghiệm mô phỏng được gắn nhãn riêng.

**Nhánh Việt Nam:** chuẩn hóa kỹ năng song ngữ và địa bàn theo version danh mục; giữ VND riêng EUR, không suy ra ngưỡng lương Việt Nam từ phân phối Pháp. Nếu thu thập log thật: có đồng ý, user ẩn danh, event timestamp, job ID, hành động, impression/session; ấn định khoảng thu thập và protocol trước phân tích. Nếu chưa đủ log, chỉ báo cáo content-based/demo và khảo sát có giới hạn, không công bố Recall@20 giả.

## 5. Câu hỏi nghiên cứu và thí nghiệm trả lời

| Mã | Câu hỏi | Thí nghiệm |
|---|---|---|
| RQ1 | Đồ thị nhiều thuộc tính tốt hơn graph cơ bản không? | Cùng encoder trên G0/G1/G2 |
| RQ2 | Encoder nào hiệu quả nhất với dữ liệu đang có? | SAGE, R-GCN, HGT cùng protocol |
| RQ3 | Gate theo user có hơn trọng số chung không? | Uniform → global learned → user-conditioned |
| RQ4 | Lịch sử quá khứ có giúp trọng số thay đổi hữu ích không? | Gate không history vs có history; nhóm ít/nhiều tương tác |
| RQ5 | Hybrid và reranking cải thiện gì, đánh đổi gì? | Thêm riêng pair features, policy, reranker |
| RQ6 | Cắt tỉa có giảm chi phí mà giữ chất lượng không? | Edge pruning/fan-out sweeps với cùng tập đánh giá |
| RQ7 | Kết quả có chuyển được sang Việt Nam không? | Nhánh kiểm chứng riêng nếu có dữ liệu; không mặc định có |

## 6. EDA và tiền xử lý

### 6.1. Checklist EDA bắt buộc

1. Schema, kiểu dữ liệu, số khóa duy nhất, FK, dòng trùng hoàn toàn và tiến trình lặp cùng cặp.
2. Missing/sentinel theo cột và theo nguồn manual/LLM; kiểm tra số âm, zero, ngoại lệ lương.
3. Số tiến trình, số việc duy nhất mỗi ứng viên, bậc nút, sparsity, thành phần liên thông.
4. Timeline theo tháng, phân phối stage, funnel và sự không nhất quán timestamp/create date.
5. Phân phối skills, ngành, kinh nghiệm, địa bàn, học vấn, lương và loại hợp đồng.
6. Số user/item đủ điều kiện mỗi split; warm/cold user/item; tỷ lệ thuộc tính unseen.
7. Số quan hệ có dữ liệu trên mỗi user; tỷ lệ lịch sử có thể dùng cho gate động.

**Biểu đồ EDA:** missing bar/heatmap; histogram kinh nghiệm; boxplot lương đúng đơn vị; Top-20 skills/ngành/địa bàn; degree log-scale; timeline; funnel stage; heatmap cạnh theo loại; histogram unique jobs/user; warm/cold split bar. Trực quan subgraph nhỏ và tổng số nút/cạnh, không vẽ toàn bộ graph thành khối không đọc được.

### 6.2. Hợp đồng tiền xử lý

- Giữ raw bất biến; ghi manifest gồm checksum, mapping đường dẫn, thời điểm audit, nguồn/license/version.
- ID và zipcode đọc string; parse date có báo lỗi, không âm thầm ép ngày lỗi thành hợp lệ.
- Tách danh sách kỹ năng bằng `;`, trim, chuẩn hóa tên/alias; giữ provenance manual/LLM và mask thiếu.
- `_rare_skill_` không coi là một kỹ năng chung thật để nối hàng nghìn người; loại khỏi semantic edges và giữ cờ rare/missing.
- Chia thời gian trước khi fit scaler, vocabulary, alias học từ corpus, salary bins và encoder dữ liệu. Vocabulary validation/test lạ dùng unknown/hash đã định nghĩa.
- Kinh nghiệm lưu interval `[lower, upper]`; nhóm mở như `+6` có upper vô hạn. Không biến khoảng thành số năm chính xác.
- Lương thiếu dùng imputation fit trên train + missing indicator; không để giá trị điền trở thành điều kiện cứng.
- Education có bảng ánh xạ và unknown; không khẳng định thứ tự văn bằng chưa được đối chiếu.
- Chuẩn hóa manual/LLM riêng; giữ source/confidence. Thí nghiệm no-LLM bắt buộc.
- Không dùng `last_stage_reached`, stage tương lai, `application_source` của mẫu đích hay regression/match features chưa chứng minh khả dụng lúc gợi ý làm feature.

**Đầu ra:** `manifest.json`, `audit.json`, `events.parquet`, `candidates.parquet`, `jobs.parquet`, vocab/scalers, mappings và split IDs; không ghi thông tin định danh ra báo cáo công khai.

## 7. Nhãn, thời gian và chống rò rỉ

### 7.1. Hai tác vụ tách biệt

**Tác vụ chính — new-pair recommendation:** mỗi cặp Candidate–Job chỉ lấy lần bắt đầu đầu tiên làm positive. Với một tiến trình, `event_time` là min các ngày stage thực có, không dùng `end_of_process_date` làm ngày bắt đầu. Giữ các application khác trong event log để phân tích, không nhân bản nhãn cùng cặp qua train/test.

**Tác vụ phụ — funnel relevance:** sau khi pipeline nhị phân ổn định, đánh giá mức độ tiến sâu trong funnel với thời hạn quan sát cố định (đề xuất 90 ngày, sensitivity 30/180). Chỉ dùng outcome trong horizon, loại hoặc đánh dấu censor các mẫu chưa đủ theo dõi; không lấy stage cuối toàn bộ dữ liệu làm relevance tại mọi thời điểm. Cấp relevance phải chốt trước chạy: bắt đầu/shortlist = 1, qualification/resume sent = 2, interview = 3, offer = 4, accepted = 5. Đây là proxy nghiệp vụ tuyển dụng, không nhãn hài lòng.

### 7.2. Split chính

1. Từ bảng cặp duy nhất, xác định hai cutoff toàn cục theo khoảng 70%/15%/15% sự kiện theo thời gian; dời cutoff để các sự kiện cùng ngày không bị tách hai phía. Ghi ngày cutoff thật trong artifact.
2. Train trước cutoff 1; validation từ cutoff 1 đến trước cutoff 2; test từ cutoff 2 trở đi. Không random split cạnh làm kết quả chính.
3. Validation dùng graph/hành vi chỉ từ train. Sau khi chọn cấu hình, có thể fit lại trên train+validation; test dùng graph đóng băng tại cutoff 2. Mọi mô hình dùng cùng chế độ.
4. Profile/nút phải đủ điều kiện tồn tại theo `create_date` ở cutoff. Bản ghi thiếu ngày, ngày mâu thuẫn do nhiễu phải có policy riêng được log; không lấy ngày xuất hiện tương lai để đưa item vào quá khứ.
5. Báo cáo các mẫu bị loại do availability và độ phủ ground truth; không âm thầm bỏ positive không đánh giá được.
6. Gate history chỉ nhận sự kiện trước cutoff tương ứng, kể cả khi tiến trình bắt đầu trước cutoff nhưng các stage sau chưa xảy ra. Không gán cạnh accepted cho thời điểm mới shortlist.
7. Báo cáo full cohort, warm-user, cold-user và cold-item theo định nghĩa đã chốt. Không ép mọi user có ít nhất ba tương tác vì sẽ loại phần lớn dữ liệu.

**Giới hạn:** thời gian JTH có nhiễu và snapshot profile không có version lịch sử. Thử sensitivity khoảng đệm 7/14 ngày quanh cutoff; báo cáo số mẫu bị bỏ, không tuyên bố split này tái hiện chính xác online.

### 7.3. Candidate set và negative sampling

- Candidate set chính: mọi job có `create_date` hợp lệ trước cutoff, trừ job user đã tương tác trước cutoff. Không dùng label tương lai để chọn một tập âm “đẹp”.
- Chưa có ngày đóng tin đáng tin cậy: gọi đây là **catalog đã biết**, không gọi là mọi tin còn đang tuyển. Phân tích thêm cửa sổ tuổi tin 90/180/365 ngày, báo cáo positive coverage.
- Chạy full-catalog chunked scoring vì catalog khoảng 6.011 job; nếu phải sampled evaluation, xuất bảng riêng với số âm và seed cố định, không so trực tiếp với full-catalog.
- Train khởi đầu BPR, mỗi positive lấy 5 job chưa quan sát trong catalog có sẵn tại thời điểm train; loại positives đã biết ở train, không nhìn test để lọc âm. Chưa tương tác là unlabeled, không phải nhãn không phù hợp chắc chắn.
- Khi tính loss theo minibatch cạnh, xóa cạnh mục tiêu và cạnh đảo tương ứng khỏi graph message passing của minibatch. Nếu loader không làm tự động, tạo disjoint supervision/message edges có kiểm soát và áp dụng giống nhau cho ba GNN; không chỉ tin `LinkNeighborLoader` sẽ tự chống leakage.
- Candidate set, cutoff, metric implementation, seeds và negatives phải dùng chung; không lọc riêng candidate set theo từng mô hình.

## 8. Biểu diễn đồ thị: ba mức để đối sánh

| Graph | Thành phần | Mục đích |
|---|---|---|
| G0 | Candidate–Job interaction hai phía | Đối chứng collaborative/LightGCN |
| G1 | G0 + Skill + Industry/JobCategory | Mô phỏng thiết kế kỹ năng–ngành cơ bản |
| G2 | G1 + ExperienceBucket + Location + SalaryBucket + Education | Kiểm tra lợi ích mở rộng thuộc tính |

G0 là bipartite; G1/G2 là heterogeneous graph, không còn là đồ thị chỉ hai loại nút. Ba graph không thay thế ba mô hình: giữ hai trục thí nghiệm độc lập.

### 8.1. Node và edge schema G2

| Node type | Quan hệ từ Candidate | Quan hệ từ Job | Lưu ý |
|---|---|---|---|
| Skill | `has_skill` | `requires_skill` | Manual trước; LLM là nhánh ablation |
| Industry/JobCategory | `has_expertise` / `in_category` | `in_industry` / `in_category` | Không gộp industry và category tùy tiện |
| ExperienceBucket | `has_experience` | `requires_experience` | Quan hệ khác nghĩa; thiếu thì không tạo cạnh |
| Location | `located_in` | `located_in` | Không đồng nghĩa candidate thích địa bàn đó |
| SalaryBucket | `has_current_salary` | `offers_salary` | Không coi là chung một sở thích; giữ unit/source |
| Education | `has_education` | `requires_education` | Phần lớn từ LLM, không hard filter |
| Job | `interacted` | — | Cạnh chỉ từ quá khứ; reverse edge đồng thời |

Tạo reverse edges có kiểm soát. Không tạo node thiếu chung nối mọi đối tượng; dùng missing indicator. Lương annual/daily có kiểu hoặc namespace riêng. Node industry có thể dùng `expertise_area` thủ công trước, `llm_industry_domains` là nguồn bổ sung được gắn cờ. Interaction stage khác nhau là edge attributes/relations phụ khi có ablation, không trộn stage tương lai vào graph chính.

**Feature node:** vector multi-hot/hash skills, category, contract; numeric đã chuẩn hóa; missing/source masks; projection theo node type về cùng dimension. Tránh one-hot ID làm feature duy nhất. Nhánh cold-start phải có content pathway và unknown policy, không dựa vào embedding ID chưa huấn luyện.

**Trực quan:** meta-graph toàn schema, bảng số node/edge theo loại và subgraph 1–2 hop của vài ứng viên ẩn danh. Trọng số attention/gate chỉ là cơ chế mô hình, không lời giải thích nhân quả.

## 9. Ba mô hình GNN chính và baseline

### 9.1. Ba encoder cần lập trình

| Model | Triển khai dự kiến | Lý do chọn | Rủi ro cần đo |
|---|---|---|---|
| Heterogeneous GraphSAGE | `SAGEConv` theo edge type trong `HeteroConv` + residual theo node type | Mốc message passing lấy mẫu, hỗ trợ content inductive | Mean aggregation chưa có gate ưu tiên theo user |
| R-GCN | Project feature từng type, đổi sang homogeneous kèm `node_type/edge_type`, dùng `RGCNConv` | Học biến đổi riêng theo quan hệ | Bộ nhớ/tham số tăng theo số relations; thử basis decomposition |
| HGT | `HGTConv`, hidden chia hết cho số heads | Attention phụ thuộc loại nút và quan hệ | Tốn chi phí; attention không tự chứng minh ưu tiên của ứng viên |

Ban đầu 2 lớp, hidden 64, dropout 0,2; HGT 4 heads. Chung feature input, scorer ban đầu `dot(z_candidate, z_job)`, loss, split và metric. Kiến trúc có tham số khác nhau nên báo cáo cả chi phí và thử budget-matched; không khẳng định cùng hidden đồng nghĩa công bằng về số tham số.

Tham khảo API: [PyG heterogeneous tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html), [PyG operator capabilities](https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html). Kiểm tra phiên bản thực cài trước code; không giả định mọi operator đều nhận `edge_weight`.

### 9.2. Baseline bắt buộc

1. Random seeded và MostPopular từ train để phát hiện lỗi pipeline.
2. Content-only: cosine/matching trên skills + category + missing-aware pair features.
3. Matrix Factorization với BPR, không có side graph.
4. Logistic Regression trên pair features, thêm LightGBM nếu môi trường cho phép; xếp hạng bằng score, AUC không thay thế Top-K.
5. LightGCN trên G0 làm baseline collaborative graph, không tính vào ba GNN nghiên cứu chính.

Tất cả dùng đúng tập đánh giá và ngân sách tuning đã công bố. Quy tắc lọc/rerank cuối phải áp dụng đối xứng cho baseline mạnh nhất để biết cải tiến đến từ encoder hay hậu xử lý.

### 9.3. Các họ mô hình cần thảo luận trong báo cáo

GCN: mốc convolution đồng nhất; GAT: attention láng giềng; GGNN/recurrent GNN: cập nhật trạng thái kiểu GRU; STGNN: kết hợp cấu trúc và thời gian, thường cần định nghĩa không gian/thời gian phù hợp. Không chọn chúng làm ba model chính vì cần tập trung đồ thị đa quan hệ và tài nguyên nhóm. HGT là graph transformer không đồng nghĩa temporal GNN. Chỉ thử GRU-history khi có đủ chuỗi và luôn so với mean pooling để chứng minh thêm độ phức tạp có ích.

## 10. Cải tiến hybrid và định nghĩa trọng số động

### 10.1. Pipeline chung

```text
Raw CSV → audit → temporal split → train-only preprocessing → G0/G1/G2
→ baseline + SAGE/R-GCN/HGT → chọn encoder trên validation
→ relation gate + content/pair scorer → Top-M retrieval
→ ConstraintPolicy → reranking đa tiêu chí → Top-20 + giải thích
```

**Tên mô hình đề xuất nội bộ:** `Dynamic-HeteroRec`. Đây là thiết kế của nhóm, chưa phải tên thuật toán chuẩn hay tuyên bố novelty đã được xác nhận. Không đặt tên “GraphSafe” như một kiến trúc đã tồn tại nếu chưa có paper của thầy.

### 10.2. Phân biệt các loại trọng số

| Loại | Có khác theo user? | Có đổi theo history/context? | Vai trò |
|---|---|---|---|
| Uniform | Không | Không | Mốc không ưu tiên |
| Global learned | Không | Chỉ đổi khi train | Mức quan trọng chung của relation |
| User-conditioned | Có | Không nếu chỉ dùng profile | Cá nhân hóa tĩnh |
| Behavior-conditioned | Có | Có khi state quá khứ thay đổi | Trọng số động thực sự theo thời điểm |
| Preference slider | Có | Có khi user nhập | Điều khiển demo; không tự có ground truth JTH |

Trọng số quan hệ `alpha` khác attention cạnh và khác ràng buộc cứng. Kỹ năng tốt không được bù một điều kiện bắt buộc đã xác minh là vi phạm.

### 10.3. Công thức đủ cụ thể để triển khai

Với relation group `r` thuộc skill, industry/category, experience, location, salary, education, và user `u` ở thời điểm `t`:

```text
q(u,t) = concat(profile(u), pooled_history(u, events strictly before t), missing_masks)
logit_r(u,t) = MLP(q(u,t))_r
alpha(u,t) = masked_softmax(logits(u,t))
compat_r(u,j) = missing-aware pair feature or projected relation-vector similarity
score(u,j,t) = base_gnn_score(u,j) + sum_r alpha_r(u,t) * compat_r(u,j)
```

Gate đặt tại **relation-aware scorer** để áp dụng giống nhau cho encoder thắng, không cần khẳng định HGT có xuất message từng relation qua API. Phiên bản sau mới thử gate trong message aggregation bằng module riêng; phải ablate độc lập, không trình bày scorer gate như đã thay đổi convolution.

Mỗi `compat_r` có quy ước scale và mask được fit trên train. Mask các nhóm không quan sát được cho user; ở từng cặp, compatibility thiếu có giá trị trung tính kèm missing flag. Nếu mọi nhóm đều không có, dùng base score, tránh softmax toàn `-inf` thành NaN. Lương hiện tại không trở thành salary preference; thành phần salary-preference chỉ bật khi có dữ liệu mong muốn xác thực, còn đối sánh lương hiện tại được đặt tên khác và ablate riêng.

History ban đầu là mean pooling hoặc time-decayed pooling embedding các job/sự kiện đã quan sát trước cutoff; decay được tuning trên validation. Không dùng final stage chưa xảy ra. Cold user không history dùng profile và missing mask. GRU là extension nếu nhóm nhiều tương tác đủ lớn.

**Học gate:** cùng loss xếp hạng với encoder/scorer; attention/gate không có nhãn “thích gần nhà” trong JTH. Có thể học proxy dự báo tương tác, nhưng không khẳng định suy ra sở thích thật. Log phân phối alpha, entropy, mức đổi sau sự kiện và collapse về một relation; chỉ thêm regularizer khi có ablation.

**Phản hồi online demo:** sự kiện mới cập nhật history state → tính lại gate → rerank; checkpoint GNN giữ nguyên. Đó là thích nghi trạng thái, không gọi là online gradient training. Có thể gom log để retrain offline sau, với đồng ý của người dùng.

### 10.4. Bộ lọc ràng buộc an toàn

Trả về ba trạng thái `pass`, `fail`, `unknown` và lý do. Chỉ loại cứng khi điều kiện bắt buộc của user/job được xác minh, có cùng đơn vị/ngữ nghĩa và đủ bằng chứng. `unknown` không tự thành `fail`.

- Kinh nghiệm theo khoảng: chỉ chắc chắn thiếu khi cận trên kinh nghiệm ứng viên thấp hơn cận dưới yêu cầu; khoảng chồng nhau là chưa chắc. Giá trị LLM chỉ phục vụ soft scoring/audit.
- Địa điểm: “candidate đang ở X” không suy ra “chỉ chấp nhận X”; hard filter chỉ khi user khai báo phạm vi bắt buộc.
- Lương: không hard filter JTH từ `actual_salary` và một mức `salary` không rõ là max offer. Demo cần user minimum và offer range/đơn vị xác thực.
- Học vấn/remote: dữ liệu LLM không đủ làm loại cứng tự động; nêu cảnh báo nguồn và cho xác nhận.
- Test các ca khoảng mở, dữ liệu thiếu, đơn vị khác nhau và yêu cầu xung đột.

**Đánh giá ràng buộc:** JTH chủ yếu dùng compatibility/violation proxy và báo coverage đủ dữ liệu. Bộ test ưu tiên/hard constraint do nhóm tạo được ghi **synthetic scenario**, không trộn vào Recall benchmark hoặc gọi là lỗi người dùng thật.

### 10.5. Vì sao cần reranking và ghép thế nào?

GNN học tín hiệu tương tác; policy xử lý điều kiện không bù trừ; reranker kiểm soát thứ tự và đa dạng. Ba vai trò khác nhau nên ghép có lý do nhưng phải chứng minh bằng ablation.

Khởi đầu lấy Top-M với `M=200` (thử 100/200/500), lọc rồi rerank bằng tổng score đã chuẩn hóa trên validation; có thể thêm diversity penalty kiểu MMR. Tránh cộng cùng compatibility hai lần: so `base + gate` rồi rerank chỉ thêm policy/diversity với phiên bản cộng lại mọi feature. Nếu chưa đủ 20, quét tiếp catalog hoặc trả ít hơn kèm lý do; không nới điều kiện cứng âm thầm. Đo retrieval Recall@M, tỷ lệ fill Top-20 và thời gian end-to-end.

Reranker học được chỉ làm sau bản rule-based; dùng nhãn từ train/validation đúng quy trình, không train trên test.

## 11. Giao thức huấn luyện và tài nguyên

| Tham số | Giá trị khởi đầu / phương án |
|---|---|
| Seed | 42, 2026, 3407; final báo trung bình ± độ lệch chuẩn |
| Hidden/layers | 64 / 2; thử hidden 128 và lớp 1/3 sau sanity check |
| Optimizer | Adam, learning rate 0,001; thử 0,0003 |
| Dropout | 0,2; thử 0 và 0,4 nếu còn ngân sách |
| Loss | BPR cho main ranking; BCE là đối chứng, không trộn kết luận |
| Batch positives | 1.024, giảm nếu OOM |
| Negatives | 5/positive, cùng protocol cho mọi model |
| Fan-out | 15/10 mỗi hop ban đầu; kiểm tra theo edge type và GPU |
| Max epochs | 100, early stopping patience 10 bằng validation NDCG@20 |
| Checkpoint | Best validation; test không dùng chọn epoch/model/hyperparameter |

Không chạy Cartesian product mọi cấu hình. Smoke test một seed trước; sau đó cùng ngân sách tối đa 6 cấu hình/model, cuối cùng 3 seed cho cấu hình được chọn. Ghi rõ số trial thất bại/OOM và budget thực tế.

**Kaggle:** input dataset gắn bằng đường dẫn cấu hình dưới `/kaggle/input/<dataset-slug>/`, output dưới `/kaggle/working/`; không hardcode slug chưa có. Ghi GPU, VRAM, CUDA, Python, Torch, PyG và thời gian từng run. Kiểm tra tương thích wheel; lưu environment lock sau khi cài thành công. Nếu tắt Internet, chuẩn bị dependency wheels theo đúng runtime. Checkpoint định kỳ và lưu processed artifacts để resume; không giả định hạn mức GPU cố định.

**Colab:** đổi config path, không viết lại mô hình; lưu output/checkpoint vào nơi người dùng chỉ định. Local Windows dùng CPU cho test nhỏ.

## 12. Ma trận thực nghiệm tối thiểu

| Mã | Thí nghiệm | Trả lời |
|---|---|---|
| E00 | Random, Popular, Content, MF-BPR, Logistic/LightGBM, LightGCN | Mốc đối sánh |
| E01–E03 | SAGE, R-GCN, HGT trên G2, scorer và loss chung | Chọn encoder |
| E04 | Encoder thắng trên G1 so G2; G0 qua cùng adapter nếu khả thi | Tác dụng graph mở rộng |
| E05 | G2 no-LLM, manual+LLM; drop từng nhóm bốn thuộc tính | Nguồn tín hiệu và độ nhiễu |
| E06 | Uniform, global learned, profile gate, profile+history gate | Trọng số động thực sự |
| E07 | Encoder vs encoder+content/pair features | Lợi ích hybrid content |
| E08 | Base, +policy, +rerank, +policy+rerank trên cùng Top-M | Phân rã hậu xử lý |
| E09 | Baseline mạnh nhất + cùng policy/rerank | Tránh gán mọi cải tiến cho GNN |
| E10 | No pruning vs edge retention 75%/50%/25%; fan-out sweeps riêng | Chất lượng–chi phí |
| E11 | Warm/cold, ít/nhiều history, no-history vs history gate | Khả năng cá nhân hóa |
| E12 | Availability windows, cutoff gap, feature snapshot sensitivity | Độ bền protocol |
| E13 | VietJobs, XING/MEDAJob hoặc Amazon nếu đủ điều kiện | Mở rộng, không chặn lõi |

Ưu tiên E00–E09 và E11; E10 tối thiểu một sweep nhỏ vì thầy yêu cầu hiệu suất; E12 tối thiểu một kiểm tra sensitivity; E13 có điều kiện. Không cần huấn luyện cả ba GNN trên mọi tổ hợp ablation.

**Cắt tỉa:** chỉ prune graph train, giữ tập query/candidate/ground truth cố định; cạnh đảo được prune cùng cạnh gốc. Không prune target label rồi đổi mẫu số. Tách graph edge pruning, neighbor sampling và model parameter pruning thành các khái niệm khác nhau; parameter pruning không nằm trong MVP. So random pruning với heuristic độ tin cậy/độ mới chỉ dựa quá khứ; đo degree bias và tỷ lệ user/item bị cô lập.

## 13. Chỉ số, biểu đồ và bảng kết quả

### 13.1. Chất lượng xếp hạng

- `Recall@K = số positive trong Top-K / tổng positive đủ điều kiện` cho từng user, sau đó macro-average.
- `Precision@K = số positive trong Top-K / K`; nếu hệ trả ít hơn K vẫn dùng K, báo thêm fill rate, không thưởng việc trả ít.
- `NDCG@K`: discount theo `log2(rank+1)`; binary relevance cho main task, graded relevance chỉ cho funnel task đã tách riêng.
- `MRR@K`: reciprocal rank positive đầu tiên trong Top-K, không có thì 0.
- K = 5, 10, **20**; chọn model chủ yếu bằng validation NDCG@20, tie-break Recall@20 và chi phí. AUC/PR-AUC/F1 chỉ bổ sung trên sampled pairs có protocol rõ.
- User không có positive đủ điều kiện không đưa vào macro ranking mean; báo riêng số lượng và lý do. Dùng chung cohort cho mọi mô hình trong từng bảng.

### 13.2. Tính khả thi và hệ thống

- `ViolationRate@20`: số job vi phạm điều kiện đã xác minh / số job được khuyến nghị có thể đánh giá điều kiện; báo riêng từng loại và overall-any-violation.
- `Assessability@20`: tỷ lệ job có đủ dữ liệu để xét điều kiện; `UnknownRate@20` là phần chưa đủ. Không thể tuyên bố an toàn nếu violation giảm chỉ vì missing tăng.
- `FillRate@20`: số kết quả thực trả / 20; `Coverage@20`: số job duy nhất được gợi ý / số job trong catalog đánh giá; thêm diversity nếu reranker tối ưu nó.
- Retrieval Recall@M trước rerank; vi phạm trước/sau lọc; Recall/NDCG trước/sau trên cùng ground truth, không loại positive bất tiện sau khi lọc.
- Thời gian preprocessing/train/epoch, thời gian scoring/rerank và latency end-to-end median/p95 trên workload cố định; peak GPU memory, model size và số tham số trainable sau khởi tạo lazy layers.
- Trung bình ± std qua 3 seed; paired bootstrap theo user để ước lượng CI chênh lệch hai hệ, nêu số mẫu/seed. Không dùng từ “có ý nghĩa thống kê” chỉ từ một lần chạy.

### 13.3. Danh sách hình kết quả bắt buộc

1. Train loss và validation NDCG@20 theo epoch, đánh dấu best checkpoint.
2. Bar Recall/NDCG/MRR@20 cho baseline và ba GNN, kèm error bars.
3. Curve metric theo K và bảng metric chính.
4. Ablation graph/features/gate/hybrid; không nhét mọi thí nghiệm vào một hình.
5. Heatmap alpha theo user và nhóm relation; ví dụ thay đổi trước/sau sự kiện đã biết.
6. Violation/unknown/fill trước và sau rerank; tách dữ liệu thật và synthetic scenario.
7. Warm/cold và nhóm lượng history; số mẫu hiện trên biểu đồ.
8. Pareto chất lượng–latency/memory/retention cho cắt tỉa.
9. Retrieval Recall@M theo M và latency end-to-end.
10. Case study thành công/thất bại, chỉ dùng ID ẩn danh và thuộc tính có nguồn.

**Schema bảng kết quả:** `run_id, dataset_hash, split_hash, graph, model, variant, seed, params, recall20, ndcg20, mrr20, coverage20, violation20, assessability20, fill20, train_seconds, p95_ms, peak_memory_mb`. Dùng null kèm lý do cho metric không có ground truth, không ghi 0 thay “không đo được”.

## 14. Độ phức tạp và đếm tham số

Ký hiệu `N` số nút, `E_r` số cạnh relation r, `R` số relation, `d` hidden, `L` số lớp, `B` batch positive, `F_l` fan-out hop l. Với neighbor sampling, số nút có thể tăng theo `B * product(F_l)`; phải đo số sampled nodes/edges thực, không chỉ viện dẫn công thức.

- SAGE kiểu mean + linear: mỗi lớp xấp xỉ phần aggregation `O(E*d)` và projection `O(N*d²)`; bản hetero nhân chi phí projection theo module/quan hệ được triển khai.
- R-GCN: phép biến đổi thông điệp dạng dense trực tiếp có chi phí cỡ `O(sum_r E_r*d²)` mỗi lớp, phụ thuộc thứ tự tối ưu transform/aggregate; tham số ma trận quan hệ xấp xỉ `R*d²`, có thể giảm bằng bases.
- HGT: tính projection theo type cộng attention/message theo cạnh và heads; không chốt một công thức chi phí duy nhất bỏ qua triển khai. Báo số type/head, sampled edges, thời gian và memory thực.
- Gate: MLP nhỏ theo user, cộng score theo số relation có mặt; re-ranking tuyến tính theo M cho score thông thường, MMR có thêm chi phí cặp tùy triển khai.
- Full-catalog dot scorer cho U user và J job có chi phí xấp xỉ `O(U*J*d)`; chunk để không giữ toàn ma trận trong VRAM.

Đếm bằng tổng `numel()` của tham số trainable **sau** một forward khởi tạo lazy modules; báo riêng encoder, gate/scorer, toàn hệ, embeddings trainable và kích thước checkpoint. LightGCN ít transformation nhưng embedding table vẫn có tham số. Nếu prune cạnh, tham số model có thể không giảm; không tuyên bố giảm parameter count từ edge pruning.

## 15. Thiết kế code và notebook từ số không

```text
GNN_for_RCM/
  data/                         # CSV raw hiện có giữ nguyên
    processed/                  # Sinh bởi pipeline, không sửa tay
  configs/
    data.yaml
    train_sage.yaml
    train_rgcn.yaml
    train_hgt.yaml
    experiments.yaml
  src/jobrec/
    data/                       # Load, audit, normalize, events, split
    graph/                      # G0/G1/G2, mappings, train-only edges
    features/                   # Node/pair features, masks, provenance
    models/                     # Baselines, SAGE, R-GCN, HGT, gate
    training/                   # Negatives, minibatch, loss, checkpoint
    evaluation/                 # Candidate sets, Top-K, bootstrap, profile
    ranking/                    # ConstraintPolicy, rerank, explanations
    inference/                  # Load artifact, recommend, history state
  notebooks/
    00_environment_and_audit.ipynb
    01_eda_and_preprocessing.ipynb
    02_split_and_graph.ipynb
    03_baselines.ipynb
    04_train_three_gnns.ipynb
    05_dynamic_hybrid_ablation.ipynb
    06_evaluation_and_pruning.ipynb
    07_demo_export.ipynb
  tests/
  app/                          # Streamlit demo dự kiến
  artifacts/                    # Manifest, splits, preprocessors, checkpoint
  reports/                      # Tables, figures, case studies
  pyproject.toml
  README.md
```

Đây là cấu trúc **dự kiến**, chưa phải các module đã được tạo. Notebook chỉ điều phối/gọi module, không sao chép ba pipeline tiền xử lý độc lập.

### 15.1. Interface phải thống nhất trước khi code model

| Thành phần | Đầu vào → đầu ra |
|---|---|
| `load_raw(config)` | File paths → bảng raw + metadata |
| `audit_data(tables)` | Raw tables → schema/missing/FK/date/duplicate report |
| `build_events(history)` | Recruitment traces → timestamped stage events + first-pair positives |
| `make_temporal_split(events, config)` | First-pair positives → cutoff, train/val/test IDs |
| `fit_preprocessors(train)` | Train-available data → vocab/scalers/bin edges |
| `build_graph(tables, cutoff, graph_level)` | Available profiles + past events → graph + mappings |
| `encode(graph, features)` | Shared graph/features → embeddings theo node type |
| `score_pairs(candidate_ids, job_ids, context)` | Pair batch → raw scores; không đọc target label |
| `evaluate(scores, candidates, positives)` | Fixed candidate set + labels → per-user và aggregate metrics |
| `recommend(profile, history, preferences, k)` | Input lúc phục vụ → job IDs, scores, lý do, unknown flags |

Checkpoint đi kèm preprocessors, node mappings, relation schema, config, version môi trường, dataset/split hashes. Không export riêng `.pt` rồi dựng lại vocab khác lúc demo.

### 15.2. Kiểm thử bắt buộc trước benchmark

- Loader đọc đúng file có dấu cách/dấu ngoặc; schema và số khóa phù hợp manifest; raw checksum không đổi.
- Event extractor không lấy end-of-process làm start; nhiều application cùng pair không xuất hiện xuyên split của main task.
- Không có sự kiện tương lai trong graph/history; reverse target edge bị loại cùng cạnh gốc khỏi minibatch messages.
- Scaler/vocab không fit trên test; availability policy, ngày lỗi/thiếu và cutoff ties có test.
- Negative sampler không lấy known train positive hoặc job chưa xuất hiện; không dùng test labels để chọn âm.
- Metric so với ví dụ Top-K tính tay: empty positives, ties, ít hơn K, multiple positives, graded relevance.
- Gate toàn missing không NaN; cold user không history; unknown categories; đơn vị lương và khoảng kinh nghiệm.
- Policy phân biệt pass/fail/unknown; không suy salary minimum hay location preference từ hồ sơ hiện tại.
- Ba model forward/backward được trên toy graph; loss hữu hạn, gradient tới gate; overfit toy dataset nhỏ trước train thật.
- Pipeline nhỏ chạy end-to-end, lưu/load checkpoint cho cùng score trong tolerance, demo dùng đúng artifact.

## 16. Web demo tương tác và RAG tùy chọn

**MVP:** Streamlit, không cần microservice phức tạp. Nhập/chọn hồ sơ; skills, experience, location, desired salary/unit, education; ưu tiên mềm và điều kiện bắt buộc hiển thị riêng. Trả Top-20, so trước/sau rerank, cho biết nguồn feature và unknown. Có view meta-graph/subgraph và biểu đồ alpha.

**Interaction log:** user/session ẩn danh, job ID, timestamp, action (view/save/apply/feedback), impression list và phiên bản model. Không gộp click demo của nhóm với JTH rồi báo như test tự nhiên. Chỉ thu thập sau thông báo và đồng ý; không hiển thị toàn bộ hồ sơ ứng viên JTH hoặc công khai raw data.

**RAG nếu còn thời gian:** lấy job attributes và graph paths đã truy xuất làm evidence, sinh giải thích có nguồn. Job node có thể liên kết chunk/source node khi có mô tả văn bản hợp pháp; không thêm “RAG node” chỉ để đủ tên. Không để LLM tự tạo kinh nghiệm/lương hoặc override điều kiện cứng. Bản không LLM dùng template là đủ MVP; RAG đánh giá groundedness/hallucination riêng, không dùng lời giải thích đẹp thay metric xếp hạng.

**Việt Nam:** demo corpus riêng, ghi rõ chế độ content/zero-shot nếu chưa có log đủ chuẩn; không giả vờ checkpoint JTH đã được chứng minh cá nhân hóa tại Việt Nam.

## 17. Lộ trình thực thi và nghiệm thu từng giai đoạn

Thời lượng đề xuất sáu tuần, điều chỉnh theo lịch nhóm. Không chuyển sang huấn luyện hàng loạt khi split/metric chưa nghiệm thu.

| Giai đoạn | Công việc code | Sản phẩm/điều kiện hoàn tất |
|---|---|---|
| Tuần 1 — dữ liệu | Scaffold, config, environment, loader, audit, event extraction, EDA | Run All audit; manifest/hash; quyết định version/duplicate/missing có log |
| Tuần 2 — protocol | Temporal split, preprocess train-only, G0/G1/G2, metrics, baselines nhẹ | Test leakage/metric pass; baseline không dùng tương lai; counts từng cohort |
| Tuần 3 — mô hình | MF/LightGCN, SAGE trước rồi R-GCN/HGT qua interface chung | Toy overfit; ba GNN một seed chạy được; checkpoint/profiling hợp lệ |
| Tuần 4 — benchmark | Tuning cùng ngân sách, ba seed, chọn encoder trên validation | Bảng baseline/ba GNN; test final được khóa, không dùng để phát triển |
| Tuần 5 — cải tiến | Gate, hybrid content, policy/rerank; ablation, cắt tỉa, sensitivity | Kết quả thành phần, costs, constraint coverage; baseline mạnh được hậu xử lý đối xứng |
| Tuần 6 — bàn giao | Freeze thiết kế, final evaluation, demo, báo cáo; Việt Nam/RAG nếu đủ điều kiện | Notebook tái chạy, bảng/hình, limitations, demo load đúng checkpoint |

**Thứ tự code ngay sau khi duyệt kế hoạch:** scaffold → `load_raw/audit_data` → event extractor → split + metrics tests → graph → baseline → SAGE → R-GCN → HGT. Chưa làm RAG/web trước khi có một baseline và đánh giá đúng.

### 17.1. Phân công đề xuất

| Thành viên | Chính | Review chéo |
|---|---|---|
| Huy | Data audit/EDA, events/split, graph builder, SAGE | Metrics, policy, cold-start/leakage |
| Phúc | Metrics/baselines, R-GCN/HGT, gate/rerank, demo | Schema, event chronology, feature provenance |
| Cả hai | Chốt protocol, chạy ablation, profiling, diễn giải kết quả và báo cáo | Mỗi thay đổi split/label phải được cả hai kiểm tra |

Nếu thời gian thiếu, bỏ RAG và dataset phụ trước; không bỏ baseline, chống leakage, ba GNN hoặc ablation cốt lõi. Nếu một model OOM, giảm budget có ghi nhận và báo kết quả/giới hạn, không giả số liệu để hoàn tất bảng.

## 18. Form theo dõi tiến độ có thể điền

| ID | Hạng mục | Phụ trách | Input | Output | Tiêu chí nghiệm thu | Trạng thái | Hạn |
|---|---|---|---|---|---|---|---|
| P01 | Environment/scaffold | Cả hai | Repo, Kaggle runtime | Config + env lock | Import/module path chạy đúng | Chưa làm | Điền |
| P02 | Audit/EDA | Huy | CSV raw | Manifest + audit + figures | Reproduce counts, FK, missing | Đã kiểm kê sơ bộ; chưa code pipeline | Điền |
| P03 | Events/split | Huy | History | Event table + split IDs | Không future/duplicate-pair leakage | Chưa làm | Điền |
| P04 | Graph/features | Huy | Train artifacts | G0/G1/G2 + mappings | Đúng schema, source/missing masks | Chưa làm | Điền |
| P05 | Metrics/baselines | Phúc | Split + candidates | Tests + baseline table | Ví dụ tính tay đúng, shared protocol | Chưa làm | Điền |
| P06 | Three GNNs | Cả hai | G2 + shared trainer | Checkpoints + logs | Forward/backward, toy overfit, equal protocol | Chưa làm | Điền |
| P07 | Dynamic hybrid | Phúc | Encoder thắng | Gate + content scorer | Không future; static vs dynamic ablation | Chưa làm | Điền |
| P08 | Policy/rerank | Phúc | Top-M + preferences | Final recommendations | Pass/fail/unknown, retrieval recall/fill | Chưa làm | Điền |
| P09 | Pruning/profiling | Cả hai | Fixed benchmark | Pareto + params/cost | Candidate/test set không đổi | Chưa làm | Điền |
| P10 | Final evaluation | Cả hai | Frozen configs | Tables + CI + figures | Test không chọn cấu hình | Chưa làm | Điền |
| P11 | Demo/report | Cả hai | Exported artifacts | Web + báo cáo | Load đúng preprocessing/checkpoint | Chưa làm | Điền |
| P12 | Việt Nam/RAG/data phụ | Cả hai | Nguồn đã xác minh | Báo cáo riêng | Không nhập nhằng nhãn thật/mô phỏng | Có điều kiện | Điền |

## 19. Tiêu chí bàn giao cuối cùng

- [ ] Tài liệu lý thuyết phân biệt GNN/RNN, heterograph/bipartite, HGT/temporal và GraphSafe/GraphSAGE.
- [ ] Data manifest/checksum, giấy phép/nguồn, version verified hoặc ghi rõ chưa verified; raw không sửa.
- [ ] EDA đầy đủ, schema đồ thị và giải thích từng thuộc tính thực có/thiếu.
- [ ] Protocol đóng băng: label, cutoff, candidate set, availability, negatives, cohorts và snapshot caveat.
- [ ] Baseline + ba GNN, shared trainer/evaluator, best checkpoint và environment tái lập.
- [ ] Hybrid/gate có định nghĩa động rõ; ablation uniform/global/profile/history và nguồn ưu tiên.
- [ ] Reranking/policy không tạo hard constraint giả từ dữ liệu thiếu/LLM; metric có assessability/fill.
- [ ] Kết quả 3 seed, bảng/biểu đồ, tham số, độ phức tạp, runtime/memory và pruning trade-off.
- [ ] Demo tương tác, có giải thích và nguồn, không rò rỉ hồ sơ raw; RAG không bắt buộc cho MVP.
- [ ] Limitations: dữ liệu Pháp, recruiter bias, sparse history, noisy dates, static profiles, inferred attributes và domain shift.
- [ ] Mọi khẳng định “cải thiện” gắn với baseline/protocol/số liệu; không hứa đạt kết quả trước thực nghiệm.

## 20. Các điểm cần xác nhận nhưng không chặn khởi động dữ liệu

1. “3 RNN” có thực sự là ba GNN như toàn bộ yêu cầu còn lại không?
2. Thầy nói “GraphSafe” là paper/thuật toán nào hay muốn GraphSAGE kết hợp constraints?
3. MEDAJob và RECSYS cụ thể là file/link/notebook nào, có được phép sử dụng không?
4. Thầy bắt buộc dữ liệu Mỹ/Việt Nam hay cho benchmark chính bằng JTH Pháp và demo Việt Nam riêng?
5. Deadline, GPU quota thực tế và phân công Huy/Phúc đã thống nhất chưa?

Mặc định vẫn bắt đầu được scaffold, audit, EDA và split draft trên dữ liệu hiện có. Chưa nên khóa thiết kế model nếu câu “RNN” là yêu cầu thực sự khác GNN.

## 21. Nguồn tham khảo đã kiểm tra

- [JTH repository và README chính thức](https://github.com/Aunsiels/JTH): nguồn đối chiếu phiên bản/schema; card cục bộ có nhãn phiên bản khác.
- [PyG heterogeneous graph learning](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html): hướng triển khai graph nhiều loại nút/cạnh.
- [PyG operator capabilities](https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html): kiểm tra giới hạn API trước thiết kế weighted propagation.
- [VietJobs repository](https://github.com/VinNLP/VietJobs): nguồn tin tuyển dụng Việt Nam, không mặc định là tương tác ứng viên.
- [XING RecSys Challenge 2016](https://2016.recsyschallenge.com/) và [schema chính thức](https://github.com/recsyschallenge/2016/blob/master/TrainingDataset.md): nguồn đối chứng có điều kiện truy cập.
- [Amazon Reviews, McAuley Lab](https://mcauleylab.ucsd.edu/public_datasets/amazon_demo/): nguồn benchmark sản phẩm phụ, không thay thế bài toán việc làm.

Các con số ở mục 3 là kết quả đọc file cục bộ, không sao chép từ số liệu tham khảo trên mạng. Các cấu hình, công thức gate, lịch trình và tiêu chí nghiệm thu là **đề xuất triển khai của nhóm**, chưa được thực nghiệm xác nhận.
