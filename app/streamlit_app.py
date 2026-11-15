import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jobrec.data import read_json
from jobrec.features import GROUPS
from jobrec.inference import Predictor


st.set_page_config(page_title="JobRec GNN", layout="wide")
st.title("JobRec: GNN + relation weights")
st.warning("Demo nghiên cứu trên JTH Pháp. Không dùng cho quyết định tuyển dụng thực tế; không công khai hồ sơ raw.")


@st.cache_resource
def load_predictor(prepared_path, checkpoint_path):
    bundle = read_json(Path(prepared_path) / "bundle.json.gz")
    return Predictor(bundle, checkpoint_path)


prepared = st.sidebar.text_input("Prepared artifacts", str(ROOT / "artifacts" / "prepared"))
checkpoint = st.sidebar.text_input("Checkpoint", str(ROOT / "artifacts" / "sage" / "best.pt"))
if not Path(checkpoint).is_file() or not (Path(prepared) / "bundle.json.gz").is_file():
    st.info("Chạy prepare và train trước, rồi nhập đường dẫn checkpoint best.pt.")
    st.stop()
try:
    predictor = load_predictor(prepared, checkpoint)
except (ValueError, OSError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

candidate_id = st.selectbox("Candidate ID (ẩn danh)", predictor.graph["nodes"]["candidate"])
session_key = f"{checkpoint}:{candidate_id}"
if st.session_state.get("session_key") != session_key:
    st.session_state.session_key = session_key
    st.session_state.saved_jobs = []
manual = st.sidebar.checkbox("Mô phỏng ưu tiên bằng slider", False)
weights = [st.sidebar.slider(group, 0.0, 1.0, 0.5) for group in GROUPS] if manual else None
st.caption("Salary preference bị tắt: lương hiện tại của ứng viên không phải lương mong muốn. Các thuộc tính JTH chưa được xác minh để loại cứng.")
if st.button("Reset tương tác phiên"):
    st.session_state.saved_jobs = []
result = predictor.recommend(candidate_id, session_job_ids=st.session_state.saved_jobs, weights=weights)
if result["weights"]:
    st.bar_chart(result["weights"])
st.caption(f"Mode: {result['mode']} | History jobs: {result['history_jobs']} | Catalog: {result['catalog_size']}")
if predictor.model.gate_mode != "history" and not manual:
    st.info("Checkpoint này không có history gate; lưu việc chỉ đổi danh sách đã xem, không tự biến thành học trọng số động.")
for rank, recommendation in enumerate(result["recommendations"], start=1):
    identifier = recommendation["job_id"]
    row = predictor.context["jobs"][identifier]
    with st.expander(f"{rank}. Job {identifier} — score {recommendation['score']:.4f}"):
        st.write({"skills": row.get("skills"), "job_category": row.get("job_category"), "zipcode": row.get("zipcode"),
                  "years_experience": row.get("years_experience"), "constraint_status": recommendation["policy"]["status"]})
        if st.button("Lưu việc (chỉ trong phiên demo)", key=f"save:{session_key}:{identifier}"):
            st.session_state.saved_jobs.append(identifier)
            st.rerun()
st.caption("Không lưu tương tác ra đĩa hoặc đưa vào benchmark. Không retrain GNN sau mỗi lần lưu; history gate chỉ cập nhật trạng thái.")
