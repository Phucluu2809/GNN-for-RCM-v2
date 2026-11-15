import json
from collections import Counter
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jobrec.data import read_json
from jobrec.features import GROUPS, attributes, compatibility, tokens
from jobrec.inference import Predictor


st.set_page_config(page_title="JobMatch | Tìm việc thông minh", page_icon="💼", layout="wide", initial_sidebar_state="expanded")

LABELS = {
    "skill": "Kỹ năng", "industry": "Ngành nghề", "experience": "Kinh nghiệm",
    "location": "Địa điểm", "salary": "Mức lương", "education": "Học vấn",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#15213a; --navy:#101b35; --blue:#2962d6; --mint:#1ea97a; --line:#e7ebf2; }
.stApp {background:#f6f8fc;color:var(--ink);font-family:'DM Sans',sans-serif}
.block-container {max-width:1440px;padding:1.5rem 3.4rem 3rem}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0d1830 0%,#132748 100%)}
[data-testid="stSidebar"] * {color:#eff5ff!important}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] select {background:#fff!important;color:#15213a!important}
[data-testid="stSidebar"] [data-baseweb="select"] span {color:#15213a!important}
[data-testid="stSidebar"] .stButton button {background:#2f6ce5!important;border:0!important;color:white!important;font-weight:700}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:#14203a!important}
.brand {font:800 22px 'Manrope';color:#fff;margin:2px 0 2px}.brand-sub{color:#aebddb;font-size:12px;margin-bottom:22px}
.topbar {display:flex;align-items:center;justify-content:space-between;margin:0 0 20px;padding-bottom:14px;border-bottom:1px solid var(--line)}
.topbar-left {font-size:14px;font-weight:700;color:#5d6a81}.topbar-left b{color:#1a315b}.status{padding:6px 11px;background:#e8fbf4;border-radius:99px;color:#16815e;font-size:12px;font-weight:700}
.hero{background:radial-gradient(circle at 82% 0,#568cf4 0,#2856b4 29%,#142d61 78%);border-radius:22px;padding:34px 38px;color:#fff;box-shadow:0 16px 38px #214a9326;margin-bottom:20px}.hero h1{color:#fff!important;font-size:37px!important;margin:7px 0 10px}.hero p{font-size:16px;color:#e0eaff;margin:0;max-width:670px}.eyebrow{font-size:11px;letter-spacing:.14em;font-weight:800;color:#a8c6ff}.hero-stat{font-size:13px;background:#ffffff1c;border:1px solid #ffffff30;padding:9px 12px;border-radius:10px;display:inline-block;margin-top:22px;color:#f5f8ff}
.search-shell{background:white;border:1px solid #e3e9f3;box-shadow:0 9px 24px #203f7410;border-radius:18px;padding:16px 18px;margin:0 0 18px}.filter-caption{font-size:12px;color:#738099;font-weight:700;margin:0 0 -8px}
.section-label{font:800 13px 'Manrope';color:#1b2c4c;letter-spacing:.02em;text-transform:uppercase;margin:16px 0 8px}.muted{color:#71809a;font-size:13px}.chip{display:inline-block;border-radius:99px;background:#edf3ff;color:#355eaa;padding:4px 9px;font-size:12px;font-weight:600;margin:2px 5px 2px 0}.chip-green{background:#e7f8f1;color:#147958}.rank{display:inline-flex;width:29px;height:29px;align-items:center;justify-content:center;border-radius:9px;background:#edf3ff;color:#2454b3;font-weight:800;font-size:13px}.job-title{font:700 18px 'Manrope';color:#162849!important;margin:0}.job-meta{font-size:13px;color:#71809a;margin:6px 0}.reason{border-left:3px solid #3c78e6;background:#f5f8ff;padding:8px 10px;border-radius:0 8px 8px 0;color:#36527d;font-size:13px;margin-top:10px}.card-caption{font-size:12px;color:#8490a4}.kpi-card{background:white;border:1px solid #e6ebf4;border-radius:15px;padding:14px 16px;min-height:95px}.kpi-value{font:800 23px 'Manrope';color:#1a3260}.kpi-label{font-size:12px;color:#71809a}.footer{font-size:12px;color:#778399;padding:14px 0}.detail-box{background:#f8faff;border:1px solid #e6edf9;border-radius:12px;padding:12px 14px}.warning-note{background:#fff8e5;border:1px solid #ffebaf;border-radius:12px;padding:11px 13px;color:#765a08;font-size:13px}
</style>
""", unsafe_allow_html=True)


def clean_text(value):
    return str(value or "").strip()


def job_title(row):
    return clean_text(row.get("job_category")) or clean_text(row.get("expertise_area")) or f"Cơ hội nghề nghiệp {row['job_id']}"


def display_salary(row):
    value = clean_text(row.get("salary"))
    if not value:
        return "Thương lượng"
    try:
        return f"€{float(value):,.0f}/năm"
    except ValueError:
        return value


def skill_list(row, limit=5):
    return sorted(tokens(row.get("skills")))[:limit]


def fit_info(candidate, row, state):
    values, observed = compatibility(candidate, row, state)
    active = [value for value, has_value in zip(values, observed) if has_value]
    percent = round(100 * sum(active) / len(active)) if active else 0
    relations = [LABELS[group] for group, value, has_value in zip(GROUPS, values, observed) if has_value and value > 0]
    candidate_skills, job_skills = tokens(candidate.get("skills")), tokens(row.get("skills"))
    shared_skills = sorted(candidate_skills & job_skills)[:4]
    return percent, relations, shared_skills


def recommendation_reason(candidate, row, state):
    _, relations, shared = fit_info(candidate, row, state)
    if shared:
        return f"Khớp kỹ năng: {', '.join(shared)}."
    if relations:
        return f"Có tín hiệu phù hợp ở: {', '.join(relations[:3]).lower()}."
    return "Được chọn theo điểm xếp hạng trên đồ thị ứng viên – việc làm."


@st.cache_resource
def load_predictor(prepared_path, checkpoint_path):
    bundle = read_json(Path(prepared_path) / "bundle.json.gz")
    return Predictor(bundle, checkpoint_path)


def personalized_recommendations(predictor, candidate_id, candidate, saved_jobs, weights):
    original = predictor.context["candidates"][candidate_id]
    predictor.context["candidates"][candidate_id] = candidate
    try:
        return predictor.recommend(candidate_id, top_k=180, session_job_ids=saved_jobs, weights=weights)
    finally:
        predictor.context["candidates"][candidate_id] = original


default_prepared = ROOT / "artifacts" / "colab_run_01" / "prepared"
default_checkpoint = ROOT / "artifacts" / "colab_run_01" / "sage_history" / "best.pt"

st.sidebar.markdown("<div class='brand'>💼 JobMatch</div><div class='brand-sub'>Gợi ý việc làm bằng đồ thị dị thể</div>", unsafe_allow_html=True)
with st.sidebar.expander("⚙ Cài đặt demo", expanded=False):
    prepared = st.text_input("Dữ liệu đã chuẩn bị", str(default_prepared if default_prepared.exists() else ROOT / "artifacts" / "prepared"))
    checkpoint = st.text_input("Checkpoint", str(default_checkpoint if default_checkpoint.exists() else ROOT / "artifacts" / "sage" / "best.pt"))
    if st.button("Tải lại mô hình", use_container_width=True):
        load_predictor.clear()
        st.rerun()

if "prepared" not in locals():
    prepared = str(default_prepared if default_prepared.exists() else ROOT / "artifacts" / "prepared")
if "checkpoint" not in locals():
    checkpoint = str(default_checkpoint if default_checkpoint.exists() else ROOT / "artifacts" / "sage" / "best.pt")
if not Path(checkpoint).is_file() or not (Path(prepared) / "bundle.json.gz").is_file():
    st.title("JobMatch")
    st.info("Chưa tìm thấy dữ liệu hoặc checkpoint. Mở Cài đặt demo ở thanh bên để nhập đường dẫn.")
    st.stop()

try:
    predictor = load_predictor(prepared, checkpoint)
except (ValueError, OSError, RuntimeError, KeyError) as error:
    st.error(f"Không tải được mô hình: {error}")
    st.stop()

candidate_ids = predictor.graph["nodes"]["candidate"]
candidate_id = st.sidebar.selectbox("👤 Hồ sơ demo", candidate_ids, format_func=lambda item: f"Ứng viên {item}")
session_key = f"{checkpoint}:{candidate_id}"
if st.session_state.get("session_key") != session_key:
    st.session_state.session_key = session_key
    st.session_state.saved_jobs = []
    st.session_state.profile_overrides = {}
    st.session_state.recommendation_cache = {}

base_candidate = predictor.context["candidates"][candidate_id]
profile_override = st.session_state.profile_overrides.get(candidate_id, {})
candidate = {**base_candidate, **profile_override}

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Danh sách quan tâm**  \\n+{len(st.session_state.saved_jobs)} việc đã lưu")
if st.session_state.saved_jobs and st.sidebar.button("Xóa việc đã lưu", use_container_width=True):
    st.session_state.saved_jobs = []
    st.rerun()
st.sidebar.caption("Dữ liệu JTH là dữ liệu nghiên cứu từ Pháp. Các thao tác trong demo chỉ tồn tại trong phiên này.")

st.markdown("<div class='topbar'><div class='topbar-left'><b>JobMatch</b> &nbsp; / &nbsp; Không gian tìm việc cá nhân</div><div class='status'>● Mô hình GNN đang sẵn sàng</div></div>", unsafe_allow_html=True)
st.markdown("""
<div class='hero'><div class='eyebrow'>PERSONALIZED CAREER DISCOVERY</div><h1>Công việc phù hợp, có lý do rõ ràng.</h1><p>Khám phá cơ hội dựa trên đồ thị ứng viên, việc làm và các quan hệ về kỹ năng, ngành nghề, kinh nghiệm, địa điểm.</p><div class='hero-stat'>✦ Gợi ý Top-K · trọng số quan hệ động · xếp hạng lại theo hồ sơ</div></div>
""", unsafe_allow_html=True)

profile_weights = st.session_state.get("profile_weights", [0.80, 0.55, 0.60, 0.45, 0.0, 0.25])
all_jobs = predictor.context["jobs"]
locations = sorted({clean_text(row.get("zipcode")) for row in all_jobs.values() if clean_text(row.get("zipcode"))})
contracts = sorted({clean_text(row.get("contract_type")) for row in all_jobs.values() if clean_text(row.get("contract_type"))})
categories = sorted({clean_text(row.get("job_category")) for row in all_jobs.values() if clean_text(row.get("job_category"))})
popular_skills = [name for name, _ in Counter(skill for row in all_jobs.values() for skill in tokens(row.get("skills"))).most_common(80)]

st.markdown("<div class='search-shell'>", unsafe_allow_html=True)
search_a, search_b, search_c, search_d = st.columns([3.2, 1.35, 1.35, 1.1])
with search_a:
    query = st.text_input("Tìm theo vị trí, kỹ năng hoặc lĩnh vực", placeholder="Ví dụ: devops, python, cloud infrastructure", label_visibility="collapsed")
with search_b:
    location_filter = st.selectbox("Địa điểm", ["Tất cả địa điểm"] + locations, label_visibility="collapsed")
with search_c:
    contract_filter = st.selectbox("Hình thức", ["Tất cả hình thức"] + contracts, label_visibility="collapsed")
with search_d:
    top_k = st.selectbox("Hiển thị", [10, 20, 50], index=0, label_visibility="collapsed")
with st.expander("Bộ lọc nâng cao và cách sắp xếp"):
    filter_a, filter_b, filter_c = st.columns([2.4, 1.3, 1.3])
    with filter_a:
        skill_filter = st.multiselect("Kỹ năng cần có", popular_skills, placeholder="Chọn kỹ năng")
    with filter_b:
        category_filter = st.multiselect("Vị trí / ngành", categories, placeholder="Chọn lĩnh vực")
    with filter_c:
        sort_mode = st.selectbox("Sắp xếp", ["Phù hợp nhất", "Lương cao hơn", "Mới hơn trong dữ liệu"])
st.markdown("</div>", unsafe_allow_html=True)

cache_key = json.dumps({"candidate": candidate, "saved": sorted(st.session_state.saved_jobs), "weights": profile_weights}, sort_keys=True, default=str)
cache = st.session_state.recommendation_cache
if cache.get("key") != cache_key:
    with st.spinner("Đang xếp hạng cơ hội phù hợp với hồ sơ của bạn..."):
        ranking = personalized_recommendations(predictor, candidate_id, candidate, st.session_state.saved_jobs, profile_weights)
    cache["key"], cache["ranking"] = cache_key, ranking
ranking = cache["ranking"]

def matches_filters(item):
    row = all_jobs[item["job_id"]]
    haystack = " ".join([job_title(row), clean_text(row.get("expertise_area")), clean_text(row.get("skills")), clean_text(row.get("contract_type"))]).lower()
    if query.strip() and query.lower().strip() not in haystack:
        return False
    if location_filter != "Tất cả địa điểm" and clean_text(row.get("zipcode")) != location_filter:
        return False
    if contract_filter != "Tất cả hình thức" and clean_text(row.get("contract_type")) != contract_filter:
        return False
    if category_filter and clean_text(row.get("job_category")) not in category_filter:
        return False
    if skill_filter and not set(skill_filter).issubset(tokens(row.get("skills"))):
        return False
    return True

filtered_jobs = [item for item in ranking["recommendations"] if matches_filters(item)]
if sort_mode == "Lương cao hơn":
    filtered_jobs.sort(key=lambda item: float(all_jobs[item["job_id"]].get("salary") or 0), reverse=True)
elif sort_mode == "Mới hơn trong dữ liệu":
    filtered_jobs.sort(key=lambda item: clean_text(all_jobs[item["job_id"]].get("create_date")), reverse=True)

tab_jobs, tab_saved, tab_profile, tab_insights = st.tabs(["🔎 Việc làm phù hợp", "♡ Việc đã lưu", "👤 Hồ sơ của tôi", "📊 Phân tích & mô hình"])

with tab_jobs:
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.markdown(f"<div class='kpi-card'><div class='kpi-value'>{len(filtered_jobs)}</div><div class='kpi-label'>việc trong kết quả đang xem</div></div>", unsafe_allow_html=True)
    metric_b.markdown(f"<div class='kpi-card'><div class='kpi-value'>{ranking['catalog_size']:,}</div><div class='kpi-label'>việc trong danh mục mô hình</div></div>", unsafe_allow_html=True)
    metric_c.markdown(f"<div class='kpi-card'><div class='kpi-value'>{len(st.session_state.saved_jobs)}</div><div class='kpi-label'>việc bạn đang quan tâm</div></div>", unsafe_allow_html=True)
    metric_d.markdown(f"<div class='kpi-card'><div class='kpi-value'>{ranking['history_jobs']}</div><div class='kpi-label'>tương tác trong hồ sơ</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Đề xuất dành riêng cho bạn</div>", unsafe_allow_html=True)
    st.caption("Kết quả đầu ra của GNN được lọc theo tiêu chí bạn chọn. Điểm tương thích nội dung là phần kỹ năng/thông tin hồ sơ khớp với tin việc, không phải xác suất được tuyển dụng.")
    if not filtered_jobs:
        st.info("Không có việc nào trong Top-180 phù hợp với bộ lọc. Bỏ bớt kỹ năng hoặc thử từ khóa khác.")
    for rank, item in enumerate(filtered_jobs[:top_k], start=1):
        identifier = item["job_id"]
        row = all_jobs[identifier]
        fit_percent, _, _ = fit_info(candidate, row, predictor.state)
        a, b = st.columns([8, 1.5])
        with a:
            st.markdown(f"<div class='job-title'><span class='rank'>{rank}</span>&nbsp; {job_title(row).title()}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='job-meta'>📍 {clean_text(row.get('zipcode')) or 'Chưa rõ'} &nbsp; · &nbsp; 💼 {clean_text(row.get('contract_type')) or 'Linh hoạt'} &nbsp; · &nbsp; ⏳ {clean_text(row.get('years_experience')) or 'Không nêu'} &nbsp; · &nbsp; 🗓 {clean_text(row.get('create_date')) or '—'}</div>", unsafe_allow_html=True)
            st.markdown("".join(f"<span class='chip'>{skill}</span>" for skill in skill_list(row)), unsafe_allow_html=True)
            st.markdown(f"<div class='reason'>✦ {recommendation_reason(candidate, row, predictor.state)}</div>", unsafe_allow_html=True)
        with b:
            st.metric("Lương", display_salary(row))
            st.metric("Tương thích", f"{fit_percent}%")
            if identifier in st.session_state.saved_jobs:
                st.button("✓ Đã lưu", key=f"saved_{identifier}", disabled=True, use_container_width=True)
            elif st.button("♡ Lưu việc", key=f"save_{identifier}", use_container_width=True):
                st.session_state.saved_jobs.append(identifier)
                st.session_state.recommendation_cache = {}
                st.rerun()
        with st.expander("Xem chi tiết & cách mô hình đánh giá"):
            st.markdown("<div class='detail-box'>", unsafe_allow_html=True)
            details_a, details_b, details_c = st.columns(3)
            details_a.write("**Lĩnh vực**", clean_text(row.get("expertise_area")) or "Chưa nêu")
            details_b.write("**Yêu cầu học vấn**", clean_text(row.get("llm_required_lowest_diploma")) or "Chưa nêu")
            details_c.write("**Mức độ senior**", clean_text(row.get("llm_seniority_level")) or "Chưa nêu")
            values, mask = compatibility(candidate, row, predictor.state)
            relation_table = [{"Quan hệ": LABELS[group], "Khả dụng": "Có" if has_value else "Thiếu dữ liệu", "Độ khớp": f"{value:.0%}" if has_value else "—"} for group, value, has_value in zip(GROUPS, values, mask)]
            st.dataframe(relation_table, hide_index=True, use_container_width=True)
            st.caption(f"GNN ranking score: {item['score']:.4f}. Giá trị này dùng để xếp hạng, không được chuẩn hóa thành xác suất.")
            st.markdown("</div>", unsafe_allow_html=True)

with tab_saved:
    st.subheader("Việc làm bạn quan tâm")
    if not st.session_state.saved_jobs:
        st.info("Bạn chưa lưu việc nào. Ở tab Việc làm phù hợp, bấm “Lưu việc” để thêm vào danh sách này.")
    else:
        for identifier in st.session_state.saved_jobs:
            row = all_jobs[identifier]
            col_a, col_b = st.columns([8, 1])
            with col_a:
                st.markdown(f"### {job_title(row).title()}")
                st.caption(f"📍 {clean_text(row.get('zipcode')) or 'Chưa rõ'} · {display_salary(row)} · {clean_text(row.get('contract_type')) or 'Linh hoạt'}")
                st.markdown("".join(f"<span class='chip chip-green'>{skill}</span>" for skill in skill_list(row)), unsafe_allow_html=True)
            with col_b:
                if st.button("Bỏ lưu", key=f"remove_{identifier}", use_container_width=True):
                    st.session_state.saved_jobs.remove(identifier)
                    st.session_state.recommendation_cache = {}
                    st.rerun()
        st.caption("Danh sách này chỉ lưu trong phiên trình duyệt hiện tại; không được ghi vào dữ liệu huấn luyện.")

with tab_profile:
    st.subheader("Cá nhân hóa hồ sơ tìm việc")
    st.caption("Chọn thông tin gần với mục tiêu hiện tại để điều chỉnh phần điểm tương thích và xếp hạng lại danh sách cho phiên này.")
    skill_vocab = sorted({value for row in all_jobs.values() for value in tokens(row.get("skills"))})
    industry_vocab = sorted({value.removeprefix("area:").removeprefix("category:") for row in all_jobs.values() for value in attributes(row, "job", predictor.state["use_llm"], predictor.state["salary_bounds"])["industry"]})
    with st.form(f"profile_form_{candidate_id}"):
        left, right = st.columns(2)
        with left:
            current_industry = tokens(candidate.get("expertise_area")) | tokens(candidate.get("job_category"))
            selected_industry = st.multiselect("Vị trí / ngành nghề bạn quan tâm", industry_vocab, default=[item for item in sorted(current_industry) if item in industry_vocab], placeholder="Ví dụ: development, data, marketing")
            current_skills = tokens(candidate.get("skills"))
            selected_skills = st.multiselect("Kỹ năng của bạn", skill_vocab, default=[item for item in sorted(current_skills) if item in skill_vocab], placeholder="Ví dụ: python, sql, communication")
        with right:
            experience_options = ["Chưa có kinh nghiệm", "Dưới 1 năm", "1–2 năm", "3–5 năm", "Trên 5 năm"]
            selected_experience = st.selectbox("Kinh nghiệm", experience_options)
            current_location = clean_text(candidate.get("zipcode"))
            location_options = ["Không giới hạn"] + locations
            location_default = location_options.index(current_location) if current_location in location_options else 0
            selected_location = st.selectbox("Khu vực làm việc", location_options, index=location_default)
            selected_education = st.text_input("Trình độ học vấn", value=clean_text(candidate.get("llm_highest_diploma")), placeholder="Ví dụ: Bachelor")
        st.markdown("**Mức ưu tiên khi xếp hạng lại**")
        weights_cols = st.columns(6)
        new_weights = []
        for col, group, value in zip(weights_cols, GROUPS, profile_weights):
            with col:
                new_weights.append(st.slider(LABELS[group], 0.0, 1.0, value, 0.05, key=f"priority_{group}"))
        submitted = st.form_submit_button("Lưu hồ sơ & cập nhật gợi ý", type="primary", use_container_width=True)
    if submitted:
        update = dict(base_candidate)
        update["skills"] = ";".join(selected_skills)
        update["expertise_area"] = ";".join(selected_industry)
        update["job_category"] = ";".join(selected_industry)
        update["years_experience"] = selected_experience
        update["zipcode"] = "" if selected_location == "Không giới hạn" else selected_location
        update["llm_highest_diploma"] = selected_education
        st.session_state.profile_overrides[candidate_id] = update
        st.session_state.profile_weights = new_weights
        st.session_state.recommendation_cache = {}
        st.rerun()
    if candidate_id in st.session_state.profile_overrides:
        if st.button("Khôi phục hồ sơ dữ liệu gốc"):
            st.session_state.profile_overrides.pop(candidate_id, None)
            st.session_state.profile_weights = [0.80, 0.55, 0.60, 0.45, 0.0, 0.25]
            st.session_state.recommendation_cache = {}
            st.rerun()
    st.markdown("<div class='section-label'>Tóm tắt hồ sơ đang dùng</div>", unsafe_allow_html=True)
    summary_cols = st.columns(3)
    summary_cols[0].write("**Kỹ năng**", clean_text(candidate.get("skills")) or "Chưa có dữ liệu")
    summary_cols[1].write("**Mục tiêu nghề nghiệp**", clean_text(candidate.get("job_category")) or "Chưa có dữ liệu")
    summary_cols[2].write("**Khu vực / kinh nghiệm**", f"{clean_text(candidate.get('zipcode')) or 'Không giới hạn'} · {clean_text(candidate.get('years_experience')) or 'Chưa nêu'}")
    st.markdown("<div class='warning-note'>Mô hình GNN đã huấn luyện vẫn dùng embedding hồ sơ gốc. Phần bạn chỉnh ở đây được áp dụng ở tầng tương thích và xếp hạng lại, giúp minh họa trọng số quan hệ động trong demo.</div>", unsafe_allow_html=True)

with tab_insights:
    summary = predictor.graph["summary"]
    st.subheader("Phân tích gợi ý & mô hình")
    st.caption(f"Graph {summary['graph_level']} · cut-off {summary['cutoff']} · {summary['feature_dim']} chiều đặc trưng · trọng số: {ranking['mode']}")
    node_cols = st.columns(min(6, len(summary["nodes"])))
    for col, (name, count) in zip(node_cols, summary["nodes"].items()):
        col.metric(f"Nút {name}", f"{count:,}")
    st.markdown("<div class='section-label'>Kỹ năng nổi bật trong các việc được đề xuất</div>", unsafe_allow_html=True)
    top_job_rows = [all_jobs[item["job_id"]] for item in ranking["recommendations"][:30]]
    demand = Counter(skill for row in top_job_rows for skill in tokens(row.get("skills")))
    candidate_skills = tokens(candidate.get("skills"))
    insight_a, insight_b = st.columns(2)
    with insight_a:
        st.bar_chart({"Số việc": dict(demand.most_common(12))})
    with insight_b:
        missing = [skill for skill, _ in demand.most_common(15) if skill not in candidate_skills]
        st.markdown("**Gợi ý bổ sung kỹ năng**")
        st.write(" · ".join(missing[:10]) if missing else "Hồ sơ đã khớp với các kỹ năng nổi bật trong Top-K.")
        st.caption("Đây là thống kê từ các tin trong Top-30 gợi ý, không phải khuyến nghị đào tạo chính thức.")
    st.markdown("<div class='section-label'>Các quan hệ trong đồ thị dị thể</div>", unsafe_allow_html=True)
    edge_rows = [{"Quan hệ": relation.replace("|", " → "), "Số cạnh": count} for relation, count in summary["edges"].items()]
    st.dataframe(edge_rows, hide_index=True, use_container_width=True)
    st.markdown("Ứng viên ↔ việc làm qua tương tác quá khứ; mỗi bên còn nối với nút kỹ năng, ngành nghề, kinh nghiệm, địa điểm, mức lương và học vấn ở graph G2. Sau GNN, tầng hybrid cộng điểm tương thích theo trọng số do hồ sơ chọn rồi xếp hạng Top-K.")
    metrics_path = Path(checkpoint).parent / "validation_metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        values = metrics.get("metrics", {})
        st.markdown("<div class='section-label'>Kết quả validation của checkpoint đang chạy</div>", unsafe_allow_html=True)
        st.caption(f"{metrics.get('model')} · history gate · best epoch {metrics.get('checkpoint_epoch')} · {metrics.get('users')} ứng viên hợp lệ; {metrics.get('coverage', {}).get('eligible_pairs', '—')}/{metrics.get('coverage', {}).get('target_pairs', '—')} positive pairs đủ điều kiện.")
        metric_view = [{"Chỉ số": name, "Giá trị": values[name]} for name in ("recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@5", "mrr@10", "mrr@20") if name in values]
        st.dataframe(metric_view, hide_index=True, use_container_width=True)
    baseline_path = ROOT / "artifacts" / "baseline" / "content_validation" / "metrics.json"
    if metrics_path.exists() and baseline_path.exists():
        baseline = read_json(baseline_path).get("metrics", {})
        compare = [{"Chỉ số": name, "GNN hiện tại": values.get(name), "Content baseline": baseline.get(name)} for name in ("recall@20", "ndcg@20", "mrr@20")]
        st.markdown("<div class='section-label'>Đối sánh với Content baseline</div>", unsafe_allow_html=True)
        st.dataframe(compare, hide_index=True, use_container_width=True)
        st.caption("Đây là artifact validation cùng dự án, dùng để minh họa đối sánh. Báo cáo nghiên cứu cần chạy nhiều seed và kiểm định paired bootstrap trước khi kết luận.")

st.divider()
st.markdown("<div class='footer'>JobMatch · demo nghiên cứu hệ gợi ý việc làm GNN. Dữ liệu JTH thuộc bối cảnh việc làm tại Pháp; không dùng kết quả demo để tự động quyết định tuyển dụng.</div>", unsafe_allow_html=True)
