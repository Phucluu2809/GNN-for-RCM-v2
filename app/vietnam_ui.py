"""Vietnam-first Streamlit experience backed by the VietJobs catalogue.

This module deliberately keeps the ranking claim modest: VietJobs is a catalogue of
job advertisements, not an interaction dataset.  Results are therefore ranked from
profile/content signals and remain clearly separate from the JTH GNN research demo.
"""

from collections import Counter
from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from jobrec.vietnam import VietJobsCatalog, clean, display_category, display_location


DATA_PATH = ROOT / "data" / "vietjobs" / "VietJobs.csv"


@st.cache_resource(show_spinner="Đang nạp danh mục việc làm Việt Nam…")
def load_catalog(path_string):
    return VietJobsCatalog(path_string)


def chip(text, green=False):
    cls = "chip chip-green" if green else "chip"
    return f"<span class='{cls}'>{text}</span>"


def salary_text(job):
    value = clean(job.get("salary"))
    if value:
        return value
    low, high = job.get("salary_min"), job.get("salary_max")
    if low is not None and high is not None:
        return f"{low:g} – {high:g} triệu/tháng"
    return "Thỏa thuận"


def reason_for(result, profile):
    parts = []
    if result["shared_skills"]:
        parts.append("khớp kỹ năng " + ", ".join(result["shared_skills"][:4]))
    if profile.get("location") and profile["location"] in result["job"]["location_groups"]:
        parts.append("đúng khu vực ưu tiên")
    if profile.get("category") and result["job"]["category"] == profile["category"]:
        parts.append("đúng nhóm ngành quan tâm")
    return "; ".join(parts) if parts else "phù hợp với bộ lọc và thông tin bạn đã chọn"


def initialize_state(catalog):
    defaults = {
        "vn_saved_jobs": [],
        "vn_applied_jobs": [],
        "vn_profile": {
            "desired_role": "",
            "skills": [],
            "location": "",
            "category": "",
            "expected_salary": 0.0,
            "experience": "Chưa chọn",
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar(catalog):
    profile = st.session_state.vn_profile
    st.sidebar.markdown("<div class='brand'>🇻🇳 JobMatch Việt Nam</div><div class='brand-sub'>Tìm việc cá nhân hóa · VietJobs</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='warning-note'>Danh mục có tin tuyển dụng; chưa có click/lưu/ứng tuyển thực tế nên phần này dùng xếp hạng hồ sơ–nội dung, không gắn nhãn GNN.</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Hồ sơ đang dùng**")
    st.sidebar.caption(profile["desired_role"] or "Chưa chọn vị trí mong muốn")
    st.sidebar.caption(f"{len(profile['skills'])} kỹ năng · {profile['location'] or 'Toàn quốc'}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Hoạt động của bạn**")
    first, second = st.sidebar.columns(2)
    first.metric("Đã lưu", len(st.session_state.vn_saved_jobs))
    second.metric("Ứng tuyển", len(st.session_state.vn_applied_jobs))
    st.sidebar.caption("Dữ liệu thao tác chỉ lưu trong phiên trình duyệt này.")


def render_profile(catalog):
    profile = st.session_state.vn_profile
    st.subheader("Hồ sơ tìm việc")
    st.caption("Thông tin này dùng để xếp hạng lại kết quả trong phiên hiện tại. Có thể chọn nhanh kỹ năng, vị trí, địa điểm và mức lương mong muốn.")

    skill_options = [skill for skill, _ in catalog.skill_counts.most_common(350)]
    location_options = [""] + catalog.locations
    category_options = [""] + catalog.categories
    with st.form("vn_profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            desired_role = st.text_input("Vị trí / công việc mong muốn", value=profile["desired_role"], placeholder="Ví dụ: Data Analyst, Nhân viên kinh doanh")
            selected_skills = st.multiselect("Kỹ năng của bạn", skill_options, default=[item for item in profile["skills"] if item in skill_options], placeholder="Chọn kỹ năng")
            experience = st.selectbox("Kinh nghiệm", ["Chưa chọn", "Chưa có kinh nghiệm", "Dưới 1 năm", "1–2 năm", "3–5 năm", "Trên 5 năm"], index=["Chưa chọn", "Chưa có kinh nghiệm", "Dưới 1 năm", "1–2 năm", "3–5 năm", "Trên 5 năm"].index(profile.get("experience", "Chưa chọn")) if profile.get("experience", "Chưa chọn") in ["Chưa chọn", "Chưa có kinh nghiệm", "Dưới 1 năm", "1–2 năm", "3–5 năm", "Trên 5 năm"] else 0)
        with c2:
            current_location = profile.get("location", "")
            location = st.selectbox("Khu vực ưu tiên", location_options, index=location_options.index(current_location) if current_location in location_options else 0, format_func=lambda value: "Toàn quốc" if not value else catalog.location_names[value])
            current_category = profile.get("category", "")
            category = st.selectbox("Lĩnh vực quan tâm", category_options, index=category_options.index(current_category) if current_category in category_options else 0, format_func=lambda value: "Tất cả lĩnh vực" if not value else display_category(value))
            expected_salary = st.number_input("Mức lương tối thiểu (triệu/tháng)", min_value=0.0, max_value=300.0, value=float(profile.get("expected_salary", 0.0)), step=1.0)
        submitted = st.form_submit_button("Lưu hồ sơ & cập nhật gợi ý", type="primary", use_container_width=True)
    if submitted:
        st.session_state.vn_profile = {
            "desired_role": desired_role,
            "skills": selected_skills,
            "location": location,
            "category": category,
            "expected_salary": expected_salary,
            "experience": experience,
        }
        st.success("Đã cập nhật hồ sơ. Mở tab “Việc làm phù hợp” để xem thứ hạng mới.")


def render_job_card(result, rank, profile):
    job = result["job"]
    left, right = st.columns([8, 1.7])
    with left:
        st.markdown(f"<div class='job-title'><span class='rank'>{rank}</span>&nbsp; {job['title']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='job-meta'>📍 {job['location_display']} &nbsp;·&nbsp; 💼 {job['contract_type'] or 'Linh hoạt'} &nbsp;·&nbsp; ⏳ {job['experience'] or 'Chưa nêu'}</div>", unsafe_allow_html=True)
        skills = job["technical_skills"][:5] or job["soft_skills"][:5]
        if skills:
            st.markdown("".join(chip(item, item.casefold() in {skill.casefold() for skill in result["shared_skills"]}) for item in skills), unsafe_allow_html=True)
        st.markdown(f"<div class='reason'>✦ Lý do: {reason_for(result, profile)}.</div>", unsafe_allow_html=True)
    with right:
        st.metric("Lương", salary_text(job))
        st.metric("Độ phù hợp", f"{round(result['score'])}/100")
        if job["id"] in st.session_state.vn_saved_jobs:
            st.button("✓ Đã lưu", key=f"vn_saved_{job['id']}", disabled=True, use_container_width=True)
        elif st.button("♡ Lưu việc", key=f"vn_save_{job['id']}", use_container_width=True):
            st.session_state.vn_saved_jobs.append(job["id"])
            st.rerun()
    with st.expander("Xem mô tả, yêu cầu và quyền lợi"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Yêu cầu**")
            st.write(job["requirements"] or "Chưa có mô tả yêu cầu.")
            if job["qualifications"]:
                st.caption("Bằng cấp: " + ", ".join(job["qualifications"][:4]))
        with d2:
            st.markdown("**Mô tả công việc**")
            st.write(job["description"] or "Chưa có mô tả công việc.")
            if job["benefits"]:
                st.caption("Quyền lợi: " + ", ".join(job["benefits"][:5]))
        if job["id"] not in st.session_state.vn_applied_jobs:
            if st.button("Đánh dấu đã ứng tuyển", key=f"vn_apply_{job['id']}", type="primary"):
                st.session_state.vn_applied_jobs.append(job["id"])
                st.success("Đã đánh dấu. Đây là thao tác demo trong phiên hiện tại.")
        else:
            st.success("Bạn đã đánh dấu ứng tuyển việc này.")


def render_jobs(catalog):
    profile = st.session_state.vn_profile
    st.markdown("<div class='hero'><div class='eyebrow'>JOBMATCH · VIỆT NAM</div><h1>Tìm việc đúng hướng, có lý do rõ ràng.</h1><p>Tìm kiếm trong danh mục VietJobs và cá nhân hóa theo kỹ năng, lĩnh vực, địa điểm cùng mức lương mong muốn.</p><div class='hero-stat'>✦ Hồ sơ → kỹ năng → tin tuyển dụng → xếp hạng lại minh bạch</div></div>", unsafe_allow_html=True)
    query_col, location_col, category_col, contract_col = st.columns([2.2, 1, 1, 1])
    with query_col:
        query = st.text_input("", placeholder="Tìm theo vị trí, kỹ năng, công ty…", label_visibility="collapsed")
    with location_col:
        location = st.selectbox("", [""] + catalog.locations, format_func=lambda value: "Tất cả địa điểm" if not value else catalog.location_names[value], label_visibility="collapsed")
    with category_col:
        category = st.selectbox("", [""] + catalog.categories, format_func=lambda value: "Tất cả lĩnh vực" if not value else display_category(value), label_visibility="collapsed")
    with contract_col:
        contract = st.selectbox("", [""] + catalog.contracts, format_func=lambda value: "Tất cả hình thức" if not value else value, label_visibility="collapsed")
    with st.expander("Bộ lọc nâng cao và cách xếp hạng"):
        filter_a, filter_b = st.columns([1, 1])
        with filter_a:
            salary_minimum = st.number_input("Lương tối thiểu (triệu/tháng)", min_value=0.0, max_value=300.0, value=float(profile.get("expected_salary", 0.0)), step=1.0)
        with filter_b:
            top_k = st.selectbox("Số kết quả hiển thị", [10, 20, 30], index=0)
        st.caption("Điểm phù hợp gồm: kỹ năng (tối đa 62), mục tiêu nghề nghiệp (28), địa điểm ưu tiên (12), lĩnh vực (12) và lương. Điểm này là điểm xếp hạng nội dung, không phải xác suất được tuyển dụng.")

    results = catalog.rank(profile, query=query, location=location, category=category, contract=contract, minimum_salary=salary_minimum or None)
    a, b, c, d = st.columns(4)
    a.markdown(f"<div class='kpi-card'><div class='kpi-value'>{len(results):,}</div><div class='kpi-label'>tin phù hợp bộ lọc</div></div>", unsafe_allow_html=True)
    b.markdown(f"<div class='kpi-card'><div class='kpi-value'>{len(catalog.jobs):,}</div><div class='kpi-label'>tin trong danh mục VietJobs</div></div>", unsafe_allow_html=True)
    c.markdown(f"<div class='kpi-card'><div class='kpi-value'>{len(st.session_state.vn_saved_jobs)}</div><div class='kpi-label'>việc đã lưu</div></div>", unsafe_allow_html=True)
    d.markdown(f"<div class='kpi-card'><div class='kpi-value'>{len(profile['skills'])}</div><div class='kpi-label'>kỹ năng trong hồ sơ</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Việc làm gợi ý cho bạn</div>", unsafe_allow_html=True)
    if not results:
        st.info("Chưa tìm được việc theo bộ lọc. Hãy giảm điều kiện lương, bỏ bớt từ khóa hoặc chọn Toàn quốc.")
    for rank, result in enumerate(results[:top_k], 1):
        render_job_card(result, rank, profile)
        st.divider()


def render_saved(catalog):
    st.subheader("Việc đã lưu và đã ứng tuyển")
    saved = [catalog.by_id[item] for item in st.session_state.vn_saved_jobs if item in catalog.by_id]
    if not saved:
        st.info("Bạn chưa lưu việc nào. Tại tab “Việc làm phù hợp”, bấm “Lưu việc” để đưa một tin vào đây.")
        return
    for job in saved:
        first, second = st.columns([8, 1])
        with first:
            state = " · Đã ứng tuyển" if job["id"] in st.session_state.vn_applied_jobs else " · Đang lưu"
            st.markdown(f"### {job['title']}")
            st.caption(f"📍 {job['location_display']} · {salary_text(job)} · {display_category(job['category'])}{state}")
            st.markdown("".join(chip(skill, True) for skill in job["technical_skills"][:5]), unsafe_allow_html=True)
        with second:
            if st.button("Bỏ lưu", key=f"vn_remove_{job['id']}", use_container_width=True):
                st.session_state.vn_saved_jobs.remove(job["id"])
                st.rerun()
        st.divider()


def render_insights(catalog):
    st.subheader("Phân tích thị trường & lộ trình GNN")
    st.caption("Các biểu đồ dưới đây mô tả danh mục tin tuyển dụng VietJobs, không phải kết quả validation của mô hình GNN.")
    cat_counts = Counter(job["category"] for job in catalog.jobs if job["category"])
    city_counts = Counter(location for job in catalog.jobs for location in job["location_groups"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Lĩnh vực nhiều tin tuyển dụng**")
        st.bar_chart({"Số tin": {display_category(key): value for key, value in cat_counts.most_common(10)}})
    with right:
        st.markdown("**Địa điểm xuất hiện nhiều nhất**")
        st.bar_chart({"Số tin": dict(city_counts.most_common(10))})
    st.markdown("<div class='section-label'>Kỹ năng thường gặp</div>", unsafe_allow_html=True)
    st.bar_chart({"Số tin": dict(catalog.skill_counts.most_common(15))})
    st.markdown("<div class='section-label'>Kết nối với đề tài GNN</div>", unsafe_allow_html=True)
    st.markdown("""
    **Có thể đưa vào đồ thị:** ứng viên, việc làm, kỹ năng, lĩnh vực, địa điểm, mức lương, kinh nghiệm và học vấn.  
    **Cần bổ sung để huấn luyện GNN tại Việt Nam:** log xem tin, lưu việc, ứng tuyển/bỏ qua và thời điểm tương tác.  
    **Quy trình đề xuất:** VietJobs làm catalogue → thu thập tương tác có sự đồng ý → split theo thời gian → so sánh Popularity/Content/LightGCN/GraphSAGE/HeteroGNN → rerank theo ràng buộc cứng (địa điểm, kinh nghiệm, lương).
    """)
    st.info("Không nên tạo tương tác giả rồi báo cáo đây là kết quả GNN thực nghiệm. Chế độ “Nghiên cứu GNN JTH” ở thanh bên vẫn giữ checkpoint đã train để minh họa pipeline đồ thị có tương tác.")


def render_vietnam():
    if not DATA_PATH.exists():
        st.error(f"Không tìm thấy VietJobs.csv tại: {DATA_PATH}")
        st.code("Tải dữ liệu VietJobs vào data/vietjobs/VietJobs.csv rồi khởi động lại Streamlit.")
        return
    catalog = load_catalog(str(DATA_PATH))
    initialize_state(catalog)
    render_sidebar(catalog)
    tab_jobs, tab_saved, tab_profile, tab_insights = st.tabs(["🔎 Việc làm phù hợp", "♡ Việc đã lưu", "👤 Hồ sơ của tôi", "📊 Phân tích & mô hình"])
    with tab_jobs:
        render_jobs(catalog)
    with tab_saved:
        render_saved(catalog)
    with tab_profile:
        render_profile(catalog)
    with tab_insights:
        render_insights(catalog)
    st.divider()
    st.caption("JobMatch Việt Nam · Demo học thuật. Không dùng kết quả để tự động ra quyết định tuyển dụng.")
