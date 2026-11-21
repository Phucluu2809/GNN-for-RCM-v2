import ast
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path


CITY_LABELS = {
    "hồ chí minh": "TP. Hồ Chí Minh",
    "tp hồ chí minh": "TP. Hồ Chí Minh",
    "thành phố hồ chí minh": "TP. Hồ Chí Minh",
    "hà nội": "Hà Nội",
    "đà nẵng": "Đà Nẵng",
    "cần thơ": "Cần Thơ",
    "hải phòng": "Hải Phòng",
}

# The source contains both province/city names and district-level spellings.  The
# UI should expose a useful, short list of metropolitan areas/provinces rather
# than ~1,000 raw strings.  A job can belong to more than one area.
LOCATION_RULES = {
    "TP. Hồ Chí Minh": (
        "hồ chí minh", "thành phố hồ chí minh", "thủ đức", "bình chánh", "củ chi",
        "hóc môn", "nhà bè", "cần giờ", "quận 1", "quận 2", "quận 3", "quận 4",
        "quận 5", "quận 6", "quận 7", "quận 8", "quận 9", "quận 10", "quận 11",
        "quận 12", "gò vấp", "tân bình", "tân phú", "bình thạnh", "phú nhuận",
    ),
    "Hà Nội": (
        "hà nội", "hà đông", "ba đình", "bắc từ liêm", "nam từ liêm", "cầu giấy",
        "đống đa", "hoàng mai", "hoàn kiếm", "hai bà trưng", "long biên", "tây hồ",
        "thanh xuân", "gia lâm", "đông anh", "sóc sơn", "mê linh", "thạch thất",
        "hoài đức", "đan phượng", "thanh trì", "thường tín", "phú xuyên",
    ),
    "Đà Nẵng": ("đà nẵng",),
    "Hải Phòng": ("hải phòng",),
    "Cần Thơ": ("cần thơ",),
}


def clean(value):
    return str(value or "").strip()


def normalized(value):
    text = unicodedata.normalize("NFKC", clean(value)).casefold()
    return re.sub(r"\s+", " ", text)


def display_location(value):
    key = normalized(value)
    if key in CITY_LABELS:
        return CITY_LABELS[key]
    return clean(value).title() or "Chưa nêu địa điểm"


def display_category(value):
    return clean(value).replace("_", " ").title() or "Khác"


def location_groups(value):
    """Return human-readable broad locations represented by a raw source value."""
    key = normalized(value)
    matches = [label for label, aliases in LOCATION_RULES.items() if any(alias in key for alias in aliases)]
    if matches:
        return tuple(matches)
    parts = [part.strip() for part in re.split(r"[,;/|]", clean(value)) if part.strip()]
    labels = [display_location(part) for part in parts]
    return tuple(dict.fromkeys(labels)) or ("Chưa nêu địa điểm",)


def parse_list(value):
    text = clean(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [clean(item) for item in parsed if clean(item)]
        except (ValueError, SyntaxError):
            pass
    return [part.strip() for part in re.split(r"[;,]", text) if part.strip()]


def words(value):
    return set(re.findall(r"[\wÀ-ỹ]+", normalized(value)))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class VietJobsCatalog:
    """Vietnamese job catalogue used for content and profile-based ranking.

    VietJobs contains job advertisements only. It does not contain applicant profiles
    or applicant-job interactions, so this class intentionally does not claim to be a
    trained collaborative/GNN recommender.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.jobs = []
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                technical = parse_list(row.get("technical_skills"))
                soft = parse_list(row.get("soft_skills"))
                languages = parse_list(row.get("languages_required"))
                qualifications = parse_list(row.get("qualifications"))
                skills = technical + soft + languages
                job = {
                    "id": str(index),
                    "title": clean(row.get("job_title")),
                    "location": clean(row.get("location")),
                    "location_key": normalized(row.get("location")),
                    "category": clean(row.get("category")),
                    "contract_type": clean(row.get("contract_type")),
                    "experience": clean(row.get("experience_required")),
                    "salary": clean(row.get("salary")),
                    "salary_min": number(row.get("salary_min")),
                    "salary_max": number(row.get("salary_max")),
                    "technical_skills": technical,
                    "soft_skills": soft,
                    "languages": languages,
                    "qualifications": qualifications,
                    "benefits": parse_list(row.get("benefits")),
                    "description": clean(row.get("description")),
                    "requirements": clean(row.get("requirements_text")),
                }
                job["location_groups"] = location_groups(job["location"])
                job["location_display"] = " · ".join(job["location_groups"])
                job["skill_keys"] = {normalized(skill) for skill in skills if normalized(skill)}
                job["searchable"] = normalized(" ".join((
                    job["title"], job["location"], job["category"], job["contract_type"],
                    " ".join(skills), job["requirements"][:700], job["description"][:700],
                )))
                self.jobs.append(job)
        self.by_id = {job["id"]: job for job in self.jobs}
        location_counts = Counter(location for job in self.jobs for location in job["location_groups"])
        # Keep the dropdown practical; obscure district spellings remain reachable
        # through the keyword search in the job title/description.
        self.locations = [location for location, count in location_counts.most_common() if count >= 15]
        self.location_names = {location: location for location in self.locations}
        self.categories = sorted({job["category"] for job in self.jobs if job["category"]})
        self.contracts = sorted({job["contract_type"] for job in self.jobs if job["contract_type"]})
        self.skill_counts = Counter(skill for job in self.jobs for skill in job["skill_keys"])

    def rank(self, profile, query="", location="", category="", contract="", minimum_salary=None):
        selected_skills = {normalized(skill) for skill in profile.get("skills", []) if normalized(skill)}
        desired_words = words(profile.get("desired_role", ""))
        query_text = normalized(query)
        selected_location = clean(location)
        selected_category = clean(category)
        selected_contract = clean(contract)
        results = []
        for job in self.jobs:
            if query_text and query_text not in job["searchable"]:
                continue
            if selected_location and selected_location not in job["location_groups"]:
                continue
            if selected_category and job["category"] != selected_category:
                continue
            if selected_contract and job["contract_type"] != selected_contract:
                continue
            if minimum_salary and (job["salary_max"] is None or job["salary_max"] < minimum_salary):
                continue
            shared = selected_skills & job["skill_keys"]
            score = 0.0
            if selected_skills:
                score += 62 * len(shared) / max(1, len(selected_skills))
            if desired_words:
                title_words = words(job["title"] + " " + job["category"])
                score += 28 * len(desired_words & title_words) / max(1, len(desired_words))
            if profile.get("location") and clean(profile["location"]) in job["location_groups"]:
                score += 12
            if profile.get("category") and job["category"] == profile["category"]:
                score += 12
            if minimum_salary and job["salary_min"] is not None:
                score += min(8, job["salary_min"] / minimum_salary)
            if not selected_skills and not desired_words:
                score += (job["salary_max"] or 0) / 100
            results.append({"job": job, "score": min(100.0, score), "shared_skills": sorted(shared)})
        return sorted(results, key=lambda item: (-item["score"], -(item["job"]["salary_max"] or 0), item["job"]["title"]))

