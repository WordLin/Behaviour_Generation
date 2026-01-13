import json
import time
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timezone
import os
import requests
from bs4 import BeautifulSoup
from jsonschema import Draft202012Validator
from openai import OpenAI


try:
    import trafilatura
except ImportError:
    trafilatura = None


# ----------------------------
# 1) Load JSON Schema
# ----------------------------
def load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
class SchemaValidationFailed(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("Schema validation failed")
        self.errors = errors

def collect_schema_errors(instance: dict, schema: dict, limit: int = 50) -> list[dict]:
    validator = Draft202012Validator(schema)
    errs = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    out = []
    for e in errs[:limit]:
        out.append({
            "path": "/" + "/".join(map(str, e.path)),
            "message": e.message,
            "schema_path": "/" + "/".join(map(str, e.schema_path)),
        })
    return out

def validate_or_raise(instance: dict, schema: dict) -> None:
    errors = collect_schema_errors(instance, schema)
    if errors:
        raise SchemaValidationFailed(errors)

# ----------------------------
# 2)新增：失败落盘（建议单独 JSONL）
# ----------------------------   

def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def record_failure(failure_jsonl: str, *, url: str, candidate: str, stage: str,
                   error_type: str, error_msg: str, extra: dict | None = None):
    rec = {
        "ts": _utc_now_iso(),
        "url": url,
        "stage": stage,              # e.g. fetch/extract/mine/fast_path/llm/schema/dedup/write
        "error_type": error_type,    # Exception class
        "error_msg": error_msg,
        "candidate": candidate[:5000],
    }
    if extra:
        rec["extra"] = extra
    append_jsonl(failure_jsonl, rec)



# ----------------------------
# 2) Crawl / Fetch
# ----------------------------
def fetch_html(url: str, timeout: int = 20) -> str:
    # 生产：加 robots.txt 检查、限速、重试、ETag/Last-Modified 缓存
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "HabitCardBot/1.0"})
    r.raise_for_status()
    return r.text


# ----------------------------
# 3) Extract main text
#从一段网页 HTML 源码 中尽量抽取出“正文主文本”（main content），并返回一个纯文本字符串。
# ----------------------------
def extract_main_text(html: str, url: str) -> str:
    if trafilatura:
        downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
        #可调整参数
        if downloaded and len(downloaded) > 200:
            return downloaded

    # fallback: basic bs4
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text("\n")
    # simple cleanup
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 2]
    return "\n".join(lines)


# ----------------------------
# 4) Candidate mining (habit bullets)
#从一段已抽取出的纯文本里，快速“挖”出可能是习惯/行动建议的候选短句（habit candidates），返回 List[str]。
# ----------------------------

# -----------------------------
# Domain lexicons (EN only)
# -----------------------------
ACTION_WORDS = [
    "recommend", "recommended", "suggest", "suggested", "advice", "tip", "tips",
    "try", "aim", "should", "avoid", "limit", "reduce", "cut down", "stop",
    "swap", "switch", "replace", "substitute", "choose", "prefer", "opt for",
    "increase", "add", "include", "drink", "eat"
]

DIET_KEYWORDS = [
    "weight loss", "fat loss", "lose weight", "cut", "cutting", "calorie", "calories", "kcal",
    "sugar", "added sugar", "blood sugar", "glucose", "insulin", "glycemic", "gi", "gl",
    "carb", "carbs", "carbohydrate", "starch", "refined carbs", "whole grain", "oats", "brown rice",
    "protein", "fiber", "high fiber", "low sugar", "sugar-free", "no sugar",
    "soda", "soft drink", "sweetened beverage", "juice", "dessert", "snack", "late-night snack",
    "vegetable", "veggies", "fruit", "nuts", "yogurt", "milk", "eggs", "fish", "chicken", "lean meat",
    "salad", "rice", "bread", "pasta", "noodles", "potato", "sweet potato", "corn"
]

MEAL_WORDS = [
    "breakfast", "lunch", "dinner", "snack", "pre-meal", "post-meal", "before bed",
    "eating out", "restaurant", "meal prep"
]

NOISE_WORDS = [
    "copyright", "all rights reserved", "disclaimer", "privacy policy", "terms of service",
    "contact us", "about us", "sign in", "log in", "register", "subscribe",
    "related posts", "recommended reading", "previous", "next", "table of contents",
    "share", "cookie", "advertisement", "sponsored"
]


# -----------------------------
# Regex helpers
# -----------------------------
BULLET_RE = re.compile(
    r"^\s*(?:"
    r"[-•·*●▪◦▶►]"                     # bullets
    r"|(?:\d{1,3})[\.\)\:]"             # 1. / 1) / 1:
    r"|[（(]?\d{1,3}[)）]"              # (1) / （1）
    r")\s+"
)

# Common substitution patterns in EN (high-signal for "diet swaps")
REPLACE_RE = re.compile(
    r"(?:replace\s+.{1,40}\s+with\s+.{1,40})"
    r"|(?:swap\s+.{1,40}\s+for\s+.{1,40})"
    r"|(?:substitute\s+.{1,40}\s+for\s+.{1,40})"
    r"|(?:use\s+.{1,40}\s+instead\s+of\s+.{1,40})"
    r"|(?:choose\s+.{1,40}\s+over\s+.{1,40})",
    re.IGNORECASE
)

# Quantity units (common in actionable nutrition instructions)
QTY_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:g|gram|grams|kg|oz|lb|ml|l|cup|cups|tbsp|tsp|tablespoon|teaspoon|serving|servings|piece|pieces)\b",
    re.IGNORECASE
)

# Sentence split (EN)
SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\;\:])\s+")


def _ends_with_strong_punct(s: str) -> bool:
    return s.endswith((".", "!", "?", ";", ":"))



# -----------------------------
# 4.x) Candidate quality gate (skip obvious non-actionable lines)
# -----------------------------
HEADER_LIKE_RE = re.compile(r".*:\s*$")  # ends with ":" -> often a section header
NAV_NOISE_RE = re.compile(
    r"(skip to main content|privacy policy|terms of service|cookies|cookie|subscribe|sign in|log in|register|copyright)",
    re.IGNORECASE
)

# “不像建议”的弱信号：没有动词/建议语气、没有替换结构、没有数量单位，且句子很像标题/说明
WEAK_NON_ACTION_RE = re.compile(
    r"^(how to|what is|why|tips|advice|examples|here are|there are|sugar's many|contents|table of contents)\b",
    re.IGNORECASE
)

def should_skip_candidate(c: str) -> bool:
    """
    Return True if candidate is very likely NOT an actionable habit instruction.
    Goal: reduce noise + avoid wasting LLM calls.
    """
    if not c:
        return True
    s = c.strip()
    sl = s.lower()

    # Too short: often headings / fragments
    if len(s) < 25:
        return True

    # Navigation / boilerplate
    if NAV_NOISE_RE.search(s):
        return True

    # Ends with ":" and looks like a header
    if HEADER_LIKE_RE.match(s) and len(s) < 140:
        return True

    # Starts like a section lead-in (often not a concrete instruction)
    if WEAK_NON_ACTION_RE.search(s) and not REPLACE_RE.search(s):
        # If it's an actual advice sentence with strong action words, keep it
        if not any(w in sl for w in ACTION_WORDS):
            return True

    # If it has neither “action tone” nor “swap/replace structure”, it's likely descriptive text
    has_action_tone = any(w in sl for w in ACTION_WORDS)
    has_swap_pattern = bool(REPLACE_RE.search(s))
    has_quantity = bool(QTY_RE.search(s))  # you already defined QTY_RE

    if (not has_action_tone) and (not has_swap_pattern) and (not has_quantity):
        return True

    # Otherwise keep
    return False


def _is_bullet(s: str) -> bool:
    return bool(BULLET_RE.match(s))


def _is_noise(s: str) -> bool:
    sl = s.lower()
    return any(w in sl for w in NOISE_WORDS)


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _split_sentences(s: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s) if p and p.strip()]
    return parts if parts else [s.strip()]


def _score_candidate(s: str) -> int:
    sl = s.lower()
    score = 0

    if _is_noise(s):
        score -= 6

    # Strong structure signal
    if _is_bullet(s):
        score += 3

    # Action / advice tone
    if any(w in sl for w in ACTION_WORDS):
        score += 2

    # Domain vocabulary
    if any(k in sl for k in DIET_KEYWORDS):
        score += 1

    # Meal context
    if any(m in sl for m in MEAL_WORDS):
        score += 1

    # Diet substitution patterns
    if REPLACE_RE.search(s):
        score += 2

    # Quantified guidance
    if QTY_RE.search(s):
        score += 1

    # Extra: "avoid/reduce/limit" + key targets boosts actionability
    if any(x in sl for x in ("avoid", "reduce", "limit", "cut down", "stop")) and any(
        y in sl for y in ("sugar", "added sugar", "carb", "carbs", "calorie", "calories", "soda", "dessert", "snack")
    ):
        score += 1

    return score


def _rebuild_blocks(text: str) -> List[str]:
    """
    Rebuild noisy line-broken text into paragraph/list-item blocks:
    - Keep bullet/numbered items as standalone blocks and merge continuation lines
    - Merge non-bullet lines into paragraph-like blocks
    """
    raw_lines = [ln.strip() for ln in text.splitlines()]
    blocks: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            blocks.append(" ".join(buf).strip())
            buf = []

    i = 0
    while i < len(raw_lines):
        ln = raw_lines[i]
        if not ln:
            flush()
            i += 1
            continue

        if _is_bullet(ln):
            flush()
            item = [ln]
            i += 1
            while i < len(raw_lines):
                nxt = raw_lines[i].strip()
                if not nxt:
                    break
                if _is_bullet(nxt):
                    break

                # Merge likely continuation lines (short-ish and previous line not clearly ended)
                if len(nxt) <= 220 and not _ends_with_strong_punct(item[-1]):
                    item.append(nxt)
                    i += 1
                    continue
                break

            blocks.append(" ".join(item).strip())
            continue

        # Non-bullet: merge into paragraph blocks
        if not buf:
            buf = [ln]
        else:
            if _ends_with_strong_punct(buf[-1]):
                flush()
                buf = [ln]
            else:
                buf.append(ln)
        i += 1

    flush()
    return blocks


def mine_habit_candidates(text: str, max_candidates: int = 50) -> List[str]:
    """
    High-recall candidate mining for diet / fat-loss / sugar-control pages (EN only lexicons):
    1) Rebuild blocks to counter messy line breaks
    2) Sentence-split long blocks to catch inline advice (not only bullets)
    3) Score candidates using: bullet + action words + diet keywords + substitution patterns + quantity units
    4) De-noise, deduplicate, sort, and truncate
    """
    MIN_LEN = 10
    MAX_LEN = 420               # allow longer actionable guidance (nutrition advice often > 200 chars)
    KEEP_POOL = max(400, max_candidates * 12)

    blocks = _rebuild_blocks(text)
    scored: List[Tuple[int, str]] = []

    for blk in blocks:
        if not blk or len(blk) < MIN_LEN:
            continue

        pieces = _split_sentences(blk) if len(blk) > MAX_LEN else [blk]

        for p in pieces:
            s = p.strip()
            if len(s) < MIN_LEN:
                continue

            # If it's extremely long and not a list item, it's likely explanatory
            if len(s) > MAX_LEN and not _is_bullet(s):
                continue

            score = _score_candidate(s)

            # Lower threshold for recall, but still require meaningful signals
            # Typical passes:
            # - bullet(+3) alone passes
            # - non-bullet: action(+2)+diet(+1) or replace(+2)+diet(+1) etc.
            if score >= 3:
                scored.append((score, s))

            if len(scored) >= KEEP_POOL:
                break

        if len(scored) >= KEEP_POOL:
            break

    # Sort: higher score first; tie-break by shorter (often more actionable)
    scored.sort(key=lambda x: (-x[0], len(x[1])))

    # Dedup by normalized text
    uniq: List[str] = []
    seen = set()
    for score, s in scored:
        if _is_noise(s):
            continue
        key = _normalize(s)
        if key in seen:
            continue
        uniq.append(s)
        seen.add(key)
        if len(uniq) >= max_candidates:
            break

    return uniq


# ----------------------------
# 5) Structured extraction
#    - Fast path: 把高频、规则可识别的饮食控糖建议，直接转成可执行的 Method Template Card，降低成本
#    - Fallback: LLM to fill missing fields
#当爬虫从网页正文里召回到一条候选句（candidate）时，它尝试用非常便宜的规则把这条候选句，失败用LLM补全。仅覆盖一个 pattern family，
#只演示“替换含糖饮料”，对“主食替换、增加蛋白、控制夜宵、减少精制碳水”等不覆盖。
#未命中的走通用流程：habit_pattern_family 分类器 / LLM 信息抽取 / schema 校验 / 去重 / 入库
# ----------------------------
def _utc_now_iso() -> str:
    # ISO 8601 with trailing Z
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _md5_8(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def _contains_any(text: str, phrases: List[str]) -> bool:
    return any(p in text for p in phrases)


def _regex_any(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)



# -----------------------------
# High-frequency pattern families (EN keywords)
# -----------------------------
# 1) Replace sugary drinks (NHS emphasizes swapping sugary drinks to water/sugar-free options)
SSB_TERMS = [
    "sugary drink", "sugary drinks", "sugar-sweetened", "sweetened beverage", "soft drink",
    "soda", "cola", "fizzy drink", "energy drink", "sports drink", "sweet tea",
    "milkshake", "cordial", "fruit juice", "juice"
]
UNSWEETENED_TERMS = [
    "water", "sparkling water", "sugar-free", "diet", "no-added-sugar", "unsweetened",
    "unsweetened tea", "herbal tea", "black coffee"
]
SWAP_ACTIONS = [
    "swap", "switch", "replace", "substitute", "choose", "opt for", "go for",
    "use", "instead of", "try"
]
REPLACE_PATTERNS = [
    re.compile(r"(replace|swap|switch|substitute)\s+.{1,60}\s+(with|for)\s+.{1,60}", re.IGNORECASE),
    re.compile(r"use\s+.{1,60}\s+instead of\s+.{1,60}", re.IGNORECASE),
    re.compile(r"(choose|opt for|go for)\s+.{1,60}\s+instead of\s+.{1,60}", re.IGNORECASE),
]


# 2) Lower-sugar breakfast cereal / porridge / oats (common in sugar-reduction guidance content)
CEREAL_TERMS = [
    "breakfast cereal", "cereals", "granola", "muesli", "porridge", "oats", "porridge oats",
    "toast", "wholemeal toast", "yogurt", "natural yoghurt", "plain yoghurt"
]
LOW_SUGAR_TERMS = [
    "lower-sugar", "low sugar", "no added sugar", "no-added-sugar", "unsweetened", "plain",
    "high in sugar", "reduce sugar", "cut down sugar"
]
FRUIT_SWAP_TERMS = [
    "add fruit", "topped with fruit", "banana", "apricot", "pear", "berries", "chopped fruit",
    "instead of sugar"
]

# 3) Swap refined grains for whole grains (CDC/Harvard highlight whole grains; limit refined grains)
REFINED_GRAIN_TERMS = [
    "refined grain", "refined grains", "white bread", "white rice", "white pasta", "refined carbs",
    "refined carbohydrate", "refined carbohydrates"
]
WHOLE_GRAIN_TERMS = [
    "whole grain", "whole grains", "whole wheat", "whole-wheat", "wholemeal", "brown rice",
    "quinoa", "oats", "whole wheat pasta", "whole-grain bread"
]

# 4) Increase fiber / non-starchy vegetables (CDC emphasizes non-starchy vegetables, fiber)
VEG_FIBER_TERMS = [
    "non-starchy vegetables", "nonstarchy vegetables", "vegetables", "veggies", "salad",
    "spinach", "broccoli", "green beans", "fiber", "high fiber", "fibre", "legumes",
    "beans", "lentils"
]
PLATE_STRUCT_TERMS = [
    "half your plate", "fill half", "start with a salad", "add vegetables", "add veggies"
]

# 5) Reduce added sugars + read labels (common “hidden sugar” advice; complements pattern 1/2)
LABEL_TERMS = [
    "added sugar", "added sugars", "read the label", "nutrition label", "ingredient list",
    "check the label", "look at the label"
]


def _base_card(
    candidate: str,
    source_url: str,
    locale: str,
    behavior_key: str,
    display_name: str,
    subdomain: List[str],
    action_verb: str,
    synonyms: List[str],
    confidence: float,
    lexical_keywords: List[str],
    semantic_queries: List[str],
) -> Dict[str, Any]:
    now = _utc_now_iso()
    card_id = f"card_{behavior_key}_v1_{_md5_8(candidate)}"

    return {"schema_version": "behavior_method_card.v1",
        "card_type": "method_template",
        "card_id": card_id,
        "version": "1.0.0",
        "status": "draft",
        "domain": "metabolic_health",
        "subdomain": subdomain,
        "locale": locale,
        "created_at": now,
        "updated_at": now,

        "behavior": {
            "behavior_key": behavior_key,
            "display_name": display_name,
            "description": candidate,
            "action_verb": action_verb,
            "object_category": "nutrition",

            "dose_policy": {
                "start_dose": {"value": 1, "unit": "serving", "unit_type": "serving", "range": [1, 1]},
                "default_dose": {"value": 1, "unit": "serving", "unit_type": "serving", "range": [1, 2]},
                "max_dose": {"value": 2, "unit": "serving", "unit_type": "serving", "range": [1, 3]},
                "progression_rule": {
                    "type": "completion_based",
                    "upgrade_condition": "completion_rate_7d >= 0.7",
                    "upgrade_step": {"value": 1, "unit": "serving", "unit_type": "serving", "range": [1, 1]},
                    "downgrade_condition": "completion_rate_7d < 0.3",
                    "downgrade_step": {"value": 1, "unit": "serving", "unit_type": "serving", "range": [1, 1]},
                },
            },

            "frequency": {
                "type": "daily",
                "times_per_day": 1,
                "rrule": "FREQ=DAILY",
                "skip_conditions": ["illness", "travel"],
            },

            "intensity": "n/a",
            "perceived_exertion": 1,
            "synonyms": synonyms,
        },

        "triggers": [
            {
                "if": {
                    "cue_id": f"cue.{behavior_key}.default",
                    "cue_type": "context",
                    "cue_event": "meal_time_or_choice_point",
                    "cue_event_key": "context:meal_or_choice",
                    "cue_event_family": "choice_point",
                },
                "then": {
                    "action": behavior_key,
                    "sub_steps": ["pick_option", "execute", "log"],
                    "bind_dose_from_policy": True,
                },
                "prompt_design": {
                    "method": "push_notification",
                    "timing": "at_choice_point",
                    "content": "Use the suggested swap or portion strategy at your next meal/drink choice.",
                    "friction_reduction": [{"type": "environment", "item": "make_healthy_options_visible"}],
                },
            }
        ],

        "execution": {
            "steps": [
                {"step_id": 1, "action": "Prepare 1–2 healthier options in advance.", "time_estimate": "3min"},
                {"step_id": 2, "action": "At the choice point, apply the swap/selection rule once.", "time_estimate": "1min"},
                {"step_id": 3, "action": "Log completion (yes/no).", "time_estimate": "10s"},
            ],
            "prep_required": True,
            "prep_items": ["healthy_options"],
            "effort_level": 1,
            "cognitive_load": 1,
            "alternatives": [
                {
                    "alt_id": "alt.no_option",
                    "condition": "no_healthy_option_available",
                    "action": "Choose the lowest-sugar/least-processed available option and reduce portion.",
                    "dose": {"value": 1, "unit": "serving", "unit_type": "serving", "range": [1, 1]},
                    "effort_level": 2,
                    "barrier_tags_solved": ["no_available_option"],
                }
            ],
        },

        "mechanism_tags": {
            "taxonomy_versions": {"comb": "v1", "bcw": "2011", "bcttv1": "2013", "barrier_taxonomy": "barrier.v1"},
            "comb_targets": ["opportunity_physical", "motivation_reflective"],
            "barrier_tags_solved": ["environment_temptation", "habitual_choice"],
            "enabler_tags_used": ["availability_healthy_option", "cue_binding"],
            "prereq_tags": [],
            "bcw_functions": ["education", "environmental_restructuring", "enablement"],
            "bct_codes": [
                {"code": "7.1", "name": "Prompts/cues"},
                {"code": "12.1", "name": "Restructuring the physical environment"},
                {"code": "8.3", "name": "Habit formation"},
            ],
        },

        "measurement": {
            "behavior_metrics": [
                {"metric": "completions", "type": "count", "target": 1, "unit": "times_per_day"},
                {"metric": "completion", "type": "binary", "target": 1, "unit": "times_per_day"},
            ],
            "proxy_outcomes": [
                {"metric": "craving_level", "type": "subjective_scale", "scale": [1, 10], "expected_change": "decrease"}
            ],
            "expected_latency": {"behavior_feedback": "immediate", "proxy_outcome": "1_day", "health_outcome": "4_weeks"},
        },

        "learning": {
            "reward_signal": {"primary": "completion_rate", "secondary": ["user_satisfaction"]},
            "reward_shaping": {"completion": 1.0, "skipped": 0.0},
            "personalization_params": [{"param": "triggers[0].prompt_design.timing", "adaptive": True}],
            "failure_modes": [{"mode": "no_option", "frequency": "medium", "next_card_hint": "Increase prep/availability."}],
        },

        "retrieval": {
            "facets": {
                "behavior_key": behavior_key,
                "domain": "metabolic_health",
                "comb_targets": ["opportunity_physical", "motivation_reflective"],
            },
            "lexical": {"keywords": lexical_keywords, "aliases": []},
            "semantic": {"query_examples": semantic_queries},
        },

        "relationships": {"prerequisites": [], "next_cards": [], "parallel_cards": [], "alternative_cards": [], "fallback_cards": []},

        "x_metadata": {
            "provenance": {
                "source_url": source_url,
                "extracted_by": "fast_path_rule_mapper",
                "confidence": confidence,
                "evidence_spans": [candidate[:220]],
            }
        },
    }





def _mk_id(prefix: str, s: str) -> str:
    return f"{prefix}_{hashlib.md5(s.encode('utf-8')).hexdigest()[:10]}"


def stable_method_template_id(card: Dict[str, Any]) -> str:
    trigger = card.get("trigger") or {}
    dp = card.get("dose_policy") or {}
    default = dp.get("default") or {}
    key = {
        "schema_version": card.get("schema_version"),
        "card_type": card.get("card_type"),
        "domain": card.get("domain"),
        "locale": card.get("locale"),
        "behavior_key": card.get("behavior_key"),
        "cue": ((trigger.get("if") or {}).get("cue")),
        "action": ((trigger.get("then") or {}).get("action")),
        "dose_unit": default.get("unit"),
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return "mt_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]



def canonicalize_method_template_card(
    card: Dict[str, Any],
    *,
    candidate: str,
    source_url: str,
    locale: str,
    extracted_by: str,
) -> Dict[str, Any]:
    """
    Canonicalize BOTH fast_path and LLM outputs:
    - force required content_slots=[]
    - override id with deterministic stable id
    - normalize provenance fields
    - remove schema-hostile nulls (e.g., constraints.no_added_sugar = None)
    """
    # 1) hard requirements for MVP
    card["card_type"] = "method_template"
    card["locale"] = locale
    card["content_slots"] = []  # 强制空数组（字段 required）

    # 2) drop null boolean fields that break schema (common failure)
    constraints = card.get("constraints")
    if isinstance(constraints, dict):
        # remove keys with None
        for k in list(constraints.keys()):
            if constraints.get(k) is None:
                constraints.pop(k, None)
        if not constraints:
            card.pop("constraints", None)

    # 3) provenance normalization
    prov = card.get("provenance") or {}
    prov.setdefault("source_url", source_url)
    prov["extracted_by"] = extracted_by
    # confidence: ensure float-like
    try:
        prov["confidence"] = float(prov.get("confidence", 0.55))
    except Exception:
        prov["confidence"] = 0.55
    ev = prov.get("evidence_spans") or []
    if not isinstance(ev, list):
        ev = []
    if candidate:
        ev.append(candidate[:256])
    # uniq + cap
    seen = set()
    ev_u = []
    for s in ev:
        if not isinstance(s, str):
            continue
        if s in seen:
            continue
        seen.add(s)
        ev_u.append(s)
        if len(ev_u) >= 20:
            break
    prov["evidence_spans"] = ev_u
    card["provenance"] = prov

    # 4) stable deterministic id
    card["id"] = stable_method_template_id(card)
    return card


def fast_path_to_method_template_mvp(
    candidate: str,
    source_url: str,
    locale: str = "en_US",
) -> Optional[Dict[str, Any]]:
    """
    Fast-path: rule-based mapping remind: output MUST conform to your MVP MethodTemplateCard.
    English keyword rules targeting fat-loss / sugar-control diet pages.
    """

    s = candidate.strip()
    s_l = s.lower()

    

    # --- pattern families (English keywords) ---
    families = [
        {
            "name": "replace_sugary_beverage",
            "hit": [
                "sugary drink", "sugar-sweetened", "soft drink", "soda", "cola",
                "sweetened tea", "milk tea", "bubble tea", "sweetened coffee",
                "energy drink", "sports drink", "sweetened juice", "frapp"
            ],
            "behavior_key": "replace_sugary_beverage",
            "title": "Replace sugary drinks with unsweetened options",
            "cue": "craving_sugary_drink",
            "action": "choose_unsweetened_beverage",
            "dose_unit": "serving",
            "primary_metric": "ssb_replacement_count",
        },
        {
            "name": "choose_low_sugar_cereal",
            "hit": [
                "breakfast cereal", "cereals are high in sugar", "lower-sugar cereal",
                "no added sugar", "no-added-sugar", "plain porridge", "porridge oats",
                "muesli", "plain shredded", "wholegrain", "wholemeal toast", "plain yoghurt"
            ],
            "behavior_key": "choose_low_sugar_breakfast",
            "title": "Choose a low-sugar breakfast (plain cereal/porridge + fruit)",
            "cue": "breakfast_choice_point",
            "action": "choose_low_sugar_breakfast",
            "dose_unit": "meal",
            "primary_metric": "low_sugar_breakfast_days",
        },
        {
            "name": "swap_sugar_in_porridge",
            "hit": [
                "add sugar to your porridge", "instead", "banana", "dried apricots",
                "chopped fruit", "mashed banana", "fruit instead of sugar"
            ],
            "behavior_key": "swap_sugar_with_fruit",
            "title": "Swap added sugar for fruit toppings",
            "cue": "breakfast_preparation",
            "action": "use_fruit_as_sweetener",
            "dose_unit": "meal",
            "primary_metric": "no_added_sugar_breakfast_days",
        },
        {
            "name": "portion_control_cereal",
            "hit": [
                "smaller portion", "eat a smaller portion", "add less sugar",
                "reduce sugar", "add less", "portion size"
            ],
            "behavior_key": "reduce_portion_or_sugar",
            "title": "Reduce portion size or added sugar gradually",
            "cue": "breakfast_serving",
            "action": "reduce_portion_or_added_sugar",
            "dose_unit": "meal",
            "primary_metric": "portion_control_days",
        },
        {
            "name": "gradual_swap_mix",
            "hit": [
                "gradual approach", "alternate days", "mix both", "mix both in the same bowl",
                "alternate", "mix sugary", "mix plain"
            ],
            "behavior_key": "gradual_cereal_swap",
            "title": "Gradually swap sugary cereal with plain cereal",
            "cue": "breakfast_choice_point",
            "action": "mix_or_alternate_cereal",
            "dose_unit": "meal",
            "primary_metric": "gradual_swap_days",
        },
    ]

    fam = None
    for f in families:
        if any(k in s_l for k in f["hit"]):
            fam = f
            break

    if fam is None:
        return None

    card_id = _mk_id(f"mt_{fam['behavior_key']}", s)

    # --- build constraints without null ---
    constraints = {
        "equipment_required": [],
        "time_cost_max_min": 10,
        "no_added_sugar": False,
        "contraindications_tags": []
    }

    # Only include the field when it's true; otherwise omit it.
    if ("sugar" in fam["behavior_key"]) or ("ssb" in fam["behavior_key"]):
        constraints["no_added_sugar"] = True

    # Minimal, schema-conformant method_template card (per your MVP schema)
    return {
        "schema_version": "behavior_method_mvp.v1",
        "card_type": "method_template",
        "id": card_id,
        "domain": "metabolic_health",
        "locale": locale,

        "behavior_key": fam["behavior_key"],
        "title": fam["title"],

        "trigger": {
            "if": {
                "cue": fam["cue"],
                "time_window": ["06:00", "11:00"] if fam["cue"].startswith("breakfast") else ["12:00", "20:00"]
            },
            "then": {"action": fam["action"]}
        },

        "dose_policy": {
            "default": {"value": 1, "unit": fam["dose_unit"]},
            "range": {"min": 1, "max": 1},
            "upgrade_rule": "if completion_rate_7d >= 0.7 then keep_or_increase_dose",
            "downgrade_rule": "if completion_rate_7d < 0.3 then simplify_dose"
        },

        "constraints": constraints,

        # Optional: you can keep empty now, run a later enrichment stage to fill mechanism_tags
        "mechanism_tags": {
            "comb_targets": ["opportunity_physical", "motivation_automatic"],
            "barrier_tags_solved": ["craving_sweet", "environment_temptation"]
        },

        # Required by schema; can be empty array
        "content_slots": [],

        "measurement": {
            "primary_metric": fam["primary_metric"],
            "secondary_metrics": ["daily_completion"]
        },

        "provenance": {
            "source_url": source_url,
            "extracted_by": "rule_parser",
            "confidence": 0.75,
            "evidence_spans": [s[:256]]
        }
    }

# ----------------------------
# LLM 兜底生成
#LLM 输出校验：OpenAI Method Template Schema
# ----------------------------

OPENAI_METHOD_TEMPLATE_SCHEMA_MIN: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["behavior_method_mvp.v1"]},
        "card_type": {"type": "string", "enum": ["method_template"]},
        "id": {"type": "string", "pattern": "^[A-Za-z0-9_\\-\\.]+$"},
        "domain": {"type": "string", "pattern": "^[a-z0-9_\\-\\.]+$"},
        "locale": {"type": "string", "pattern": "^[a-z]{2}_[A-Z]{2}$"},
        "behavior_key": {"type": "string", "pattern": "^[a-z0-9_\\-\\.]+$"},
        "title": {"type": "string", "minLength": 2, "maxLength": 120},

        "trigger": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "if": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "cue": {"type": "string", "pattern": "^[a-z0-9_\\-\\.]+$"}
                    },
                    # 只保留 cue，避免 time_window optional 引发 required 约束问题
                    "required": ["cue"],
                },
                "then": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {"type": "string", "pattern": "^[a-z0-9_\\-\\.]+$"}
                    },
                    "required": ["action"],
                }
            },
            "required": ["if", "then"],
        },

        "dose_policy": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "default": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string", "minLength": 1, "maxLength": 16}
                    },
                    "required": ["value", "unit"],
                },
                "range": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "min": {"type": "number"},
                        "max": {"type": "number"}
                    },
                    "required": ["min", "max"],
                }
            },
            "required": ["default", "range"],
        },

        "measurement": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "primary_metric": {"type": "string", "pattern": "^[a-z0-9_\\-\\.]+$"}
            },
            "required": ["primary_metric"],
        },
    },
    "required": [
        "schema_version",
        "card_type",
        "id",
        "domain",
        "locale",
        "behavior_key",
        "title",
        "trigger",
        "dose_policy",
        "measurement",
    ],
}


def llm_extract_to_method_template(candidate: str, source_url: str, locale: str = "en_US") -> Dict[str, Any]:
    """
    LLM fallback: Structured Outputs (json_schema) -> returns a minimal method_template dict.
    Official pattern: text.format.type=json_schema, strict=true, schema=...  :contentReference[oaicite:8]{index=8}
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system = (
        "You extract ONE habit method_template from diet / fat-loss / sugar-control advice.\n"
        "Output MUST be valid JSON and MUST conform to the given schema.\n"
        "All tag-like fields (domain, behavior_key, trigger.if.cue, trigger.then.action, measurement.*) "
        "MUST be lowercase snake_case.\n"
        "You MUST output ALL fields required by the schema.\n"
        "If uncertain, use safe defaults:\n"
        "- trigger.if.time_window: use a reasonable window like [\"08:00\",\"22:00\"]\n"
        "- constraints: use empty arrays, booleans, and a small time_cost_max_min (e.g., 5-15)\n"
        "- fallbacks: []\n"
        "- mechanism_tags: comb_targets [] and barrier_tags_solved [] if unknown\n"
        "- dose_policy.upgrade_rule/downgrade_rule: can be empty string if unknown\n"
        "- measurement.secondary_metrics: []\n"
    )


    user = (
        f"Source URL: {source_url}\n"
        f"Candidate advice snippet:\n{candidate}\n\n"
        f"Locale must be {locale}.\n"
        "Generate ONE method_template card."
    )

    # Use a model snapshot that supports Structured Outputs with json_schema format.
    # The Structured Outputs guide calls out supported snapshots such as gpt-4o-mini and gpt-4o-2024-08-06. :contentReference[oaicite:9]{index=9}
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "method_template",
                "strict": True,
                "schema": OPENAI_METHOD_TEMPLATE_SCHEMA_MIN,
            }
        },
        temperature=0,
    )

    # The SDK exposes output_text; it should be the JSON string when generation completes. :contentReference[oaicite:10]{index=10}
    card = json.loads(resp.output_text)

    # --------- post-fill fields to satisfy your FULL cards_schema.json ---------

    # content_slots is REQUIRED by full schema -> always fill here (and canonicalize also enforces it)
    card["content_slots"] = []

    # optional fields: keep stable defaults so full-schema passes and later enrichment can refine
    card.setdefault("constraints", {
        "equipment_required": [],
        "time_cost_max_min": 10,
        "contraindications_tags": [],
    })
    # 如果你希望糖控相关默认 no_added_sugar=True，可用简单启发式：
    if "sugar" in candidate.lower():
        card["constraints"].setdefault("no_added_sugar", True)

    card.setdefault("fallbacks", [])
    card.setdefault("mechanism_tags", {"comb_targets": [], "barrier_tags_solved": []})

    # upgrade/downgrade_rule 在 full schema 里是可选，但 dose_policy 里你可能希望统一补齐便于后续执行
    dp = card.get("dose_policy", {})
    dp.setdefault("upgrade_rule", "if completion_rate_7d >= 0.7 then keep_or_increase_dose")
    dp.setdefault("downgrade_rule", "if completion_rate_7d < 0.3 then simplify_dose")
    card["dose_policy"] = dp

    # time_window 在 full schema 里是可选；如果你希望统一有，可在这里补默认
    trig_if = (card.get("trigger") or {}).get("if") or {}
    if "time_window" not in trig_if:
        trig_if["time_window"] = ["08:00", "22:00"]
        card["trigger"]["if"] = trig_if

    # provenance（full schema 里是可选；但强烈建议统一填，便于合并与审计）
    card["provenance"] = {
        "source_url": source_url,
        "extracted_by": "llm",
        "confidence": 0.55,
        "evidence_spans": [candidate[:256]],
    }

    # ensure locale
    card["locale"] = locale
    return card



# ----------------------------
# 6) Persist (JSONL)
# ----------------------------
def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def record_failure(
    failure_jsonl: str,
    *,
    url: str,
    stage: str,
    candidate: str,
    error_type: str,
    error_msg: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    rec = {
        "ts": _utc_now_iso(),
        "url": url,
        "stage": stage,           # fetch/extract/mine/fast_path/llm/canonicalize/schema/upsert/write
        "error_type": error_type,
        "error_msg": error_msg,
        "candidate": (candidate or "")[:5000],
    }
    if extra:
        rec["extra"] = extra
    append_jsonl(failure_jsonl, rec)


# ----------------------------
# 7) Orchestration (closed loop)
# ----------------------------
from jsonschema import ValidationError

def validate_or_raise(instance: dict, schema: dict) -> None:
    errors = collect_schema_errors(instance, schema)
    if errors:
        raise SchemaValidationFailed(errors)

#写入方式从 append 改成 upsert（一次性重写 JSONL）

def load_jsonl_as_map(path: str, key_field: str = "id") -> Dict[str, Dict[str, Any]]:
    """Load existing JSONL as {id: obj}. If duplicated ids exist, keep the last one."""
    mp: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return mp
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = obj.get(key_field)
            if k:
                mp[k] = obj
    return mp


def write_jsonl_atomic(path: str, items: List[Dict[str, Any]]) -> None:
    """Atomic rewrite: write to .tmp then os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def merge_method_templates(old: Dict[str, Any], new: Dict[str, Any], max_evidence: int = 20) -> Dict[str, Any]:
    """
    Merge two MethodTemplateCard with the same id.
    Strategy:
      - keep old as base
      - provenance.confidence = max(old,new)
      - provenance.evidence_spans = union(old,new) with cap
      - provenance.extracted_by -> 'hybrid' if conflict
      - optionally keep old fields unless missing
    """
    merged = dict(old)

    old_p = dict(merged.get("provenance") or {})
    new_p = dict(new.get("provenance") or {})

    # source_url: keep existing; fill if missing
    if not old_p.get("source_url") and new_p.get("source_url"):
        old_p["source_url"] = new_p["source_url"]

    # confidence: max
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    old_p["confidence"] = max(_to_float(old_p.get("confidence")), _to_float(new_p.get("confidence")))

    # evidence_spans: union + cap
    old_ev = old_p.get("evidence_spans") or []
    new_ev = new_p.get("evidence_spans") or []
    if not isinstance(old_ev, list):
        old_ev = []
    if not isinstance(new_ev, list):
        new_ev = []

    union = []
    seen = set()
    for s in old_ev + new_ev:
        if not isinstance(s, str):
            continue
        if s in seen:
            continue
        seen.add(s)
        union.append(s)
        if len(union) >= max_evidence:
            break
    old_p["evidence_spans"] = union

    # extracted_by: conflict -> hybrid
    ob = old_p.get("extracted_by")
    nb = new_p.get("extracted_by")
    if ob and nb and ob != nb:
        old_p["extracted_by"] = "hybrid"
    elif not ob and nb:
        old_p["extracted_by"] = nb

    merged["provenance"] = old_p

    # If some key fields are missing in old but present in new, fill them
    # (避免早期脏卡缺字段导致长期留存)
    for k in ["title", "description", "behavior_key", "dose_policy", "trigger", "constraints", "tags", "content_slots"]:
        if merged.get(k) is None and new.get(k) is not None:
            merged[k] = new[k]

    return merged



def ingest_url_to_cards(
    url: str,
    full_schema: Dict[str, Any],
    out_jsonl: str,
    use_llm_fallback: bool = True,
    locale: str = "en_US",
    failure_jsonl: Optional[str] = None,
) -> Dict[str, Any]:
    if failure_jsonl is None:
        failure_jsonl = out_jsonl + ".failures.jsonl"

    # Upsert store: read existing library
    store = load_jsonl_as_map(out_jsonl, key_field="id")

    stats = {
        "url": url,
        "candidates": 0,
        "skipped": 0,          # ✅ 新增
        "inserted": 0,
        "merged": 0,
        "saved_total": 0,
        "failed": 0,
        "llm_used": 0,
        "store_before": len(store),
        "store_after": None,
    }

    # Stage-level protection for fetch/extract
    stage = "fetch"
    try:
        html = fetch_html(url)
        stage = "extract"
        text = extract_main_text(html, url=url)
        stage = "mine"
        candidates = mine_habit_candidates(text)
        stats["candidates"] = len(candidates)
    except Exception as e:
        stats["failed"] += 1
        record_failure(
            failure_jsonl,
            url=url,
            stage=stage,
            candidate="",
            error_type=type(e).__name__,
            error_msg=str(e),
        )
        return stats
    
    stats.setdefault("skipped", 0)
    
    for c in candidates:
    # ---------- NEW: quality gate ----------
        if should_skip_candidate(c):
            stats["skipped"] += 1
            continue

        stage = "fast_path"
        card = None
        extracted_by = None

        try:
            card = fast_path_to_method_template_mvp(c, source_url=url, locale=locale)
            extracted_by = "rule_parser" if card is not None else None

            if card is None and use_llm_fallback:
                stage = "llm"
                stats["llm_used"] += 1  # 建议：在调用前计数，避免异常导致一直是 0
                card = llm_extract_to_method_template(c, source_url=url, locale=locale)
                extracted_by = "llm"

            if card is None:
                continue

            stage = "canonicalize"
            card = canonicalize_method_template_card(
                card,
                candidate=c,
                source_url=url,
                locale=locale,
                extracted_by=extracted_by or "hybrid",
            )

            stage = "schema"
            validate_or_raise(card, full_schema)

            # Upsert + merge
            stage = "upsert"
            cid = card.get("id")
            if not cid:
                raise ValueError("card.id missing after canonicalize")

            if cid in store:
                store[cid] = merge_method_templates(store[cid], card, max_evidence=20)
                stats["merged"] += 1
            else:
                store[cid] = card
                stats["inserted"] += 1

            stats["saved_total"] = stats["inserted"] + stats["merged"]

        except SchemaValidationFailed as e:
            stats["failed"] += 1
            record_failure(
                failure_jsonl,
                url=url,
                stage=stage,
                candidate=c,
                error_type="SchemaValidationFailed",
                error_msg="schema validation failed",
                extra={
                    "schema_errors": e.errors,
                    "card_preview": card,
                },
            )
            continue

        except Exception as e:
            stats["failed"] += 1
            record_failure(
                failure_jsonl,
                url=url,
                stage=stage,
                candidate=c,
                error_type=type(e).__name__,
                error_msg=str(e),
            )
            continue

        time.sleep(0.2)

    # Atomic rewrite JSONL
    stage = "write"
    try:
        items = list(store.values())
        items.sort(key=lambda x: str(x.get("id", "")))
        write_jsonl_atomic(out_jsonl, items)
        stats["store_after"] = len(store)
    except Exception as e:
        stats["failed"] += 1
        record_failure(
            failure_jsonl,
            url=url,
            stage=stage,
            candidate="",
            error_type=type(e).__name__,
            error_msg=str(e),
        )

    return stats
