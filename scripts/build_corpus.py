"""Build a license-verified sports-science corpus from Europe PMC.

Run with: python -m scripts.build_corpus

Every document in the corpus is a real peer-reviewed paper fetched from Europe
PMC's open-access full-text API. Nothing is written unless the API reports a
redistributable Creative Commons license — the license check is a hard gate, not
a note, so the corpus stays safe to commit to a public repo.

Each output file carries YAML frontmatter with the citation (authors, journal,
year, DOI, PMCID, license) so retrieved chunks can name their source, and the
critic can verify groundedness against a real reference instead of vibes.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).parent.parent))

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

CORPUS_DIR = Path(__file__).parent.parent / "rag" / "knowledge_base" / "corpus"

# Licenses we may redistribute. NC/ND variants are excluded deliberately: this
# repo is open source and may be used commercially, and ND forbids the chunking
# that RAG requires.
ALLOWED_LICENSES = {"cc by", "cc by-sa", "cc0", "cc by 4.0", "cc by 3.0"}

# Topics map to the decision domains the agents actually reason about. The eval
# found the previous single-file KB had nothing on endurance, so intensity
# distribution and concurrent training are first-class here.
TARGETS = [
    ("training_load", 'TITLE:"training load" AND (injury OR monitoring)'),
    ("training_load", 'TITLE:"acute:chronic workload ratio"'),
    ("training_load", 'TITLE:"acute chronic workload ratio" AND TITLE:"injury"'),
    ("training_load", 'TITLE:"session RPE" AND TITLE:"training load"'),
    ("recovery", 'TITLE:"overtraining" AND TITLE:"syndrome"'),
    ("recovery", 'TITLE:"tapering" AND (performance OR "training")'),
    ("recovery", 'TITLE:"recovery" AND TITLE:"resistance training"'),
    ("recovery", 'TITLE:"sleep" AND TITLE:"athletic performance"'),
    ("resistance_training", 'TITLE:"resistance training" AND TITLE:"hypertrophy" AND TITLE:"volume"'),
    ("resistance_training", 'TITLE:"training frequency" AND TITLE:"hypertrophy"'),
    ("resistance_training", 'TITLE:"resistance training" AND TITLE:"dose-response"'),
    ("resistance_training", 'TITLE:"progressive overload"'),
    ("resistance_training", 'TITLE:"repetitions in reserve"'),
    ("resistance_training", 'TITLE:"rating of perceived exertion" AND TITLE:"resistance training"'),
    ("resistance_training", 'TITLE:"muscle strength" AND TITLE:"training volume" AND TITLE:"meta-analysis"'),
    ("resistance_training", 'TITLE:"detraining" AND (strength OR muscle)'),
    ("resistance_training", 'TITLE:"deload" OR TITLE:"training cessation" AND TITLE:"strength"'),
    ("resistance_training", 'TITLE:"velocity loss" AND TITLE:"resistance training"'),
    ("resistance_training", 'TITLE:"periodization" AND TITLE:"strength"'),
    ("endurance", 'TITLE:"training intensity distribution" AND TITLE:"endurance"'),
    ("endurance", 'TITLE:"polarized training"'),
    ("endurance", 'TITLE:"high-intensity interval training" AND TITLE:"endurance"'),
    ("endurance", 'TITLE:"concurrent training" AND (strength OR endurance)'),
    ("endurance", 'TITLE:"running economy" AND TITLE:"strength training"'),
    ("nutrition", 'TITLE:"protein" AND TITLE:"resistance training" AND TITLE:"muscle"'),
    ("nutrition", 'TITLE:"nutrient timing"'),
    ("nutrition", 'TITLE:"carbohydrate" AND TITLE:"endurance performance"'),
    ("nutrition", 'TITLE:"energy availability" AND TITLE:"athletes"'),
    ("nutrition", 'TITLE:"body composition" AND TITLE:"energy restriction" AND TITLE:"athletes"'),
]

MAX_PER_QUERY = 2
MIN_BODY_CHARS = 3000

# Search terms collide with clinical literature ("tapering" matches steroid
# tapering, "flare" matches rheumatology), so every paper must also look like
# sports science in its title or abstract.
DOMAIN_TERMS = re.compile(
    r"\b(athlet|exercis|training|sport|muscle|strength|endurance|resistance|aerobic|"
    r"hypertroph|fitness|performance|workload|VO2|runner|cycl|lifting|physical activity)",
    re.I,
)
# Clinical-population papers that mention exercise incidentally.
OFF_TOPIC = re.compile(
    r"\b(rheumatoid|arthritis|prednisolone|chemotherap|oncolog|cancer patient|"
    r"nephrotic|dialysis|schizophren|dementia|COPD patient)", re.I,
)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def search(query: str, page_size: int = 12) -> list[dict]:
    """Search Europe PMC for open-access papers with full text in PMC."""
    full_query = f'({query}) AND OPEN_ACCESS:Y AND IN_PMC:Y AND HAS_FT:Y'
    url = SEARCH_URL + "?" + urllib.parse.urlencode(
        {"query": full_query, "format": "json", "resultType": "core",
         "pageSize": str(page_size), "sort": "CITED desc"}
    )
    try:
        return _get_json(url).get("resultList", {}).get("result", [])
    except Exception as e:
        print(f"    search failed: {e}")
        return []


def fetch_fulltext(pmcid: str) -> str | None:
    try:
        with urllib.request.urlopen(FULLTEXT_URL.format(pmcid=pmcid), timeout=90) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    full text fetch failed for {pmcid}: {e}")
        return None


def _text_of(node) -> str:
    """Flatten an element's text, dropping xref/table/figure clutter."""
    parts = []
    for child in node.iter():
        if child.tag in ("xref", "table-wrap", "fig", "graphic", "inline-formula", "disp-formula"):
            continue
        if child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Sections that carry method detail or bookkeeping rather than findings.
SKIP_SECTIONS = re.compile(
    r"^(references?|acknowledg|competing interest|conflict|funding|author contribution|"
    r"supplementary|abbreviation|ethic|data availability|statistical analys)", re.I
)


def xml_to_markdown(xml: str) -> tuple[str, str]:
    """Return (abstract, body_markdown) extracted from JATS XML."""
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as e:
        print(f"    XML parse error: {e}")
        return "", ""

    abstract_parts = []
    for abstract in root.iter("abstract"):
        for p in abstract.iter("p"):
            t = _text_of(p)
            if t:
                abstract_parts.append(t)
    abstract = "\n\n".join(abstract_parts)

    body = root.find(".//body")
    if body is None:
        return abstract, ""

    lines = []
    for sec in body.findall("sec"):
        title_el = sec.find("title")
        title = _text_of(title_el) if title_el is not None else ""
        if title and SKIP_SECTIONS.match(title):
            continue
        if title:
            lines.append(f"\n## {title}\n")
        for child in sec:
            if child.tag == "p":
                t = _text_of(child)
                if t:
                    lines.append(t + "\n")
            elif child.tag == "sec":
                sub_title_el = child.find("title")
                sub_title = _text_of(sub_title_el) if sub_title_el is not None else ""
                if sub_title and SKIP_SECTIONS.match(sub_title):
                    continue
                if sub_title:
                    lines.append(f"\n### {sub_title}\n")
                for p in child.findall("p"):
                    t = _text_of(p)
                    if t:
                        lines.append(t + "\n")
    return abstract, "\n".join(lines)


def slugify(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:limit].rstrip("_")


def build() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    seen_pmcids: set[str] = set()
    manifest = []
    rejected = []

    for topic, query in TARGETS:
        print(f"\n[{topic}] {query}")
        results = search(query)
        kept = 0
        for r in results:
            if kept >= MAX_PER_QUERY:
                break
            pmcid = r.get("pmcid")
            license_raw = (r.get("license") or "").strip().lower()
            title = (r.get("title") or "").rstrip(".")

            if not pmcid or pmcid in seen_pmcids:
                continue
            # HARD GATE: only redistributable Creative Commons licenses.
            if license_raw not in ALLOWED_LICENSES:
                rejected.append((title[:70], license_raw or "unknown"))
                continue

            xml = fetch_fulltext(pmcid)
            time.sleep(0.4)  # be polite to the API
            if not xml:
                continue
            abstract, body = xml_to_markdown(xml)
            if len(body) < MIN_BODY_CHARS:
                print(f"    skipped (thin body): {title[:60]}")
                continue

            relevance_text = f"{title} {abstract}"
            if not DOMAIN_TERMS.search(relevance_text) or OFF_TOPIC.search(relevance_text):
                print(f"    skipped (off topic): {title[:60]}")
                continue

            year = r.get("pubYear", "")
            authors = r.get("authorString", "").rstrip(".")
            journal = (r.get("journalInfo", {}) or {}).get("journal", {}).get("title") or ""
            doi = r.get("doi", "")
            citation = f"{authors} ({year}). {title}. {journal}."

            first_author = (authors.split(",")[0] if authors else "unknown").split()[0]
            filename = f"{topic}__{slugify(first_author + '_' + year + '_' + title)}.md"

            front = [
                "---",
                f'title: "{title.replace(chr(34), chr(39))}"',
                f'authors: "{authors}"',
                f'journal: "{journal}"',
                f"year: {year}",
                f'doi: "{doi}"',
                f'pmcid: "{pmcid}"',
                f'license: "{license_raw.upper()}"',
                f'source_url: "https://europepmc.org/article/PMC/{pmcid}"',
                f"topic: {topic}",
                f'citation: "{citation}"',
                "---",
                "",
                f"# {title}",
                "",
                f"**Source:** {citation} DOI: {doi}. License: {license_raw.upper()}.",
                "",
            ]
            if abstract:
                front += ["## Abstract", "", abstract, ""]
            content = "\n".join(front) + body

            (CORPUS_DIR / filename).write_text(content, encoding="utf-8")
            seen_pmcids.add(pmcid)
            manifest.append({
                "file": filename, "topic": topic, "title": title, "authors": authors,
                "journal": journal, "year": year, "doi": doi, "pmcid": pmcid,
                "license": license_raw.upper(), "chars": len(content),
            })
            kept += 1
            print(f"    + [{license_raw.upper()}] {title[:66]}")

    # Manifest + human-readable license register.
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Corpus sources and licenses",
        "",
        f"{len(manifest)} peer-reviewed open-access papers fetched from Europe PMC by "
        "`python -m scripts.build_corpus`.",
        "",
        "Every paper below was verified via the Europe PMC API to carry a redistributable "
        "Creative Commons license before being written to disk; anything else was rejected. "
        "Attribution for each work is in its file's frontmatter and in the table below.",
        "",
        "| Topic | Citation | License | Source |",
        "|---|---|---|---|",
    ]
    for m in sorted(manifest, key=lambda x: (x["topic"], x["year"])):
        cite = f"{m['authors']} ({m['year']}). {m['title']}. *{m['journal']}*."
        lines.append(
            f"| {m['topic']} | {cite} | {m['license']} | "
            f"[PMC](https://europepmc.org/article/PMC/{m['pmcid']}) |"
        )
    (CORPUS_DIR / "LICENSES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"Corpus: {len(manifest)} papers, "
          f"{sum(m['chars'] for m in manifest):,} characters")
    by_topic: dict[str, int] = {}
    for m in manifest:
        by_topic[m["topic"]] = by_topic.get(m["topic"], 0) + 1
    for t, n in sorted(by_topic.items()):
        print(f"  {t:22s} {n}")
    if rejected:
        print(f"\nRejected {len(rejected)} papers on license grounds, e.g.:")
        for title, lic in rejected[:5]:
            print(f"  [{lic}] {title}")
    print(f"\nWritten to {CORPUS_DIR}")


if __name__ == "__main__":
    build()
