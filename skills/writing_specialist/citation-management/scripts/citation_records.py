#!/usr/bin/env python3
"""Convert selected Crossref metadata to one bibliography file (BibTeX by default).

No discovery, venue filtering, automatic citation insertion or evidence scoring.
Metadata candidates stay unassessed until their relevant sources are inspected.
JSON, TSV, RIS, ENW and Zotero RDF remain explicit alternative formats.
"""
# Derived from Yuan1z0825/nature-skills (Apache-2.0); see ../LICENSE.nature-skills.
# Retains CatMaster's claim-relative metadata export fixes.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape, quoteattr

ZOTERO_RDF_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "z": "http://www.zotero.org/namespaces/export#",
    "dcterms": "http://purl.org/dc/terms/",
    "bib": "http://purl.org/net/biblio#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "prism": "http://prismstandard.org/namespaces/1.2/basic/",
}


NATURE_EXACT = {
    "Nature",
    "Nature Biotechnology",
    "Nature Cancer",
    "Nature Cardiovascular Research",
    "Nature Cell Biology",
    "Nature Chemical Biology",
    "Nature Chemistry",
    "Nature Climate Change",
    "Nature Communications",
    "Nature Computational Science",
    "Nature Ecology & Evolution",
    "Nature Electronics",
    "Nature Energy",
    "Nature Food",
    "Nature Genetics",
    "Nature Geoscience",
    "Nature Human Behaviour",
    "Nature Immunology",
    "Nature Machine Intelligence",
    "Nature Materials",
    "Nature Medicine",
    "Nature Metabolism",
    "Nature Methods",
    "Nature Microbiology",
    "Nature Nanotechnology",
    "Nature Neuroscience",
    "Nature Photonics",
    "Nature Physics",
    "Nature Plants",
    "Nature Protocols",
    "Nature Reviews Cancer",
    "Nature Reviews Cardiology",
    "Nature Reviews Chemistry",
    "Nature Reviews Clinical Oncology",
    "Nature Reviews Drug Discovery",
    "Nature Reviews Earth & Environment",
    "Nature Reviews Endocrinology",
    "Nature Reviews Gastroenterology & Hepatology",
    "Nature Reviews Genetics",
    "Nature Reviews Immunology",
    "Nature Reviews Materials",
    "Nature Reviews Microbiology",
    "Nature Reviews Molecular Cell Biology",
    "Nature Reviews Nephrology",
    "Nature Reviews Neurology",
    "Nature Reviews Neuroscience",
    "Nature Reviews Physics",
    "Nature Reviews Psychology",
    "Nature Reviews Rheumatology",
    "Nature Structural & Molecular Biology",
    "Scientific Reports",
}


SCIENCE_EXACT = {
    "Science",
    "Science Advances",
    "Science Immunology",
    "Science Robotics",
    "Science Signaling",
    "Science Translational Medicine",
}


CELL_EXACT = {
    "Cell",
    "Cancer Cell",
    "Cell Chemical Biology",
    "Cell Genomics",
    "Cell Host & Microbe",
    "Cell Metabolism",
    "Cell Reports",
    "Cell Reports Medicine",
    "Cell Reports Methods",
    "Cell Reports Physical Science",
    "Cell Stem Cell",
    "Cell Systems",
    "Chem",
    "Current Biology",
    "Developmental Cell",
    "Immunity",
    "Joule",
    "Med",
    "Molecular Cell",
    "Neuron",
    "One Earth",
    "Patterns",
    "Structure",
    "The Innovation",
}


CELL_TRENDS_EXACT = {
    "Trends in Biochemical Sciences",
    "Trends in Biotechnology",
    "Trends in Cancer",
    "Trends in Cell Biology",
    "Trends in Chemistry",
    "Trends in Cognitive Sciences",
    "Trends in Ecology & Evolution",
    "Trends in Endocrinology & Metabolism",
    "Trends in Genetics",
    "Trends in Immunology",
    "Trends in Microbiology",
    "Trends in Molecular Medicine",
    "Trends in Neurosciences",
    "Trends in Parasitology",
    "Trends in Pharmacological Sciences",
    "Trends in Plant Science",
}


FLAGSHIP = {"Nature", "Science", "Cell"}


@dataclass
class Segment:
    id: str
    text: str
    search_query: str
    order: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "order": self.order,
            "text": self.text,
            "search_query": self.search_query,
        }


@dataclass
class Candidate:
    title: str
    journal: str
    family: str
    year: str
    y1: str
    doi: str
    url: str
    volume: str
    issue: str
    start_page: str
    end_page: str
    issn: str
    authors: list[str]
    abstract: str
    type: str
    retrieval_score: float
    source_query: str

    @property
    def doi_url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else self.url

    @property
    def key(self) -> str:
        if self.doi:
            return self.doi.lower()
        return f"{self.title.lower()}|{self.journal.lower()}"

    @property
    def first_author(self) -> str:
        if not self.authors:
            return "Unknown author"
        return self.authors[0].split(",", 1)[0]

    @property
    def citation_marker(self) -> str:
        if self.year:
            return f"({self.first_author} et al., {self.year})"
        return f"({self.first_author} et al.)"

    @property
    def page_range(self) -> str:
        if self.start_page and self.end_page:
            return f"{self.start_page}-{self.end_page}"
        return self.start_page

    @property
    def identifier_url(self) -> str:
        return self.doi_url or self.url

    @property
    def article_resource(self) -> str:
        if self.identifier_url:
            return self.identifier_url
        return f"urn:candidate:{stable_hash(self.key or self.title or 'candidate')}"

    @property
    def journal_resource(self) -> str:
        return build_journal_resource(self)

    @property
    def zotero_citation_key(self) -> str:
        return build_zotero_citation_key(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "journal": self.journal,
            "family": self.family,
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            "doi_url": self.doi_url,
            "volume": self.volume,
            "issue": self.issue,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "issn": self.issn,
            "authors": self.authors,
            "abstract": self.abstract,
            "type": self.type,
            "source_query": self.source_query,
            "citation_marker": self.citation_marker,
            "claim_relation": "unassessed",
            "access_depth": "metadata",
            "screening_note": "Inspect abstract/publisher page before citing this paper as support.",
            "enw_record": build_enw_record(self),
            "ris_record": build_ris_record(self),
            "journal_resource": self.journal_resource,
            "zotero_rdf_article": build_zotero_rdf_article(self),
            "zotero_rdf_journal": build_zotero_rdf_journal(self),
        }


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip()


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug or "item"


def zotero_date_value(item: Candidate) -> str:
    if item.y1:
        return item.y1.replace("/", "-")
    return item.year


def split_author_parts(name: str) -> tuple[str, str]:
    if "," in name:
        family, given = name.split(",", 1)
        return family.strip(), given.strip()
    parts = [part for part in name.split() if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[-1], " ".join(parts[:-1])


def build_journal_resource(item: Candidate) -> str:
    parts: list[str] = []
    if item.issn:
        parts.append(f"issn:{slugify(item.issn)}")
    elif item.journal:
        parts.append(f"title:{slugify(item.journal)}")
    else:
        parts.append(f"record:{stable_hash(item.key or item.title or 'journal')}")
    if item.volume:
        parts.append(f"vol:{slugify(item.volume)}")
    if item.issue:
        parts.append(f"issue:{slugify(item.issue)}")
    return "urn:" + ":".join(parts)


def build_zotero_citation_key(item: Candidate) -> str:
    first_author = slugify(item.first_author)
    title_words = re.findall(r"[A-Za-z0-9]+", item.title)[:3]
    title_part = "".join(word.capitalize() for word in title_words) or "Item"
    year = item.year or "n.d."
    return f"{first_author}{title_part}{year}"


def journal_family(journal: str) -> str | None:
    journal = normalize_title(journal)
    if not journal:
        return None
    if journal in NATURE_EXACT or journal.startswith("Nature ") or journal.startswith("npj "):
        return "Nature Portfolio"
    if journal in SCIENCE_EXACT:
        return "Science family"
    if journal in CELL_EXACT or journal in CELL_TRENDS_EXACT:
        return "Cell Press"
    return None


def first(values: list[Any] | None, default: str = "") -> str:
    if not values:
        return default
    value = values[0]
    if isinstance(value, str):
        return value
    return default


def date_parts(item: dict[str, Any]) -> list[int]:
    for key in ("published-print", "published-online", "published", "issued"):
        parts = item.get(key, {}).get("date-parts")
        if parts and parts[0]:
            return parts[0]
    return []


def year_from_item(item: dict[str, Any]) -> str:
    parts = date_parts(item)
    return str(parts[0]) if parts else ""


def y1_from_item(item: dict[str, Any]) -> str:
    parts = date_parts(item)
    if not parts:
        return ""
    year = f"{parts[0]:04d}"
    month = f"{parts[1]:02d}" if len(parts) > 1 else "01"
    day = f"{parts[2]:02d}" if len(parts) > 2 else "01"
    return f"{year}/{month}/{day}"


def author_name(author: dict[str, Any]) -> str:
    family = author.get("family", "").strip()
    given = author.get("given", "").strip()
    if family and given:
        return f"{family}, {given}"
    return family or given or author.get("name", "").strip()


def pages(item: dict[str, Any]) -> tuple[str, str]:
    page = item.get("page", "") or item.get("article-number", "")
    if not page:
        return "", ""
    if "-" in page:
        start, end = page.split("-", 1)
        return start.strip(), end.strip()
    return page.strip(), ""


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ris_escape(text: str) -> str:
    return clean_text(text).replace("\n", " ").replace("\r", " ")


def candidate_from_crossref(item: dict[str, Any], source_query: str) -> Candidate | None:
    journal = first(item.get("container-title"))
    if not journal:
        return None
    family = journal_family(journal) or ""
    start, end = pages(item)
    authors = [author_name(author) for author in item.get("author", [])]
    authors = [author for author in authors if author]
    return Candidate(
        title=clean_text(first(item.get("title"))),
        journal=normalize_title(journal),
        family=family,
        year=year_from_item(item),
        y1=y1_from_item(item),
        doi=item.get("DOI", ""),
        url=item.get("URL", ""),
        volume=item.get("volume", ""),
        issue=item.get("issue", ""),
        start_page=start,
        end_page=end,
        issn=first(item.get("ISSN")),
        authors=authors,
        abstract=clean_text(item.get("abstract", "")),
        type=item.get("type", ""),
        retrieval_score=float(item.get("score", 0.0) or 0.0),
        source_query=source_query,
    )


def build_ris_record(item: Candidate) -> str:
    lines: list[str] = []
    lines.append("TY  - JOUR")
    if item.title:
        lines.append(f"TI  - {ris_escape(item.title)}")
    for author in item.authors:
        lines.append(f"AU  - {ris_escape(author)}")
    if item.journal:
        lines.append(f"T2  - {ris_escape(item.journal)}")
        lines.append(f"JO  - {ris_escape(item.journal)}")
    if item.year:
        lines.append(f"PY  - {ris_escape(item.year)}")
    if item.y1:
        lines.append(f"Y1  - {ris_escape(item.y1)}")
    if item.volume:
        lines.append(f"VL  - {ris_escape(item.volume)}")
    if item.issue:
        lines.append(f"IS  - {ris_escape(item.issue)}")
    if item.start_page:
        lines.append(f"SP  - {ris_escape(item.start_page)}")
    if item.end_page:
        lines.append(f"EP  - {ris_escape(item.end_page)}")
    if item.doi:
        lines.append(f"DO  - {ris_escape(item.doi)}")
    if item.doi_url:
        lines.append(f"UR  - {ris_escape(item.doi_url)}")
    if item.issn:
        lines.append(f"SN  - {ris_escape(item.issn)}")
    if item.abstract:
        lines.append(f"AB  - {ris_escape(item.abstract)}")
    lines.append("N1  - Metadata-only candidate. Inspect abstract or publisher page before citing as support.")
    lines.append("ER  -")
    return "\n".join(lines)


def write_ris(candidates: list[Candidate], path: Path) -> None:
    lines: list[str] = []
    for item in candidates:
        lines.append(build_ris_record(item))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_enw_record(item: Candidate) -> str:
    lines: list[str] = []
    lines.append("%0 Journal Article")
    if item.title:
        lines.append(f"%T {ris_escape(item.title)}")
    for author in item.authors:
        lines.append(f"%A {ris_escape(author)}")
    if item.journal:
        lines.append(f"%J {ris_escape(item.journal)}")
    if item.volume:
        lines.append(f"%V {ris_escape(item.volume)}")
    if item.issue:
        lines.append(f"%N {ris_escape(item.issue)}")
    if item.start_page and item.end_page:
        lines.append(f"%P {ris_escape(item.start_page)}-{ris_escape(item.end_page)}")
    elif item.start_page:
        lines.append(f"%P {ris_escape(item.start_page)}")
    if item.year:
        lines.append(f"%D {ris_escape(item.year)}")
    if item.issn:
        lines.append(f"%@ {ris_escape(item.issn)}")
    if item.doi:
        lines.append(f"%R {ris_escape(item.doi)}")
    if item.doi_url:
        lines.append(f"%U {ris_escape(item.doi_url)}")
    if item.abstract:
        lines.append(f"%X {ris_escape(item.abstract)}")
    return "\n".join(lines)


def write_enw(candidates: list[Candidate], path: Path) -> None:
    lines: list[str] = []
    for item in candidates:
        lines.append(build_enw_record(item))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_zotero_rdf_article(item: Candidate) -> str:
    lines: list[str] = [f'    <bib:Article rdf:about={quoteattr(item.article_resource)}>']
    lines.append("        <z:itemType>journalArticle</z:itemType>")
    if item.journal:
        lines.append(f'        <dcterms:isPartOf rdf:resource={quoteattr(item.journal_resource)}/>')
    if item.authors:
        lines.append("        <bib:authors>")
        lines.append("            <rdf:Seq>")
        for author in item.authors:
            family, given = split_author_parts(author)
            lines.append("                <rdf:li>")
            lines.append("                    <foaf:Person>")
            if family:
                lines.append(f"                        <foaf:surname>{xml_escape(family)}</foaf:surname>")
            if given:
                lines.append(f"                        <foaf:givenName>{xml_escape(given)}</foaf:givenName>")
            lines.append("                    </foaf:Person>")
            lines.append("                </rdf:li>")
        lines.append("            </rdf:Seq>")
        lines.append("        </bib:authors>")
    if item.title:
        lines.append(f"        <dc:title>{xml_escape(item.title)}</dc:title>")
    date_value = zotero_date_value(item)
    if date_value:
        lines.append(f"        <dc:date>{xml_escape(date_value)}</dc:date>")
    lines.append("        <z:libraryCatalog>Crossref</z:libraryCatalog>")
    if item.identifier_url:
        lines.append("        <dc:identifier>")
        lines.append("            <dcterms:URI>")
        lines.append(f"                <rdf:value>{xml_escape(item.identifier_url)}</rdf:value>")
        lines.append("            </dcterms:URI>")
        lines.append("        </dc:identifier>")
    if item.doi:
        lines.append(f"        <dc:identifier>{xml_escape(f'DOI {item.doi}')}</dc:identifier>")
    if item.page_range:
        lines.append(f"        <bib:pages>{xml_escape(item.page_range)}</bib:pages>")
    lines.append(f"        <z:citationKey>{xml_escape(item.zotero_citation_key)}</z:citationKey>")
    lines.append("    </bib:Article>")
    return "\n".join(lines)


def build_zotero_rdf_journal(item: Candidate) -> str:
    lines: list[str] = [f'    <bib:Journal rdf:about={quoteattr(item.journal_resource)}>']
    if item.volume:
        lines.append(f"        <prism:volume>{xml_escape(item.volume)}</prism:volume>")
    if item.journal:
        lines.append(f"        <dc:title>{xml_escape(item.journal)}</dc:title>")
    if item.issue:
        lines.append(f"        <prism:number>{xml_escape(item.issue)}</prism:number>")
    if item.issn:
        lines.append(f"        <dc:identifier>{xml_escape(f'ISSN {item.issn}')}</dc:identifier>")
    lines.append("    </bib:Journal>")
    return "\n".join(lines)


def build_zotero_rdf_document(candidates: list[Candidate]) -> str:
    root_open = [
        "<rdf:RDF",
        *(f' xmlns:{prefix}="{uri}"' for prefix, uri in ZOTERO_RDF_NS.items()),
        ">",
    ]
    journal_map: dict[str, str] = {}
    article_blocks: list[str] = []
    for item in candidates:
        article_blocks.append(build_zotero_rdf_article(item))
        if item.journal and item.journal_resource not in journal_map:
            journal_map[item.journal_resource] = build_zotero_rdf_journal(item)
    sections = ["".join(root_open), *article_blocks, *journal_map.values(), "</rdf:RDF>"]
    return "\n".join(section for section in sections if section)


def write_zotero_rdf(candidates: list[Candidate], path: Path) -> None:
    path.write_text(build_zotero_rdf_document(candidates), encoding="utf-8")


def write_mapping_tsv(mapping: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "segment_id",
        "segment_order",
        "segment_text",
        "search_query",
        "suggested_insert_text",
        "citation_marker",
        "claim_relation",
        "access_depth",
        "title",
        "journal",
        "family",
        "year",
        "doi",
        "doi_url",
        "authors",
        "screening_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for entry in mapping:
            segment: Segment = entry["segment"]
            if not entry["references"]:
                writer.writerow(
                    {
                        "segment_id": segment.id,
                        "segment_order": segment.order,
                        "segment_text": segment.text,
                        "search_query": segment.search_query,
                        "suggested_insert_text": "",
                        "claim_relation": "unassessed",
                        "access_depth": "metadata",
                        "screening_note": "No metadata candidate supplied.",
                    }
                )
            for candidate in entry["references"]:
                writer.writerow(
                    {
                        "segment_id": segment.id,
                        "segment_order": segment.order,
                        "segment_text": segment.text,
                        "search_query": segment.search_query,
                        "suggested_insert_text": entry["suggested_insert_text"],
                        "citation_marker": candidate.citation_marker,
                        "claim_relation": "unassessed",
                        "access_depth": "metadata",
                        "title": candidate.title,
                        "journal": candidate.journal,
                        "family": candidate.family,
                        "year": candidate.year,
                        "doi": candidate.doi,
                        "doi_url": candidate.doi_url,
                        "authors": "; ".join(candidate.authors),
                        "screening_note": "Inspect abstract/publisher page before citing this paper as support.",
                    }
                )



def write_bibtex(candidates: list[Candidate], path: Path) -> None:
    """Write complete candidate metadata without retrieval scores or support claims."""
    entries = []
    used_keys: set[str] = set()
    for item in candidates:
        base_key = item.zotero_citation_key
        key = base_key
        suffix = 2
        while key in used_keys:
            key = f"{base_key}{suffix}"
            suffix += 1
        used_keys.add(key)
        fields = {
            "title": item.title,
            "author": " and ".join(item.authors),
            "journal": item.journal,
            "year": item.year,
            "volume": item.volume,
            "number": item.issue,
            "pages": item.page_range.replace("-", "--"),
            "doi": item.doi,
            "url": item.identifier_url,
            "issn": item.issn,
            "abstract": item.abstract,
        }
        lines = [f"@article{{{key},"]
        for name, value in fields.items():
            if value:
                escaped = value.replace("{", "\\{").replace("}", "\\}")
                lines.append(f"  {name} = {{{escaped}}},")
        lines.append("}")
        entries.append("\n".join(lines))
    path.write_text("\n\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON list of selected Crossref works, or a Crossref message")
    parser.add_argument("output", type=Path, help="Output path, normally references.bib")
    parser.add_argument("--format", choices=("bib", "json", "tsv", "ris", "enw", "zotero-rdf"), default="bib",
                        help="One output format; default: bib")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("message", payload)
    if isinstance(payload, dict):
        payload = payload.get("items", [payload])
    candidates = []
    for work in payload:
        candidate = candidate_from_crossref(work, source_query="")
        if candidate is None:
            raise ValueError("A supplied work lacks journal metadata; no records were written.")
        candidates.append(candidate)
    if args.format == "json":
        args.output.write_text(json.dumps([item.as_dict() for item in candidates], ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.format == "tsv":
        write_mapping_tsv([{
            "segment": Segment(str(index), "", "", index),
            "references": [item],
            "suggested_insert_text": "",
        } for index, item in enumerate(candidates, 1)], args.output)
    else:
        {"bib": write_bibtex, "ris": write_ris, "enw": write_enw, "zotero-rdf": write_zotero_rdf}[args.format](candidates, args.output)


if __name__ == "__main__":
    main()
