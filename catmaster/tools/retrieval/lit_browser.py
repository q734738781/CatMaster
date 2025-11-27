"""
Selenium based literature search/capture helper.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

from ruamel.yaml import YAML
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@dataclass(slots=True)
class EvidenceAnchor:
    title: str
    authors: List[str]
    publication: str
    year: str
    doi: Optional[str]
    url: str
    selector: Optional[str]
    snippet: str
    acquired_at: str


class LiteratureBrowser:
    def __init__(self, chromedriver_path: Optional[str], headless: bool = True, timeout: int = 20) -> None:
        self.chromedriver_path = chromedriver_path
        self.timeout = timeout
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        if chromedriver_path:
            service = Service(executable_path=chromedriver_path)
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            # Selenium Manager (>=4.6) auto-resolves the proper driver
            self.driver = webdriver.Chrome(options=options)

    def close(self) -> None:
        if self.driver:
            self.driver.quit()

    def __enter__(self) -> "LiteratureBrowser":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def search(self, query: str, site: Optional[str] = None, max_results: int = 10) -> List[EvidenceAnchor]:
        search_query = f"{query}"
        if site:
            search_query = f"site:{site} {query}"
        url = "https://scholar.google.com/scholar?q=" + urllib.parse.quote_plus(search_query)
        self.driver.get(url)
        WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".gs_r"))
        )
        cards = self.driver.find_elements(By.CSS_SELECTOR, ".gs_r")
        results: List[EvidenceAnchor] = []
        for card in cards[:max_results]:
            title_elem = card.find_element(By.CSS_SELECTOR, ".gs_rt")
            title_link = title_elem.find_element(By.TAG_NAME, "a") if title_elem else None
            title = title_link.text if title_link else title_elem.text
            url = title_link.get_attribute("href") if title_link else ""
            snippet_elem = card.find_elements(By.CSS_SELECTOR, ".gs_rs")
            snippet = snippet_elem[0].text if snippet_elem else ""
            author_info_elem = card.find_elements(By.CSS_SELECTOR, ".gs_a")
            author_info = author_info_elem[0].text if author_info_elem else ""
            year = ""
            venue = ""
            if "-" in author_info:
                parts = author_info.split("-")
                if len(parts) >= 2:
                    venue = parts[1].strip()
                year = "".join(ch for ch in author_info if ch.isdigit())[-4:]
            results.append(
                EvidenceAnchor(
                    title=title,
                    authors=_split_authors(author_info),
                    publication=venue,
                    year=year,
                    doi=None,
                    url=url,
                    selector=None,
                    snippet=snippet,
                    acquired_at=_utc_now(),
                )
            )
        return results

    def capture_page(self, target_url: str, wait_seconds: int = 5) -> EvidenceAnchor:
        self.driver.get(target_url)
        time.sleep(wait_seconds)
        title = self.driver.title
        body_elem = self.driver.find_element(By.TAG_NAME, "body")
        snippet = body_elem.text[:1000]
        return EvidenceAnchor(
            title=title,
            authors=[],
            publication="",
            year="",
            doi=None,
            url=target_url,
            selector="body",
            snippet=snippet,
            acquired_at=_utc_now(),
        )


def save_results(results: Iterable[EvidenceAnchor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2, ensure_ascii=False)


def _split_authors(author_info: str) -> List[str]:
    if not author_info:
        return []
    parts = author_info.split("-")[0].split(",")
    return [p.strip() for p in parts if p.strip()]


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


yaml = YAML(typ="safe")


def load_search_config(path: Path = Path("configs/search.yaml")) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.load(fh) or {}


def search(
    query: str,
    site: Optional[str] = None,
    top_k: int = 10,
    headless: Optional[bool] = None,
    config_path: Path = Path("configs/orchestrator.yaml"),
) -> dict:
    config = load_search_config(Path("configs/search.yaml"))
    chromedriver = config.get("literature", {}).get("chromedriver_path") or _chromedriver_from_config(config_path)
    if headless is None:
        headless = bool(config.get("literature", {}).get("headless", True))
    with LiteratureBrowser(chromedriver_path=chromedriver, headless=headless) as browser:
        results = browser.search(query, site=site, max_results=top_k)
    return {"results": [asdict(r) for r in results]}


def capture(
    url: str,
    wait_seconds: int = 5,
    headless: Optional[bool] = None,
    config_path: Path = Path("configs/orchestrator.yaml"),
) -> dict:
    config = load_search_config(Path("configs/search.yaml"))
    chromedriver = config.get("literature", {}).get("chromedriver_path") or _chromedriver_from_config(config_path)
    if headless is None:
        headless = bool(config.get("literature", {}).get("headless", True))
    with LiteratureBrowser(chromedriver_path=chromedriver, headless=headless) as browser:
        result = browser.capture_page(url, wait_seconds=wait_seconds)
    return asdict(result)


def _chromedriver_from_config(config_path: Path) -> Optional[str]:
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.load(fh) or {}
    return data.get("pc", {}).get("chromedriver_path")


