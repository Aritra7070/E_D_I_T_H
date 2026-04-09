from __future__ import annotations

import re
import time
import warnings
from urllib.parse import urlparse

from duckduckgo_search import DDGS

from models.base import DetectorResult, calibrate_score, clamp

warnings.filterwarnings("ignore", category=RuntimeWarning)


CLICKBAIT_WORDS = {
    "shocking",
    "breaking",
    "secret",
    "leaked",
    "bombshell",
    "urgent",
    "coverup",
    "viral",
    "exclusive",
}

EMOTIONAL_WORDS = {
    "panic",
    "panicking",
    "terrifying",
    "outrageous",
    "massive",
    "explosive",
    "stunning",
    "everyone",
    "never",
    "always",
    "chaos",
    "crisis",
}

GENERIC_NEWS_TOKENS = {
    "media",
    "advisory",
    "match",
    "news",
    "report",
    "reported",
    "reports",
    "according",
    "published",
    "announced",
    "official",
    "officials",
    "season",
    "article",
    "code",
    "conduct",
    "league",
    "premier",
    "indian",
    "tata",
    "stadium",
    "captain",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "this",
    "that",
    "is",
    "are",
    "was",
    "were",
    "it",
    "with",
    "by",
    "on",
    "at",
    "from",
    "as",
    "be",
    "been",
    "before",
    "after",
    "during",
    "against",
    "his",
    "her",
    "its",
    "no",
    "our",
    "they",
    "their",
    "them",
    "team",
    "teams",
    "has",
    "have",
    "had",
    "will",
    "would",
    "could",
    "should",
}

TRUSTED_DOMAIN_SCORES = {
    "apnews.com": 0.95,
    "bbc.com": 0.92,
    "bbc.co.uk": 0.92,
    "reuters.com": 0.96,
    "npr.org": 0.9,
    "pbs.org": 0.89,
    "nytimes.com": 0.87,
    "wsj.com": 0.87,
    "washingtonpost.com": 0.86,
    "theguardian.com": 0.85,
    "usatoday.com": 0.8,
    "cnn.com": 0.78,
    "abcnews.go.com": 0.8,
    "cbsnews.com": 0.8,
    "nbcnews.com": 0.8,
    "bcci.tv": 0.94,
    "iplt20.com": 0.94,
    "espncricinfo.com": 0.9,
    "cricbuzz.com": 0.86,
    "thehindu.com": 0.85,
    "indianexpress.com": 0.84,
    "hindustantimes.com": 0.79,
    "ndtv.com": 0.78,
}

TRUSTED_SOURCE_SCORES = {
    "associated press": 0.95,
    "reuters": 0.96,
    "bbc": 0.92,
    "indian premier league official website": 0.94,
    "ipl": 0.92,
    "sportstar": 0.86,
    "espncricinfo": 0.9,
    "cricbuzz": 0.86,
    "india today": 0.8,
    "the hindu": 0.85,
    "ndtv": 0.78,
}


class TextDetector:
    def __init__(self) -> None:
        self.model_backend = "duckduckgo-search:DDGS.text"

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
        if not denominator:
            return default
        return min(1.0, max(0.0, numerator / denominator))

    def _normalize_text(self, text: str) -> str:
        normalized = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
        normalized = re.sub(r"\bNo\.\s+(?=\d)", "No ", normalized)
        normalized = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", normalized)
        normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", normalized)
        normalized = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", normalized)
        normalized = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", normalized)
        normalized = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _clean_claim_sentence(self, sentence: str) -> str:
        cleaned = sentence.strip(" -")
        for pattern in [
            r"^(?:[A-Z]{2,}(?:\s+[A-Z]{2,}){1,5}\s+)",
            r"^(?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\s+)",
            r"^(?:\d{1,2},\s+\d{4}\s+)",
            r"^(?:Match\s+\d+,\s*[A-Z]{1,5}\s+vs\s+[A-Z]{1,5}\s*-\s*)",
            r"^(?:Code of Conduct\s+)",
        ]:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" -")
        return cleaned or sentence.strip()

    def _extract_claim(self, text: str) -> str:
        normalized_text = self._normalize_text(text)
        sentences = [segment.strip(" -") for segment in re.split(r"(?<=[.!?])\s+", normalized_text) if segment.strip()]
        if not sentences:
            return normalized_text.strip()

        def claim_score(sentence: str) -> float:
            lowered = sentence.lower()
            tokens = sentence.split()
            number_bonus = 0.22 if re.search(r"\d", sentence) else 0.0
            source_bonus = sum(
                lowered.count(token)
                for token in ["said", "reported", "according", "confirmed", "published", "announced", "fined", "denied", "won"]
            ) * 0.12
            entity_bonus = min(len(re.findall(r"\b(?:[A-Z][a-z]+|[A-Z]{2,5})\b", sentence)) * 0.06, 0.3)
            verb_bonus = 0.16 if re.search(r"\b(is|are|was|were|has|have|had|said|fined|won|lost|confirmed|announced|reported)\b", lowered) else 0.0
            uppercase_penalty = min(sum(1 for word in sentence.split() if len(word) > 3 and word.isupper()) * 0.06, 0.3)
            header_penalty = 0.2 if lowered.startswith(("media advisory", "press release", "match ")) else 0.0
            subordinate_penalty = 0.18 if lowered.startswith(("as ", "after ", "because ", "while ", "when ")) else 0.0
            short_penalty = 0.12 if len(tokens) < 8 else 0.0
            return (len(tokens) * 0.03) + number_bonus + source_bonus + entity_bonus + verb_bonus - uppercase_penalty - header_penalty - subordinate_penalty - short_penalty

        return self._clean_claim_sentence(max(sentences, key=claim_score))

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9']+", text.lower()) if token not in STOPWORDS]

    def _build_query(self, claim: str) -> str:
        normalized_claim = self._normalize_text(claim)
        if len(normalized_claim.split()) <= 14:
            return normalized_claim[:140]

        named_terms: list[str] = []
        for match in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,5})\b", normalized_claim):
            term = re.sub(r"^(?:Captain|Match)\s+", "", match.group(0).strip(), flags=re.IGNORECASE)
            lowered = term.lower()
            term_tokens = [token.lower() for token in re.findall(r"[A-Za-z]+", term)]
            if not term_tokens:
                continue
            if (
                lowered in GENERIC_NEWS_TOKENS
                or lowered in STOPWORDS
                or all(token in GENERIC_NEWS_TOKENS or token in STOPWORDS for token in term_tokens)
                or term in named_terms
            ):
                continue
            named_terms.append(term)
            if len(named_terms) >= 5:
                break

        named_term_tokens = {
            token.lower()
            for term in named_terms
            for token in re.findall(r"[A-Za-z]+", term)
        }
        keyword_terms: list[str] = []
        for token in self._tokenize(normalized_claim):
            if token in GENERIC_NEWS_TOKENS or token in named_term_tokens or len(token) <= 2:
                continue
            if token not in keyword_terms:
                keyword_terms.append(token)
            if len(keyword_terms) >= 6:
                break

        number_terms = re.findall(r"\d+(?:\.\d+)?", normalized_claim)[:3]
        query_terms = named_terms + keyword_terms + number_terms
        deduped_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in query_terms:
            term_key = term.lower()
            if term_key in seen_terms:
                continue
            deduped_terms.append(term)
            seen_terms.add(term_key)

        if deduped_terms:
            return " ".join(deduped_terms[:10])
        return normalized_claim[:140]

    def _highlight_terms(self, text: str) -> list[dict[str, object]]:
        highlights: list[dict[str, object]] = []
        for word in CLICKBAIT_WORDS | EMOTIONAL_WORDS:
            for match in re.finditer(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE):
                reason = "Clickbait language detected" if word in CLICKBAIT_WORDS else "Emotional language detected"
                highlights.append({"word": text[match.start() : match.end()], "start": match.start(), "end": match.end(), "reason": reason})
        return sorted(highlights, key=lambda item: (int(item["start"]), int(item["end"])))

    # ------------------------------------------------------------------
    # NEW: Claim coverage — do search results actually mention the claim?
    # ------------------------------------------------------------------
    def _claim_coverage(self, claim: str, articles: list[dict[str, object]]) -> float:
        """Fraction of claim keywords that appear across all search results."""
        claim_tokens = set(self._tokenize(claim))
        if not claim_tokens:
            return 0.0
        covered: set[str] = set()
        for article in articles:
            text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
            covered |= (claim_tokens & set(self._tokenize(text)))
        return self._safe_ratio(len(covered), len(claim_tokens), 0.0)

    # ------------------------------------------------------------------
    # NEW: Contradiction detection — do results debunk / deny the claim?
    # ------------------------------------------------------------------
    def _detect_contradictions(self, claim: str, articles: list[dict[str, object]]) -> tuple[bool, list[str]]:
        """Check if search results contain negation or debunking language."""
        pattern = re.compile(
            r"\b(?:not true|false|fake|debunked|misleading|incorrect|denied|hoax|rumou?r|fabricated|baseless|unfounded|no evidence|disputed|disproven)\b",
            re.IGNORECASE,
        )
        contradicting: list[str] = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            if pattern.search(text):
                contradicting.append(str(article.get("title", "Unknown source")))
        return len(contradicting) > 0, contradicting

    def _domain_trust(self, href: str) -> tuple[str, float]:
        domain = urlparse(href).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        if domain in TRUSTED_DOMAIN_SCORES:
            return domain, TRUSTED_DOMAIN_SCORES[domain]

        for trusted_domain, score in TRUSTED_DOMAIN_SCORES.items():
            if domain.endswith(f".{trusted_domain}") or domain == trusted_domain:
                return domain, score

        return domain, 0.45

    def _source_trust(self, source_name: str, domain: str, domain_trust: float) -> float:
        source_key = source_name.strip().lower()
        source_trust = TRUSTED_SOURCE_SCORES.get(source_key, 0.0)
        return max(domain_trust, source_trust)

    def _search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search the web via DuckDuckGo. Returns empty list on any failure.
        Compatible with duckduckgo-search versions that expose DDGS as a context manager.
        """
        results: list[dict] = []
        try:
            with DDGS(timeout=15) as ddgs:
                for row in ddgs.text(query, region="us-en", safesearch="moderate", max_results=max_results):
                    results.append(row)
        except TypeError:
            try:
                ddgs = DDGS(timeout=15)
                for row in ddgs.text(query, region="us-en", safesearch="moderate", max_results=max_results):
                    results.append(row)
            except Exception as exc:
                print(f"[TextDetector] Web search failed (non-fatal): {exc}")
                results = []
        except Exception as exc:
            print(f"[TextDetector] Web search failed (non-fatal): {exc}")
            results = []
        return results

    def _search_news(self, query: str, max_results: int = 5) -> list[dict]:
        results: list[dict] = []
        try:
            with DDGS(timeout=15) as ddgs:
                for row in ddgs.news(query, region="us-en", safesearch="moderate", max_results=max_results):
                    results.append(row)
        except TypeError:
            try:
                ddgs = DDGS(timeout=15)
                for row in ddgs.news(query, region="us-en", safesearch="moderate", max_results=max_results):
                    results.append(row)
            except Exception as exc:
                print(f"[TextDetector] News search failed (non-fatal): {exc}")
                results = []
        except Exception as exc:
            print(f"[TextDetector] News search failed (non-fatal): {exc}")
            results = []
        return results

    def _search_related_articles(self, query: str, claim: str) -> tuple[list[dict[str, object]], str | None]:
        claim_tokens = set(self._tokenize(claim))
        claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim))
        results: list[dict[str, object]] = []

        # Build multiple query variations for better resilience
        search_queries: list[str] = []
        for candidate in [query, f"\"{claim[:140]}\""]:
            candidate = candidate.strip()
            if candidate and candidate not in search_queries:
                search_queries.append(candidate)
        shortened_query = " ".join(query.split()[:6]).strip()
        if shortened_query and shortened_query not in search_queries:
            search_queries.append(shortened_query)
        # Entity-focused: pick capitalized proper nouns + key verbs
        proper_nouns = re.findall(r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,5})\b", claim)
        if len(proper_nouns) >= 2:
            entity_query = " ".join(dict.fromkeys(proper_nouns[:6]))
            if entity_query not in search_queries:
                search_queries.append(entity_query)
        keyword_query = " ".join(self._build_query(claim).split()[:8]).strip()
        if keyword_query and keyword_query not in search_queries:
            search_queries.append(keyword_query)

        raw_results: list[dict] = []
        last_error: str | None = None
        MAX_RETRIES = 3

        news_results = self._search_news(query, max_results=5)
        if news_results:
            raw_results = news_results

        for attempt in range(MAX_RETRIES):
            if raw_results:
                break
            if attempt > 0:
                time.sleep(1.5)  # breathing room for rate limits

            aggregated_rows: list[dict] = []
            seen_hrefs: set[str] = set()
            for search_query in search_queries:
                try:
                    batch_results = self._search_web(search_query, max_results=5)
                    for row in batch_results:
                        href = str(row.get("href", row.get("url", ""))).strip()
                        if not href or href in seen_hrefs:
                            continue
                        aggregated_rows.append(row)
                        seen_hrefs.add(href)
                    if len(aggregated_rows) >= 8:
                        break
                except Exception as exc:
                    last_error = str(exc)

            if aggregated_rows:
                raw_results = aggregated_rows
                break

        if not raw_results and last_error:
            return [], last_error

        for index, row in enumerate(raw_results):
            title = str(row.get("title", "")).strip()
            href = str(row.get("href", row.get("url", ""))).strip()
            snippet = str(row.get("body", row.get("snippet", ""))).strip()
            if not href:
                continue

            domain, trust = self._domain_trust(href)
            source_name = str(row.get("source") or domain)
            trust = self._source_trust(source_name, domain, trust)
            article_text = " ".join(part for part in [title, snippet, href] if part)
            article_tokens = set(self._tokenize(article_text))
            overlap = claim_tokens & article_tokens
            overlap_ratio = self._safe_ratio(len(overlap), len(claim_tokens), 0.0)

            title_tokens = set(self._tokenize(title))
            title_overlap = self._safe_ratio(len(claim_tokens & title_tokens), min(len(claim_tokens), 6), 0.0)
            snippet_overlap = self._safe_ratio(len(claim_tokens & set(self._tokenize(snippet))), min(len(claim_tokens), 8), 0.0)
            snippet_numbers = set(re.findall(r"\d+(?:\.\d+)?", snippet))
            number_alignment = 0.12 if claim_numbers and claim_numbers & snippet_numbers else 0.0

            match_score = clamp((overlap_ratio * 0.48) + (title_overlap * 0.24) + (snippet_overlap * 0.16) + (trust * 0.12) + number_alignment)
            results.append(
                {
                    "id": f"search-{index}",
                    "source": source_name,
                    "domain": domain,
                    "title": title or href,
                    "summary": snippet or "No snippet returned from live search.",
                    "href": href,
                    "trust": round(trust, 4),
                    "match_score": round(match_score, 4),
                    "overlap_terms": sorted(overlap),
                }
            )

        results.sort(key=lambda item: (float(item["match_score"]), float(item["trust"])), reverse=True)
        return results[:5], None

    def analyze(self, text: str) -> DetectorResult:
        print(f"[TextDetector] Analyzing text ({len(text)} chars)")
        try:
            return self._analyze_internal(text)
        except Exception as exc:
            import traceback

            print(f"[TextDetector] CRITICAL ERROR: {exc}")
            traceback.print_exc()
            return DetectorResult(
                fake_score=0.5,
                confidence=0.4,
                explanation="Text analysis encountered an internal error. Results are inconclusive.",
                details={
                    "error": str(exc),
                    "model_backend": "heuristic:fallback",
                    "reason_labels": ["Analysis pipeline error - manual review recommended"],
                },
                component_scores={
                    "clickbait_score": 0.0,
                    "emotional_score": 0.0,
                    "source_credibility": 0.0,
                },
            )

    def _analyze_internal(self, text: str) -> DetectorResult:
        cleaned_text = self._normalize_text(text)
        if not cleaned_text:
            raise ValueError("Text input is empty.")

        claim = self._extract_claim(cleaned_text)
        query = self._build_query(claim)
        related_articles, search_error = self._search_related_articles(query, claim)
        highlights = self._highlight_terms(cleaned_text)
        tokens = self._tokenize(cleaned_text)

        clickbait_signal = clamp(sum(1 for token in tokens if token in CLICKBAIT_WORDS) / 3.0)
        emotional_signal = clamp(sum(1 for token in tokens if token in EMOTIONAL_WORDS) / 3.0)
        uppercase_signal = clamp(sum(1 for word in claim.split() if len(word) > 4 and word.isupper()) / 6.0)
        punctuation_signal = clamp((cleaned_text.count("!") + cleaned_text.count("?")) / 6.0)
        pressure_signal = clamp((clickbait_signal * 0.45) + (emotional_signal * 0.25) + (uppercase_signal * 0.2) + (punctuation_signal * 0.1))

        # --- Stricter match threshold (was 0.46) ---
        strong_matches = [article for article in related_articles if float(article["match_score"]) >= 0.50 and float(article["trust"]) >= 0.72]
        moderate_matches = [article for article in related_articles if float(article["match_score"]) >= 0.3]
        trusted_matches = [article for article in related_articles if float(article["trust"]) >= 0.72]
        unique_strong_sources = {str(article["source"]) for article in strong_matches}
        average_match = sum(float(article["match_score"]) for article in related_articles) / len(related_articles) if related_articles else 0.0
        average_trust = sum(float(article["trust"]) for article in related_articles) / len(related_articles) if related_articles else 0.0
        best_match = max((float(article["match_score"]) for article in related_articles), default=0.0)
        best_trust = max((float(article["trust"]) for article in related_articles), default=0.0)
        official_or_high_confidence_support = best_match >= 0.62 and best_trust >= 0.88

        # --- NEW: Claim coverage and contradiction analysis ---
        claim_coverage = self._claim_coverage(claim, related_articles)
        claim_found = claim_coverage >= 0.35
        has_contradictions, contradicting_sources = self._detect_contradictions(claim, related_articles)

        # --- Evidence (more skeptical) ---
        evidence: list[str] = []
        if search_error:
            evidence.append("Live search could not verify the claim")
        elif not related_articles:
            print("[TextDetector] No web results - using language-only scoring")
            language_score = (clickbait_signal * 0.5) + (emotional_signal * 0.5)
            final_score = calibrate_score(0.1 + language_score * 0.6)
            evidence.append("No web sources found - scored by language patterns only")
            if clickbait_signal > 0.25:
                evidence.append("Clickbait language detected")
            if emotional_signal > 0.25 or uppercase_signal > 0.2 or punctuation_signal > 0.25:
                evidence.append("Emotional or sensational language detected")
            return DetectorResult(
                fake_score=final_score,
                confidence=0.4,
                explanation="No web sources were found, so EDITH fell back to language-pattern scoring only.",
                details={
                    "text": cleaned_text,
                    "claim": claim,
                    "query": query,
                    "highlights": highlights,
                    "related_articles": [],
                    "search_status": "empty",
                    "search_error": search_error,
                    "claim_coverage": 0.0,
                    "has_contradictions": False,
                    "contradicting_sources": [],
                    "fact_check": {
                        "status": "Suspicious",
                        "confidence": 0.4,
                        "evidence": evidence,
                    },
                    "status_override": "Suspicious",
                    "reason_labels": evidence[:4],
                    "model_backend": self.model_backend,
                },
                component_scores={
                    "weak_evidence_signal": 1.0,
                    "low_trust_signal": 1.0,
                    "low_match_signal": 1.0,
                    "claim_gap_signal": 1.0,
                    "clickbait_signal": clickbait_signal,
                    "emotional_signal": clamp(emotional_signal + (uppercase_signal * 0.6) + (punctuation_signal * 0.4)),
                },
            )
        elif not claim_found:
            evidence.append("Claim not found in any search results")
        elif has_contradictions:
            evidence.append("Conflicting information found in sources")
            for source in contradicting_sources[:2]:
                evidence.append(f"Contradicted by: {source}")
        elif len(strong_matches) >= 2 and len(unique_strong_sources) >= 2:
            evidence.append("Multiple independent sources confirm the claim")
        elif official_or_high_confidence_support and claim_found:
            evidence.append("A highly relevant trusted source supports the claim")
        elif strong_matches:
            evidence.append("Only one strong source match was found")
        elif moderate_matches:
            evidence.append("Search results mention the topic but do not strongly verify the claim")
        else:
            evidence.append("No strong evidence found in live search")

        if trusted_matches and len(strong_matches) < 2:
            evidence.append("Trusted coverage exists but corroboration is limited")
        if clickbait_signal > 0.25:
            evidence.append("Clickbait language detected")
        if emotional_signal > 0.25 or uppercase_signal > 0.2 or punctuation_signal > 0.25:
            evidence.append("Emotional or sensational language detected")

        # --- STRICT status determination ---
        # 1. No results / search error → Unverified
        # 2. Claim not found in results → Suspicious (hard rule)
        # 3. Contradictions found → Unverified
        # 4. 2+ strong independent sources + good coverage → Authentic
        # 5. Everything else → Suspicious (default skeptical)
        if search_error or not related_articles:
            status = "Unverified"
            signal_score = 0.72
        elif not claim_found:
            status = "Suspicious"
            signal_score = calibrate_score(0.58 + (pressure_signal * 0.18))
        elif has_contradictions:
            status = "Unverified"
            signal_score = calibrate_score(0.62 + (pressure_signal * 0.12))
        elif (
            len(strong_matches) >= 2
            and len(unique_strong_sources) >= 2
            and average_match >= 0.45
            and claim_coverage >= 0.45
            and pressure_signal < 0.20
        ):
            status = "Authentic Signals"
            signal_score = calibrate_score(0.16 + (pressure_signal * 0.14) + max(0.0, 0.18 - average_match * 0.2))
        elif official_or_high_confidence_support and claim_coverage >= 0.45 and pressure_signal < 0.22:
            status = "Authentic Signals"
            signal_score = calibrate_score(0.2 + (pressure_signal * 0.12) + max(0.0, 0.16 - best_match * 0.18))
        elif strong_matches or (moderate_matches and average_match >= 0.34):
            status = "Unverified"
            signal_score = calibrate_score(0.41 + (pressure_signal * 0.16) + max(0.0, 0.12 - average_match * 0.08))
        elif len(strong_matches) < 2:
            status = "Suspicious"
            signal_score = calibrate_score(0.52 + (pressure_signal * 0.20) + max(0.0, 0.16 - average_match * 0.1))
        else:
            status = "Suspicious"
            signal_score = calibrate_score(0.48 + (pressure_signal * 0.20) + max(0.0, 0.18 - average_match * 0.1))

        confidence = clamp(
            0.56
            + (0.16 if related_articles else 0.0)
            + min(average_match, 0.22)
            + min(abs(signal_score - 0.5), 0.18)
            - (0.12 if search_error else 0.0)
            - (0.06 if has_contradictions else 0.0)
        )

        if status == "Authentic Signals":
            if official_or_high_confidence_support and len(strong_matches) < 2:
                explanation = (
                    "Live search found a highly relevant trusted source that strongly supports the extracted claim. "
                    + "The wording is comparatively restrained, so EDITH reports authentic signals."
                )
            else:
                explanation = (
                    "Live search found multiple relevant sources that independently confirm the extracted claim. "
                    + "The wording is comparatively restrained, so EDITH reports authentic signals."
                )
        elif status == "Unverified":
            if has_contradictions:
                explanation = (
                    "EDITH found conflicting information across search results for this claim. "
                    + "Some sources contradict the claim, making verification inconclusive."
                )
            elif strong_matches or moderate_matches:
                explanation = (
                    "Live search found some relevant support for the extracted claim, but the corroboration is still incomplete. "
                    + "EDITH reports this as unverified rather than suspicious."
                )
            else:
                explanation = (
                    "EDITH could not gather enough live evidence to verify the extracted claim in real time. "
                    + "This is unverified rather than authentic because the search layer did not return dependable support."
                )
        else:
            if not claim_found:
                explanation = (
                    "The specific claim could not be found in any live search results. "
                    + "EDITH marks this as suspicious because the claim lacks verifiable online presence."
                )
            elif len(strong_matches) < 2:
                explanation = (
                    "Live search did not produce enough independent sources to corroborate the claim. "
                    + "EDITH requires at least two reliable sources before marking content as authentic."
                )
            else:
                explanation = (
                    "Live search did not produce strong corroboration for the extracted claim, or the wording adds pressure cues. "
                    + "EDITH marks this as suspicious because there is not enough strong evidence to call it authentic."
                )

        return DetectorResult(
            fake_score=signal_score,
            confidence=confidence,
            explanation=explanation,
            details={
                "text": cleaned_text,
                "claim": claim,
                "query": query,
                "highlights": highlights,
                "related_articles": related_articles,
                "search_status": "error" if search_error else "ok",
                "search_error": search_error,
                "claim_coverage": round(claim_coverage, 4),
                "has_contradictions": has_contradictions,
                "contradicting_sources": contradicting_sources,
                "fact_check": {
                    "status": status,
                    "confidence": round(confidence, 4),
                    "evidence": evidence,
                },
                "status_override": status,
                "reason_labels": evidence[:4],
                "model_backend": self.model_backend,
            },
            component_scores={
                "weak_evidence_signal": clamp(1.0 - min(len(strong_matches) / 2.0, 1.0)),
                "low_trust_signal": clamp(1.0 - average_trust),
                "low_match_signal": clamp(1.0 - average_match),
                "claim_gap_signal": clamp(1.0 - claim_coverage),
                "clickbait_signal": clickbait_signal,
                "emotional_signal": clamp(emotional_signal + (uppercase_signal * 0.6) + (punctuation_signal * 0.4)),
            },
        )