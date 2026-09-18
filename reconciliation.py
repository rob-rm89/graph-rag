"""Entity reconciliation and cross-paper citation linking.

LightRAG merges graph nodes only on exact name equality, so the same author,
venue or paper written slightly differently in two documents becomes two
nodes.  This module keeps a persistent registry of canonical names and
resolves incoming bibliographic records against it so that

* ``"E. Marchetti"`` and ``"Elena Marchetti"`` become one author node,
* a reference to an already ingested paper links to that paper's node
  (cross-paper citation linking) instead of spawning a ``CitedWork`` twin,
* a paper ingested *after* being cited upgrades its ``CitedWork`` node to a
  ``Paper`` node,
* venues and organisations collapse on normalised names.

A graph-wide pass (:meth:`Reconciler.plan_graph_merges`) additionally finds
near-duplicates produced by LightRAG's own extraction; the ingestion engine
executes the resulting plans with ``LightRAG.amerge_entities``.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.utils import logger

from config import normalize_type

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ingestion import BibliographicRecord

KIND_PAPER = "paper"
KIND_CITEDWORK = "citedwork"
KIND_AUTHOR = "author"
KIND_VENUE = "venue"
KIND_ORGANIZATION = "organization"
WORK_KINDS = frozenset({KIND_PAPER, KIND_CITEDWORK})
NAMED_KINDS = frozenset({KIND_VENUE, KIND_ORGANIZATION})

REGISTRY_FILENAME = "litgraph_registry.json"
TITLE_MATCH_THRESHOLD = 0.92
CONTAINMENT_MIN_CHARS = 25

_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)
_YEAR_PAREN_RE = re.compile(r"\(\s*(?:1[5-9]|20)\d{2}[a-z]?\s*\)")
_TRAILING_QUALIFIER_RE = re.compile(r"\s*\([^()]*\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_PARTICLES = frozenset(
    {
        "van",
        "von",
        "de",
        "der",
        "den",
        "del",
        "della",
        "di",
        "da",
        "du",
        "la",
        "le",
        "dos",
        "das",
    }
)

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalize_doi(value: Any) -> str | None:
    """Extract and canonicalise a DOI (``10.xxxx/...``) from any string."""
    if not value:
        return None
    match = _DOI_RE.search(str(value))
    if not match:
        return None
    return match.group(1).rstrip(".,;:)]}").lower()


def normalize_text(value: Any) -> str:
    """NFKC, casefold, punctuation to spaces, single spaces."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(_NON_ALNUM_RE.sub(" ", text).split())


def normalize_title(value: Any) -> str:
    """Title key: drop parenthesised years, then :func:`normalize_text`."""
    return normalize_text(_YEAR_PAREN_RE.sub(" ", str(value or "")))


def title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def titles_match(a: str, b: str, threshold: float = TITLE_MATCH_THRESHOLD) -> bool:
    """Compare two *normalised* titles."""
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= CONTAINMENT_MIN_CHARS and f" {shorter} " in f" {longer} ":
        return True
    if len(shorter) < 0.7 * len(longer):
        return False
    return title_similarity(a, b) >= threshold


# --------------------------------------------------------------------------- #
# Person names
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PersonName:
    raw: str
    given: tuple[str, ...]
    family: str  # normalised

    @property
    def completeness(self) -> int:
        """Number of spelled-out given names (initials count as zero)."""
        return sum(1 for token in self.given if len(normalize_text(token)) > 1)

    @property
    def key(self) -> str:
        return f"{self.family}|{' '.join(normalize_text(t) for t in self.given)}"

    @property
    def initials_key(self) -> str:
        return f"{self.family}|{''.join(normalize_text(t)[:1] for t in self.given)}"


def parse_person(name: Any) -> PersonName | None:
    """Split a personal name into given tokens and a normalised family name.

    Handles ``"Family, Given"``, glued initials (``"E.M."``), common surname
    particles (``"van der Berg"``) and a trailing disambiguation qualifier
    (``"John Smith (Other University)"``).
    """
    text = unicodedata.normalize("NFKC", str(name or "")).strip()
    text = _TRAILING_QUALIFIER_RE.sub("", text).strip() or text
    if not text:
        return None
    if "," in text:
        family_part, given_part = (part.strip() for part in text.split(",", 1))
    else:
        tokens = text.replace(".", ". ").split()
        if len(tokens) == 1:
            return PersonName(raw=text, given=(), family=normalize_text(tokens[0]))
        idx = len(tokens) - 1
        while idx - 1 > 0 and tokens[idx - 1].casefold().strip(".") in _PARTICLES:
            idx -= 1
        family_part = " ".join(tokens[idx:])
        given_part = " ".join(tokens[:idx])
    given = tuple(
        token.strip(".")
        for token in given_part.replace(".", ". ").split()
        if token.strip(".")
    )
    return PersonName(raw=text, given=given, family=normalize_text(family_part))


def persons_compatible(a: PersonName, b: PersonName) -> bool:
    """True when the names could denote the same person.

    Same family name and, position by position, either matching initials or
    matching spelled-out given names.  ``"E. Marchetti"`` is compatible with
    ``"Elena Marchetti"``; ``"Eva Marchetti"`` is not.
    """
    if not a.family or a.family != b.family:
        return False
    for token_a, token_b in zip(a.given, b.given, strict=False):
        norm_a, norm_b = normalize_text(token_a), normalize_text(token_b)
        if len(norm_a) > 1 and len(norm_b) > 1:
            if norm_a != norm_b:
                return False
        elif norm_a[:1] != norm_b[:1]:
            return False
    return True


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass
class RegistryEntry:
    name: str
    kind: str
    doi: str | None = None
    aliases: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    # External identifiers such as "openalex:A123", "orcid:0000-...", "ror:...".
    ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "doi": self.doi,
            "aliases": self.aliases,
            "docs": self.docs,
            "ids": self.ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryEntry:
        return cls(
            name=str(data["name"]),
            kind=str(data.get("kind") or KIND_CITEDWORK),
            doi=data.get("doi"),
            aliases=list(data.get("aliases") or []),
            docs=list(data.get("docs") or []),
            ids=list(data.get("ids") or []),
        )


def identity_conflict(known: Iterable[str], given: Iterable[str]) -> bool:
    """True when both sides carry identifiers and none agree (distinct entities)."""
    known_set, given_set = set(known), set(given)
    return bool(known_set) and bool(given_set) and known_set.isdisjoint(given_set)


class IdentityUnionFind:
    """Union-find that refuses to bridge clusters with conflicting identifiers."""

    def __init__(self, names: Iterable[str], identity: dict[str, Iterable[str]]):
        self._parent = {name: name for name in names}
        self._ids = {name: set(identity.get(name, ())) for name in self._parent}

    def find(self, name: str) -> str:
        while self._parent[name] != name:
            self._parent[name] = self._parent[self._parent[name]]
            name = self._parent[name]
        return name

    def compatible(self, a: str, b: str) -> bool:
        return not identity_conflict(self._ids[self.find(a)], self._ids[self.find(b)])

    def union(self, a: str, b: str) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return True
        if not self.compatible(root_a, root_b):
            return False
        self._parent[root_b] = root_a
        self._ids[root_a] |= self._ids[root_b]
        return True

    def clusters(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in self._parent:
            grouped[self.find(name)].append(name)
        return sorted(
            (sorted(members) for members in grouped.values()), key=lambda m: m[0]
        )


class EntityRegistry:
    """Persistent map of canonical entity names with alias/DOI indexes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._entries: dict[str, RegistryEntry] = {}
        if path is not None and path.exists():
            self._load()
        self._reindex()

    # -- persistence -------------------------------------------------------- #

    def _load(self) -> None:
        assert self.path is not None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for item in data.get("entries", []):
                entry = RegistryEntry.from_dict(item)
                self._entries[entry.name] = entry
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not read entity registry %s: %s", self.path, exc)
            self._entries = {}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "entries": [
                e.to_dict()
                for e in sorted(self._entries.values(), key=lambda e: e.name)
            ],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- indexes ------------------------------------------------------------ #

    def _reindex(self) -> None:
        self._by_doi: dict[str, RegistryEntry] = {}
        self._by_id: dict[str, RegistryEntry] = {}
        self._work_by_norm: dict[str, RegistryEntry] = {}
        self._work_buckets: dict[str, list[tuple[str, RegistryEntry]]] = defaultdict(
            list
        )
        self._people_by_family: dict[str, list[RegistryEntry]] = defaultdict(list)
        self._named_by_norm: dict[tuple[str, str], RegistryEntry] = {}
        for entry in self._entries.values():
            if entry.doi:
                self._by_doi[entry.doi] = entry
            for identifier in entry.ids:
                self._by_id.setdefault(identifier, entry)
            if entry.kind in WORK_KINDS:
                for label in (entry.name, *entry.aliases):
                    norm = normalize_title(label)
                    if not norm:
                        continue
                    self._work_by_norm.setdefault(norm, entry)
                    self._work_buckets[norm.split(" ", 1)[0]].append((norm, entry))
            elif entry.kind == KIND_AUTHOR:
                person = parse_person(entry.name)
                if person and person.family:
                    self._people_by_family[person.family].append(entry)
            elif entry.kind in NAMED_KINDS:
                for label in (entry.name, *entry.aliases):
                    self._named_by_norm.setdefault(
                        (entry.kind, normalize_text(label)), entry
                    )

    # -- queries ------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def get(self, name: str) -> RegistryEntry | None:
        return self._entries.get(name)

    def entries(self, kind: str | None = None) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if kind is None or e.kind == kind]

    def resolve_work(self, title: Any, doi: Any = None) -> RegistryEntry | None:
        """Find a paper or cited work by DOI, exact normalised title, or fuzzy title."""
        for candidate in (normalize_doi(doi), normalize_doi(title)):
            if candidate and candidate in self._by_doi:
                return self._by_doi[candidate]
        norm = normalize_title(title)
        if not norm:
            return None
        exact = self._work_by_norm.get(norm)
        if exact is not None:
            return exact
        best: tuple[float, RegistryEntry] | None = None
        for other_norm, entry in self._work_buckets.get(norm.split(" ", 1)[0], []):
            if titles_match(norm, other_norm):
                score = title_similarity(norm, other_norm)
                if best is None or score > best[0]:
                    best = (score, entry)
        return best[1] if best else None

    def resolve_by_id(
        self, kind: str, ids: Iterable[str] | None
    ) -> RegistryEntry | None:
        """Exact identity match on an external identifier of the given kind."""
        for identifier in ids or ():
            entry = self._by_id.get(identifier)
            if entry is not None and entry.kind == kind:
                return entry
        return None

    def resolve_person(
        self, name: Any, ids: Iterable[str] | None = None
    ) -> RegistryEntry | None:
        """Find an author by external identifier, exact normalised name, or
        unique compatible initials.

        A shared identifier is decisive regardless of spelling.  Candidates that
        carry identifiers disjoint from the given ones are different people and
        are never matched by name.
        """
        given_ids = [identifier for identifier in ids or () if identifier]
        by_id = self.resolve_by_id(KIND_AUTHOR, given_ids)
        if by_id is not None:
            return by_id
        person = parse_person(name)
        if person is None or not person.family:
            return None
        candidates = [
            entry
            for entry in self._people_by_family.get(person.family, [])
            if not identity_conflict(entry.ids, given_ids)
        ]
        # Prefer an entry literally named like the query over alias matches.
        candidates.sort(key=lambda entry: entry.name != str(name).strip())
        for entry in candidates:
            for label in (entry.name, *entry.aliases):
                known = parse_person(label)
                if known and known.key == person.key:
                    return entry
        compatible = [
            entry
            for entry in candidates
            if (known := parse_person(entry.name)) and persons_compatible(person, known)
        ]
        if len(compatible) == 1:
            return compatible[0]
        if len(compatible) > 1:
            logger.info(
                "Author %r is ambiguous between %s; keeping it separate",
                name,
                [e.name for e in compatible],
            )
        return None

    def resolve_named(
        self, kind: str, name: Any, ids: Iterable[str] | None = None
    ) -> RegistryEntry | None:
        given_ids = [identifier for identifier in ids or () if identifier]
        by_id = self.resolve_by_id(kind, given_ids)
        if by_id is not None:
            return by_id
        entry = self._named_by_norm.get((kind, normalize_text(name)))
        if entry is not None and identity_conflict(entry.ids, given_ids):
            return None
        return entry

    # -- mutations ---------------------------------------------------------- #

    def register(
        self,
        name: str,
        kind: str,
        *,
        doi: Any = None,
        doc: str | None = None,
        ids: Iterable[str] | None = None,
    ) -> RegistryEntry:
        entry = self._entries.get(name)
        if entry is None:
            entry = RegistryEntry(name=name, kind=kind, doi=normalize_doi(doi))
            self._entries[name] = entry
        elif doi and not entry.doi:
            entry.doi = normalize_doi(doi)
        if doc and doc not in entry.docs:
            entry.docs.append(doc)
        for identifier in ids or ():
            if identifier and identifier not in entry.ids:
                entry.ids.append(identifier)
        self._reindex()
        return entry

    def add_alias(
        self,
        entry: RegistryEntry,
        alias: str,
        doc: str | None = None,
        ids: Iterable[str] | None = None,
    ) -> None:
        if alias != entry.name and alias not in entry.aliases:
            entry.aliases.append(alias)
        if doc and doc not in entry.docs:
            entry.docs.append(doc)
        for identifier in ids or ():
            if identifier and identifier not in entry.ids:
                entry.ids.append(identifier)
        self._reindex()

    def set_kind(self, name: str, kind: str) -> None:
        entry = self._entries.get(name)
        if entry is not None and entry.kind != kind:
            entry.kind = kind
            self._reindex()

    def merge_into(
        self, source: str, target: str, kind: str | None = None
    ) -> RegistryEntry:
        """Fold ``source`` into ``target`` (creating ``target`` if unknown)."""
        src = self._entries.pop(source, None)
        dst = self._entries.get(target)
        if dst is None:
            dst = RegistryEntry(
                name=target,
                kind=kind or (src.kind if src else KIND_CITEDWORK),
                doi=src.doi if src else None,
            )
            self._entries[target] = dst
        if kind:
            dst.kind = kind
        if src is not None:
            dst.doi = dst.doi or src.doi
            for alias in (src.name, *src.aliases):
                if alias != dst.name and alias not in dst.aliases:
                    dst.aliases.append(alias)
            for doc in src.docs:
                if doc not in dst.docs:
                    dst.docs.append(doc)
            for identifier in src.ids:
                if identifier not in dst.ids:
                    dst.ids.append(identifier)
        elif source != dst.name and source not in dst.aliases:
            dst.aliases.append(source)
        self._reindex()
        return dst


# --------------------------------------------------------------------------- #
# Reconciler
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MergePlan:
    """Merge ``sources`` into the existing graph node ``target``."""

    sources: tuple[str, ...]
    target: str
    reason: str


@dataclass
class Reconciliation:
    record: BibliographicRecord
    existing: frozenset[str]
    merges: list[MergePlan] = field(default_factory=list)
    linked_papers: list[str] = field(default_factory=list)
    new_cited_works: list[str] = field(default_factory=list)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


class Reconciler:
    """Resolve bibliographic records against an :class:`EntityRegistry`."""

    def __init__(self, registry: EntityRegistry) -> None:
        self.registry = registry

    # -- record-level ------------------------------------------------------- #

    def reconcile(self, record: BibliographicRecord, doc_id: str) -> Reconciliation:
        reg = self.registry
        merges: list[MergePlan] = []
        existing: set[str] = set()
        linked: list[str] = []
        new_cited: list[str] = []

        title = self._reconcile_title(record, doc_id, merges)
        authors, author_map = self._reconcile_authors(record, doc_id, merges, existing)
        venue = self._reconcile_named(KIND_VENUE, record.venue, doc_id, existing)

        def canonical_org(org: str) -> str | None:
            return self._reconcile_named(
                KIND_ORGANIZATION,
                org,
                doc_id,
                existing,
                ids=record.institution_ids.get(org),
            )

        org_map = {org: canonical_org(org) for org in record.affiliations}
        affiliations = _dedupe([name for name in org_map.values() if name])

        # Per-author institutions: remap both keys (authors) and values
        # (organisations) to their canonical names.
        author_affiliations: dict[str, list[str]] = {}
        for author, institutions in record.author_affiliations.items():
            canonical_author = author_map.get(author)
            if canonical_author is None:
                resolved = reg.resolve_person(author, ids=record.author_ids.get(author))
                canonical_author = resolved.name if resolved else author
            names = [
                org_map[org] if org in org_map else canonical_org(org)
                for org in institutions
            ]
            merged = author_affiliations.setdefault(canonical_author, [])
            merged.extend(name for name in names if name)
            author_affiliations[canonical_author] = _dedupe(merged)

        author_ids: dict[str, list[str]] = {}
        for author, ids in record.author_ids.items():
            canonical_author = author_map.get(author, author)
            author_ids[canonical_author] = _dedupe(
                author_ids.get(canonical_author, []) + ids
            )
        institution_ids: dict[str, list[str]] = {}
        for org, ids in record.institution_ids.items():
            canonical = org_map.get(org) or canonical_org(org) or org
            institution_ids[canonical] = _dedupe(
                institution_ids.get(canonical, []) + ids
            )

        references: list[str] = []
        reference_dois: dict[str, str] = {}
        for reference in record.references:
            doi = record.reference_dois.get(reference)
            entry = reg.resolve_work(reference, doi=doi)
            if entry is None:
                reg.register(
                    reference, KIND_CITEDWORK, doi=doi or reference, doc=doc_id
                )
                references.append(reference)
                new_cited.append(reference)
                if doi:
                    reference_dois[reference] = doi
                continue
            if entry.name == title:
                continue  # a reference to the document itself
            reg.add_alias(entry, reference, doc=doc_id)
            if doi and not entry.doi:
                reg.register(entry.name, entry.kind, doi=doi)
            references.append(entry.name)
            existing.add(entry.name)
            if entry.doi:
                reference_dois[entry.name] = entry.doi
            if entry.kind == KIND_PAPER:
                linked.append(entry.name)
        references = _dedupe(references)

        reconciled = replace(
            record,
            title=title,
            authors=authors,
            venue=venue,
            affiliations=affiliations,
            references=references,
            reference_dois=reference_dois,
            author_affiliations=author_affiliations,
            author_ids=author_ids,
            institution_ids=institution_ids,
        )
        if linked:
            logger.info("Cross-paper citations from %r -> %s", title, linked)
        return Reconciliation(
            record=reconciled,
            existing=frozenset(existing),
            merges=merges,
            linked_papers=_dedupe(linked),
            new_cited_works=new_cited,
        )

    def _reconcile_title(
        self, record: BibliographicRecord, doc_id: str, merges: list[MergePlan]
    ) -> str:
        reg = self.registry
        title = record.title.strip()
        entry = reg.resolve_work(title, record.doi)
        if entry is None:
            reg.register(title, KIND_PAPER, doi=record.doi, doc=doc_id)
            return title
        if entry.kind == KIND_CITEDWORK:
            # The paper is now ingested: its own title is authoritative.
            if entry.name != title:
                merges.append(
                    MergePlan(
                        (entry.name,), title, "cited work is now an ingested paper"
                    )
                )
                reg.merge_into(entry.name, title, kind=KIND_PAPER)
            else:
                reg.set_kind(title, KIND_PAPER)
            reg.register(title, KIND_PAPER, doi=record.doi, doc=doc_id)
            logger.info("Upgraded cited work %r to paper %r", entry.name, title)
            return title
        if entry.name != title:
            logger.warning(
                "Title %r resolves to ingested paper %r; treating as the same paper",
                title,
                entry.name,
            )
            reg.add_alias(entry, title, doc=doc_id)
            return entry.name
        reg.register(title, KIND_PAPER, doi=record.doi, doc=doc_id)
        return title

    def _reconcile_authors(
        self,
        record: BibliographicRecord,
        doc_id: str,
        merges: list[MergePlan],
        existing: set[str],
    ) -> tuple[list[str], dict[str, str]]:
        """Return canonical author names and the original -> canonical mapping."""
        reg = self.registry
        canonical: list[str] = []
        mapping: dict[str, str] = {}
        for author in record.authors:
            ids = list(record.author_ids.get(author) or [])
            person = parse_person(author)
            entry = reg.resolve_person(author, ids=ids)
            if entry is None or person is None:
                name = author
                taken = reg.get(author)
                if taken is not None and (
                    taken.kind != KIND_AUTHOR or identity_conflict(taken.ids, ids)
                ):
                    # Same spelling, different identifiers: a distinct person.
                    name = self._disambiguate(author, record, ids)
                    logger.warning(
                        "Author %r (%s) differs from existing %r; recorded as %r",
                        author,
                        ", ".join(ids),
                        taken.name,
                        name,
                    )
                reg.register(name, KIND_AUTHOR, doc=doc_id, ids=ids)
                canonical.append(name)
                mapping[author] = name
                continue
            known = parse_person(entry.name)
            rename_blocked = author in reg and reg.get(author) is not entry
            if (
                known is not None
                and person.completeness > known.completeness
                and not rename_blocked
            ):
                # The new spelling is more complete: it becomes the node, the old
                # node is merged into it once the new node exists.
                merges.append(MergePlan((entry.name,), author, "author name completed"))
                reg.merge_into(entry.name, author, kind=KIND_AUTHOR)
                reg.register(author, KIND_AUTHOR, doc=doc_id, ids=ids)
                canonical.append(author)
                mapping[author] = author
                logger.info("Author %r completed to %r", entry.name, author)
            else:
                reg.add_alias(entry, author, doc=doc_id, ids=ids)
                canonical.append(entry.name)
                mapping[author] = entry.name
                existing.add(entry.name)
        return _dedupe(canonical), mapping

    def _disambiguate(
        self, name: str, record: BibliographicRecord, ids: list[str]
    ) -> str:
        """Qualify a homonym with its institution, else its identifier."""
        institutions = record.author_affiliations.get(name) or []
        qualifier = institutions[0] if institutions else None
        if not qualifier and ids:
            qualifier = ids[0].split(":", 1)[-1]
        candidate = f"{name} ({qualifier})" if qualifier else f"{name} (2)"
        counter = 2
        while candidate in self.registry:
            counter += 1
            candidate = f"{name} ({qualifier or ''}{' ' if qualifier else ''}{counter})"
        return candidate

    def _reconcile_named(
        self,
        kind: str,
        name: str | None,
        doc_id: str,
        existing: set[str],
        ids: Iterable[str] | None = None,
    ) -> str | None:
        if not name:
            return None
        given_ids = [identifier for identifier in ids or () if identifier]
        entry = self.registry.resolve_named(kind, name, ids=given_ids)
        if entry is None:
            canonical = name
            taken = self.registry.get(name)
            if taken is not None and (
                taken.kind != kind or identity_conflict(taken.ids, given_ids)
            ):
                qualifier = given_ids[0].split(":", 1)[-1] if given_ids else "2"
                canonical = f"{name} ({qualifier})"
                logger.warning(
                    "%s %r differs from existing entry; recorded as %r",
                    kind,
                    name,
                    canonical,
                )
            self.registry.register(canonical, kind, doc=doc_id, ids=given_ids)
            return canonical
        self.registry.add_alias(entry, name, doc=doc_id, ids=given_ids)
        existing.add(entry.name)
        return entry.name

    def _identity(self, name: str) -> frozenset[str]:
        """Identifiers the registry knows for a graph node name (DOI for works)."""
        entry = self.registry.get(name)
        if entry is None:
            return frozenset()
        ids = set(entry.ids)
        if entry.doi:
            ids.add(f"doi:{entry.doi}")
        return frozenset(ids)

    # -- graph-wide --------------------------------------------------------- #

    def plan_graph_merges(self, nodes: list[dict[str, Any]]) -> list[MergePlan]:
        """Plan merges for near-duplicate nodes already in the graph."""
        typed = [
            (str(node["id"]), normalize_type(node.get("entity_type")))
            for node in nodes
            if node.get("id")
        ]
        plans: list[MergePlan] = []
        plans.extend(
            self._plan_work_merges([(n, t) for n, t in typed if t in WORK_KINDS])
        )
        plans.extend(
            self._plan_author_merges([n for n, t in typed if t == KIND_AUTHOR])
        )
        for kind in sorted(NAMED_KINDS):
            plans.extend(
                self._plan_named_merges([n for n, t in typed if t == kind], kind)
            )
        if plans:
            logger.info("Graph reconciliation planned %d merge(s)", len(plans))
        return plans

    def _prefer(self, names: list[str]) -> str:
        """Deterministic target choice: registry canonical, then longest, then A-Z."""
        canonical = [n for n in names if n in self.registry]
        pool = canonical or names
        return sorted(pool, key=lambda n: (-len(n), n))[0]

    def _union_shared_identifiers(
        self, uf: IdentityUnionFind, names: list[str]
    ) -> None:
        """Names that share any external identifier denote the same entity."""
        by_id: dict[str, list[str]] = defaultdict(list)
        for name in names:
            for identifier in self._identity(name):
                by_id[identifier].append(name)
        for members in by_id.values():
            for other in members[1:]:
                uf.union(members[0], other)

    def _plan_work_merges(self, works: list[tuple[str, str]]) -> list[MergePlan]:
        names = sorted(name for name, _ in works)
        uf = IdentityUnionFind(names, {name: self._identity(name) for name in names})
        self._union_shared_identifiers(uf, names)

        norms = {name: normalize_title(name) for name in names}
        buckets: dict[str, list[str]] = defaultdict(list)
        for name, norm in norms.items():
            if norm:
                buckets[norm.split(" ", 1)[0]].append(name)
        for members in buckets.values():
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    if titles_match(norms[a], norms[b]) and not uf.union(a, b):
                        logger.info(
                            "Works %r and %r look alike but have distinct DOIs", a, b
                        )

        kinds = dict(works)
        plans: list[MergePlan] = []
        for members in uf.clusters():
            if len(members) < 2:
                continue
            papers = [m for m in members if kinds[m] == KIND_PAPER]
            target = self._prefer(papers or members)
            sources = tuple(sorted(m for m in members if m != target))
            plans.append(MergePlan(sources, target, "duplicate work titles"))
        return plans

    def _plan_author_merges(self, names: list[str]) -> list[MergePlan]:
        names = sorted(names)
        parsed = {name: parse_person(name) for name in names}
        uf = IdentityUnionFind(names, {name: self._identity(name) for name in names})
        # 1. Shared identifiers are decisive, whatever the spelling.
        self._union_shared_identifiers(uf, names)

        # 2. Name heuristics, never across distinct identifiers.
        by_family: dict[str, list[str]] = defaultdict(list)
        for name, person in parsed.items():
            if person and person.family:
                by_family[person.family].append(name)
        for family in sorted(by_family):
            members = by_family[family]
            full = [n for n in members if parsed[n].completeness > 0]  # type: ignore[union-attr]
            partial = [n for n in members if parsed[n].completeness == 0]  # type: ignore[union-attr]
            for i, a in enumerate(full):
                for b in full[i + 1 :]:
                    if parsed[a].key == parsed[b].key:  # type: ignore[union-attr]
                        uf.union(a, b)
            for name in partial:
                person = parsed[name]
                assert person is not None
                roots = {
                    uf.find(other)
                    for other in full
                    if persons_compatible(person, parsed[other])  # type: ignore[arg-type]
                    and uf.compatible(name, other)
                }
                if len(roots) == 1:
                    uf.union(name, next(iter(roots)))
                elif not roots:
                    for other in partial:
                        if (
                            other != name
                            and parsed[other].initials_key == person.initials_key  # type: ignore[union-attr]
                        ):
                            uf.union(name, other)
                else:
                    logger.info("Author %r ambiguous in graph; left unmerged", name)

        plans: list[MergePlan] = []
        for cluster in uf.clusters():
            if len(cluster) < 2:
                continue
            target = sorted(
                cluster,
                key=lambda n: (
                    n not in self.registry,
                    -parsed[n].completeness,  # type: ignore[union-attr]
                    -len(n),
                    n,
                ),
            )[0]
            sources = tuple(sorted(n for n in cluster if n != target))
            plans.append(MergePlan(sources, target, "duplicate author names"))
        return plans

    def _plan_named_merges(self, names: list[str], kind: str) -> list[MergePlan]:
        names = sorted(names)
        uf = IdentityUnionFind(names, {name: self._identity(name) for name in names})
        self._union_shared_identifiers(uf, names)
        groups: dict[str, list[str]] = defaultdict(list)
        for name in names:
            norm = normalize_text(name)
            if norm:
                groups[norm].append(name)
        for members in groups.values():
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    uf.union(a, b)  # refused when identifiers conflict
        plans: list[MergePlan] = []
        for members in uf.clusters():
            if len(members) < 2:
                continue
            target = self._prefer(members)
            sources = tuple(sorted(m for m in members if m != target))
            plans.append(MergePlan(sources, target, f"duplicate {kind} names"))
        return plans

    def absorb_merge(self, plan: MergePlan, kind: str | None = None) -> None:
        """Record an executed merge in the registry."""
        for source in plan.sources:
            self.registry.merge_into(source, plan.target, kind=kind)
