from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RawDoc:
    source: str
    relative_path: str
    raw_text: str

def load_source(source_dir: Path, source_name: str) -> list[RawDoc]:
    """Recursively find *.md under source_dir, return RawDoc records."""
    docs: list[RawDoc] = []

    for file_path in source_dir.rglob("*.md"):
        if not file_path.is_file():
            continue

        try:
            rel_path = str(file_path.relative_to(source_dir))
            text = file_path.read_text(encoding="utf-8")

            if not text.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                continue

            docs.append(
                RawDoc(
                    source=source_name,
                    relative_path=rel_path,
                    raw_text=text
                )
            )
        except (UnicodeDecodeError, PermissionError) as e:
            logger.warning(f"Skipping {file_path} due to read error: {e}")

    return docs




def load_all_sources(raw_root: Path, source_names: list[str]) -> list[RawDoc]:
    """Call load_source for each configured source, concatenate."""
    all_docs: list[RawDoc] = []

    for source in source_names:
        source_dir = raw_root / source
        if not source_dir.exists() or not source_dir.is_dir():
            logger.warning(f"Source directory does not exist: {source_dir}")
            continue

        docs = load_source(source_dir, source)
        logger.info(f"Loaded {len(docs)} documents from source: {source_dir}")
        all_docs.extend(docs)

    return all_docs