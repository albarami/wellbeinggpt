"""
DOCX Reader Module

Extracts paragraphs from Arabic .docx files deterministically.

Non-negotiable contract for this phase:
- DOCX is source of truth (only ingestion reads it)
- Anchors are stable within the extracted run: source_anchor = "para_<index>"
- Hash is SHA256 of file bytes
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P


@dataclass
class ParsedParagraph:
    """
    A parsed paragraph from the DOCX.

    Attributes:
        para_index: Zero-based paragraph index in document order.
        text: The paragraph text content.
        style: Word style name (e.g., "Heading 1") if available.
        doc_name: Original file name.
        doc_hash: SHA256 hash of file bytes.
        source_anchor: "para_<index>"
    """

    para_index: int
    text: str
    style: Optional[str] = None
    doc_name: str = ""
    doc_hash: str = ""
    source_anchor: str = ""
    runs: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Set source anchor if not provided."""
        if not self.source_anchor:
            self.source_anchor = f"para_{self.para_index}"


@dataclass
class ParsedDocument:
    """
    A fully parsed document with metadata.

    Attributes:
        doc_name: Original file name.
        doc_hash: SHA256 hash of the file content.
        paragraphs: List of parsed paragraphs.
        total_paragraphs: Total number of paragraphs.
    """

    doc_name: str
    doc_hash: str
    paragraphs: list[ParsedParagraph]
    total_paragraphs: int = 0

    def __post_init__(self):
        """Set total paragraphs if not provided."""
        if not self.total_paragraphs:
            self.total_paragraphs = len(self.paragraphs)


class DocxReader:
    """
    Reader for Arabic .docx files.

    This reader:
    1. Parses all paragraphs in document order
    2. Computes SHA256 file hash (doc_hash)
    3. Produces stable within-run anchors: source_anchor = "para_<index>"
    4. Preserves style info and run-level formatting (optional)
    """

    def __init__(self):
        """Initialize the DOCX reader."""
        pass

    def read(self, file_path: str | Path) -> ParsedDocument:
        """
        Read and parse a .docx file.

        Args:
            file_path: Path to the .docx file.

        Returns:
            ParsedDocument: The parsed document with all paragraphs.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid .docx file.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.suffix.lower() == ".docx":
            raise ValueError(f"Not a .docx file: {file_path}")

        # Compute file hash
        doc_hash = self._compute_file_hash(path)

        # Parse document with python-docx
        doc = Document(str(path))
        paragraphs = self._parse_paragraphs(doc, doc_name=path.name, doc_hash=doc_hash)

        return ParsedDocument(
            doc_name=path.name,
            doc_hash=doc_hash,
            paragraphs=paragraphs,
        )

    def read_bytes(self, content: bytes, file_name: str) -> ParsedDocument:
        """
        Read and parse .docx content from bytes.

        Args:
            content: The .docx file content as bytes.
            file_name: Name to use for the document.

        Returns:
            ParsedDocument: The parsed document.
        """
        import io

        # Compute hash from content
        doc_hash = hashlib.sha256(content).hexdigest()

        # Create a BytesIO object for python-docx
        doc_stream = io.BytesIO(content)

        # Parse document
        doc = Document(doc_stream)
        paragraphs = self._parse_paragraphs(doc, doc_name=file_name, doc_hash=doc_hash)

        return ParsedDocument(
            doc_name=file_name,
            doc_hash=doc_hash,
            paragraphs=paragraphs,
        )

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _parse_paragraphs(self, doc: Document, doc_name: str, doc_hash: str) -> list[ParsedParagraph]:
        """
        Parse all paragraphs from a python-docx Document, including table cell paragraphs,
        preserving document order.

        Args:
            doc: The python-docx Document object.

        Returns:
            List of ParsedParagraph objects.
        """
        def iter_block_items(parent) -> list[Paragraph | Table]:
            """
            Yield paragraphs and tables in document order.
            """
            out: list[Paragraph | Table] = []
            body = parent.element.body
            for child in body.iterchildren():
                if isinstance(child, CT_P):
                    out.append(Paragraph(child, parent))
                elif isinstance(child, CT_Tbl):
                    out.append(Table(child, parent))
            return out

        out_paras: list[ParsedParagraph] = []
        global_idx = 0

        for block in iter_block_items(doc):
            if isinstance(block, Paragraph):
                text = (block.text or "").strip()
                style = block.style.name if getattr(block, "style", None) else None
                runs = [r.text for r in block.runs if r.text] if getattr(block, "runs", None) else []
                out_paras.append(
                    ParsedParagraph(
                        para_index=global_idx,
                        text=text,
                        style=style,
                        doc_name=doc_name,
                        doc_hash=doc_hash,
                        source_anchor=f"para_{global_idx}",
                        runs=runs,
                    )
                )
                global_idx += 1
                continue

            # Table: include cell paragraphs row-major.
            if isinstance(block, Table):
                for row in block.rows:
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            text = (p.text or "").strip()
                            style = p.style.name if getattr(p, "style", None) else None
                            runs = [r.text for r in p.runs if r.text] if getattr(p, "runs", None) else []
                            out_paras.append(
                                ParsedParagraph(
                                    para_index=global_idx,
                                    text=text,
                                    style=style,
                                    doc_name=doc_name,
                                    doc_hash=doc_hash,
                                    source_anchor=f"para_{global_idx}",
                                    runs=runs,
                                )
                            )
                            global_idx += 1

        return out_paras

