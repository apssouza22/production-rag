"""Convert PDFs in a directory to Markdown with Docling."""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.domain.pdf_parser.factory import make_pdf_parser_service
from src.domain.pdf_parser.parser import PDFParserService

logger = logging.getLogger(__name__)


async def convert_pdfs(input_dir: Path, output_dir: Path, pdf_parser: PDFParserService) -> tuple[int, int]:
    """Convert each PDF in ``input_dir`` to a Markdown file in ``output_dir``.

    Returns a pair containing the number of successfully converted and failed PDFs.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.info("No PDF files found in %s", input_dir)
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    failed = 0

    for pdf_path in pdf_paths:
        try:
            pdf_content = await pdf_parser.parse_pdf(pdf_path)
            if pdf_content is None:
                logger.warning("Parser skipped %s", pdf_path)
                failed += 1
                continue

            markdown: Optional[str] = pdf_content.markdown
            if markdown is None:
                logger.warning("Parser returned no Markdown for %s", pdf_path)
                failed += 1
                continue

            markdown_path = output_dir / f"{pdf_path.stem}.md"
            markdown_path.write_text(markdown, encoding="utf-8")
            logger.info("Converted %s -> %s", pdf_path.name, markdown_path)
            converted += 1
        except Exception:
            logger.exception("Failed to convert %s", pdf_path)
            failed += 1

    return converted, failed


def main() -> int:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown using Docling.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=project_root / "data" / "arxiv_pdfs",
        help="Directory containing PDFs (default: data/arxiv_pdfs)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=project_root / "data" / "arxiv_md",
        help="Directory for generated Markdown (default: data/arxiv_md)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        pdf_parser = make_pdf_parser_service()
        converted, failed = asyncio.run(convert_pdfs(args.input_dir, args.output_dir, pdf_parser))
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Finished: %d converted, %d failed", converted, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
