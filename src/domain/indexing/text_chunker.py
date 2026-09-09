import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from .models import ChunkMetadata, TextChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """Service for chunking text into overlapping segments.

    Uses word-based chunking with configurable chunk size and overlap.
    Default: 600 words per chunk with 100 word overlap.
    """

    def __init__(self, chunk_size: int = 600, overlap_size: int = 100, min_chunk_size: int = 100):
        """Initialize text chunker.

        :param chunk_size: Target number of words per chunk
        :param overlap_size: Number of overlapping words between chunks
        :param min_chunk_size: Minimum words for a chunk to be valid
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be less than chunk size")

        logger.info(
            f"Text chunker initialized: chunk_size={chunk_size}, overlap_size={overlap_size}, min_chunk_size={min_chunk_size}"
        )

    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words while preserving whitespace information.

        :param text: Input text
        :returns: List of words
        """
        # Split on whitespace while keeping the words
        words = re.findall(r"\S+", text)
        return words

    def _reconstruct_text(self, words: List[str]) -> str:
        """Reconstruct text from words.

        :param words: List of words
        :returns: Reconstructed text
        """
        return " ".join(words)

    def chunk_paper(
        self,
        title: str,
        abstract: str,
        full_text: str,
        arxiv_id: str,
        paper_id: str,
        sections: Optional[Union[Dict[str, str], str, list]] = None,
    ) -> List[TextChunk]:
        """Chunk a paper using hybrid section-based approach.

        Strategy:
        - For sections 100-800 words: Use as single chunk with title+abstract
        - For sections <100 words: Combine with adjacent sections
        - For sections >800 words: Split using traditional word-based chunking
        - Fallback to traditional chunking if no sections available

        :param title: Paper title
        :param abstract: Paper abstract
        :param full_text: Full text content
        :param arxiv_id: ArXiv ID of the paper
        :param paper_id: Database ID of the paper
        :param sections: Dictionary or JSON string of sections
        :returns: List of text chunks with metadata
        """
        # Try section-based chunking first
        if sections:
            try:
                section_chunks = self._chunk_by_sections(title, abstract, arxiv_id, paper_id, sections)
                if section_chunks:
                    logger.info(f"Created {len(section_chunks)} section-based chunks for {arxiv_id}")
                    return section_chunks
            except Exception as e:
                logger.warning(f"Section-based chunking failed for {arxiv_id}: {e}")

        # Fallback to traditional word-based chunking
        logger.info(f"Using traditional word-based chunking for {arxiv_id}")
        return self.chunk_text(full_text, arxiv_id, paper_id)

    def chunk_text(self, text: str, arxiv_id: str, paper_id: str) -> List[TextChunk]:
        """Chunk text into overlapping segments.

        :param text: Full text to chunk
        :param arxiv_id: ArXiv ID of the paper
        :param paper_id: Database ID of the paper
        :returns: List of text chunks with metadata
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for paper {arxiv_id}")
            return []

        # Split text into words
        words = self._split_into_words(text)

        if len(words) < self.min_chunk_size:
            logger.warning(f"Text for paper {arxiv_id} has only {len(words)} words, less than minimum {self.min_chunk_size}")
            # Return single chunk if text is too small
            if words:
                return [
                    TextChunk(
                        text=self._reconstruct_text(words, text),
                        metadata=ChunkMetadata(
                            chunk_index=0,
                            start_char=0,
                            end_char=len(text),
                            word_count=len(words),
                            overlap_with_previous=0,
                            overlap_with_next=0,
                        ),
                        arxiv_id=arxiv_id,
                        paper_id=paper_id,
                    )
                ]
            return []

        chunks = []
        chunk_index = 0
        current_position = 0

        while current_position < len(words):
            # Calculate chunk boundaries
            chunk_start = current_position
            chunk_end = min(current_position + self.chunk_size, len(words))

            # Extract chunk words
            chunk_words = words[chunk_start:chunk_end]
            chunk_text = self._reconstruct_text(chunk_words)

            # Calculate character offsets (approximate)
            start_char = len(" ".join(words[:chunk_start])) if chunk_start > 0 else 0
            end_char = len(" ".join(words[:chunk_end]))

            # Calculate overlaps
            overlap_with_previous = min(self.overlap_size, chunk_start) if chunk_start > 0 else 0
            overlap_with_next = self.overlap_size if chunk_end < len(words) else 0

            # Create chunk
            chunk = TextChunk(
                text=chunk_text,
                metadata=ChunkMetadata(
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    word_count=len(chunk_words),
                    overlap_with_previous=overlap_with_previous,
                    overlap_with_next=overlap_with_next,
                    section_title=None,  # Could be enhanced with section detection
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            chunks.append(chunk)

            # Move to next chunk position (with overlap)
            current_position += self.chunk_size - self.overlap_size
            chunk_index += 1

            # Break if we've processed all words
            if chunk_end >= len(words):
                break

        logger.info(f"Chunked paper {arxiv_id}: {len(words)} words -> {len(chunks)} chunks")

        return chunks

    def _chunk_by_sections(
        self, title: str, abstract: str, arxiv_id: str, paper_id: str, sections: Union[Dict[str, str], str, list]
    ) -> List[TextChunk]:
        """Implement hybrid section-based chunking strategy.

        :param title: Paper title
        :param abstract: Paper abstract
        :param arxiv_id: ArXiv ID
        :param paper_id: Database ID
        :param sections: Sections data
        :returns: List of text chunks
        """
        # Parse sections data
        section_items = self._parse_sections(sections)
        if not section_items:
            return []

        # Filter and clean sections
        section_items = self._filter_sections(section_items, abstract)
        if not section_items:
            logger.warning(f"No meaningful sections found after filtering for {arxiv_id}")
            return []

        # Create header (title + abstract)
        header = f"{title}\n\nAbstract: {abstract}\n\n"

        # Process sections using hybrid strategy
        chunks = []
        small_sections = []  # Buffer for combining small sections

        for i, section in enumerate(section_items):
            section_title = section["title"]
            section_breadcrumb = section.get("breadcrumb")
            section_label = section_breadcrumb or section_title
            content_str = str(section["content"]) if section["content"] else ""
            section_words = len(content_str.split())

            if section_words < 100:
                # Collect small sections to combine later
                small_sections.append((section_title, content_str, section_words, section_breadcrumb))

                # If this is the last section or next section is large, process accumulated small sections
                next_content = str(section_items[i + 1]["content"]) if i < len(section_items) - 1 else ""
                if i == len(section_items) - 1 or len(next_content.split()) >= 100:
                    chunks.extend(self._create_combined_chunk(header, small_sections, chunks, arxiv_id, paper_id))
                    small_sections = []

            elif 100 <= section_words <= 800:
                # Perfect size - create single chunk
                chunk_text = f"{header}Section: {section_label}\n\n{content_str}"
                chunk = self._create_section_chunk(
                    chunk_text, section_title, len(chunks), arxiv_id, paper_id, section_breadcrumb
                )
                chunks.append(chunk)

            else:
                # Large section - split using traditional chunking
                section_text = f"Section: {section_label}\n\n{content_str}"
                full_section_text = f"{header}{section_text}"

                # Use traditional chunking but with section context
                section_chunks = self._split_large_section(
                    full_section_text,
                    header,
                    section_title,
                    len(chunks),
                    arxiv_id,
                    paper_id,
                    section_breadcrumb,
                )
                chunks.extend(section_chunks)

        return chunks

    def _section_label(self, title: str, breadcrumb: Optional[str]) -> str:
        """Return the display label for a section, preferring breadcrumb when available."""
        return breadcrumb or title

    def _normalize_section_dict(
        self,
        title: str,
        content: str,
        level: int = 1,
        breadcrumb: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build a normalized section dictionary."""
        return {
            "title": title,
            "content": content,
            "level": level,
            "breadcrumb": breadcrumb,
        }

    def _parse_sections(self, sections: Union[Dict[str, str], str, list]) -> list[dict[str, Any]]:
        """Parse sections data into a list of section dictionaries."""
        if isinstance(sections, dict):
            return [self._normalize_section_dict(title, content) for title, content in sections.items()]

        if isinstance(sections, list):
            return self._parse_section_list(sections)

        if isinstance(sections, str):
            try:
                parsed = json.loads(sections)
            except json.JSONDecodeError:
                logger.warning("Failed to parse sections JSON")
                return []

            if isinstance(parsed, dict):
                return [self._normalize_section_dict(title, content) for title, content in parsed.items()]
            if isinstance(parsed, list):
                return self._parse_section_list(parsed)

        return []

    def _parse_section_list(self, sections: list) -> list[dict[str, Any]]:
        """Parse a list of section objects into normalized section dictionaries."""
        result: list[dict[str, Any]] = []
        for i, section in enumerate(sections):
            if isinstance(section, dict):
                result.append(
                    self._normalize_section_dict(
                        title=section.get("title", section.get("heading", f"Section {i + 1}")),
                        content=section.get("content", section.get("text", "")),
                        level=section.get("level", 1),
                        breadcrumb=section.get("breadcrumb"),
                    )
                )
            else:
                result.append(self._normalize_section_dict(f"Section {i + 1}", str(section)))
        return result

    def _filter_sections(self, section_items: list[dict[str, Any]], abstract: str) -> list[dict[str, Any]]:
        """Filter out unwanted sections and avoid duplication.

        :param section_items: Parsed section dictionaries
        :param abstract: Paper abstract for duplication check
        :returns: Filtered section dictionaries
        """
        filtered: list[dict[str, Any]] = []
        abstract_words = set(abstract.lower().split())

        for section in section_items:
            section_title = section["title"]
            content_str = str(section["content"]).strip()

            # Skip empty sections
            if not content_str:
                continue

            # Skip metadata/header sections based on title
            if self._is_metadata_section(section_title):
                continue

            # Skip sections that are duplicates of the abstract
            if self._is_duplicate_abstract(content_str, abstract, abstract_words):
                logger.debug(f"Skipping duplicate abstract section: {section_title}")
                continue

            # Skip sections that are too small and contain only metadata
            if len(content_str.split()) < 20 and self._is_metadata_content(content_str):
                logger.debug(f"Skipping metadata section: {section_title}")
                continue

            filtered.append(section)

        return filtered

    def _is_metadata_section(self, section_title: str) -> bool:
        """Check if a section title indicates metadata/header content."""
        title_lower = section_title.lower().strip()

        metadata_indicators = [
            "content",
            "header",
            "authors",
            "author",
            "affiliation",
            "email",
            "arxiv",
            "preprint",
            "submitted",
            "received",
            "accepted",
        ]

        # Exact matches or very short titles that are likely metadata
        if title_lower in metadata_indicators or len(title_lower) < 5:
            return True

        # Check if title contains only metadata indicators
        for indicator in metadata_indicators:
            if indicator in title_lower and len(title_lower) < 20:
                return True

        return False

    def _is_duplicate_abstract(self, content: str, abstract: str, abstract_words: set) -> bool:
        """Check if section content is a duplicate of the abstract."""
        content_lower = content.lower().strip()
        abstract_lower = abstract.lower().strip()

        # Direct string match (allowing for minor formatting differences)
        if abstract_lower in content_lower or content_lower in abstract_lower:
            return True

        # Word overlap check - if >80% of words overlap, likely duplicate
        content_words = set(content_lower.split())

        if len(abstract_words) > 10:  # Only check for substantial abstracts
            overlap = len(abstract_words.intersection(content_words))
            overlap_ratio = overlap / len(abstract_words)

            if overlap_ratio > 0.8:
                return True

        return False

    def _is_metadata_content(self, content: str) -> bool:
        """Check if content contains only metadata (emails, arxiv IDs, etc.)."""
        content_lower = content.lower()

        # Check for common metadata patterns
        metadata_patterns = [
            "@",  # Email addresses
            "arxiv:",  # ArXiv IDs
            "university",
            "institute",
            "department",
            "college",
            "gmail.com",
            "edu",
            "ac.uk",
            "preprint",
        ]

        # If content is mostly metadata patterns
        word_count = len(content.split())
        if word_count < 30:  # Short content
            metadata_word_count = sum(1 for pattern in metadata_patterns if pattern in content_lower)
            if metadata_word_count >= 2:  # Contains multiple metadata indicators
                return True

        return False

    def _create_combined_chunk(
        self, header: str, small_sections: List, existing_chunks: List, arxiv_id: str, paper_id: str
    ) -> List[TextChunk]:
        """Create chunks by combining small sections."""
        if not small_sections:
            return []

        # Combine all small sections
        combined_content = []
        total_words = 0

        for section_title, content, word_count, section_breadcrumb in small_sections:
            section_label = self._section_label(section_title, section_breadcrumb)
            combined_content.append(f"Section: {section_label}\n\n{content}")
            total_words += word_count

        combined_text = f"{header}{'\n\n'.join(combined_content)}"

        # If still too small, combine with previous chunk if possible
        if total_words + len(header.split()) < 200 and existing_chunks:
            # Try to merge with previous chunk
            prev_chunk = existing_chunks[-1]
            merged_text = f"{prev_chunk.text}\n\n{'\n\n'.join(combined_content)}"

            # Update the previous chunk
            existing_chunks[-1] = TextChunk(
                text=merged_text,
                metadata=ChunkMetadata(
                    chunk_index=prev_chunk.metadata.chunk_index,
                    start_char=0,
                    end_char=len(merged_text),
                    word_count=len(merged_text.split()),
                    overlap_with_previous=0,
                    overlap_with_next=0,
                    section_title=f"{prev_chunk.metadata.section_title} + Combined",
                    section_breadcrumb=prev_chunk.metadata.section_breadcrumb,
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            return []

        # Create new chunk with combined content
        sections_titles = [title for title, _, _, _ in small_sections]
        combined_title = " + ".join(sections_titles[:3])  # Limit title length
        if len(sections_titles) > 3:
            combined_title += f" + {len(sections_titles) - 3} more"

        breadcrumbs = [breadcrumb for _, _, _, breadcrumb in small_sections if breadcrumb]
        combined_breadcrumb = " + ".join(breadcrumbs[:3]) if breadcrumbs else None
        if breadcrumbs and len(breadcrumbs) > 3:
            combined_breadcrumb = f"{combined_breadcrumb} + {len(breadcrumbs) - 3} more"

        chunk = self._create_section_chunk(
            combined_text, combined_title, len(existing_chunks), arxiv_id, paper_id, combined_breadcrumb
        )
        return [chunk]

    def _create_section_chunk(
        self,
        chunk_text: str,
        section_title: str,
        chunk_index: int,
        arxiv_id: str,
        paper_id: str,
        section_breadcrumb: Optional[str] = None,
    ) -> TextChunk:
        """Create a single section-based chunk."""
        return TextChunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                word_count=len(chunk_text.split()),
                overlap_with_previous=0,
                overlap_with_next=0,
                section_title=section_title,
                section_breadcrumb=section_breadcrumb,
            ),
            arxiv_id=arxiv_id,
            paper_id=paper_id,
        )

    def _split_large_section(
        self,
        full_section_text: str,
        header: str,
        section_title: str,
        base_chunk_index: int,
        arxiv_id: str,
        paper_id: str,
        section_breadcrumb: Optional[str] = None,
    ) -> List[TextChunk]:
        """Split large sections using traditional word-based chunking."""
        # Remove header from section text for chunking, then add back to each chunk
        section_only = full_section_text[len(header) :]

        # Use traditional chunking on section content
        traditional_chunks = self.chunk_text(section_only, arxiv_id, paper_id)

        # Add header to each chunk and update metadata
        enhanced_chunks = []
        for i, chunk in enumerate(traditional_chunks):
            enhanced_text = f"{header}{chunk.text}"

            enhanced_chunk = TextChunk(
                text=enhanced_text,
                metadata=ChunkMetadata(
                    chunk_index=base_chunk_index + i,
                    start_char=chunk.metadata.start_char,
                    end_char=chunk.metadata.end_char + len(header),
                    word_count=len(enhanced_text.split()),
                    overlap_with_previous=chunk.metadata.overlap_with_previous,
                    overlap_with_next=chunk.metadata.overlap_with_next,
                    section_title=f"{section_title} (Part {i + 1})",
                    section_breadcrumb=section_breadcrumb,
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks
