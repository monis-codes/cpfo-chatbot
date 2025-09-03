#!/usr/bin/env python3

"""
Enhanced Data Processing Script for LLM Fine-tuning
Transforms raw PDFs and TXT files into clean, chunked data for RAG pipeline
"""

import os
import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("Warning: No PDF library available. Install PyMuPDF or pdfplumber for PDF processing")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw PDFs and TXT files into clean, chunked data"""
    
    def __init__(self, raw_data_dir: str = "./data/raw", processed_data_dir: str = "./data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced chunking parameters
        self.max_chunk_size = 800  # Reduced from 500
        self.min_chunk_size = 200  # Increased from 100
        self.overlap_size = 100    # Overlap between chunks for context
        self.max_chunks_per_doc = 200  # Prevent documents with excessive chunks
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'processing_errors': [],
            'skipped_chunks': 0,
            'merged_chunks': 0
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using available library"""
        if not PDF_AVAILABLE:
            raise ImportError("No PDF processing library available")
        
        text_content = ""
        
        try:
            # Try PyMuPDF first
            if 'fitz' in globals():
                doc = fitz.open(pdf_path)
                for page in doc:
                    text_content += page.get_text()
                doc.close()
            
            # Fallback to pdfplumber
            elif 'pdfplumber' in globals():
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            
            logger.info(f"Successfully extracted text from PDF: {pdf_path.name}")
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise
    
    def extract_text_from_txt(self, txt_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            logger.info(f"Successfully read TXT file: {txt_path.name}")
            return content
            
        except Exception as e:
            logger.error(f"Error reading TXT file {txt_path}: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning and normalization"""
        if not text:
            return ""
        
        # Remove common PDF artifacts and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove table of contents patterns
        text = re.sub(r'^\d+\.?\s+[A-Z][^.]*\.\.\.\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove repetitive header/footer patterns
        text = re.sub(r'^[A-Z\s]+$', '', text, flags=re.MULTILINE)
        
        # Normalize whitespace more aggressively
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces after newlines
        
        # Remove excessive repetitive patterns
        text = re.sub(r'(\.\s*){4,}', '... ', text)  # Multiple dots
        text = re.sub(r'(-\s*){4,}', '--- ', text)   # Multiple dashes
        text = re.sub(r'(_\s*){4,}', '___ ', text)   # Multiple underscores
        
        # Handle encoding issues
        text = text.replace('\x00', '')  # Remove null bytes
        text = text.replace('\ufeff', '')  # Remove BOM
        text = text.replace('\r\n', '\n')  # Normalize line endings
        text = text.replace('\r', '\n')
        
        # Remove very short standalone lines (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3 or not line:  # Keep empty lines and lines with >3 chars
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        return text.strip()
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking control"""
        # Simple sentence splitting (can be enhanced with nltk if needed)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_semantic_chunks(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Enhanced semantic chunking with better size control"""
        if not text:
            return []
        
        chunks = []
        
        # First, split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Check if this single paragraph is too large
            if len(paragraph) > self.max_chunk_size * 2:
                # If we have a current chunk, save it first
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunk = self._create_chunk_dict(current_chunk, source_file, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = ""
                
                # Split the large paragraph by sentences
                sentences = self.split_by_sentences(paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) > self.max_chunk_size:
                        if temp_chunk and len(temp_chunk) >= self.min_chunk_size:
                            chunk = self._create_chunk_dict(temp_chunk, source_file, chunk_index)
                            chunks.append(chunk)
                            chunk_index += 1
                            # Add overlap
                            temp_chunk = sentence
                        else:
                            temp_chunk += " " + sentence if temp_chunk else sentence
                    else:
                        temp_chunk += " " + sentence if temp_chunk else sentence
                
                if temp_chunk:
                    current_chunk = temp_chunk
            
            # Normal paragraph processing
            elif current_chunk and len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = self._create_chunk_dict(current_chunk, source_file, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Add overlap from the end of previous chunk
                    overlap = self._get_chunk_overlap(current_chunk)
                    current_chunk = overlap + "\n\n" + paragraph if overlap else paragraph
                else:
                    current_chunk += "\n\n" + paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk = self._create_chunk_dict(current_chunk, source_file, chunk_index)
            chunks.append(chunk)
        elif current_chunk and not chunks:
            # If no chunks were created but we have text, create one anyway
            chunk = self._create_chunk_dict(current_chunk, source_file, 0)
            chunks.append(chunk)
        
        # Post-process chunks to ensure quality
        chunks = self._post_process_chunks(chunks, source_file)
        
        # Limit chunks per document
        if len(chunks) > self.max_chunks_per_doc:
            logger.warning(f"Document {source_file} has {len(chunks)} chunks, limiting to {self.max_chunks_per_doc}")
            chunks = self._merge_excessive_chunks(chunks, source_file)
        
        return chunks
    
    def _get_chunk_overlap(self, chunk: str) -> str:
        """Get overlap text from the end of a chunk"""
        words = chunk.split()
        if len(words) > 20:
            overlap_words = words[-10:]  # Last 10 words
            return " ".join(overlap_words)
        return ""
    
    def _post_process_chunks(self, chunks: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
        """Post-process chunks to improve quality"""
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk["content"]
            
            # Skip chunks that are too repetitive
            if self._is_repetitive_content(content):
                self.stats['skipped_chunks'] += 1
                logger.debug(f"Skipping repetitive chunk in {source_file}")
                continue
            
            # Skip chunks with very low information content
            if self._is_low_information_content(content):
                self.stats['skipped_chunks'] += 1
                logger.debug(f"Skipping low-information chunk in {source_file}")
                continue
            
            # Update metadata
            chunk["metadata"]["word_count"] = len(content.split())
            chunk["metadata"]["char_count"] = len(content)
            chunk["metadata"]["quality_score"] = self._calculate_chunk_quality(content)
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _merge_excessive_chunks(self, chunks: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
        """Merge chunks when there are too many"""
        if len(chunks) <= self.max_chunks_per_doc:
            return chunks
        
        # Calculate how many chunks to merge
        target_chunks = self.max_chunks_per_doc
        merge_ratio = len(chunks) / target_chunks
        
        merged_chunks = []
        current_merged = ""
        merged_count = 0
        chunk_index = 0
        
        for i, chunk in enumerate(chunks):
            if merged_count < merge_ratio and current_merged:
                current_merged += "\n\n" + chunk["content"]
                merged_count += 1
            else:
                if current_merged:
                    merged_chunk = self._create_chunk_dict(current_merged, source_file, chunk_index)
                    merged_chunks.append(merged_chunk)
                    chunk_index += 1
                    self.stats['merged_chunks'] += merged_count
                
                current_merged = chunk["content"]
                merged_count = 1
        
        # Add the last merged chunk
        if current_merged:
            merged_chunk = self._create_chunk_dict(current_merged, source_file, chunk_index)
            merged_chunks.append(merged_chunk)
            self.stats['merged_chunks'] += merged_count
        
        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks for {source_file}")
        return merged_chunks
    
    def _is_repetitive_content(self, content: str) -> bool:
        """Check if content is too repetitive"""
        words = content.lower().split()
        if len(words) < 10:
            return False
        
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        
        return repetition_ratio < 0.3  # Less than 30% unique words
    
    def _is_low_information_content(self, content: str) -> bool:
        """Check if content has low information value"""
        # Check for content that's mostly numbers, punctuation, or very short
        if len(content.strip()) < 50:
            return True
        
        # Check if mostly numbers and basic punctuation
        alphanumeric = re.sub(r'[^a-zA-Z0-9\s]', '', content)
        if len(alphanumeric) / len(content) < 0.5:
            return True
        
        # Check for table-like content (lots of numbers and minimal text)
        words = content.split()
        number_words = sum(1 for word in words if re.match(r'^\d+\.?\d*$', word))
        if len(words) > 0 and number_words / len(words) > 0.5:
            return True
        
        return False
    
    def _calculate_chunk_quality(self, content: str) -> float:
        """Calculate quality score for a chunk"""
        if not content:
            return 0.0
        
        score = 1.0
        
        # Length factor
        if len(content) < self.min_chunk_size:
            score *= 0.5
        elif len(content) > self.max_chunk_size * 2:
            score *= 0.7
        
        # Word count factor
        words = content.split()
        if len(words) < 20:
            score *= 0.6
        
        # Sentence structure
        sentences = self.split_by_sentences(content)
        if len(sentences) < 2:
            score *= 0.8
        
        # EPFO relevance
        epfo_score = self._calculate_epfo_relevance(content)
        score *= (0.5 + epfo_score * 0.5)  # Boost EPFO relevant content
        
        return min(score, 1.0)
    
    def _create_chunk_dict(self, content: str, source_file: str, chunk_index: int) -> Dict[str, Any]:
        """Create a standardized chunk dictionary"""
        chunk_id = f"{Path(source_file).stem}_{chunk_index}_{str(uuid.uuid4())[:8]}"
        
        return {
            "id": chunk_id,
            "content": content.strip(),
            "metadata": {
                "source_file": source_file,
                "chunk_index": chunk_index,
                "word_count": len(content.split()),
                "char_count": len(content),
                "created_at": datetime.now().isoformat(),
                "quality_score": self._calculate_chunk_quality(content)
            },
            "source": source_file,
            "timestamp": datetime.now().isoformat(),
            "chunk_index": chunk_index
        }
    
    def process_single_file(self, file_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process a single file and return chunks and metadata"""
        try:
            logger.info(f"Processing file: {file_path.name}")
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                if not PDF_AVAILABLE:
                    raise ImportError("PDF processing not available")
                raw_text = self.extract_text_from_pdf(file_path)
                source_type = "pdf"
            elif file_path.suffix.lower() == '.txt':
                raw_text = self.extract_text_from_txt(file_path)
                source_type = "txt"
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return [], {}
            
            # Clean the text
            cleaned_text = self.clean_text(raw_text)
            
            if not cleaned_text:
                logger.warning(f"No text extracted from {file_path.name}")
                return [], {}
            
            # Check if document is too large
            if len(cleaned_text) > 100000:  # 100k characters
                logger.warning(f"Large document detected: {file_path.name} ({len(cleaned_text)} chars)")
            
            # Create chunks
            chunks = self.create_semantic_chunks(cleaned_text, file_path.name)
            
            # Calculate quality metrics
            chunk_qualities = [chunk["metadata"]["quality_score"] for chunk in chunks]
            avg_quality = sum(chunk_qualities) / len(chunk_qualities) if chunk_qualities else 0
            
            # Create document metadata
            doc_metadata = {
                "source_file": file_path.name,
                "source_type": source_type,
                "title": file_path.stem,
                "content": cleaned_text[:1000] + "..." if len(cleaned_text) > 1000 else cleaned_text,
                "quality_metrics": {
                    "raw_text_length": len(raw_text),
                    "cleaned_text_length": len(cleaned_text),
                    "word_count": len(cleaned_text.split()),
                    "chunk_count": len(chunks),
                    "avg_chunk_size": sum(len(c["content"]) for c in chunks) / len(chunks) if chunks else 0,
                    "avg_chunk_quality": avg_quality,
                    "epfo_relevance_score": self._calculate_epfo_relevance(cleaned_text)
                },
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "file_size": file_path.stat().st_size,
                    "processing_version": "v2.0"
                },
                "chunk_ids": [chunk["id"] for chunk in chunks]
            }
            
            logger.info(f"Successfully processed {file_path.name}: {len(chunks)} chunks created (avg quality: {avg_quality:.2f})")
            return chunks, doc_metadata
            
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            logger.error(error_msg)
            self.stats['processing_errors'].append(error_msg)
            raise
    
    def _calculate_epfo_relevance(self, text: str) -> float:
        """Calculate EPFO relevance score based on keyword presence"""
        epfo_keywords = [
            'epfo', 'employee provident fund', 'provident fund', 'pf',
            'pension', 'eps', 'employee pension scheme', 'withdrawal',
            'claim', 'settlement', 'contribution', 'employer', 'employee',
            'uan', 'universal account number', 'kyc', 'nomination',
            'transfer', 'balance', 'passbook', 'circular', 'notification',
            'member', 'subscriber', 'beneficiary', 'pensioner'
        ]
        
        text_lower = text.lower()
        matches = 0
        total_weight = 0
        
        # Weighted scoring
        keyword_weights = {
            'epfo': 3, 'employee provident fund': 3, 'provident fund': 2,
            'pension': 2, 'eps': 2, 'employee pension scheme': 3,
            'uan': 2, 'universal account number': 2
        }
        
        for keyword in epfo_keywords:
            weight = keyword_weights.get(keyword, 1)
            count = text_lower.count(keyword)
            matches += count * weight
            total_weight += weight
        
        # Normalize score between 0 and 1
        relevance_score = min(matches / 20, 1.0)  # Adjusted threshold
        return relevance_score
    
    def process_all_files(self) -> Dict[str, Any]:
        """Process all files in the raw data directory"""
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")
        
        all_chunks = []
        all_documents = []
        
        # Get all supported files
        supported_extensions = ['.pdf', '.txt']
        files_to_process = []
        
        for ext in supported_extensions:
            files_to_process.extend(self.raw_data_dir.glob(f"*{ext}"))
        
        self.stats['total_files'] = len(files_to_process)
        logger.info(f"Found {len(files_to_process)} files to process")
        
        if not files_to_process:
            logger.warning("No supported files found in raw data directory")
            return self._create_processing_summary(all_chunks, all_documents)
        
        # Process each file
        for file_path in files_to_process:
            try:
                chunks, doc_metadata = self.process_single_file(file_path)
                
                if chunks:
                    all_chunks.extend(chunks)
                    all_documents.append(doc_metadata)
                    self.stats['processed_files'] += 1
                    self.stats['total_chunks'] += len(chunks)
                else:
                    logger.warning(f"No chunks created for {file_path.name}")
                    self.stats['failed_files'] += 1
                    
            except Exception as e:
                self.stats['failed_files'] += 1
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
        
        # Save processed data
        self._save_processed_data(all_chunks, all_documents)
        
        # Create and return processing summary
        return self._create_processing_summary(all_chunks, all_documents)
    
    def _save_processed_data(self, chunks: List[Dict[str, Any]], documents: List[Dict[str, Any]]):
        """Save processed chunks and documents to JSON files"""
        
        # Save chunks
        chunks_file = self.processed_data_dir / "document_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Save documents
        documents_file = self.processed_data_dir / "processed_documents.json"
        with open(documents_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")
        logger.info(f"Saved {len(documents)} documents to {documents_file}")
    
    def _create_processing_summary(self, chunks: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create processing summary with statistics"""
        
        # Calculate chunk size statistics
        chunk_sizes = [len(chunk["content"]) for chunk in chunks]
        word_counts = [chunk["metadata"]["word_count"] for chunk in chunks]
        quality_scores = [chunk["metadata"]["quality_score"] for chunk in chunks]
        
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "processing_parameters": {
                "max_chunk_size": self.max_chunk_size,
                "min_chunk_size": self.min_chunk_size,
                "overlap_size": self.overlap_size,
                "max_chunks_per_doc": self.max_chunks_per_doc
            },
            "statistics": self.stats.copy(),
            "data_summary": {
                "total_documents": len(documents),
                "total_chunks": len(chunks),
                "avg_chunks_per_document": len(chunks) / len(documents) if documents else 0,
                "document_types": {},
                "total_content_length": sum(chunk_sizes)
            },
            "quality_metrics": {
                "chunk_size_stats": {
                    "min": min(chunk_sizes) if chunk_sizes else 0,
                    "max": max(chunk_sizes) if chunk_sizes else 0,
                    "avg": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                    "median": sorted(chunk_sizes)[len(chunk_sizes)//2] if chunk_sizes else 0
                },
                "word_count_stats": {
                    "min": min(word_counts) if word_counts else 0,
                    "max": max(word_counts) if word_counts else 0,
                    "avg": sum(word_counts) / len(word_counts) if word_counts else 0
                },
                "quality_stats": {
                    "avg_chunk_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                    "min_quality": min(quality_scores) if quality_scores else 0,
                    "max_quality": max(quality_scores) if quality_scores else 0
                },
                "avg_document_relevance": sum(doc["quality_metrics"]["epfo_relevance_score"] for doc in documents) / len(documents) if documents else 0
            }
        }
        
        # Document type distribution
        if documents:
            doc_types = {}
            for doc in documents:
                doc_type = doc.get("source_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            summary["data_summary"]["document_types"] = doc_types
        
        # Save summary
        summary_file = self.processed_data_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing summary saved to {summary_file}")
        return summary


def main():
    """Main execution function"""
    print("Starting Enhanced Data Processing Pipeline...")
    print("=" * 50)
    
    # Initialize processor
    processor = DataProcessor()
    
    try:
        # Process all files
        summary = processor.process_all_files()
        
        # Print results
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"Files processed: {summary['statistics']['processed_files']}/{summary['statistics']['total_files']}")
        print(f"Total chunks created: {summary['statistics']['total_chunks']}")
        print(f"Failed files: {summary['statistics']['failed_files']}")
        print(f"Skipped low-quality chunks: {summary['statistics']['skipped_chunks']}")
        print(f"Merged chunks: {summary['statistics']['merged_chunks']}")
        
        # Print quality metrics
        quality_metrics = summary['quality_metrics']
        print(f"\nChunk Size Stats:")
        print(f"  Min: {quality_metrics['chunk_size_stats']['min']} chars")
        print(f"  Max: {quality_metrics['chunk_size_stats']['max']} chars")
        print(f"  Avg: {quality_metrics['chunk_size_stats']['avg']:.0f} chars")
        
        print(f"\nQuality Stats:")
        print(f"  Avg chunk quality: {quality_metrics['quality_stats']['avg_chunk_quality']:.2f}")
        print(f"  Avg document relevance: {quality_metrics['avg_document_relevance']:.2f}")
        
        if summary['statistics']['processing_errors']:
            print("\nErrors encountered:")
            for error in summary['statistics']['processing_errors']:
                print(f"  - {error}")
        
        print(f"\nProcessed data saved to: {processor.processed_data_dir}")
        print("Files created:")
        print("  - document_chunks.json")
        print("  - processed_documents.json") 
        print("  - processing_summary.json")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        print(f"\nProcessing failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())