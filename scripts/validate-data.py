#!/usr/bin/env python3
"""
Data Validation Script for LLM Fine-tuning
Validates processed JSON files and generates comprehensive quality reports
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import logging
import statistics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate processed data quality and generate reports"""
    
    def __init__(self, processed_data_dir: str = "./data/processed"):
        self.processed_data_dir = Path(processed_data_dir)
        self.validation_results = {}
        
    def load_processed_data(self) -> Tuple[List[Dict], List[Dict], Dict]:
        """Load processed documents and chunks"""
        documents_file = self.processed_data_dir / "processed_documents.json"
        chunks_file = self.processed_data_dir / "document_chunks.json"
        summary_file = self.processed_data_dir / "processing_summary.json"
        
        documents = []
        chunks = []
        summary = {}
        
        try:
            with open(documents_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            logger.info(f"Loaded {len(documents)} documents from {documents_file}")
        except FileNotFoundError:
            logger.error(f"Processed documents file not found: {documents_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in documents file: {e}")
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        except FileNotFoundError:
            logger.error(f"Document chunks file not found: {chunks_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in chunks file: {e}")
        
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            logger.info(f"Loaded processing summary from {summary_file}")
        except FileNotFoundError:
            logger.warning(f"Processing summary file not found: {summary_file}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in summary file: {e}")
        
        return documents, chunks, summary
    
    def validate_data_quality(self, documents: List[Dict], chunks: List[Dict]) -> Dict[str, Any]:
        """Validate data quality and generate comprehensive metrics"""
        logger.info("Starting comprehensive data quality validation")
        
        validation_results = {
            'total_documents': len(documents),
            'total_chunks': len(chunks),
            'validation_timestamp': datetime.now().isoformat(),
            'issues': [],
            'warnings': [],
            'quality_metrics': {}
        }
        
        # Document-level validation
        logger.info("Validating documents...")
        doc_validation = self._validate_documents(documents)
        validation_results.update(doc_validation)
        
        # Chunk-level validation
        logger.info("Validating chunks...")
        chunk_validation = self._validate_chunks(chunks)
        validation_results.update(chunk_validation)
        
        # Cross-validation between documents and chunks
        logger.info("Cross-validating documents and chunks...")
        cross_validation = self._cross_validate(documents, chunks)
        validation_results.update(cross_validation)
        
        # Content quality analysis
        logger.info("Analyzing content quality...")
        content_analysis = self._analyze_content_quality(chunks)
        validation_results.update(content_analysis)
        
        # Calculate overall quality score
        validation_results['overall_quality_score'] = self._calculate_quality_score(validation_results)
        
        logger.info("Data quality validation complete")
        return validation_results
    
    def _validate_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """Validate document-level data structure and content"""
        issues = []
        warnings = []
        
        required_fields = ['source_file', 'source_type', 'title', 'content', 'quality_metrics', 'metadata', 'chunk_ids']
        
        # Document structure validation
        for i, doc in enumerate(documents):
            # Check required fields
            missing_fields = [field for field in required_fields if field not in doc]
            if missing_fields:
                issues.append(f"Document {i} ({doc.get('source_file', 'unknown')}): Missing fields {missing_fields}")
            
            # Check content quality
            if 'content' in doc:
                content_length = len(doc['content'])
                if content_length < 100:
                    warnings.append(f"Document {i}: Very short content ({content_length} chars)")
                elif content_length == 0:
                    issues.append(f"Document {i}: Empty content")
            
            # Check quality metrics
            if 'quality_metrics' in doc:
                metrics = doc['quality_metrics']
                word_count = metrics.get('word_count', 0)
                chunk_count = metrics.get('chunk_count', 0)
                relevance_score = metrics.get('epfo_relevance_score', 0)
                
                if word_count < 50:
                    warnings.append(f"Document {i}: Low word count ({word_count})")
                
                if chunk_count == 0:
                    issues.append(f"Document {i}: No chunks created")
                elif chunk_count > 100:
                    warnings.append(f"Document {i}: Very high chunk count ({chunk_count})")
                
                if relevance_score < 0.001:
                    warnings.append(f"Document {i}: Very low EPFO relevance score ({relevance_score:.3f})")
            
            # Check chunk_ids consistency
            if 'chunk_ids' in doc:
                if not doc['chunk_ids']:
                    issues.append(f"Document {i}: Empty chunk_ids list")
                elif not isinstance(doc['chunk_ids'], list):
                    issues.append(f"Document {i}: chunk_ids is not a list")
        
        # Document type and source analysis
        doc_types = Counter(doc.get('source_type', 'unknown') for doc in documents)
        source_files = Counter(doc.get('source_file', 'unknown') for doc in documents)
        
        # Check for duplicate source files
        duplicates = [file for file, count in source_files.items() if count > 1]
        if duplicates:
            issues.extend([f"Duplicate source file: {file}" for file in duplicates])
        
        # Calculate document statistics
        content_lengths = [len(doc.get('content', '')) for doc in documents]
        word_counts = [doc.get('quality_metrics', {}).get('word_count', 0) for doc in documents]
        relevance_scores = [doc.get('quality_metrics', {}).get('epfo_relevance_score', 0) for doc in documents]
        
        return {
            'document_validation': {
                'issues': issues,
                'warnings': warnings,
                'document_types': dict(doc_types),
                'source_files': dict(source_files),
                'statistics': {
                    'avg_content_length': statistics.mean(content_lengths) if content_lengths else 0,
                    'median_content_length': statistics.median(content_lengths) if content_lengths else 0,
                    'avg_word_count': statistics.mean(word_counts) if word_counts else 0,
                    'avg_relevance_score': statistics.mean(relevance_scores) if relevance_scores else 0,
                    'content_length_range': {
                        'min': min(content_lengths) if content_lengths else 0,
                        'max': max(content_lengths) if content_lengths else 0
                    }
                }
            }
        }
    
    def _validate_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Validate chunk-level data structure and content"""
        issues = []
        warnings = []
        
        required_fields = ['id', 'content', 'metadata', 'source', 'timestamp', 'chunk_index']
        
        chunk_ids = set()
        source_chunk_counts = defaultdict(int)
        
        for i, chunk in enumerate(chunks):
            # Check required fields
            missing_fields = [field for field in required_fields if field not in chunk]
            if missing_fields:
                issues.append(f"Chunk {i}: Missing fields {missing_fields}")
            
            # Check for duplicate IDs
            chunk_id = chunk.get('id')
            if chunk_id:
                if chunk_id in chunk_ids:
                    issues.append(f"Chunk {i}: Duplicate ID {chunk_id}")
                else:
                    chunk_ids.add(chunk_id)
            
            # Check content quality
            content = chunk.get('content', '')
            content_length = len(content)
            
            if content_length == 0:
                issues.append(f"Chunk {i}: Empty content")
            elif content_length < 50:
                warnings.append(f"Chunk {i}: Very short content ({content_length} chars)")
            elif content_length > 2000:
                warnings.append(f"Chunk {i}: Very long content ({content_length} chars)")
            
            # Check content quality indicators
            if content:
                # Check for repetitive content
                words = content.lower().split()
                if len(words) > 10:
                    word_freq = Counter(words)
                    most_common_word_freq = word_freq.most_common(1)[0][1]
                    if most_common_word_freq > len(words) * 0.3:
                        warnings.append(f"Chunk {i}: Highly repetitive content")
                
                # Check for meaningful content
                if len(set(words)) < len(words) * 0.3:  # Low vocabulary diversity
                    warnings.append(f"Chunk {i}: Low vocabulary diversity")
            
            # Check metadata consistency
            metadata = chunk.get('metadata', {})
            if isinstance(metadata, dict):
                source_file = metadata.get('source_file')
                if source_file:
                    source_chunk_counts[source_file] += 1
                
                # Check word count accuracy
                stated_word_count = metadata.get('word_count', 0)
                actual_word_count = len(content.split()) if content else 0
                if abs(stated_word_count - actual_word_count) > 2:  # Allow small margin
                    warnings.append(f"Chunk {i}: Word count mismatch (stated: {stated_word_count}, actual: {actual_word_count})")
            
            # Validate chunk index
            chunk_index = chunk.get('chunk_index')
            if chunk_index is not None and not isinstance(chunk_index, int):
                issues.append(f"Chunk {i}: Invalid chunk_index type")
        
        # Calculate chunk statistics
        chunk_sizes = [len(chunk.get('content', '')) for chunk in chunks]
        chunk_word_counts = [len(chunk.get('content', '').split()) for chunk in chunks]
        
        return {
            'chunk_validation': {
                'issues': issues,
                'warnings': warnings,
                'unique_chunks': len(chunk_ids),
                'chunks_per_source': dict(source_chunk_counts),
                'size_distribution': {
                    'min': min(chunk_sizes) if chunk_sizes else 0,
                    'max': max(chunk_sizes) if chunk_sizes else 0,
                    'avg': statistics.mean(chunk_sizes) if chunk_sizes else 0,
                    'median': statistics.median(chunk_sizes) if chunk_sizes else 0,
                    'std_dev': statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0
                },
                'word_count_distribution': {
                    'min': min(chunk_word_counts) if chunk_word_counts else 0,
                    'max': max(chunk_word_counts) if chunk_word_counts else 0,
                    'avg': statistics.mean(chunk_word_counts) if chunk_word_counts else 0,
                    'median': statistics.median(chunk_word_counts) if chunk_word_counts else 0
                }
            }
        }
    
    def _cross_validate(self, documents: List[Dict], chunks: List[Dict]) -> Dict[str, Any]:
        """Cross-validate consistency between documents and chunks"""
        issues = []
        warnings = []
        
        # Collect chunk IDs referenced by documents
        doc_chunk_ids = set()
        doc_chunk_mapping = {}
        
        for i, doc in enumerate(documents):
            chunk_ids = doc.get('chunk_ids', [])
            doc_chunk_ids.update(chunk_ids)
            
            source_file = doc.get('source_file', f'doc_{i}')
            doc_chunk_mapping[source_file] = set(chunk_ids)
        
        # Collect actual chunk IDs
        actual_chunk_ids = {chunk.get('id') for chunk in chunks if chunk.get('id')}
        
        # Find orphaned chunks (chunks not referenced by any document)
        orphaned_chunks = actual_chunk_ids - doc_chunk_ids
        if orphaned_chunks:
            warnings.append(f"Found {len(orphaned_chunks)} orphaned chunks not referenced by documents")
        
        # Find missing chunks (chunks referenced by documents but don't exist)
        missing_chunks = doc_chunk_ids - actual_chunk_ids
        if missing_chunks:
            issues.append(f"Found {len(missing_chunks)} missing chunks referenced by documents but not found")
        
        # Validate chunk source consistency
        chunk_source_mapping = defaultdict(set)
        for chunk in chunks:
            source = chunk.get('source') or chunk.get('metadata', {}).get('source_file')
            if source:
                chunk_source_mapping[source].add(chunk.get('id'))
        
        # Compare document chunk counts with actual chunks
        chunk_count_mismatches = []
        for doc in documents:
            source_file = doc.get('source_file')
            expected_count = doc.get('quality_metrics', {}).get('chunk_count', 0)
            actual_count = len(chunk_source_mapping.get(source_file, set()))
            
            if expected_count != actual_count:
                chunk_count_mismatches.append({
                    'source_file': source_file,
                    'expected': expected_count,
                    'actual': actual_count
                })
        
        if chunk_count_mismatches:
            for mismatch in chunk_count_mismatches:
                warnings.append(f"Chunk count mismatch for {mismatch['source_file']}: expected {mismatch['expected']}, found {mismatch['actual']}")
        
        return {
            'cross_validation': {
                'issues': issues,
                'warnings': warnings,
                'orphaned_chunks': len(orphaned_chunks),
                'missing_chunks': len(missing_chunks),
                'chunk_count_mismatches': chunk_count_mismatches,
                'consistency_stats': {
                    'documents_with_chunks': len([d for d in documents if d.get('chunk_ids')]),
                    'sources_with_chunks': len(chunk_source_mapping),
                    'total_doc_chunk_refs': len(doc_chunk_ids),
                    'total_actual_chunks': len(actual_chunk_ids)
                }
            }
        }
    
    def _analyze_content_quality(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze content quality across all chunks"""
        if not chunks:
            return {'content_analysis': {'issues': ['No chunks to analyze'], 'warnings': [], 'quality_indicators': {}}}
        
        issues = []
        warnings = []
        
        # Content quality indicators
        total_chars = 0
        total_words = 0
        language_indicators = defaultdict(int)
        readability_scores = []
        
        # EPFO-specific content analysis
        epfo_keywords = [
            'epfo', 'provident fund', 'pension', 'employee', 'employer', 
            'contribution', 'withdrawal', 'claim', 'uan', 'pf', 'eps'
        ]
        
        keyword_presence = defaultdict(int)
        chunks_with_keywords = 0
        
        # Content patterns
        empty_chunks = 0
        very_short_chunks = 0
        very_long_chunks = 0
        repetitive_chunks = 0
        
        for chunk in chunks:
            content = chunk.get('content', '')
            
            if not content.strip():
                empty_chunks += 1
                continue
            
            content_length = len(content)
            word_count = len(content.split())
            
            total_chars += content_length
            total_words += word_count
            
            # Size analysis
            if content_length < 100:
                very_short_chunks += 1
            elif content_length > 1500:
                very_long_chunks += 1
            
            # Keyword analysis
            content_lower = content.lower()
            chunk_has_keywords = False
            for keyword in epfo_keywords:
                if keyword in content_lower:
                    keyword_presence[keyword] += 1
                    chunk_has_keywords = True
            
            if chunk_has_keywords:
                chunks_with_keywords += 1
            
            # Language and readability indicators
            sentences = re.split(r'[.!?]+', content)
            if len(sentences) > 1:
                avg_sentence_length = word_count / len(sentences)
                readability_scores.append(avg_sentence_length)
            
            # Check for repetitive content
            words = content_lower.split()
            if len(words) > 20:
                word_freq = Counter(words)
                most_common_freq = word_freq.most_common(1)[0][1]
                if most_common_freq > len(words) * 0.2:  # More than 20% repetition
                    repetitive_chunks += 1
            
            # Language detection (basic)
            if re.search(r'[a-zA-Z]', content):
                language_indicators['english'] += 1
            if re.search(r'[^\x00-\x7F]', content):
                language_indicators['non_ascii'] += 1
        
        # Generate quality warnings
        if empty_chunks > 0:
            issues.append(f"{empty_chunks} chunks are empty")
        
        if very_short_chunks > len(chunks) * 0.3:
            warnings.append(f"{very_short_chunks} chunks are very short (may impact quality)")
        
        if very_long_chunks > len(chunks) * 0.1:
            warnings.append(f"{very_long_chunks} chunks are very long (may impact retrieval)")
        
        if repetitive_chunks > len(chunks) * 0.2:
            warnings.append(f"{repetitive_chunks} chunks appear to have repetitive content")
        
        if chunks_with_keywords < len(chunks) * 0.5:
            warnings.append(f"Only {chunks_with_keywords}/{len(chunks)} chunks contain EPFO-related keywords")
        
        # Calculate quality indicators
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        avg_word_count = total_words / len(chunks) if chunks else 0
        avg_readability = statistics.mean(readability_scores) if readability_scores else 0
        keyword_coverage = chunks_with_keywords / len(chunks) if chunks else 0
        
        return {
            'content_analysis': {
                'issues': issues,
                'warnings': warnings,
                'quality_indicators': {
                    'avg_chunk_size': avg_chunk_size,
                    'avg_word_count': avg_word_count,
                    'avg_sentence_length': avg_readability,
                    'keyword_coverage': keyword_coverage,
                    'language_distribution': dict(language_indicators),
                    'content_patterns': {
                        'empty_chunks': empty_chunks,
                        'very_short_chunks': very_short_chunks,
                        'very_long_chunks': very_long_chunks,
                        'repetitive_chunks': repetitive_chunks
                    },
                    'keyword_frequency': dict(keyword_presence)
                }
            }
        }

    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:

        score = 100.0  # Start with perfect score
        
        # Count issues and warnings by severity
        critical_issues = 0
        moderate_issues = 0  
        minor_warnings = 0
        
        # Categorize issues and warnings by severity
        for section in ['document_validation', 'chunk_validation', 'cross_validation', 'content_analysis']:
            if section in validation_results:
                section_data = validation_results[section]
                issues = section_data.get('issues', [])
                warnings = section_data.get('warnings', [])
                
                # Critical issues (data corruption, missing data, etc.)
                critical_patterns = ['missing', 'empty', 'duplicate', 'not found', 'invalid']
                for issue in issues:
                    if any(pattern in issue.lower() for pattern in critical_patterns):
                        critical_issues += 1
                    else:
                        moderate_issues += 1
                
                # Categorize warnings by severity
                for warning in warnings:
                    warning_lower = warning.lower()
                    # High chunk count warnings are less severe now with better chunking
                    if 'very high chunk count' in warning_lower:
                        # Only penalize if extremely high (>300)
                        if 'chunk count (' in warning_lower:
                            try:
                                chunk_count = int(warning_lower.split('chunk count (')[1].split(')')[0])
                                if chunk_count > 300:
                                    minor_warnings += 1
                            except:
                                pass
                    # Very long chunks are less severe if within reasonable bounds
                    elif 'very long' in warning_lower and 'chunk' in warning_lower:
                        if 'chars)' in warning_lower:
                            try:
                                char_count = int(warning_lower.split('(')[1].split(' chars')[0])
                                if char_count > 3000:  # Only penalize truly excessive chunks
                                    minor_warnings += 1
                            except:
                                minor_warnings += 0.5  # Half penalty for parsing issues
                    # Other warnings
                    else:
                        minor_warnings += 1
        
        # Apply penalties with adjusted rates
        score -= (critical_issues * 10)   # 10 points per critical issue
        score -= (moderate_issues * 3)    # 3 points per moderate issue  
        score -= (minor_warnings * 0.5)   # 0.5 points per minor warning (much reduced)
        
        # Quality bonuses based on actual metrics
        if 'content_analysis' in validation_results:
            content_analysis = validation_results['content_analysis']
            quality_indicators = content_analysis.get('quality_indicators', {})
            
            # Bonus for good keyword coverage (your data has 87.4%)
            keyword_coverage = quality_indicators.get('keyword_coverage', 0)
            if keyword_coverage > 0.85:
                score += 10  # Excellent coverage
            elif keyword_coverage > 0.70:
                score += 5   # Good coverage
            elif keyword_coverage > 0.50:
                score += 2   # Acceptable coverage
            
            # Bonus for reasonable chunk sizes (your data: avg 1086, median 790)
            avg_chunk_size = quality_indicators.get('avg_chunk_size', 0)
            if 600 <= avg_chunk_size <= 1200:  # Your range is good
                score += 8
            elif 400 <= avg_chunk_size <= 1500:  # Acceptable range
                score += 4
            
            # Bonus for content diversity
            patterns = quality_indicators.get('content_patterns', {})
            repetitive_chunks = patterns.get('repetitive_chunks', 0)
            total_chunks = validation_results.get('total_chunks', 1)
            
            repetitive_ratio = repetitive_chunks / total_chunks
            if repetitive_ratio < 0.05:  # Less than 5% repetitive
                score += 5
            elif repetitive_ratio < 0.10:  # Less than 10% repetitive
                score += 3
        
        # Bonus for good document relevance (your avg: 0.868)
        if 'document_validation' in validation_results:
            doc_stats = validation_results['document_validation'].get('statistics', {})
            avg_relevance = doc_stats.get('avg_relevance_score', 0)
            if avg_relevance > 0.85:
                score += 8  # Excellent relevance
            elif avg_relevance > 0.70:
                score += 5  # Good relevance
            elif avg_relevance > 0.50:
                score += 2  # Acceptable relevance
        
        # Bonus for good chunk size distribution (your range: 200-4014 is much better)
        if 'chunk_validation' in validation_results:
            chunk_val = validation_results['chunk_validation']
            size_dist = chunk_val.get('size_distribution', {})
            min_size = size_dist.get('min', 0)
            max_size = size_dist.get('max', 0)
            
            # Bonus for reasonable min/max bounds
            if min_size >= 150 and max_size <= 5000:
                score += 5
            elif min_size >= 100 and max_size <= 10000:
                score += 2
        
        # Cross-validation bonus (no orphaned/missing chunks)
        if 'cross_validation' in validation_results:
            cross_val = validation_results['cross_validation']
            orphaned = cross_val.get('orphaned_chunks', 0)
            missing = cross_val.get('missing_chunks', 0)
            
            if orphaned == 0 and missing == 0:
                score += 5  # Perfect consistency
        
        # Final adjustments
        # Don't penalize too heavily for having many documents with reasonable chunk counts
        total_chunks = validation_results.get('total_chunks', 0)
        total_docs = validation_results.get('total_documents', 1)
        avg_chunks_per_doc = total_chunks / total_docs
        
        if 50 <= avg_chunks_per_doc <= 150:  # Reasonable range
            score += 3
        elif avg_chunks_per_doc <= 200:  # Acceptable
            score += 1
        
        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, score))

    def generate_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report"""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("DATA VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Validation completed: {validation_results['validation_timestamp']}")
        report_lines.append(f"Overall Quality Score: {validation_results['overall_quality_score']:.1f}/100")
        report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Documents: {validation_results['total_documents']}")
        report_lines.append(f"Total Chunks: {validation_results['total_chunks']}")
        
        # Count issues and warnings
        total_issues = len(validation_results.get('issues', []))
        total_warnings = len(validation_results.get('warnings', []))
        
        for section in ['document_validation', 'chunk_validation', 'cross_validation', 'content_analysis']:
            if section in validation_results:
                section_data = validation_results[section]
                total_issues += len(section_data.get('issues', []))
                total_warnings += len(section_data.get('warnings', []))
        
        report_lines.append(f"Total Issues: {total_issues}")
        report_lines.append(f"Total Warnings: {total_warnings}")
        report_lines.append("")
        
        # Document Validation
        if 'document_validation' in validation_results:
            doc_val = validation_results['document_validation']
            report_lines.append("DOCUMENT VALIDATION")
            report_lines.append("-" * 40)
            
            # Statistics
            stats = doc_val.get('statistics', {})
            report_lines.append(f"Average content length: {stats.get('avg_content_length', 0):.0f} characters")
            report_lines.append(f"Average word count: {stats.get('avg_word_count', 0):.0f} words")
            report_lines.append(f"Average relevance score: {stats.get('avg_relevance_score', 0):.3f}")
            
            # Document types
            doc_types = doc_val.get('document_types', {})
            if doc_types:
                report_lines.append("Document types:")
                for doc_type, count in doc_types.items():
                    report_lines.append(f"  - {doc_type}: {count}")
            
            # Issues and warnings
            if doc_val.get('issues'):
                report_lines.append("Issues:")
                for issue in doc_val['issues'][:10]:  # Limit to first 10
                    report_lines.append(f"  - {issue}")
                if len(doc_val['issues']) > 10:
                    report_lines.append(f"  ... and {len(doc_val['issues']) - 10} more")
            
            if doc_val.get('warnings'):
                report_lines.append("Warnings:")
                for warning in doc_val['warnings'][:10]:
                    report_lines.append(f"  - {warning}")
                if len(doc_val['warnings']) > 10:
                    report_lines.append(f"  ... and {len(doc_val['warnings']) - 10} more")
            
            report_lines.append("")
        
        # Chunk Validation
        if 'chunk_validation' in validation_results:
            chunk_val = validation_results['chunk_validation']
            report_lines.append("CHUNK VALIDATION")
            report_lines.append("-" * 40)
            
            # Size distribution
            size_dist = chunk_val.get('size_distribution', {})
            report_lines.append(f"Chunk size distribution:")
            report_lines.append(f"  - Min: {size_dist.get('min', 0)} characters")
            report_lines.append(f"  - Max: {size_dist.get('max', 0)} characters")
            report_lines.append(f"  - Average: {size_dist.get('avg', 0):.0f} characters")
            report_lines.append(f"  - Median: {size_dist.get('median', 0):.0f} characters")
            
            # Word count distribution
            word_dist = chunk_val.get('word_count_distribution', {})
            report_lines.append(f"Word count distribution:")
            report_lines.append(f"  - Min: {word_dist.get('min', 0)} words")
            report_lines.append(f"  - Max: {word_dist.get('max', 0)} words")
            report_lines.append(f"  - Average: {word_dist.get('avg', 0):.0f} words")
            
            report_lines.append(f"Unique chunks: {chunk_val.get('unique_chunks', 0)}")
            
            # Issues and warnings (limited)
            if chunk_val.get('issues'):
                report_lines.append("Top issues:")
                for issue in chunk_val['issues'][:5]:
                    report_lines.append(f"  - {issue}")
                if len(chunk_val['issues']) > 5:
                    report_lines.append(f"  ... and {len(chunk_val['issues']) - 5} more")
            
            report_lines.append("")
        
        # Content Analysis
        if 'content_analysis' in validation_results:
            content_analysis = validation_results['content_analysis']
            report_lines.append("CONTENT QUALITY ANALYSIS")
            report_lines.append("-" * 40)
            
            quality_indicators = content_analysis.get('quality_indicators', {})
            
            report_lines.append(f"Keyword coverage: {quality_indicators.get('keyword_coverage', 0):.1%}")
            report_lines.append(f"Average sentence length: {quality_indicators.get('avg_sentence_length', 0):.1f} words")
            
            # Content patterns
            patterns = quality_indicators.get('content_patterns', {})
            if patterns:
                report_lines.append("Content patterns:")
                for pattern, count in patterns.items():
                    if count > 0:
                        report_lines.append(f"  - {pattern.replace('_', ' ').title()}: {count}")
            
            # Top keywords
            keyword_freq = quality_indicators.get('keyword_frequency', {})
            if keyword_freq:
                report_lines.append("Top EPFO keywords:")
                sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
                for keyword, count in sorted_keywords[:10]:
                    report_lines.append(f"  - {keyword}: {count} occurrences")
            
            report_lines.append("")
        
        # Cross Validation
        if 'cross_validation' in validation_results:
            cross_val = validation_results['cross_validation']
            report_lines.append("CROSS VALIDATION")
            report_lines.append("-" * 40)
            
            consistency_stats = cross_val.get('consistency_stats', {})
            report_lines.append(f"Documents with chunks: {consistency_stats.get('documents_with_chunks', 0)}")
            report_lines.append(f"Sources with chunks: {consistency_stats.get('sources_with_chunks', 0)}")
            report_lines.append(f"Orphaned chunks: {cross_val.get('orphaned_chunks', 0)}")
            report_lines.append(f"Missing chunks: {cross_val.get('missing_chunks', 0)}")
            
            mismatches = cross_val.get('chunk_count_mismatches', [])
            if mismatches:
                report_lines.append(f"Chunk count mismatches: {len(mismatches)}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        score = validation_results['overall_quality_score']
        if score >= 90:
            report_lines.append("✓ Excellent data quality! Ready for fine-tuning.")
        elif score >= 75:
            report_lines.append("✓ Good data quality with minor issues to address.")
        elif score >= 60:
            report_lines.append("⚠ Moderate data quality. Review and fix critical issues.")
        else:
            report_lines.append("⚠ Poor data quality. Significant cleanup required.")
        
        if total_issues > 0:
            report_lines.append("• Address critical issues before proceeding with fine-tuning")
        
        if total_warnings > 10:
            report_lines.append("• Review warnings to improve data quality")
        
        # Add specific recommendations based on content analysis
        if 'content_analysis' in validation_results:
            quality_indicators = validation_results['content_analysis'].get('quality_indicators', {})
            
            keyword_coverage = quality_indicators.get('keyword_coverage', 0)
            if keyword_coverage < 0.5:
                report_lines.append("• Low EPFO keyword coverage - ensure data is domain-relevant")
            
            patterns = quality_indicators.get('content_patterns', {})
            if patterns.get('very_short_chunks', 0) > validation_results['total_chunks'] * 0.2:
                report_lines.append("• Many chunks are too short - consider adjusting chunking strategy")
            
            if patterns.get('repetitive_chunks', 0) > validation_results['total_chunks'] * 0.1:
                report_lines.append("• Repetitive content detected - review text cleaning process")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_validation_results(self, validation_results: Dict[str, Any]) -> Path:
        """Save validation results to JSON file"""
        output_file = self.processed_data_dir / "validation_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation results saved to {output_file}")
        return output_file
    
    def save_validation_report(self, report: str) -> Path:
        """Save validation report to text file"""
        output_file = self.processed_data_dir / "validation_report.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to {output_file}")
        return output_file
    
    def validate_and_report(self) -> Tuple[Dict[str, Any], str]:
        """Complete validation workflow with report generation"""
        # Load data
        documents, chunks, summary = self.load_processed_data()
        
        if not documents and not chunks:
            error_msg = "No data found to validate. Please run data processing first."
            logger.error(error_msg)
            return {}, error_msg
        
        # Validate data
        validation_results = self.validate_data_quality(documents, chunks)
        
        # Generate report
        report = self.generate_report(validation_results)
        
        # Save results
        self.save_validation_results(validation_results)
        self.save_validation_report(report)
        
        return validation_results, report


def main():
    """Main execution function"""
    print("Starting Data Validation Pipeline...")
    print("=" * 50)
    
    # Initialize validator
    validator = DataValidator()
    
    try:
        # Run validation and generate report
        validation_results, report = validator.validate_and_report()
        
        if not validation_results:
            print("No data found to validate. Please run data processing first.")
            return 1
        
        # Print report to console
        print(report)
        
        # Print file locations
        print("\nValidation files saved:")
        print(f"  - {validator.processed_data_dir / 'validation_results.json'}")
        print(f"  - {validator.processed_data_dir / 'validation_report.txt'}")
        
        # Return appropriate exit code based on quality score
        quality_score = validation_results.get('overall_quality_score', 0)
        if quality_score >= 75:
            print(f"\n✓ Data validation passed (Score: {quality_score:.1f}/100)")
            return 0
        else:
            print(f"\n⚠ Data validation concerns (Score: {quality_score:.1f}/100)")
            return 1
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        print(f"\nValidation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())