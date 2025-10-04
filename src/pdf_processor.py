"""PDF processing utilities with OCR support."""

import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import pandas as pd


class PDFProcessor:
    """Extract text and tables from PDF files with OCR fallback."""
    
    def __init__(self):
        """Initialize PDF processor with OCR configuration."""
        # Set Tesseract path if provided
        tesseract_path = os.getenv("TESSERACT_PATH")
        if tesseract_path and Path(tesseract_path).exists():
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Tesseract configured at: {tesseract_path}")
        else:
            logger.warning("Tesseract path not configured or invalid")
        
        # Set Poppler path for pdf2image
        self.poppler_path = os.getenv("POPPLER_PATH")
        if self.poppler_path and Path(self.poppler_path).exists():
            logger.info(f"Poppler configured at: {self.poppler_path}")
        else:
            logger.warning("Poppler path not configured")
            self.poppler_path = None
        
        logger.info("PDF Processor initialized")
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- Page {page_num} ---\n{text}")
                    else:
                        logger.warning(f"No text found on page {page_num}")
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from {pdf_path}")
            return full_text
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    def extract_tables(self, pdf_path: str) -> List[List[List]]:
        """
        Extract tables from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tables (each table is a list of rows)
        """
        try:
            all_tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    if tables:
                        logger.info(f"Found {len(tables)} tables on page {page_num}")
                        all_tables.extend(tables)
            
            logger.info(f"Extracted {len(all_tables)} tables from {pdf_path}")
            return all_tables
            
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []
    
    def extract_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text using OCR for scanned PDFs.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            OCR extracted text
        """
        try:
            # Convert PDF to images
            convert_kwargs = {}
            if self.poppler_path:
                convert_kwargs['poppler_path'] = self.poppler_path
            
            images = convert_from_path(pdf_path, **convert_kwargs)
            
            text_content = []
            for i, image in enumerate(images, 1):
                logger.info(f"Processing page {i} with OCR...")
                text = pytesseract.image_to_string(image, lang='eng')
                if text.strip():
                    text_content.append(f"--- Page {i} (OCR) ---\n{text}")
            
            full_text = "\n\n".join(text_content)
            logger.info(f"OCR extracted {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    def extract_hybrid(self, pdf_path: str) -> str:
        """
        Hybrid extraction: try pdfplumber first, fallback to OCR.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text using best available method
        """
        try:
            # Try pdfplumber first
            text = self.extract_text(pdf_path)
            
            # Check if we got meaningful content
            if len(text.strip()) > 100:  # Arbitrary threshold
                logger.info("Using pdfplumber extraction")
                return text
            else:
                logger.info("Pdfplumber extraction insufficient, trying OCR")
                return self.extract_with_ocr(pdf_path)
                
        except Exception as e:
            logger.warning(f"Pdfplumber failed, trying OCR: {e}")
            return self.extract_with_ocr(pdf_path)
    
    def analyze_pdf(self, pdf_path: str) -> Dict:
        """
        Comprehensive PDF analysis with multiple extraction methods.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with text, tables, and metadata
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            # Extract text using hybrid method
            text = self.extract_hybrid(str(pdf_path))
            
            # Extract tables
            tables = self.extract_tables(str(pdf_path))
            
            # Get PDF metadata
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_count = len(pdf.pages)
                metadata = pdf.metadata or {}
            
            analysis = {
                "text": text,
                "tables": tables,
                "path": str(pdf_path),
                "filename": pdf_path.name,
                "page_count": page_count,
                "metadata": metadata,
                "text_length": len(text),
                "table_count": len(tables)
            }
            
            logger.info(f"PDF analysis complete: {page_count} pages, {len(text)} chars, {len(tables)} tables")
            return analysis
            
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            raise
    
    def tables_to_dataframes(self, tables: List[List[List]]) -> List[pd.DataFrame]:
        """
        Convert extracted tables to pandas DataFrames.
        
        Args:
            tables: List of table data
            
        Returns:
            List of pandas DataFrames
        """
        dataframes = []
        
        for i, table in enumerate(tables):
            try:
                if not table or len(table) < 2:
                    continue
                
                # Use first row as headers
                headers = table[0]
                data = table[1:]
                
                df = pd.DataFrame(data, columns=headers)
                
                # Clean up the DataFrame
                df = df.dropna(how='all')  # Remove empty rows
                df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
                
                dataframes.append(df)
                logger.info(f"Converted table {i+1} to DataFrame: {df.shape}")
                
            except Exception as e:
                logger.warning(f"Failed to convert table {i+1} to DataFrame: {e}")
                continue
        
        return dataframes
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Detect if PDF is scanned (image-based) or text-based.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if PDF appears to be scanned
        """
        try:
            text = self.extract_text(pdf_path)
            # If very little text is extracted, it's likely scanned
            return len(text.strip()) < 50
        except Exception:
            return True  # Assume scanned if extraction fails
