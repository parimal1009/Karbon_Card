"""Main AI Agent for automatic parser generation."""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from loguru import logger
import time

from .llm_client import LLMClient
from .pdf_processor import PDFProcessor
from .code_executor import CodeExecutor
from .prompts import (
    SYSTEM_PROMPT,
    PARSER_GENERATION_PROMPT,
    SELF_FIX_PROMPT,
    CODE_REVIEW_PROMPT,
    OPTIMIZATION_PROMPT
)


class BankParserAgent:
    """AI Agent that generates bank statement parsers automatically."""
    
    def __init__(
        self,
        max_iterations: int = 3,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1
    ):
        """
        Initialize the agent.
        
        Args:
            max_iterations: Maximum self-correction attempts
            model_name: LLM model to use
            temperature: LLM temperature setting
        """
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.current_bank = None
        self.status = "idle"
        self.progress = 0
        
        # Initialize components
        try:
            self.llm_client = LLMClient(
                model_name=model_name,
                temperature=temperature
            )
            self.pdf_processor = PDFProcessor()
            self.code_executor = CodeExecutor()
            
            logger.info(f"Agent initialized with {max_iterations} max iterations")
            logger.info(f"Using model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def analyze_bank_statement(
        self,
        pdf_path: str,
        csv_path: str
    ) -> Dict:
        """
        Analyze bank statement PDF and expected CSV output.
        
        Args:
            pdf_path: Path to sample PDF
            csv_path: Path to expected CSV
            
        Returns:
            Analysis dictionary with extracted info
        """
        logger.info(f"Analyzing bank statement: {pdf_path}")
        self.status = "analyzing"
        self.progress = 20
        
        try:
            # Validate input files
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # Extract PDF content
            pdf_data = self.pdf_processor.analyze_pdf(pdf_path)
            
            # Load expected CSV
            expected_df = pd.read_csv(csv_path)
            
            # Prepare structured data for the prompt
            pdf_content = pdf_data["text"]
            tables = pdf_data.get("tables", [])
            table_content = "\n\n".join([pd.DataFrame(t).to_string() for t in tables])
            
            analysis = {
                "pdf_content": pdf_content,
                "table_content": table_content,
                "csv_schema": list(expected_df.columns),
                "sample_data": expected_df.head(3).to_string(index=False),
                "row_count": len(expected_df),
                "pdf_path": pdf_path,
                "csv_path": csv_path,
                "expected_df": expected_df,
                "pdf_metadata": {
                    "page_count": pdf_data.get("page_count", 0),
                    "text_length": pdf_data.get("text_length", 0),
                    "table_count": pdf_data.get("table_count", 0)
                }
            }
            
            logger.info(f"Analysis complete: {analysis['row_count']} expected rows, "
                       f"{analysis['pdf_metadata']['page_count']} pages")
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def generate_parser_code(
        self,
        bank_name: str,
        analysis: Dict
    ) -> str:
        """
        Generate initial parser code using LLM.
        
        Args:
            bank_name: Name of the bank
            analysis: PDF/CSV analysis data
            
        Returns:
            Generated Python code
        """
        logger.info(f"Generating parser for {bank_name}")
        self.status = "generating"
        self.progress = 40
        
        try:
            user_prompt = PARSER_GENERATION_PROMPT.format(
                bank_name=bank_name,
                pdf_content=analysis["pdf_content"],
                table_content=analysis["table_content"],
                csv_schema=", ".join(analysis["csv_schema"]),
                sample_data=analysis["sample_data"]
            )
            
            logger.debug(f"Sending prompt to LLM (length: {len(user_prompt)} chars)")
            response = self.llm_client.invoke(SYSTEM_PROMPT, user_prompt)
            
            code = self.code_executor.extract_code_from_response(response)
            
            # Validate syntax
            is_valid, syntax_error = self.code_executor.validate_code_syntax(code)
            if not is_valid:
                logger.warning(f"Generated code has syntax errors: {syntax_error}")
                # Try to fix basic syntax issues
                code = self._fix_basic_syntax(code)
            
            logger.info("Initial parser code generated successfully")
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise
    
    def _fix_basic_syntax(self, code: str) -> str:
        """Fix basic syntax issues in generated code."""
        try:
            lines = code.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Fix common issues
                line = line.replace('```', '')  # Remove any remaining markdown
                line = line.replace('python', '')  # Remove language specifier
                
                # Skip empty lines at the beginning
                if not fixed_lines and not line.strip():
                    continue
                    
                fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
        except Exception:
            return code  # Return original if fixing fails
    
    def self_correct(
        self,
        code: str,
        error: str,
        analysis: Dict,
        attempt: int
    ) -> str:
        """
        Self-correct parser code based on error.
        
        Args:
            code: Current parser code
            error: Error message
            analysis: Analysis data
            attempt: Current attempt number
            
        Returns:
            Corrected code
        """
        logger.info(f"Self-correcting (attempt {attempt}/{self.max_iterations})")
        self.status = f"correcting_attempt_{attempt}"
        self.progress = 40 + (attempt * 20)
        
        try:
            # Prepare context for self-correction
            user_prompt = SELF_FIX_PROMPT.format(
                code=code,
                error=error,
                expected_schema=", ".join(analysis["csv_schema"]),
                actual_output="Parser execution failed",
                pdf_sample=analysis["pdf_content"],
                table_sample=analysis["table_content"]
            )
            
            logger.debug(f"Sending correction prompt to LLM")
            response = self.llm_client.invoke(SYSTEM_PROMPT, user_prompt)
            
            corrected_code = self.code_executor.extract_code_from_response(response)
            
            # Validate corrected syntax
            is_valid, syntax_error = self.code_executor.validate_code_syntax(corrected_code)
            if not is_valid:
                logger.warning(f"Corrected code still has syntax errors: {syntax_error}")
                corrected_code = self._fix_basic_syntax(corrected_code)
            
            logger.info(f"Code corrected for attempt {attempt}")
            return corrected_code
            
        except Exception as e:
            logger.error(f"Self-correction failed: {e}")
            return code  # Return original code if correction fails
    
    def optimize_parser(
        self,
        code: str,
        performance_stats: Dict
    ) -> str:
        """
        Optimize working parser code for better performance.
        
        Args:
            code: Working parser code
            performance_stats: Performance statistics
            
        Returns:
            Optimized code
        """
        try:
            user_prompt = OPTIMIZATION_PROMPT.format(
                code=code,
                processing_time=performance_stats.get('execution_time', 0),
                memory_usage=performance_stats.get('memory_usage', 0),
                success_rate=performance_stats.get('success_rate', 100)
            )
            
            response = self.llm_client.invoke(SYSTEM_PROMPT, user_prompt)
            optimized_code = self.code_executor.extract_code_from_response(response)
            
            logger.info("Parser code optimized")
            return optimized_code
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            return code  # Return original if optimization fails
    
    def run(
        self,
        bank_name: str,
        pdf_path: str,
        csv_path: str,
        output_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Main agent loop: analyze → generate → test → self-correct.
        
        Args:
            bank_name: Name of the bank
            pdf_path: Sample PDF path
            csv_path: Expected CSV path
            output_path: Where to save parser
            
        Returns:
            Tuple of (success, error_message)
        """
        start_time = time.time()
        self.current_bank = bank_name
        self.status = "starting"
        self.progress = 0
        
        logger.info(f"Starting agent run for {bank_name}")
        
        try:
            # Step 1: Analyze input files
            self.progress = 10
            analysis = self.analyze_bank_statement(pdf_path, csv_path)
            
            # Step 2: Generate initial parser
            self.progress = 30
            code = self.generate_parser_code(bank_name, analysis)
            
            # Step 3: Test and self-correct loop
            for iteration in range(self.max_iterations):
                self.iteration_count = iteration
                self.status = f"testing_iteration_{iteration + 1}"
                self.progress = 50 + (iteration * 15)
                
                logger.info(f"Testing parser (iteration {iteration + 1}/{self.max_iterations})")
                
                # Save current version
                if not self.code_executor.save_parser(code, output_path):
                    return False, "Failed to save parser code"
                
                # Test parser
                success, result_df, error = self.code_executor.test_parser(
                    output_path,
                    pdf_path,
                    csv_path
                )
                
                if success:
                    # Success! Optionally optimize the code
                    execution_time = time.time() - start_time
                    logger.info(f"✓ Parser generation successful on iteration {iteration + 1}")
                    logger.info(f"Total execution time: {execution_time:.2f}s")
                    
                    self.status = "completed"
                    self.progress = 100
                    
                    # Try to optimize if we have time and it's working well
                    if iteration == 0:  # Only optimize if we succeeded on first try
                        try:
                            stats = self.code_executor.get_execution_stats()
                            if stats.get('execution_time', 0) > 5:  # If slow, try to optimize
                                logger.info("Attempting to optimize parser performance")
                                optimized_code = self.optimize_parser(code, stats)
                                
                                # Test optimized version
                                self.code_executor.save_parser(optimized_code, output_path)
                                opt_success, _, _ = self.code_executor.test_parser(
                                    output_path, pdf_path, csv_path
                                )
                                
                                if not opt_success:
                                    # Revert to working version
                                    logger.warning("Optimization failed, reverting to working version")
                                    self.code_executor.save_parser(code, output_path)
                                else:
                                    logger.info("Parser successfully optimized")
                        except Exception as e:
                            logger.warning(f"Optimization attempt failed: {e}")
                    
                    return True, None
                
                # If not last iteration, try to fix
                if iteration < self.max_iterations - 1:
                    logger.warning(f"Iteration {iteration + 1} failed: {error}")
                    code = self.self_correct(code, error, analysis, iteration + 1)
                else:
                    logger.error(f"Failed after {self.max_iterations} attempts")
                    self.status = "failed"
                    self.progress = 0
                    return False, error
            
            return False, "Max iterations reached without success"
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(error_msg)
            self.status = "failed"
            self.progress = 0
            return False, error_msg
        
        finally:
            execution_time = time.time() - start_time
            logger.info(f"Agent run completed in {execution_time:.2f}s")
    
    def get_status(self) -> Dict:
        """Get current agent status and progress."""
        return {
            "iteration": self.iteration_count,
            "max_iterations": self.max_iterations,
            "current_bank": self.current_bank,
            "status": self.status,
            "progress": self.progress,
            "last_error": self.code_executor.last_error,
            "model_info": self.llm_client.get_model_info(),
            "execution_stats": self.code_executor.get_execution_stats()
        }
    
    def validate_setup(self) -> List[str]:
        """
        Validate agent setup and dependencies.
        
        Returns:
            List of validation issues (empty if all good)
        """
        issues = []
        
        try:
            # Check LLM client
            if not self.llm_client.api_key:
                issues.append("GROQ_API_KEY not configured")
            
            # Check PDF processor dependencies
            try:
                import pytesseract
                import pdf2image
                import pdfplumber
            except ImportError as e:
                issues.append(f"Missing PDF processing dependency: {e}")
            
            # Check OCR configuration
            tesseract_path = os.getenv("TESSERACT_PATH")
            if tesseract_path and not Path(tesseract_path).exists():
                issues.append(f"Tesseract path not found: {tesseract_path}")
            
            poppler_path = os.getenv("POPPLER_PATH")
            if poppler_path and not Path(poppler_path).exists():
                issues.append(f"Poppler path not found: {poppler_path}")
            
            logger.info(f"Setup validation complete: {len(issues)} issues found")
            
        except Exception as e:
            issues.append(f"Validation failed: {str(e)}")
        
        return issues
    
    def reset(self):
        """Reset agent state for new run."""
        self.iteration_count = 0
        self.current_bank = None
        self.status = "idle"
        self.progress = 0
        self.code_executor.last_error = None
        self.code_executor.last_output = None
        logger.info("Agent state reset")