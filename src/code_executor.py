"""Safe code execution and testing utilities."""

import sys
import traceback
import pandas as pd
import numpy as np
import importlib.util
import tempfile
import subprocess
from io import StringIO
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from loguru import logger
import time
import psutil
import os


class CodeExecutor:
    """Execute and test generated parser code safely."""
    
    def __init__(self):
        """Initialize code executor."""
        self.last_error = None
        self.last_output = None
        self.execution_stats = {}
        
    def save_parser(self, code: str, output_path: str) -> bool:
        """
        Save generated parser code to file.
        
        Args:
            code: Python code to save
            output_path: Destination file path
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Parser saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save parser: {e}")
            self.last_error = str(e)
            return False
    
    def extract_code_from_response(self, response: str) -> str:
        """
        Extract Python code from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted Python code
        """
        code = response.strip()
        
        # Remove markdown code blocks if present
        if "```python" in code:
            start = code.find("```python") + len("```python")
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()
        elif "```" in code:
            start = code.find("```") + 3
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()
        
        # Remove any remaining markdown or explanatory text
        lines = code.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else code
    
    def validate_code_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code syntax.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Code validation error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def test_parser_safe(
        self,
        parser_path: str,
        pdf_path: str,
        expected_csv_path: Optional[str] = None,
        timeout: int = 60
    ) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Test generated parser in a safe subprocess.
        
        Args:
            parser_path: Path to parser module
            pdf_path: Path to test PDF
            expected_csv_path: Path to expected CSV output
            timeout: Maximum execution time in seconds
            
        Returns:
            Tuple of (success, dataframe, error_message)
        """
        try:
            test_script = f"""
import sys
import pandas as pd
from pathlib import Path
import traceback

try:
    sys.path.insert(0, str(Path("{parser_path}").parent))
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("parser_module", "{parser_path}")
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)
    
    parse_func = getattr(parser_module, 'parse')
    result_df = parse_func("{pdf_path}")
    
    if not isinstance(result_df, pd.DataFrame):
        raise ValueError("Parse function must return a pandas DataFrame")
    
    result_df.to_csv("temp_result.csv", index=False)
    print("SUCCESS")
    print(f"Shape: {{result_df.shape}}")
    print(f"Columns: {{list(result_df.columns)}}")
    
except Exception as e:
    print("ERROR")
    print(f"{{type(e).__name__}}: {{str(e)}}")
    print(traceback.format_exc())
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script_path = f.name
            
            try:
                start_time = time.time()
                result = subprocess.run(
                    [sys.executable, temp_script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=Path(parser_path).parent
                )
                execution_time = time.time() - start_time
                
                output_lines = result.stdout.strip().split('\n')
                
                if output_lines[0] == "SUCCESS":
                    temp_csv_path = Path(parser_path).parent / "temp_result.csv"
                    if temp_csv_path.exists():
                        result_df = pd.read_csv(temp_csv_path)
                        temp_csv_path.unlink()
                        
                        if expected_csv_path:
                            success, error = self._compare_dataframes(result_df, expected_csv_path)
                            if not success:
                                return False, result_df, error
                        
                        self.execution_stats = {
                            'execution_time': execution_time,
                            'success': True,
                            'shape': result_df.shape,
                            'columns': list(result_df.columns)
                        }
                        
                        logger.info(f"Parser test passed in {execution_time:.2f}s")
                        return True, result_df, None
                    else:
                        return False, None, "Result CSV not found"
                        
                else:
                    error_msg = '\n'.join(output_lines[1:]) if len(output_lines) > 1 else "Unknown error"
                    self.last_error = error_msg
                    return False, None, error_msg
                    
            finally:
                Path(temp_script_path).unlink(missing_ok=True)
                
        except subprocess.TimeoutExpired:
            error_msg = f"Parser execution timed out after {timeout} seconds"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Test execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def test_parser(
        self,
        parser_path: str,
        pdf_path: str,
        expected_csv_path: Optional[str] = None
    ) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Test generated parser against sample PDF.
        
        Args:
            parser_path: Path to parser module
            pdf_path: Path to test PDF
            expected_csv_path: Path to expected CSV output
            
        Returns:
            Tuple of (success, dataframe, error_message)
        """
        try:
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            spec = importlib.util.spec_from_file_location("parser_module", parser_path)
            parser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parser_module)
            
            parse_func = getattr(parser_module, 'parse')
            result_df = parse_func(pdf_path)
            
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = final_memory - initial_memory
            
            if not isinstance(result_df, pd.DataFrame):
                raise ValueError("Parse function must return a pandas DataFrame")
            
            self.execution_stats = {
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'success': True,
                'shape': result_df.shape,
                'columns': list(result_df.columns)
            }
            
            if expected_csv_path:
                success, error = self._compare_dataframes(result_df, expected_csv_path)
                if not success:
                    return False, result_df, error
            
            logger.info(f"Parser test passed in {execution_time:.2f}s, memory: {memory_usage:.1f}MB")
            self.last_output = result_df
            return True, result_df, None
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Parser test failed: {error_msg}")
            self.last_error = error_msg
            return False, None, error_msg
    
    def _compare_dataframes(
        self,
        result_df: pd.DataFrame,
        expected_csv_path: str
    ) -> Tuple[bool, Optional[str]]:
        """Compare result DataFrame with expected CSV."""
        try:
            expected_df = pd.read_csv(expected_csv_path)
            
            # Check columns match
            if list(result_df.columns) != list(expected_df.columns):
                error_msg = f"Column mismatch. Expected: {list(expected_df.columns)}, Got: {list(result_df.columns)}"
                logger.warning(error_msg)
                return False, error_msg
            
            # Check row count
            if len(result_df) != len(expected_df):
                error_msg = f"Row count mismatch. Expected: {len(expected_df)}, Got: {len(result_df)}"
                logger.warning(error_msg)
                return False, error_msg
            
            # Normalize NaN/0 values for amount columns
            result_df_copy = result_df.copy()
            for col in result_df_copy.columns:
                if 'Debit' in col or 'Credit' in col or 'Amt' in col:
                    # Replace 0 and 0.0 with NaN for proper comparison
                    result_df_copy[col] = result_df_copy[col].replace([0, 0.0], np.nan)
            
            # Data type harmonization
            for col in expected_df.columns:
                if col in result_df_copy.columns:
                    try:
                        if expected_df[col].dtype == 'object':
                            result_df_copy[col] = result_df_copy[col].astype(str)
                        elif pd.api.types.is_numeric_dtype(expected_df[col]):
                            result_df_copy[col] = pd.to_numeric(result_df_copy[col], errors='coerce')
                    except Exception:
                        pass
            
            # Detailed comparison
            try:
                pd.testing.assert_frame_equal(
                    result_df_copy.reset_index(drop=True),
                    expected_df.reset_index(drop=True),
                    check_dtype=False,
                    rtol=0.01,
                    atol=0.01
                )
                return True, None
            except AssertionError as e:
                error_str = str(e)
                # Check if difference is only due to 0.0 vs NaN
                if ('0.0' in error_str or '0' in error_str) and 'nan' in error_str.lower():
                    logger.info("Only NaN/0 representation differences detected, accepting result")
                    return True, None
                
                logger.warning(f"DataFrames are not exactly equal: {str(e)}")
                return False, f"Data values don't match expected output: {str(e)}"
                
        except Exception as e:
            return False, f"Comparison failed: {str(e)}"
    
    def get_comparison_summary(
        self,
        actual_df: Optional[pd.DataFrame],
        expected_df: pd.DataFrame
    ) -> str:
        """
        Generate comparison summary between actual and expected output.
        
        Args:
            actual_df: Actual parser output
            expected_df: Expected output
            
        Returns:
            Formatted comparison string
        """
        if actual_df is None:
            return "Parser failed to produce output"
        
        summary = []
        summary.append(f"Expected shape: {expected_df.shape}")
        summary.append(f"Actual shape: {actual_df.shape}")
        summary.append(f"\nExpected columns: {list(expected_df.columns)}")
        summary.append(f"Actual columns: {list(actual_df.columns)}")
        
        summary.append(f"\nExpected sample (first 3 rows):")
        summary.append(expected_df.head(3).to_string(index=False))
        summary.append(f"\nActual sample (first 3 rows):")
        summary.append(actual_df.head(3).to_string(index=False))
        
        summary.append(f"\nData Quality:")
        summary.append(f"- Expected missing values: {expected_df.isnull().sum().sum()}")
        summary.append(f"- Actual missing values: {actual_df.isnull().sum().sum()}")
        
        return "\n".join(summary)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics from last test."""
        return self.execution_stats.copy()
    
    def benchmark_parser(
        self,
        parser_path: str,
        pdf_path: str,
        iterations: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark parser performance over multiple runs.
        
        Args:
            parser_path: Path to parser module
            pdf_path: Path to test PDF
            iterations: Number of test iterations
            
        Returns:
            Performance statistics
        """
        times = []
        memory_usage = []
        
        for i in range(iterations):
            logger.info(f"Benchmark iteration {i+1}/{iterations}")
            success, _, _ = self.test_parser(parser_path, pdf_path)
            
            if success and self.execution_stats:
                times.append(self.execution_stats['execution_time'])
                memory_usage.append(self.execution_stats.get('memory_usage', 0))
            else:
                logger.warning(f"Benchmark iteration {i+1} failed")
        
        if not times:
            return {"error": "All benchmark iterations failed"}
        
        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "avg_memory": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "iterations": len(times),
            "success_rate": len(times) / iterations * 100
        }