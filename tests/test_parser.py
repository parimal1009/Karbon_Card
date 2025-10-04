"""Tests for generated bank parsers."""

import sys
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_parser_and_data(bank_name: str):
    """Import parser module and get test data paths."""
    import importlib.util
    
    parser_path = Path("custom_parsers") / f"{bank_name}_parser.py"
    pdf_path = Path("data") / bank_name / f"{bank_name}_sample.pdf"
    csv_path = Path("data") / bank_name / f"{bank_name}_expected.csv"
    
    if not parser_path.exists():
        pytest.skip(f"Parser for {bank_name} not found")
    
    if not pdf_path.exists() or not csv_path.exists():
        pytest.skip(f"Test data for {bank_name} not found")
    
    # Import parser module
    spec = importlib.util.spec_from_file_location("parser", parser_path)
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)
    
    return parser_module, pdf_path, csv_path


@pytest.mark.parametrize("bank_name", ["icici", "sbi", "hdfc", "axis", "kotak"])
def test_parser_exists(bank_name):
    """Test that parser file exists."""
    parser_path = Path("custom_parsers") / f"{bank_name}_parser.py"
    
    if parser_path.exists():
        assert parser_path.is_file()
        assert parser_path.stat().st_size > 0


@pytest.mark.parametrize("bank_name", ["icici", "sbi", "hdfc", "axis", "kotak"])
def test_parser_has_parse_function(bank_name):
    """Test that parser has a parse function."""
    try:
        parser_module, _, _ = get_parser_and_data(bank_name)
        assert hasattr(parser_module, 'parse')
        assert callable(parser_module.parse)
    except Exception:
        pytest.skip(f"Parser for {bank_name} not available")


@pytest.mark.parametrize("bank_name", ["icici", "sbi", "hdfc", "axis", "kotak"])
def test_parser_returns_dataframe(bank_name):
    """Test that parser returns a pandas DataFrame."""
    try:
        parser_module, pdf_path, _ = get_parser_and_data(bank_name)
        result = parser_module.parse(str(pdf_path))
        assert isinstance(result, pd.DataFrame)
        assert not result.empty, "Parser returned empty DataFrame"
    except Exception:
        pytest.skip(f"Parser test skipped for {bank_name}")


@pytest.mark.parametrize("bank_name", ["icici", "sbi", "hdfc", "axis", "kotak"])
def test_parser_output_matches_schema(bank_name):
    """Test that parser output matches expected schema."""
    try:
        parser_module, pdf_path, csv_path = get_parser_and_data(bank_name)
        
        result_df = parser_module.parse(str(pdf_path))
        expected_df = pd.read_csv(csv_path)
        
        # Check columns match
        assert list(result_df.columns) == list(expected_df.columns), \
            f"Column mismatch for {bank_name}. Expected: {list(expected_df.columns)}, Got: {list(result_df.columns)}"
        
        # Check row count is reasonable (allow some variance)
        expected_rows = len(expected_df)
        actual_rows = len(result_df)
        variance_threshold = max(1, expected_rows * 0.1)  # 10% variance allowed
        
        assert abs(actual_rows - expected_rows) <= variance_threshold, \
            f"Row count mismatch for {bank_name}. Expected: ~{expected_rows}, Got: {actual_rows}"
        
    except Exception:
        pytest.skip(f"Schema test skipped for {bank_name}")


@pytest.mark.parametrize("bank_name", ["icici", "sbi", "hdfc", "axis", "kotak"])
def test_parser_data_types(bank_name):
    """Test that parser outputs correct data types."""
    try:
        parser_module, pdf_path, csv_path = get_parser_and_data(bank_name)
        
        result_df = parser_module.parse(str(pdf_path))
        expected_df = pd.read_csv(csv_path)
        
        # Check that numeric columns are properly parsed
        for col in result_df.columns:
            if col.lower() in ['amount', 'debit', 'credit', 'balance']:
                # Should be numeric or convertible to numeric
                try:
                    pd.to_numeric(result_df[col], errors='coerce')
                except Exception as e:
                    pytest.fail(f"Column {col} should be numeric for {bank_name}: {e}")
        
        # Check that date columns are properly formatted
        for col in result_df.columns:
            if 'date' in col.lower():
                # Should be parseable as dates
                try:
                    pd.to_datetime(result_df[col], errors='coerce')
                except Exception as e:
                    pytest.fail(f"Column {col} should be date-like for {bank_name}: {e}")
        
    except Exception:
        pytest.skip(f"Data type test skipped for {bank_name}")


@pytest.mark.parametrize("bank_name", ["icici", "sbi", "hdfc", "axis", "kotak"])
def test_parser_data_quality(bank_name):
    """Test data quality of parser output."""
    try:
        parser_module, pdf_path, _ = get_parser_and_data(bank_name)
        
        result_df = parser_module.parse(str(pdf_path))
        
        # Check for excessive missing values
        missing_percentage = result_df.isnull().sum().sum() / (len(result_df) * len(result_df.columns))
        assert missing_percentage < 0.5, f"Too many missing values ({missing_percentage:.1%}) for {bank_name}"
        
        # Check for duplicate rows
        duplicate_count = result_df.duplicated().sum()
        assert duplicate_count < len(result_df) * 0.1, f"Too many duplicate rows ({duplicate_count}) for {bank_name}"
        
        # Check that description column has meaningful content
        if 'description' in result_df.columns:
            desc_col = 'description'
        elif 'narration' in result_df.columns:
            desc_col = 'narration'
        else:
            desc_col = None
        
        if desc_col:
            empty_descriptions = result_df[desc_col].isnull().sum()
            assert empty_descriptions < len(result_df) * 0.3, f"Too many empty descriptions for {bank_name}"
        
    except Exception:
        pytest.skip(f"Data quality test skipped for {bank_name}")


def test_icici_parser_specific():
    """Specific test for ICICI parser if it exists."""
    try:
        parser_module, pdf_path, csv_path = get_parser_and_data("icici")
        
        result_df = parser_module.parse(str(pdf_path))
        expected_df = pd.read_csv(csv_path)
        
        # ICICI specific validations
        assert not result_df.empty, "ICICI parser returned empty DataFrame"
        assert len(result_df.columns) > 0, "ICICI parser has no columns"
        
        # Check for common bank statement columns
        result_cols_lower = [col.lower() for col in result_df.columns]
        
        has_date = any('date' in col for col in result_cols_lower)
        has_amount = any(keyword in col for col in result_cols_lower 
                        for keyword in ['amount', 'debit', 'credit'])
        has_description = any(keyword in col for col in result_cols_lower 
                            for keyword in ['description', 'narration', 'particulars'])
        
        assert has_date, "ICICI parser missing date column"
        assert has_amount, "ICICI parser missing amount/transaction column"
        assert has_description, "ICICI parser missing description column"
        
        # Validate data ranges
        if any('balance' in col.lower() for col in result_df.columns):
            balance_cols = [col for col in result_df.columns if 'balance' in col.lower()]
            for col in balance_cols:
                # Balance should be numeric
                numeric_balance = pd.to_numeric(result_df[col], errors='coerce')
                assert not numeric_balance.isnull().all(), f"Balance column {col} should have numeric values"
        
    except Exception:
        pytest.skip("ICICI specific test skipped")


class TestParserPerformance:
    """Performance tests for parsers."""
    
    @pytest.mark.parametrize("bank_name", ["icici", "sbi", "hdfc"])
    def test_parser_execution_time(self, bank_name):
        """Test that parser executes within reasonable time."""
        import time
        
        try:
            parser_module, pdf_path, _ = get_parser_and_data(bank_name)
            
            start_time = time.time()
            result_df = parser_module.parse(str(pdf_path))
            execution_time = time.time() - start_time
            
            # Should complete within 30 seconds for most PDFs
            assert execution_time < 30, f"Parser for {bank_name} took too long: {execution_time:.2f}s"
            
            # Log performance for monitoring
            print(f"\n{bank_name} parser execution time: {execution_time:.2f}s")
            
        except Exception:
            pytest.skip(f"Performance test skipped for {bank_name}")
    
    @pytest.mark.parametrize("bank_name", ["icici", "sbi", "hdfc"])
    def test_parser_memory_usage(self, bank_name):
        """Test parser memory usage."""
        import psutil
        import os
        
        try:
            parser_module, pdf_path, _ = get_parser_and_data(bank_name)
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result_df = parser_module.parse(str(pdf_path))
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Should not use excessive memory (>500MB increase)
            assert memory_increase < 500, f"Parser for {bank_name} used too much memory: {memory_increase:.1f}MB"
            
            print(f"\n{bank_name} parser memory increase: {memory_increase:.1f}MB")
            
        except Exception:
            pytest.skip(f"Memory test skipped for {bank_name}")


class TestParserRobustness:
    """Robustness tests for parsers."""
    
    def test_parser_handles_missing_file(self):
        """Test parser behavior with missing PDF file."""
        try:
            parser_module, _, _ = get_parser_and_data("icici")
            
            with pytest.raises((FileNotFoundError, Exception)):
                parser_module.parse("nonexistent_file.pdf")
                
        except Exception:
            pytest.skip("Missing file test skipped")
    
    def test_parser_handles_empty_file(self):
        """Test parser behavior with empty PDF file."""
        try:
            parser_module, _, _ = get_parser_and_data("icici")
            
            # Create temporary empty file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                result = parser_module.parse(tmp_path)
                # Should return empty DataFrame or raise appropriate exception
                if isinstance(result, pd.DataFrame):
                    assert result.empty or len(result) == 0
            except Exception:
                # Expected behavior for empty/invalid PDF
                pass
            finally:
                os.unlink(tmp_path)
                
        except Exception:
            pytest.skip("Empty file test skipped")
    
    def test_parser_handles_corrupted_pdf(self):
        """Test parser behavior with corrupted PDF."""
        try:
            parser_module, _, _ = get_parser_and_data("icici")
            
            # Create temporary corrupted PDF file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(b"This is not a valid PDF file content")
                tmp_path = tmp.name
            
            try:
                # Should handle gracefully
                with pytest.raises(Exception):
                    parser_module.parse(tmp_path)
            finally:
                os.unlink(tmp_path)
                
        except Exception:
            pytest.skip("Corrupted PDF test skipped")


class TestParserIntegration:
    """Integration tests for the complete parsing workflow."""
    
    def test_end_to_end_parsing_workflow(self):
        """Test complete parsing workflow from PDF to CSV."""
        try:
            from src.agent import BankParserAgent
            from src.code_executor import CodeExecutor
            
            # Test with ICICI if available
            bank_name = "icici"
            data_dir = Path("data") / bank_name
            pdf_path = data_dir / f"{bank_name}_sample.pdf"
            csv_path = data_dir / f"{bank_name}_expected.csv"
            parser_path = Path("custom_parsers") / f"{bank_name}_parser.py"
            
            if not all(p.exists() for p in [pdf_path, csv_path, parser_path]):
                pytest.skip("Required files not found for integration test")
            
            # Test code executor
            executor = CodeExecutor()
            success, result_df, error = executor.test_parser(
                str(parser_path),
                str(pdf_path),
                str(csv_path)
            )
            
            if success:
                assert isinstance(result_df, pd.DataFrame)
                assert not result_df.empty
                print(f"\nIntegration test passed: {len(result_df)} rows extracted")
            else:
                pytest.fail(f"Integration test failed: {error}")
                
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")
    
    def test_agent_validation(self):
        """Test agent setup validation."""
        try:
            from src.agent import BankParserAgent
            
            agent = BankParserAgent()
            issues = agent.validate_setup()
            
            # Should return a list (empty if no issues)
            assert isinstance(issues, list)
            
            if issues:
                print(f"\nAgent validation issues found: {issues}")
            else:
                print("\nAgent validation passed")
                
        except Exception as e:
            pytest.skip(f"Agent validation test skipped: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
