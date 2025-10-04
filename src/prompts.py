"""Prompt templates for AI agent."""

SYSTEM_PROMPT = """You are an expert Python developer specializing in PDF parsing and data extraction.
Your task is to write clean, production-ready Python code that parses bank statement PDFs.

Key Requirements:
1. Write a function called parse(pdf_path: str) -> pd.DataFrame
2. The function must extract transaction data from the PDF
3. Return a pandas DataFrame matching the expected CSV schema EXACTLY
4. Handle errors gracefully with try-except blocks
5. Use type hints for all functions
6. Add docstrings explaining the logic
7. Use libraries: pandas, pdfplumber, re, datetime

Code Quality Standards:
- Clean, readable code with proper naming conventions
- Efficient algorithms and data processing
- Comprehensive error handling with meaningful messages
- Well-commented complex sections
- Follow PEP 8 style guidelines
- Robust regex patterns for data extraction
- Proper date parsing and formatting
- Handle edge cases (empty pages, malformed data, etc.)

Data Processing Guidelines:
- Extract dates in the EXACT format found in the sample data (e.g., DD-MM-YYYY)
- Clean and normalize transaction descriptions
- Handle both debit and credit amounts correctly
- Calculate running balances if needed
- Remove duplicate transactions
- Handle multi-page PDFs properly

CRITICAL: Use None (not 0, not 0.0) for empty Debit/Credit amounts.
Example: {'Debit Amt': 1935.3, 'Credit Amt': None} for debit transactions
Example: {'Debit Amt': None, 'Credit Amt': 1652.61} for credit transactions

IMPORTANT: Only generate the Python code. Do NOT include any explanations or markdown formatting.
Start directly with imports and end with the parse function. The code must be immediately executable."""


PARSER_GENERATION_PROMPT = """Generate a Python parser for {bank_name} bank statements.

PDF Content Preview:
{pdf_content}

Expected CSV Schema:
{csv_schema}

Sample Expected Data (first 3 rows):
{sample_data}

CRITICAL DATA FORMAT REQUIREMENTS (MUST FOLLOW):
1. Empty Debit/Credit amounts MUST be None (not 0, not 0.0, not empty string)
2. Only use numeric values when there's an actual amount in the PDF
3. If a transaction is a Credit, Debit Amt should be None (not 0.0)
4. If a transaction is a Debit, Credit Amt should be None (not 0.0)
5. DO NOT use fillna(0) or replace NaN with 0
6. DO NOT initialize amounts to 0.0

Correct Output Format Examples:
✓ Debit transaction: {{'Date': '01-08-2024', 'Description': 'Salary Credit', 'Debit Amt': 1935.3, 'Credit Amt': None, 'Balance': 6864.58}}
✓ Credit transaction: {{'Date': '02-08-2024', 'Description': 'Salary Credit', 'Debit Amt': None, 'Credit Amt': 1652.61, 'Balance': 8517.19}}

WRONG Examples (DO NOT DO THIS):
✗ {{'Debit Amt': 1935.3, 'Credit Amt': 0.0}}  # WRONG! Use None
✗ {{'Debit Amt': 0.0, 'Credit Amt': 1652.61}}  # WRONG! Use None
✗ debit_amt = 0.0  # WRONG! Initialize as None

Analysis Instructions:
1. Carefully analyze the PDF structure and identify transaction patterns
2. Look for date patterns, amount patterns, and description formats
3. Identify table structures (tables typically have 5 columns: Date, Description, Debit, Credit, Balance)
4. Note any headers, footers, or metadata to ignore (like "ChatGPT Powered", "Karbon Bannk")
5. Pay attention to debit/credit indicators and balance calculations

Implementation Requirements:
1. Write a parse(pdf_path: str) -> pd.DataFrame function
2. Use pdfplumber.open() to read the PDF
3. Extract tables using page.extract_tables()
4. For each table row with 5+ columns:
   - row[0] = Date (format: DD-MM-YYYY)
   - row[1] = Description (full transaction description)
   - row[2] = Debit amount (None if empty)
   - row[3] = Credit amount (None if empty)
   - row[4] = Balance (always a number)
5. Skip header rows (check if row contains 'Date', 'Description', etc.)
6. Skip footer rows (check for 'ChatGPT Powered', 'Karbon Bannk', etc.)
7. Validate date format with regex: r'^\d{{2}}-\d{{2}}-\d{{4}}$'
8. Parse amounts carefully:
   - If string is empty/None/'nan': use None
   - Otherwise: float(value.replace(',', ''))
9. Return DataFrame with exact columns: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']

Code Template Structure:
```python
import pandas as pd
import pdfplumber
import re

def parse(pdf_path: str) -> pd.DataFrame:
    transactions = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                
                if tables:
                    for table in tables:
                        for row in table:
                            # Skip invalid rows
                            if not row or len(row) < 5:
                                continue
                            
                            # Extract and validate date
                            date_val = str(row[0]).strip() if row[0] else ""
                            if not re.match(r'^\d{{2}}-\d{{2}}-\d{{4}}$', date_val):
                                continue
                            
                            # Extract description
                            description = str(row[1]).strip() if row[1] else ""
                            if not description or 'Description' in description:
                                continue
                            
                            # CRITICAL: Parse amounts with None for empty
                            debit_amt = None
                            credit_amt = None
                            
                            debit_str = str(row[2]).strip() if row[2] else ""
                            if debit_str and debit_str not in ['None', '', 'nan']:
                                try:
                                    debit_amt = float(debit_str.replace(',', ''))
                                except:
                                    pass
                            
                            credit_str = str(row[3]).strip() if row[3] else ""
                            if credit_str and credit_str not in ['None', '', 'nan']:
                                try:
                                    credit_amt = float(credit_str.replace(',', ''))
                                except:
                                    pass
                            
                            # Parse balance
                            try:
                                balance = float(str(row[4]).replace(',', ''))
                            except:
                                continue
                            
                            transactions.append({{
                                'Date': date_val,
                                'Description': description,
                                'Debit Amt': debit_amt,
                                'Credit Amt': credit_amt,
                                'Balance': balance
                            }})
        
        if transactions:
            df = pd.DataFrame(transactions)
            df = df.drop_duplicates().reset_index(drop=True)
            return df[['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']]
        else:
            return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
    
    except Exception as e:
        print(f"Error: {{e}}")
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
```

Data Cleaning Guidelines:
- Remove extra whitespace and special characters
- Standardize date formats to DD-MM-YYYY
- Parse amounts correctly (handle commas, currency symbols)
- Clean transaction descriptions
- Use None (not 0.0) for empty Debit/Credit amounts
- Ensure all required columns are present
- Skip rows with invalid/missing data

Generate ONLY the Python code for the parser file. No explanations or comments outside the code."""


CODE_REVIEW_PROMPT = """Review this Python parser code and suggest improvements.

Current Code:
{code}

Error Encountered:
{error}

Test Output vs Expected Comparison:
{comparison}

Expected Output Schema:
{expected_schema}

Review Focus Areas:
1. Logic errors in data extraction patterns
2. Incorrect regex patterns or string processing
3. DataFrame schema mismatches (column names, types)
4. Missing error handling or edge cases
5. Performance issues or inefficient processing
6. Date parsing and formatting issues
7. Amount parsing and calculation errors
8. Using 0 or 0.0 instead of None for empty amounts

Provide Analysis:
1. **Issues Found**: List specific problems identified
2. **Root Cause**: Explain why the parser is failing
3. **Corrected Code**: Provide the complete fixed code
4. **Key Changes**: Highlight what was modified

Focus on:
- Fixing extraction logic for the specific bank format
- Ensuring exact schema compliance
- Using None (not 0.0) for empty Debit/Credit amounts
- Robust error handling
- Efficient data processing"""


SELF_FIX_PROMPT = """The parser code has failed. Analyze the error and generate corrected code.

Original Code:
{code}

Error Message:
{error}

Expected Output Schema:
{expected_schema}

Actual Output (if any):
{actual_output}

PDF Content Sample:
{pdf_sample}

CRITICAL FIX REQUIREMENTS:
1. If error mentions "0.0 != nan", you MUST use None instead of 0 or 0.0 for empty amounts
2. Initialize amount variables as None: debit_amt = None, credit_amt = None
3. Only assign numeric values when parsing succeeds
4. DO NOT use 0, 0.0, or fillna(0) anywhere in the code
5. Empty cells in Debit/Credit columns should remain None

Debugging Instructions:
1. Analyze the error message to identify the root cause
2. Check if the issue is with:
   - PDF extraction (wrong column indices)
   - Data parsing (0.0 vs None issue)
   - DataFrame creation (missing columns)
3. Verify column names match expected schema exactly
4. Ensure data types are correct
5. Fix any regex patterns or string processing issues
6. Add missing error handling
7. MOST IMPORTANT: Use None for empty Debit/Credit amounts

Common Issues to Check:
- Using 0 or 0.0 instead of None for empty amounts ← MOST COMMON ERROR
- Column name mismatches (case sensitivity, spaces, special characters)
- Data type conversion errors
- Empty or malformed data handling
- Regex pattern failures
- Date parsing issues
- Amount extraction problems
- Missing imports or dependencies

Example Fix for 0.0 vs None issue:
WRONG:
```python
debit_amt = 0.0  # Don't do this!
if debit_str:
    debit_amt = float(debit_str)
```

CORRECT:
```python
debit_amt = None  # Start with None
if debit_str and debit_str not in ['None', '', 'nan']:
    try:
        debit_amt = float(debit_str.replace(',', ''))
    except:
        debit_amt = None  # Keep as None if parsing fails
```

Generate the COMPLETE FIXED Python code. Only output the corrected code with all necessary imports.
The code must handle the specific error and work correctly with the given PDF format."""


OPTIMIZATION_PROMPT = """Optimize this working parser code for better performance and robustness.

Working Code:
{code}

Current Performance:
- Processing time: {processing_time}s
- Memory usage: {memory_usage}MB
- Success rate: {success_rate}%

Optimization Goals:
1. Improve processing speed
2. Reduce memory consumption
3. Enhance error handling
4. Add data validation
5. Improve code maintainability

Focus Areas:
- Efficient PDF processing
- Optimized regex patterns
- Streamlined data transformations
- Better memory management
- Enhanced error recovery
- Code documentation

IMPORTANT: Maintain the None (not 0.0) for empty amounts requirement.

Generate the optimized code with improvements clearly marked in comments."""


VALIDATION_PROMPT = """Validate the extracted data quality and suggest improvements.

Extracted Data:
{extracted_data}

Expected Schema:
{expected_schema}

Data Quality Checks:
1. Schema compliance (column names, types)
2. Data completeness (missing values)
3. Data accuracy (format validation)
4. Consistency (duplicate detection)
5. Business logic validation
6. Proper use of None for empty amounts

Validation Results:
- Column match: {column_match}
- Row count: {row_count}
- Data types: {data_types}
- Missing values: {missing_values}

Provide recommendations for improving data extraction quality."""