import pandas as pd
import pdfplumber
import re

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parse ICICI bank statement PDF and return a pandas DataFrame.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    pd.DataFrame: DataFrame containing transaction data.
    """
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
                            if not re.match(r'^\d{2}-\d{2}-\d{4}$', date_val):
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
                            
                            transactions.append({
                                'Date': date_val,
                                'Description': description,
                                'Debit Amt': debit_amt,
                                'Credit Amt': credit_amt,
                                'Balance': balance
                            })
        
        if transactions:
            df = pd.DataFrame(transactions)
            df = df.drop_duplicates().reset_index(drop=True)
            return df[['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']]
        else:
            return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
    
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])