import os
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
# from langgraph.prebuilt import create_react_agent
from datetime import datetime
import requests
import getpass
from dotenv import load_dotenv
import time
import altair as alt
import logging
from typing import TypedDict, Optional, List
from pathlib import Path
import traceback

class GraphState(TypedDict):
    input: str
    thread_id: str
    messages: List[dict]
    portfolio_output: Optional[str]
    market_report_output: Optional[str]
    conversational_output: Optional[str]
    user_name: Optional[str]
    market_charts: Optional[List[dict]]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Before loading any environment variables, print current state
print("\nBefore load_dotenv:")
print(f"OPENAI_API_KEY from env: {'Found' if os.getenv('OPENAI_API_KEY') else 'Not found'}")

# Load from .env file
print("\nTrying to load .env file...")
DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

# After loading .env, print state
print("\nAfter load_dotenv:")
print(f"OPENAI_API_KEY from .env: {'Found' if os.getenv('OPENAI_API_KEY') else 'Not found'}")

# Get API key and print partial key for verification
openai_key = os.getenv('OPENAI_API_KEY')
if openai_key:
    print(f"\nUsing OpenAI key that starts with: {openai_key[:7]}...")
    print(f"Key ends with: ...{openai_key[-4:]}")
else:
    print("\nNo OpenAI API key found!")

# Print which .env file is being loaded
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    print(f".env file found at: {env_path}")
else:
    print("No .env file found in script directory")

# Set up environment variables for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "financial-assistant"

# Initialize global variables
context_tracker = {
    "last_index_discussed": None,
    "last_year_discussed": None
}

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================
# Set path to Knowledge_base directory
KB_PATH = Path(__file__).parent.parent / "Knowledge_base"


def load_financial_knowledge():
    try:
        kb_path = KB_PATH / "financial_knowledge.txt"
        with open(kb_path, "r", encoding="utf-8") as file:
            content = file.read()
            metrics_knowledge = ""
            detailed_explanations = {}
            faq = {}
            current_section = None
            for line in content.split('\n'):
                if line.strip() == "##FINANCIAL METRICS KNOWLEDGE:":
                    current_section = "metrics"
                    continue
                elif line.strip() == "# DETAILED EXPLANATIONS":
                    current_section = "explanations"
                    continue
                elif line.strip() == "# FAQ":
                    current_section = "faq"
                    continue
                if current_section == "metrics" and line.strip():
                    metrics_knowledge += line + "\n"
                elif current_section == "explanations" and ":" in line:
                    key, value = line.split(":", 1)
                    detailed_explanations[key.strip().lower()] = value.strip()
                elif current_section == "faq" and ":" in line:
                    key, value = line.split(":", 1)
                    faq[key.strip().lower()] = value.strip()
            return metrics_knowledge, detailed_explanations, faq
    except Exception as e:
        print(f"Error loading financial knowledge: {str(e)}")
        return "Financial metrics knowledge not available.", {}, {}


def load_market_data():
    excel_file = KB_PATH / "market_data.xlsx"
    try:
        market_data = pd.read_excel(excel_file, sheet_name="historical_data")
        data_description = pd.read_excel(excel_file, sheet_name="data_description")
        market_data.iloc[:, 0] = pd.to_datetime(market_data.iloc[:, 0])
        market_data.set_index(market_data.columns[0], inplace=True)
        market_data = market_data.sort_index()
        data_info = {}
        for _, row in data_description.iterrows():
            data_info[row['Market_data']] = {
                'asset_type': row['Asset_type'],
                'description': row['Description'],
                'currency': row['Currency']
            }
        print(f"Successfully loaded market_data.xlsx with {len(market_data)} rows and {len(market_data.columns)} columns")
        print(f"Available indices: {', '.join(market_data.columns)}")
        print(f"Date range: {market_data.index.min().strftime('%Y-%m-%d')} to {market_data.index.max().strftime('%Y-%m-%d')}")
        asset_types = {}
        for col, info in data_info.items():
            asset_type = info['asset_type']
            if asset_type not in asset_types:
                asset_types[asset_type] = []
            asset_types[asset_type].append(col)
        print(f"Available asset types: {', '.join(asset_types.keys())}")
        return market_data, data_info, asset_types
    except Exception as e:
        print(f"Error loading market_data.xlsx: {str(e)}")
        return None, {}, {}


METRICS_KNOWLEDGE, DETAILED_EXPLANATIONS, FAQ = load_financial_knowledge()
market_data, data_info, asset_types = load_market_data()


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@tool
def get_financial_knowledge(query: str) -> str:
    """
    Get information about financial metrics, terms, or concepts.
    Input should be a specific financial term or concept you want to learn about.
    """
    query = query.lower().strip()
    
    # Check if the query is in detailed explanations
    if query in DETAILED_EXPLANATIONS:
        return DETAILED_EXPLANATIONS[query]
    
    # Check if the query is in FAQ
    if query in FAQ:
        return FAQ[query]
    
    # If not found directly, search for partial matches
    for key, value in DETAILED_EXPLANATIONS.items():
        if query in key:
            return f"{key.capitalize()}: {value}"
    
    # If still not found, return the general metrics knowledge
    return f"I couldn't find specific information about '{query}', but here's some general information about financial metrics:\n\n{METRICS_KNOWLEDGE}"

@tool
def get_data_info(column_name: str) -> str:
    """
    Get information about a specific market data column.
    Input should be the exact name of the column you want information about.
    """
    if market_data is None:
        return "Market data is not available."
    
    if column_name in data_info:
        info = data_info[column_name]
        return f"Information about {column_name}:\n- Asset Type: {info['asset_type']}\n- Description: {info['description']}\n- Currency: {info['currency']}"
    else:
        # Try to find partial matches
        matches = [col for col in data_info.keys() if column_name.lower() in col.lower()]
        if matches:
            return f"Exact match not found. Did you mean one of these? {', '.join(matches)}"
        else:
            return f"No information found for '{column_name}'. Use get_available_columns to see available data."

@tool
def get_available_columns(params: str = "") -> str:
    """
    Get a list of available data columns, optionally filtered by a search term.
    Input can be a JSON string with a "search" key, or just a simple string to search for.
    Example: {"search": "Equity"} or just "Equity"
    """
    global market_data
    
    if market_data is None or market_data.empty:
        if not initialize_sample_data():
            return "Market data is not available."
    
    try:
        search_term = ""
        
        # Handle both JSON string and direct string inputs
        if params.strip():
            if params.strip().startswith('{'):
                try:
                    # Try to parse as JSON
                    params_dict = json.loads(params)
                    search_term = str(params_dict.get("search", "")).strip().lower()
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat the whole string as a search term
                    search_term = params.strip().lower()
            else:
                # Direct string input
                search_term = params.strip().lower()
        
        # Get all columns
        all_columns = market_data.columns.tolist()
        
        # Filter columns if search term is provided
        if search_term:
            filtered_columns = [col for col in all_columns if search_term in col.lower()]
            
            if not filtered_columns:
                return f"No columns found matching '{search_term}'. Available columns: {', '.join(all_columns)}"
            
            return f"Columns matching '{search_term}': {', '.join(filtered_columns)}"
        else:
            # Group columns by category
            categories = {}
            for col in all_columns:
                category = col.split()[0] if ' ' in col else 'Other'
                if category not in categories:
                    categories[category] = []
                categories[category].append(col)
            
            result = "Available data columns:\n\n"
            for category, cols in categories.items():
                result += f"**{category}**:\n"
                for col in cols:
                    result += f"- {col}\n"
                result += "\n"
            
            return result
    
    except Exception as e:
        return f"Error retrieving columns: {str(e)}"

@tool
def calculate_performance(params: str) -> str:
    """
    Calculate performance metrics for a specific market data column over a time period.
    Input should be a JSON string with keys: start_date, end_date, column_name.
    Example: {"start_date": "2019-12-31", "end_date": "2020-12-31", "column_name": "S&P 500"}
    """
    global market_data
    
    if market_data is None or market_data.empty:
        if not initialize_sample_data():
            return "Market data is not available."
    
    try:
        # Handle both JSON string and direct string inputs
        if params.strip().startswith('{'):
            try:
                # Try to parse as JSON
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract parameters manually
                params_dict = {}
                for param in params.strip("{}").split(","):
                    if ":" in param:
                        key, value = param.split(":", 1)
                        params_dict[key.strip().strip('"').strip("'")] = value.strip().strip('"').strip("'")
        else:
            return "Invalid input format. Please provide a JSON string with start_date, end_date, and column_name."
        
        # Extract parameters
        start_date = str(params_dict.get("start_date", "")).strip()
        end_date = str(params_dict.get("end_date", "")).strip()
        column_name = str(params_dict.get("column_name", "")).strip()
        
        # Validate parameters
        if not start_date or not end_date or not column_name:
            return "Missing required parameters. Please provide start_date, end_date, and column_name."
        
        # Find the best matching column name
        if column_name not in market_data.columns:
            # Try to find a close match
            best_match = None
            best_score = 0
            
            for col in market_data.columns:
                # Check for exact match (case-insensitive)
                if col.lower() == column_name.lower():
                    best_match = col
                    break
                
                # Check for partial match
                if column_name.lower() in col.lower() or col.lower() in column_name.lower():
                    score = len(set(col.lower()) & set(column_name.lower())) / max(len(col), len(column_name))
                    if score > best_score:
                        best_score = score
                        best_match = col
            
            if best_match and best_score > 0.5:
                column_name = best_match
                print(f"Using closest match: '{column_name}' for input '{params_dict.get('column_name')}'")
            else:
                return f"Column '{column_name}' not found. Available columns: {', '.join(market_data.columns)}"
        
        # Parse dates
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD format."
        
        # Ensure end_date is not in the future
        current_date = pd.to_datetime(datetime.now().date())
        if end_date > current_date:
            end_date = current_date
        
        # Get data for the specified period
        data = market_data.loc[(market_data.index >= start_date) & (market_data.index <= end_date), column_name].copy()
        
        if data.empty:
            return f"No data available for {column_name} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}."
        
        # Sort data chronologically (oldest to newest)
        data = data.sort_index()
        
        # Get all years in the data range
        start_year = start_date.year
        end_year = end_date.year
        all_years = range(start_year - 1, end_year + 1)  # Include previous year for first year's return
        
        # Get December values for each year in the entire dataset (for annual returns)
        december_values = {}
        
        # First, get all December values from the entire dataset
        for year in all_years:
            # Create a date range for December of the year
            dec_start = pd.Timestamp(year=year, month=12, day=1)
            dec_end = pd.Timestamp(year=year, month=12, day=31)
            
            # Find the last available date in December that has data
            december_data = market_data.loc[(market_data.index >= dec_start) & 
                                           (market_data.index <= dec_end), column_name]
            
            if not december_data.empty:
                # Use the last date in December
                last_dec_date = december_data.index[-1]
                december_values[year] = {
                    'date': last_dec_date,
                    'value': december_data.iloc[-1]
                }
            else:
                # If no December data, find the closest date before December
                before_dec = market_data.loc[(market_data.index < dec_start), column_name]
                if not before_dec.empty:
                    closest_date = before_dec.index[-1]
                    december_values[year] = {
                        'date': closest_date,
                        'value': before_dec.iloc[-1],
                        'note': f'No December data available, using closest previous date ({closest_date.strftime("%Y-%m-%d")})'
                    }
        
        # Calculate annual returns using December-to-December values
        annual_returns = {}
        
        for year in range(start_year, end_year + 1):
            if year in december_values and (year-1) in december_values:
                current_dec_value = december_values[year]['value']
                prev_dec_value = december_values[year-1]['value']
                annual_return = (current_dec_value / prev_dec_value) - 1
                annual_returns[year] = annual_return * 100  # Convert to percentage
        
        # Calculate total return using EXACT start and end dates provided by the user
        total_return = 0
        annualized_return = 0
        years = 0
        
        # Find the exact values for the start and end dates
        start_value = None
        end_value = None
        start_exact_date = None
        end_exact_date = None
        
        # For the start date, find the exact date or the closest date before
        if start_date.month == 1 and start_date.day == 1:
            # If start date is January 1st, use previous year's December value
            prev_year_dec_start = pd.Timestamp(year=start_date.year-1, month=12, day=1)
            prev_year_dec_end = pd.Timestamp(year=start_date.year-1, month=12, day=31)
            prev_dec_data = market_data.loc[(market_data.index >= prev_year_dec_start) & 
                                          (market_data.index <= prev_year_dec_end), column_name]
            if not prev_dec_data.empty:
                start_exact_date = prev_dec_data.index[-1]
                start_value = prev_dec_data.iloc[-1]
            else:
                # If no December data, find the closest date before
                before_dec = market_data.loc[market_data.index < prev_year_dec_start, column_name]
                if not before_dec.empty:
                    start_exact_date = before_dec.index[-1]
                    start_value = before_dec.iloc[-1]
        else:
            start_data = market_data.loc[market_data.index >= start_date, column_name]
            if not start_data.empty:
                start_exact_date = start_data.index[0]
                start_value = start_data.iloc[0]
        
        # For the end date, find the exact date or the closest date before
        end_data = market_data.loc[market_data.index <= end_date, column_name]
        if not end_data.empty:
            end_exact_date = end_data.index[-1]
            end_value = end_data.iloc[-1]
        
        # Calculate total return using the exact start and end values
        if start_value is not None and end_value is not None:
            total_return = (end_value / start_value) - 1
            
            # Calculate annualized return
            days = (end_exact_date - start_exact_date).days
            years = days / 365.25
            annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
        
        # Calculate monthly returns for volatility and other metrics
        monthly_returns = data.pct_change().dropna()
        
        # Calculate volatility (annualized from monthly data)
        volatility = monthly_returns.std() * (12 ** 0.5) if not monthly_returns.empty else 0
        
        # Calculate maximum drawdown
        max_drawdown = 0
        if not monthly_returns.empty:
            cumulative_returns = (1 + monthly_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max) - 1
            max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        sharpe_ratio = 0
        if not monthly_returns.empty:
            risk_free_rate = 0.02
            monthly_risk_free = (1 + risk_free_rate) ** (1 / 12) - 1
            excess_returns = monthly_returns - monthly_risk_free
            if excess_returns.std() > 0:
                sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (12 ** 0.5)
        
        # Calculate annual volatilities, max drawdowns, and Sharpe ratios
        annual_volatilities = {}
        annual_max_drawdowns = {}
        annual_sharpe_ratios = {}
        
        for year in range(start_year, end_year + 1):
            # Get data for the year
            year_start = max(start_date, pd.Timestamp(year=year, month=1, day=1))
            year_end = min(end_date, pd.Timestamp(year=year, month=12, day=31))
            
            year_data = data[(data.index >= year_start) & (data.index <= year_end)]
            if not year_data.empty and len(year_data) > 1:
                year_returns = year_data.pct_change().dropna()
                
                if not year_returns.empty:
                    # Calculate annualized volatility from monthly data
                    annual_volatilities[year] = year_returns.std() * (12 ** 0.5) * 100  # Convert to percentage
                    
                    year_cumulative_returns = (1 + year_returns).cumprod()
                    year_running_max = year_cumulative_returns.cummax()
                    year_drawdown = (year_cumulative_returns / year_running_max) - 1
                    annual_max_drawdowns[year] = year_drawdown.min() * 100  # Convert to percentage
                    
                    year_excess_returns = year_returns - monthly_risk_free
                    if year_excess_returns.std() > 0:
                        annual_sharpe_ratios[year] = (year_excess_returns.mean() / year_excess_returns.std()) * (12 ** 0.5)
                    else:
                        annual_sharpe_ratios[year] = 0
        
        # Format the result
        result = f"Performance metrics for {column_name} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:\n\n"
        
        # Overall metrics
        result += f"- **Total Return**: {total_return * 100:.2f}%\n"
        result += f"- **Annualized Return**: {annualized_return * 100:.2f}%\n"
        result += f"- **Volatility (Annualized)**: {volatility * 100:.2f}%\n"
        result += f"- **Maximum Drawdown**: {max_drawdown * 100:.2f}%\n"
        result += f"- **Sharpe Ratio**: {sharpe_ratio:.2f}\n\n"
        
        # Annual returns summary
        result += "### Annual Returns Summary\n\n"
        
        for year in range(start_year, end_year + 1):
            if year in annual_returns:
                ret = annual_returns[year]
                result += f"- **{year}**: {ret:.2f}%\n"
        
        # Annual returns table
        result += "\n### Annual Performance Details\n\n"
        result += "| Year | December Date | December Value | Previous December | Annual Return (%) | Volatility (%) | Max Drawdown (%) | Sharpe Ratio |\n"
        result += "|------|--------------|---------------|-------------------|------------------|----------------|------------------|-------------|\n"
        
        for year in range(start_year, end_year + 1):
            if year in december_values:
                date = december_values[year]['date'].strftime('%Y-%m-%d')
                value = december_values[year]['value']
                
                # Previous December info
                prev_date = "-"
                prev_value = "-"
                if (year-1) in december_values:
                    prev_date = december_values[year-1]['date'].strftime('%Y-%m-%d')
                    prev_value = f"{december_values[year-1]['value']:.2f}"
                
                # Metrics
                ret = annual_returns.get(year, float('nan'))
                vol = annual_volatilities.get(year, float('nan'))
                mdd = annual_max_drawdowns.get(year, float('nan'))
                sr = annual_sharpe_ratios.get(year, float('nan'))
                
                # Format strings
                ret_str = f"{ret:.2f}" if not pd.isna(ret) else "-"
                vol_str = f"{vol:.2f}" if not pd.isna(vol) else "-"
                mdd_str = f"{mdd:.2f}" if not pd.isna(mdd) else "-"
                sr_str = f"{sr:.2f}" if not pd.isna(sr) else "-"
                
                result += f"| {year} | {date} | {value:.2f} | {prev_value} ({prev_date}) | {ret_str} | {vol_str} | {mdd_str} | {sr_str} |\n"
        
        # Add detailed calculations
        result += "\n### Detailed Calculations\n\n"
        
        # Total return calculation details
        result += f"**Total Return Calculation**:\n"
        if start_exact_date is not None and end_exact_date is not None:
            result += f"Start Value ({start_exact_date.strftime('%Y-%m-%d')}): {start_value:.2f}\n"
            result += f"End Value ({end_exact_date.strftime('%Y-%m-%d')}): {end_value:.2f}\n"
            result += f"Total Return: ({end_value:.2f} / {start_value:.2f}) - 1 = {total_return * 100:.2f}%\n\n"
        
        # Annual returns calculation details
        result += "**Annual Returns** (December to December):\n"
        for year in range(start_year, end_year + 1):
            if year in annual_returns:
                current_value = december_values[year]['value']
                previous_value = december_values[year-1]['value']
                ret = annual_returns[year]
                
                result += f"**{year}**: ({current_value:.2f} / {previous_value:.2f}) - 1 = {ret:.2f}%\n"
        
        # Calculate CAGR if we have multiple years
        if years > 1:
            result += f"\n**Compound Annual Growth Rate (CAGR)**: {annualized_return * 100:.2f}%\n"
            result += f"CAGR calculation: (({end_value:.2f} / {start_value:.2f}) ^ (1 / {years:.2f})) - 1 = {annualized_return * 100:.2f}%\n"
        
        return result
        
    except Exception as e:
        return f"Error calculating performance: {str(e)}\nPlease check that the date format is correct (YYYY-MM-DD) and the column name exists."

@tool
def generate_data_table(params: str) -> str:
    """
    Generate a table of market data for specified columns and date range.
    Input should be a JSON string with keys: columns, start_date, end_date, frequency.
    Example: {"columns": ["S&P 500", "NASDAQ"], "start_date": "2020-01-01", "end_date": "2020-12-31", "frequency": "monthly"}
    Frequency options: daily, weekly, monthly, quarterly, annual
    """
    if market_data is None:
        return "Market data is not available."
    
    try:
        # Handle both JSON string and direct string inputs
        if params.strip().startswith('{'):
            try:
                # Try to parse as JSON
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract parameters manually
                params_dict = {}
                for param in params.strip("{}").split(","):
                    if ":" in param:
                        key, value = param.split(":", 1)
                        params_dict[key.strip().strip('"').strip("'")] = value.strip().strip('"').strip("'")
        else:
            # Direct string input - try to parse key-value pairs
            params_dict = {}
            parts = params.split()
            for i in range(0, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    key = parts[i].strip(":")
                    value = parts[i + 1]
                    params_dict[key] = value
        
        # Extract parameters
        columns_input = params_dict.get("columns", [])
        start_date = params_dict.get("start_date", "")
        end_date = params_dict.get("end_date", "")
        frequency = params_dict.get("frequency", "monthly")
        
        # Handle string input for columns
        if isinstance(columns_input, str):
            if columns_input.startswith("[") and columns_input.endswith("]"):
                # Parse JSON array
                columns_input = json.loads(columns_input)
            else:
                # Single column as string
                columns_input = [columns_input]
        
        # Validate parameters
        if not columns_input or not start_date or not end_date:
            return "Missing required parameters. Please provide columns, start_date, and end_date."
        
        # Find matching columns (case-insensitive)
        columns = []
        for col_name in columns_input:
            matching_cols = [col for col in market_data.columns if col_name.lower() in col.lower()]
            if matching_cols:
                columns.append(matching_cols[0])
        
        if not columns:
            return f"None of the specified columns were found. Use get_available_columns to see available data."
        
        # Convert dates to datetime objects
        start_date_obj = pd.to_datetime(start_date)
        end_date_obj = pd.to_datetime(end_date)
        
        # Filter data by date range using boolean mask
        mask = (market_data.index >= start_date_obj) & (market_data.index <= end_date_obj)
        filtered_data = market_data.loc[mask, columns]
        
        if filtered_data.empty:
            return f"No data available for the specified columns between {start_date} and {end_date}."
        
        # Resample data based on frequency
        if frequency.lower() == "daily":
            resampled_data = filtered_data
        elif frequency.lower() == "weekly":
            resampled_data = filtered_data.resample('WE').last()
        elif frequency.lower() == "monthly":
            resampled_data = filtered_data.resample('ME').last()
        elif frequency.lower() == "quarterly":
            resampled_data = filtered_data.resample('QE').last()
        elif frequency.lower() == "annual":
            resampled_data = filtered_data.resample('YE').last()  # Changed from 'Y' to 'YE' to avoid deprecation warning
        else:
            return f"Invalid frequency: {frequency}. Options are: daily, weekly, monthly, quarterly, annual."
        
        # Format the table
        table = "| Date |"
        for col in columns:
            table += f" {col} |"
        table += "\n|------|"
        for _ in columns:
            table += "------|"
        table += "\n"
        
        # Add data rows, filtering out rows with all NaN values
        valid_rows = 0
        for date, row in resampled_data.iterrows():
            # Skip rows where all values are NaN
            if row.isna().all():
                continue
                
            date_str = date.strftime('%Y-%m-%d')
            table += f"| {date_str} |"
            for col in columns:
                value = row[col]
                if pd.isna(value):
                    table += " N/A |"
                else:
                    table += f" {value:.2f} |"
            table += "\n"
            valid_rows += 1
            
            # Limit to 20 rows for readability if there are many rows
            if valid_rows >= 20 and len(resampled_data) > 25:
                table += f"| ... | ... |\n"
                break
        
        # If we limited the rows, add the last few rows
        if valid_rows >= 20 and len(resampled_data) > 25:
            last_rows = resampled_data.tail(3)
            for date, row in last_rows.iterrows():
                if row.isna().all():
                    continue
                    
                date_str = date.strftime('%Y-%m-%d')
                table += f"| {date_str} |"
                for col in columns:
                    value = row[col]
                    if pd.isna(value):
                        table += " N/A |"
                    else:
                        table += f" {value:.2f} |"
                table += "\n"
        
        return table
        
    except Exception as e:
        return f"Error generating data table: {str(e)}\nPlease check that the date format is correct (YYYY-MM-DD) and the column names exist."

@tool(return_direct=True)
def respond(message: str) -> str:
    """
    Respond to the user's message without using any specific data or tools.
    Use this for general questions, clarifications, or when no other tool is appropriate.
    """
    if message.lower().strip() in ["hi", "hello", "hey", "initial_greeting"]:
        return "Hi, I'm Ashley from Stashly, your financial assistant. How can I assist you today?"
    
    return message

# @tool(return_direct=True)
# def respond(message: str) -> str:
#     """
#     Respond to the user's message without using any specific data or tools.
#     Use this for general questions, clarifications, or when no other tool is appropriate.
#     """
#     if message.lower().strip() in ["hi", "hello", "hey", "initial_greeting"]:
#         return """Thought: I should respond with our standard greeting
# Action: respond
# Action Input: Hi, I'm Ashley from Stashley, your financial assistant. How can I assist you today?
# Observation: Using standard greeting
# Thought: I now know the final answer
# Final Answer: Hi, I'm Ashley from Stashley, your financial assistant. How can I assist you today?"""
    
#     return f"""Thought: I should respond to this message
# Action: respond
# Action Input: {message}
# Observation: Processing response
# Thought: I now know the final answer
# Final Answer: {message}"""

@tool
def calculate_annual_returns(params: str) -> str:
    """
    Calculate annual returns for a specific market data column over multiple years.
    Input should be a JSON string with keys: start_year, end_year, column_name.
    Example: {"start_year": 2020, "end_year": 2024, "column_name": "Equity Sweden"}
    """
    global market_data
    
    if market_data is None or market_data.empty:
        if not initialize_sample_data():
            return "Market data is not available."
    
    try:
        # Parse input parameters
        if params.strip().startswith('{'):
            try:
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract parameters manually
                params_dict = {}
                for param in params.strip("{}").split(","):
                    if ":" in param:
                        key, value = param.split(":", 1)
                        params_dict[key.strip().strip('"').strip("'")] = value.strip().strip('"').strip("'")
        else:
            return "Invalid input format. Please provide a JSON string with start_year, end_year, and column_name."
        
        # Extract parameters
        start_year = params_dict.get("start_year", "")
        end_year = params_dict.get("end_year", "")
        column_name = str(params_dict.get("column_name", "")).strip()
        
        # Handle column_name or columns parameter
        if not column_name and "columns" in params_dict:
            columns = params_dict.get("columns", [])
            if isinstance(columns, list) and len(columns) > 0:
                column_name = columns[0]
            elif isinstance(columns, str):
                column_name = columns
        
        # Validate parameters
        if not start_year or not end_year or not column_name:
            return "Missing required parameters. Please provide start_year, end_year, and column_name (or columns)."
        
        if column_name not in market_data.columns:
            return f"Column '{column_name}' not found. Available columns: {', '.join(market_data.columns)}"
        
        try:
            start_year = int(start_year)
            end_year = int(end_year)
        except ValueError:
            return "Invalid year format. Please provide years as integers (e.g., 2020)."
        
        # Ensure end_year is not in the future
        current_year = datetime.now().year
        if end_year > current_year:
            end_year = current_year
        
        # Get December values for each year
        december_values = {}
        
        for year in range(start_year - 1, end_year + 1):  # Include previous year for first year's return
            # Create a date range for December of the year
            dec_start = pd.Timestamp(year=year, month=12, day=1)
            dec_end = pd.Timestamp(year=year, month=12, day=31)
            
            # Find the last available date in December that has data
            december_data = market_data.loc[(market_data.index >= dec_start) & 
                                           (market_data.index <= dec_end), column_name]
            
            if not december_data.empty:
                # Use the last date in December
                last_dec_date = december_data.index[-1]
                december_values[year] = {
                    'date': last_dec_date,
                    'value': december_data.iloc[-1]
                }
            else:
                # If no December data, find the closest date before December
                before_dec = market_data.loc[market_data.index < dec_start, column_name]
                if not before_dec.empty:
                    closest_date = before_dec.index[-1]
                    december_values[year] = {
                        'date': closest_date,
                        'value': before_dec.iloc[-1],
                        'note': f'No December data available, using closest previous date ({closest_date.strftime("%Y-%m-%d")})'
                    }
        
        # Calculate annual returns
        annual_returns = {}
        
        for year in range(start_year, end_year + 1):
            if year in december_values and (year-1) in december_values:
                current_value = december_values[year]['value']
                previous_value = december_values[year-1]['value']
                annual_return = (current_value / previous_value) - 1
                annual_returns[year] = annual_return * 100  # Convert to percentage
        
        # Format the result
        result = f"Annual returns for {column_name} from {start_year} to {end_year}:\n\n"
        result += "| Year | December Date | December Value | Previous December | Annual Return (%) |\n"
        result += "|------|--------------|---------------|-------------------|------------------|\n"
        
        for year in range(start_year, end_year + 1):
            if year in december_values:
                date = december_values[year]['date'].strftime('%Y-%m-%d')
                value = december_values[year]['value']
                
                if year in annual_returns:
                    prev_year = year - 1
                    prev_date = december_values[prev_year]['date'].strftime('%Y-%m-%d')
                    prev_value = december_values[prev_year]['value']
                    ret = annual_returns[year]
                    
                    result += f"| {year} | {date} | {value:.2f} | {prev_value:.2f} ({prev_date}) | {ret:.2f} |\n"
                else:
                    result += f"| {year} | {date} | {value:.2f} | - | - |\n"
        
        # Add detailed calculations
        result += "\n### Detailed Calculations\n\n"
        
        for year in range(start_year, end_year + 1):
            if year in annual_returns:
                current_value = december_values[year]['value']
                previous_value = december_values[year-1]['value']
                ret = annual_returns[year]
                
                result += f"**{year}**: ({current_value:.2f} / {previous_value:.2f}) - 1 = {ret:.2f}%\n"
        
        return result
        
    except Exception as e:
        return f"Error calculating annual returns: {str(e)}"

@tool
def get_market_data_for_chart(params: str) -> str:
    """
    Get market data for a chart in the format needed by create_market_chart.
    Input should be a JSON string with keys: column_name, start_date, end_date.
    Example: {"column_name": "OMX30", "start_date": "2020-01-01", "end_date": "2024-01-01"}
    """
    global market_data
    
    if market_data is None or market_data.empty:
        return json.dumps({"error": "Market data is not available."})
    
    try:
        # Parse input parameters
        if params.strip().startswith('{'):
            try:
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract parameters manually
                params_dict = {}
                for param in params.strip("{}").split(","):
                    if ":" in param:
                        key, value = param.split(":", 1)
                        params_dict[key.strip().strip('"').strip("'")] = value.strip().strip('"').strip("'")
        else:
            return json.dumps({"error": "Invalid input format. Please provide a JSON string with column_name, start_date, and end_date."})
        
        # Extract parameters
        column_name = str(params_dict.get("column_name", "")).strip()
        start_date = str(params_dict.get("start_date", "")).strip()
        end_date = str(params_dict.get("end_date", "")).strip()
        
        # Validate parameters
        if not column_name or not start_date or not end_date:
            return json.dumps({"error": "Missing required parameters. Please provide column_name, start_date, and end_date."})
        
        # Try to find the column (case-insensitive match)
        matched_column = None
        for col in market_data.columns:
            if column_name.lower() in col.lower():
                matched_column = col
                break
                
        if not matched_column:
            return json.dumps({"error": f"Column '{column_name}' not found in market data."})
            
        # Parse dates
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except ValueError:
            return json.dumps({"error": "Invalid date format. Please use YYYY-MM-DD format."})
            
        # Filter data by date range
        mask = (market_data.index >= start_date) & (market_data.index <= end_date)
        filtered_data = market_data.loc[mask, matched_column]
        
        if filtered_data.empty:
            return json.dumps({"error": f"No data available for {matched_column} between {start_date} and {end_date}."})
            
        # Convert to the format needed by create_market_chart
        result_data = {}
        result_data[matched_column] = {}
        
        for date, value in filtered_data.items():
            if not pd.isna(value):
                result_data[matched_column][date.strftime('%Y-%m-%d')] = float(value)
                
        # Return as JSON string
        return json.dumps(result_data)
        
    except Exception as e:
        return json.dumps({"error": f"Error getting market data: {str(e)}"})


@tool
def create_market_chart(data_str: str) -> Dict[str, Any]:
    """Create a Vega-Lite chart from a JSON/dict time series: {\"OMX30\": {\"2022-01-31\": 2134.07, ...}}"""
    try:
        import json
        
        # Parse the input, which might be JSON
        if isinstance(data_str, str):
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                print(f"Error parsing JSON: {data_str[:100]}...")
                return None
        else:
            data = data_str
            
        print(f"Data type: {type(data)}, Data: {str(data)[:100]}...")
        
        # Extract the index name (first key in data)
        index_name = list(data.keys())[0] if data and isinstance(data, dict) else "Market Index"
        
        # Create values array in the format expected by Vega-Lite
        values = []
        for idx_name, timeseries in data.items():
            if isinstance(timeseries, dict):
                for date_str, value in timeseries.items():
                    if not pd.isna(value):
                        values.append({
                            "date": date_str,
                            "value": float(value)
                        })
        
        # Sort values by date
        values.sort(key=lambda x: x["date"])
        
        # Calculate min value for y-axis scaling (90% of minimum)
        if values:
            min_value = min(item["value"] for item in values) * 0.9
        else:
            min_value = 0
            
        # Check if we have values to plot
        if not values:
            print("No valid data points found for chart")
            return None
        
        # Create the Vega-Lite spec directly
        start_date = values[0]["date"] if values else "N/A"
        end_date = values[-1]["date"] if values else "N/A"
        
        vega_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": f"{index_name} ({start_date} to {end_date})",
            "mark": "line",
            "width": 600,
            "height": 400,
            "data": {
                "values": values
            },
            "encoding": {
                "x": {
                    "field": "date",
                    "type": "temporal",
                    "title": "Date",
                    "axis": {
                        "format": "%Y-%m"
                    }
                },
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "title": "Price",
                    "scale": {
                        "domainMin": min_value
                    }
                }
            }
        }
        
        # Print the Vega-Lite spec for debugging
        print(f"Generated Vega-Lite spec with {len(values)} data points")
        
        return {
            "vega_lite_spec": vega_spec,
            "description": "Here's your requested chart!"
        }
        
    except Exception as e:
        print(f"Error creating chart: {str(e)}\n{traceback.format_exc()}")
        return None

# ============================================================================
# AGENT SETUP
# ============================================================================

# Update the prompt template
prompt_template = """You are Ashley, a helpful and knowledgeable financial analyst assistant from Stashly. For any greeting or 'initial_greeting' message, you must respond with exactly: "Hi, I'm Ashley, your financial assistant from Stashly. How can I assist you today?"

Previous conversation:
{chat_history}

You have access to the following tools:

{tools}

Tool names: {tool_names} and use llm to generate the response

##PERSONA:
You are Ashley, a vibrant, enthusiastic Financial Analyst and AI Assistant from Stashly.
You're an exceptional teacher who makes complex financial concepts simple and engaging.
Your approach is pedagogical, breaking down information into digestible pieces.
You're positive, encouraging, and adapt your explanations to different learning styles.
Your tone is warm, supportive, funny and occasionally playful (using emojis sparingly).
Always remember and refer to information from previous messages in the conversation.

##RESPONSE STRUCTURE:
1. Use information from the chat history to personalize responses and maintain context
2. Answer the user's question thoroughly using your built-in knowledge and tools
3. Provide one (only one)relevant follow-up suggestion based on the context, such as:
   - Related analyses you could perform
   - Additional metrics to consider
   - Complementary market data to examine
   - Other investment perspectives
4. Always end your response with: "Is there anything else I can help you with today?"

IMPORTANT FORMATTING INSTRUCTIONS:
- When using get_available_columns, provide the asset type as a simple string without quotes, e.g., "Equity" not "'Equity'"
- When using calculate_performance, format the input as a proper JSON string, e.g., "start_date": "2020-12-31", "end_date": "2021-12-31", "column_name": "Equity Sweden"
- When using generate_data_table, format the input as a proper JSON string, e.g., "columns": ["Equity Sweden"], "start_date": "2020-12-31", "end_date": "2021-12-31", "frequency": "monthly"
- When using calculate_annual_returns, format the input as a proper JSON string, e.g., "start_year": 2020, "end_year": 2024, "column_name": "Equity Sweden"
- When creating charts, first use get_market_data_for_chart to get real market data, then pass the result to create_market_chart
- ALWAYS follow the exact format below, including the "Thought:", "Action:", "Action Input:", "Observation:", and "Final Answer:" prefixes
- After each "Thought:", you MUST include either an "Action:" or "Final Answer:" - never end with just a "Thought:"
- When providing your final answer, ALWAYS use this exact format:
  Thought: I now know the final answer
  Final Answer: [your detailed response here]

  Would you like me to: [Suggestion 1], or is there anything else I can help you with today?

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


Begin!

Question: {input}
{agent_scratchpad}
"""

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=openai_key,
    model="gpt-4o-mini", #not working for gpt-4o-mini for some reason, cant figure out thought action
    temperature=0.6,
    request_timeout=120,
    max_retries=5
)

# Initialize memory for conversation history
memory = ConversationBufferWindowMemory(
    return_messages=True,
    max_token_limit=10000,  # Trim to last ~1000 tokens
    k=10  # Keep last 10 messages
)

# Define the tools
tools = [
    calculate_performance,
    get_data_info,
    get_financial_knowledge,
    respond,
    get_available_columns,
    generate_data_table,
    calculate_annual_returns,
    create_market_chart,
    get_market_data_for_chart,  # Add this new tool
]

# Create the agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=PromptTemplate.from_template(prompt_template)
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,
    return_intermediate_steps=True,
    early_stopping_method="force", # force generate if stuck
    timeout=120
)

# ============================================================================
# CHAT FUNCTIONALITY
# ============================================================================

def get_chat_history() -> str:
    """Get formatted chat history for the prompt."""
    try:
        # Get the messages from memory
        messages = memory.load_memory_variables({}).get("history", [])
        
        # If there's no history, return empty string
        if not messages:
            return ""
            
        # Format the messages into a string
        formatted_history = "\n".join([
            f"Human: {msg.content}" if msg.type == 'human' else f"Assistant: {msg.content}"
            for msg in messages
        ])
        
        return formatted_history
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return ""

def filter_messages(messages: List[Dict[str, str]], max_tokens: int = 4000) -> List[Dict[str, str]]:
    """
    Filter messages to keep under token limit while preserving context.
    """
    # Keep system message and last few exchanges
    filtered = []
    token_count = 0
    
    # Always keep the most recent messages
    for msg in reversed(messages):
        # Rough token estimation (4 chars ≈ 1 token)
        estimated_tokens = len(msg['content']) // 4
        
        if token_count + estimated_tokens <= max_tokens:
            filtered.insert(0, msg)
            token_count += estimated_tokens
        else:
            break
    
    return filtered

async def process_user_input(user_input: str, thread_id: str = "default") -> str:
    """Process user input and return the agent's response."""
    try:
        # Check for initial greeting
        if user_input.lower().strip() == "initial_greeting":
            response = {
                "output": "Hi, I'm Ashley from Stashly, your financial assistant. How can I assist you today?"
            }
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response["output"])
            return response["output"]
            
        # Get chat history
        chat_history = get_chat_history()
        
        # For all messages, use the agent with proper ReAct format
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history,
            "thread_id": thread_id  # Add thread_id to context
        })
        
        # Extract response
        response = result["output"]
        
        # Add to memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)
        
        return response
        
    except Exception as e:
        error_message = f"I apologize, but I encountered an error processing your request: {str(e)}"
        logger.error(f"Error in process_user_input: {str(e)}")
        memory.chat_memory.add_ai_message(error_message)
        return error_message

def chat():
    """Main chat loop function."""
    print("\nHi, I'm Ashley from Stashly, your financial assistant. How can I help you today? Type 'quit' to exit.\n")
    
    while True:
        # Get user input
        user_input = input("\nUser: ")
        
        # Check for quit command
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nThanks for chatting today! Talk to you soon!")
            break
        
        try:
            # Process user input through the agent with history
            result = process_user_input(user_input)
            print("\nAI:", result)
        except Exception as e:
            print(f"\nAI: I'm sorry, I encountered an error: {str(e)}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    chat()

async def run_portfolio_agent(state: GraphState) -> GraphState:
    print("Portfolio agent processing:", state["input"])
    try:
        # Get memory-based chat history
        chat_history = get_chat_history()

        # Run the ReAct agent with a timeout
        try:
            result = await agent_executor.ainvoke({
                "input": state["input"],
                "chat_history": chat_history,
                "thread_id": state.get("thread_id", "default")
            })
        except Exception as agent_error:
            if "iteration limit" in str(agent_error) or "time limit" in str(agent_error):
                # Create a graceful response for timeout
                result = {
                    "output": "I started analyzing this, but it's a complex question requiring more computation than I can handle at once. Could we break this down into smaller steps? For example, we could first look at the basic metrics, then dive deeper into specific analyses.",
                    "intermediate_steps": []
                }
            else:
                # Re-raise other errors
                raise agent_error

        # Update assistant memory
        memory.chat_memory.add_user_message(state["input"])
        memory.chat_memory.add_ai_message(result["output"])

        # Update state with portfolio output
        state["portfolio_output"] = result.get("output", "No response from portfolio agent.")
        
        import json
        from pprint import pprint

        # Avoid printing full result with unserializable types
        safe_result = {k: v for k, v in result.items() if k != "intermediate_steps"}
        print("\n🔍 Agent result (safe):")
        print(json.dumps(safe_result, indent=2))

        # Extract from intermediate_steps (tool output)
        steps = result.get("intermediate_steps", [])
        for action, observation in steps:
            if isinstance(observation, dict) and "vega_lite_spec" in observation:
                state["market_charts"] = [observation["vega_lite_spec"]]
                break

        # Fallback: check if it's directly on result
        if not state.get("market_charts"):
            if "vega_lite_spec" in result:
                state["market_charts"] = [result["vega_lite_spec"]]
            elif "market_charts" in result:
                state["market_charts"] = result["market_charts"]

        return state

        
           # 🚨 NEW: If the ReAct chain returns a vega-lite spec, store it
        # if "vega_lite_spec" in result:
        #     state["market_charts"] = [result["vega_lite_spec"]]
        # elif "market_charts" in result:
        #     state["market_charts"] = result["market_charts"]     
        
        return state

    except Exception as e:
        error_message = f"Portfolio Agent Error: {str(e)}"
        print(error_message)
        state["portfolio_output"] = error_message
        return state

# Make sure to export the function
__all__ = ['run_portfolio_agent', 'PortfolioAgent', 'memory']
