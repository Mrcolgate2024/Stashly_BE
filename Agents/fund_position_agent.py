from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import tool
import pandas as pd
import difflib
import os
from dotenv import load_dotenv
from typing import TypedDict, Optional, List
from pathlib import Path
import re
import requests
from langchain.prompts import ChatPromptTemplate
from Agents.fund_loader import load_all_fund_positions
import traceback


# Load .env from project root
# print("\nTrying to load .env file...")
DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

class GraphState(TypedDict):
    input: str
    thread_id: str
    messages: List[dict]
    portfolio_output: Optional[str]
    market_report_output: Optional[str]
    conversational_output: Optional[str]
    websearch_output: Optional[str]
    stock_output: Optional[str]
    chat_output: Optional[str]
    fund_output: Optional[str]
    market_charts: Optional[List[dict]]
    user_name: Optional[str]
    last_ran_agent: Optional[str]

# -----------------------------
# Load Fund Position Data Once
# -----------------------------
FUND_DF = load_all_fund_positions()
# Replace empty strings only in object (string) columns
for col in FUND_DF.select_dtypes(include="object").columns:
    FUND_DF[col] = FUND_DF[col].fillna("")

FUND_ALIASES = {
    "lf": "länsförsäkringar",
    "lf usa": "Länsförsäkringar USA Index",
    "lf europe": "Länsförsäkringar Europa Index",
    "lf sweden": "Länsförsäkringar Sverige Index",
    "lf global": "Länsförsäkringar Global Index",
    "länsförsäkringar global index": "Länsförsäkringar Global Index",
    "länsförsäkringar usa index": "Länsförsäkringar USA Index",
    "länsförsäkringar europa index": "Länsförsäkringar Europa Index",
    "länsförsäkringar sverige index": "Länsförsäkringar Sverige Index"
}

# -----------------------------
# Helper Functions
# -----------------------------

def get_sector_and_industry_from_figi(isin: str) -> str:
    """
    Lookup sector and industry from OpenFIGI using ISIN, with fallback to yfinance if needed.
    """
    api_key = os.getenv("OPENFIGI_API_KEY")
    if not api_key:
        return "❌ Missing OpenFIGI API key. Set OPENFIGI_API_KEY in your environment."

    headers = {
        "Content-Type": "application/json",
        "X-OPENFIGI-APIKEY": api_key
    }

    body = [{"idType": "ID_ISIN", "idValue": isin}]
    try:
        response = requests.post("https://api.openfigi.com/v3/mapping", json=body, headers=headers)
        if response.status_code != 200:
            return f"❌ Error from OpenFIGI API: {response.status_code} - {response.text}"

        results = response.json()
        data = results[0].get("data")
        if not data:
            return f"❌ No data found for ISIN {isin}"

        record = data[0]
        name = record.get("name", "N/A")
        ticker = record.get("ticker", "N/A")
        sector = record.get("sector") or "⚠️ Not provided"
        industry = record.get("industry") or "⚠️ Not provided"

        # Fallback: try yfinance if missing sector/industry
        if ticker != "N/A" and (sector == "⚠️ Not provided" or industry == "⚠️ Not provided"):
            try:
                yf_info = yf.Ticker(ticker).info
                sector_yf = yf_info.get("sector")
                industry_yf = yf_info.get("industry")
                if sector_yf:
                    sector = sector_yf
                if industry_yf:
                    industry = industry_yf
            except Exception:
                pass

        return (
            f"🔍 Lookup result for ISIN `{isin}`:\n\n"
            f"- Name: {name}\n"
            f"- Ticker: {ticker}\n"
            f"- Sector: {sector}\n"
            f"- Industry: {industry}"
        )

    except Exception as e:
        return f"❌ Error during OpenFIGI lookup: {str(e)}"


def find_closest_fund(name: str) -> str:
    name = name.lower().strip()
    
    # First check if the input is already a full fund name
    if name in [f.lower() for f in FUND_DF["fund_name"].dropna().unique()]:
        return name.title()
    
    # Alias match (strongest)
    for alias, full_name in FUND_ALIASES.items():
        if alias == name:
            return full_name
        if name.startswith(alias):
            return full_name
        # Also check if the input contains the alias
        if alias in name:
            return full_name

    # Fuzzy fallback
    options = FUND_DF["fund_name"].dropna().unique().tolist()
    closest = difflib.get_close_matches(name, options, n=1, cutoff=0.7)
    return closest[0] if closest else ""

def find_holdings_for_fund(fund_name: str) -> pd.DataFrame:
    return FUND_DF[FUND_DF["fund_name"].str.lower() == fund_name.lower()]

def calculate_exposure(fund_name: str, company: str, ownership_pct: float) -> str:
    df = find_holdings_for_fund(fund_name)
    if df.empty:
        return f"Fund '{fund_name}' not found."

    matches = df[df["Instrument Name"].str.contains(company, case=False, na=False)]
    if matches.empty:
        return f"No holdings matching '{company}' in fund '{fund_name}'."

    # Get the report date
    report_date = matches['Report Date'].iloc[0] if not matches.empty else None
    if report_date:
        try:
            report_date = pd.to_datetime(report_date).strftime('%Y-%m-%d')
        except:
            pass

    exposure = 0.0
    for _, row in matches.iterrows():
        pct = row.get("% of Fund AUM", 0)
        exposure += (ownership_pct / 100.0) * pct

    summary = f"Your estimated exposure to '{company}' through {fund_name} at {ownership_pct}% ownership is {exposure:.4f}% of the fund's assets."
    if report_date:
        summary += f"\n*Report Date: {report_date}*"
    return summary

def calculate_total_exposure(company: str, fund_weights: dict) -> str:
    total_exposure = 0.0
    rows = []
    report_dates = set()
    
    for fund_name, ownership_pct in fund_weights.items():
        # Get raw exposure for this fund first
        result = fund_exposure_lookup(f"{fund_name}, {company}")
        
        # Parse the result to extract exposure
        if "❌" in result or "No holdings matching" in result:  # Error or no holdings case
            rows.append((fund_name, ownership_pct, 0.0, 0.0))  # Add row with zero exposure
            continue
            
        # Try to extract raw fund exposure first
        match = re.search(r"represents ([\d.]+)% of", result)
        if match:
            fund_exposure = float(match.group(1))
            user_exposure = (ownership_pct / 100.0) * fund_exposure
            
            total_exposure += user_exposure
            rows.append((fund_name, ownership_pct, fund_exposure, user_exposure))
            
            # Extract report date if present
            if "*Report Date:" in result:
                report_date = result.split("*Report Date: ")[1].split("*")[0]
                report_dates.add(report_date)
    
    # Format the response table
    table = "| Fund | Allocation (%) | Exposure in Fund (%) | Total Exposure (%) |\n"
    table += "|------|-----------------|------------------------|---------------------|\n"
    for name, alloc, exp_fund, exp_total in rows:
        table += f"| {name} | {alloc:.2f} | {exp_fund:.4f} | {exp_total:.4f} |\n"

    summary = f"📊 Your total exposure to {company} is approximately {total_exposure:.4f}%\n\n{table}"
    if report_dates:
        summary += f"\n*Report Date(s): {', '.join(sorted(report_dates))}*"
    
    return summary

def calculate_fund_risk(stddev_24m: float) -> str:
    """Calculate risk level based on 24-month standard deviation."""
    if stddev_24m is None:
        return "Unknown"
    if stddev_24m < 5:
        return "Low"
    elif stddev_24m < 15:
        return "Medium"
    else:
        return "High"

def calculate_portfolio_risk(fund_names: List[str]) -> str:
    """Calculate the overall risk level of a portfolio based on its constituent funds."""
    try:
        total_risk = 0
        total_weight = 0
        risk_details = []
        
        for fund_name in fund_names:
            fund_data = FUND_DF[FUND_DF['fund_name'].str.lower() == fund_name.lower()]
            if not fund_data.empty:
                fund_metadata = fund_data.iloc[0]
                stddev_24m = fund_metadata.get('stddev_24m')
                if stddev_24m is not None:
                    # Use fund AUM as weight
                    weight = fund_metadata.get('fund_aum', 1)
                    total_risk += stddev_24m * weight
                    total_weight += weight
                    risk_level = calculate_fund_risk(stddev_24m)
                    risk_details.append(f"{fund_name}: {risk_level} risk ({stddev_24m:.2f}% stddev)")
        
        if total_weight == 0:
            return "Could not calculate portfolio risk - no valid fund data found"
        
        # Calculate weighted average risk
        weighted_avg_risk = total_risk / total_weight
        portfolio_risk_level = calculate_fund_risk(weighted_avg_risk)
        
        response = f"Portfolio Risk Level: {portfolio_risk_level}\n"
        response += f"Weighted Average Standard Deviation: {weighted_avg_risk:.2f}%\n\n"
        response += "Individual Fund Risk Levels:\n"
        response += "\n".join(risk_details)
        
        return response
    except Exception as e:
        return f"Error calculating portfolio risk: {str(e)}"

# -----------------------------
# LangChain Tools
# -----------------------------

@tool
def total_exposure_to_company(input: str = "") -> str:
    """Use this to calculate exposure to a company across MULTIPLE funds. Format: '25% LF Global, 25% LF USA, Tesla'."""
    try:
        parts = input.split(",")
        if len(parts) < 2:
            return "Please specify fund allocations and company name, e.g. '25% LF Global, 25% LF USA, Tesla'"

        company = parts[-1].strip()
        total_exposure = 0.0
        report_dates = set()
        rows = []
        errors = []

        # Process fund allocations
        fund_weights = {}
        for p in parts[:-1]:
            m = re.match(r"(\d+(\.\d+)?)%\s+(.+)", p.strip())
            if m:
                pct = float(m.group(1))
                fund_name = m.group(3).strip()
                resolved_fund = find_closest_fund(fund_name)
                if resolved_fund:
                    fund_weights[resolved_fund] = pct
                else:
                    errors.append(f"❌ Fund not found: {fund_name}")

        # Calculate exposure for each fund
        for fund_name, ownership_pct in fund_weights.items():
            # Get raw exposure for this fund
            result = fund_exposure_lookup(f"{fund_name}, {company}")
            
            # Parse the result to extract exposure
            if "❌" in result:  # Error case
                errors.append(result)
                continue
                
            # Try to extract raw fund exposure
            match = re.search(r"represents ([\d.]+)% of", result)
            if match:
                fund_exposure = float(match.group(1))
                user_exposure = (ownership_pct / 100.0) * fund_exposure
                
                total_exposure += user_exposure
                rows.append((fund_name, ownership_pct, fund_exposure, user_exposure))
                
                # Extract report date if present
                if "*Report Date:" in result:
                    report_date = result.split("*Report Date: ")[1].split("*")[0]
                    report_dates.add(report_date)

        if not rows:
            if errors:
                return "\n".join(errors)
            return f"❌ No valid exposures found for {company} in the specified funds"

        # Format the response table
        table = "| Fund | Allocation (%) | Exposure in Fund (%) | Total Exposure (%) |\n"
        table += "|------|-----------------|------------------------|---------------------|\n"
        for name, alloc, exp_fund, exp_total in rows:
            table += f"| {name} | {alloc:.2f} | {exp_fund:.4f} | {exp_total:.4f} |\n"

        summary = f"📊 Your total exposure to {company} is approximately {total_exposure:.4f}%\n\n{table}"
        if report_dates:
            summary += f"\n*Report Date(s): {', '.join(sorted(report_dates))}*"
        if errors:
            summary += "\n\n⚠️ Note: Some funds were skipped due to errors:\n" + "\n".join(errors)
        return summary
    except Exception as e:
        return f"❌ Error in exposure calculation: {str(e)}"

@tool
def fund_exposure_lookup(fund_name: str) -> str:
    """Look up a fund's holdings and exposure details."""
    try:
        # Try exact match first
        fund_data = FUND_DF[FUND_DF['fund_name'].str.lower() == fund_name.lower()]
        
        # If no exact match, try fuzzy matching
        if fund_data.empty:
            all_funds = FUND_DF['fund_name'].unique()
            matches = difflib.get_close_matches(fund_name, all_funds, n=1, cutoff=0.6)
            if matches:
                fund_data = FUND_DF[FUND_DF['fund_name'].str.lower() == matches[0].lower()]
        
        if fund_data.empty:
            return f"Could not find fund: {fund_name}"
        
        # Get fund metadata
        fund_metadata = fund_data.iloc[0]
        fund_name = fund_metadata['fund_name']
        fund_company = fund_metadata['fund_company']
        fund_aum = fund_metadata['fund_aum']
        stddev_24m = fund_metadata.get('stddev_24m')
        active_risk = fund_metadata.get('active_risk')
        benchmark = fund_metadata.get('benchmark', '')
        
        # Calculate risk level
        risk_level = calculate_fund_risk(stddev_24m)
        
        # Format the response
        response = f"Fund: {fund_name}\n"
        response += f"Company: {fund_company}\n"
        response += f"AUM: {fund_aum:,.2f} SEK\n"
        if stddev_24m is not None:
            response += f"24-month Standard Deviation: {stddev_24m:.2f}%\n"
        if active_risk is not None:
            response += f"Active Risk: {active_risk:.2f}%\n"
        response += f"Risk Level: {risk_level}\n"
        if benchmark:
            response += f"Benchmark: {benchmark}\n"
        
        # Get top holdings
        holdings = fund_data[['Instrument Name', '% of Fund AUM']].sort_values('% of Fund AUM', ascending=False)
        if not holdings.empty:
            response += "\nTop Holdings:\n"
            for _, row in holdings.head(5).iterrows():
                response += f"{row['Instrument Name']}: {row['% of Fund AUM']:.2f}%\n"
        
        return response
    except Exception as e:
        return f"Error looking up fund: {str(e)}"

@tool
def largest_position_across_funds(input: str = "") -> str:
    """Find the largest single position across all funds."""
    # Check if the input contains a fund name
    fund_name = find_closest_fund(input)
    if fund_name:
        # If a fund is specified, only look in that fund
        df = FUND_DF[FUND_DF["fund_name"].str.lower() == fund_name.lower()]
    else:
        # If no fund specified, look across all funds
        df = FUND_DF
    
    if df.empty:
        return f"No holdings found for fund '{input}'."
    
    row = df.loc[df["Market Value"].astype(float).idxmax()]
    report_date = row['Report Date'] if 'Report Date' in row else None
    if report_date:
        try:
            report_date = pd.to_datetime(report_date).strftime('%Y-%m-%d')
        except:
            pass
    
    fund_context = f" in {fund_name}" if fund_name else " across all funds"
    summary = f"Largest holding{fund_context}: {row['Instrument Name']} ({row['ISIN']}) worth {row['Market Value']} in {row['fund_name']}"
    if report_date:
        summary += f"\n*Report Date: {report_date}*"
    return summary

@tool
def top_holdings_for_fund(input: str = "") -> str:
    """List the top 10 holdings in a specific fund."""
    fund_name = find_closest_fund(input)
    df = find_holdings_for_fund(fund_name)
    if df.empty:
        return f"Fund '{input}' not found."
    
    # Get the report date
    report_date = df['Report Date'].iloc[0] if not df.empty else None
    
    top = df.sort_values("Market Value", ascending=False).head(10)
    table = "| Holding | % of Fund AUM |\n|---------|----------------|\n"
    for _, row in top.iterrows():
        table += f"| {row['Instrument Name']} | {row['% of Fund AUM']:.2f}% |\n"
    
    summary = f"Top 10 holdings in {fund_name}"
    if report_date:
        summary += f" (Report Date: {report_date})"
    summary += ":\n\n" + table
    return summary

@tool
def sector_exposure_lookup(input: str = "") -> str:
    """Find total exposure across all funds to a specific sector (English or Swedish name)."""
    sector_query = input.lower()
    
    # Check if the input contains a fund name
    fund_name = find_closest_fund(input)
    if fund_name:
        # If a fund is specified, only look in that fund
        matches = FUND_DF[
            (FUND_DF["fund_name"].str.lower() == fund_name.lower()) &
            (FUND_DF["Industry Name"].str.lower().str.contains(sector_query) |
             FUND_DF["Bransch_namn"].str.lower().str.contains(sector_query))
        ]
    else:
        # If no fund specified, look across all funds
        matches = FUND_DF[
            FUND_DF["Industry Name"].str.lower().str.contains(sector_query) |
            FUND_DF["Bransch_namn"].str.lower().str.contains(sector_query)
        ]
    
    if matches.empty:
        return f"No sector matching '{sector_query}' found."
    
    # Get the report date
    report_date = matches['Report Date'].iloc[0] if not matches.empty else None
    if report_date:
        try:
            report_date = pd.to_datetime(report_date).strftime('%Y-%m-%d')
        except:
            pass
    
    total_value = matches["Market Value"].astype(float).sum()
    total_aum = matches["fund_aum"].astype(float).sum()
    pct = (total_value / total_aum * 100) if total_aum > 0 else 0
    
    fund_context = f" in {fund_name}" if fund_name else " across all funds"
    summary = f"Total reported exposure{fund_context} to sector '{sector_query}' is approximately {pct:.2f}% of fund assets."
    if report_date:
        summary += f"\n*Report Date: {report_date}*"
    return summary

@tool
def funds_holding_company(input: str = "") -> str:
    """List all funds that hold a specific company."""
    # Check if the input contains a fund name
    fund_name = find_closest_fund(input)
    if fund_name:
        # If a fund is specified, only look in that fund
        matches = FUND_DF[
            (FUND_DF["fund_name"].str.lower() == fund_name.lower()) &
            (FUND_DF["Instrument Name"].str.contains(input, case=False, na=False))
        ]
    else:
        # If no fund specified, look across all funds
        matches = FUND_DF[FUND_DF["Instrument Name"].str.contains(input, case=False, na=False)]
    
    if matches.empty:
        return f"No funds found holding '{input}'."
    
    # Get the report date
    report_date = matches['Report Date'].iloc[0] if not matches.empty else None
    if report_date:
        try:
            report_date = pd.to_datetime(report_date).strftime('%Y-%m-%d')
        except:
            pass
    
    fund_context = f" in {fund_name}" if fund_name else ""
    summary = f"Funds holding '{input}'{fund_context}"
    if report_date:
        summary += f" (Report Date: {report_date})"
    summary += ":\n\n" + "\n".join(f"- {name}" for name in sorted(matches["fund_name"].unique().tolist()))
    return summary

@tool
def funds_not_holding_company(input: str = "") -> str:
    """List all funds that do NOT hold a specific company."""
    # Check if the input contains a fund name
    fund_name = find_closest_fund(input)
    if fund_name:
        # If a fund is specified, only look in that fund
        holding_funds = set(FUND_DF[
            (FUND_DF["fund_name"].str.lower() == fund_name.lower()) &
            (FUND_DF["Instrument Name"].str.contains(input, case=False, na=False))
        ]["fund_name"].unique())
        non_holding_funds = [fund_name] if fund_name not in holding_funds else []
    else:
        # If no fund specified, look across all funds
        all_funds = set(FUND_DF["fund_name"].unique())
        holding_funds = set(FUND_DF[FUND_DF["Instrument Name"].str.contains(input, case=False, na=False)]["fund_name"].unique())
        non_holding_funds = sorted(all_funds - holding_funds)
    
    # Get the report date
    report_date = FUND_DF['Report Date'].iloc[0] if not FUND_DF.empty else None
    if report_date:
        try:
            report_date = pd.to_datetime(report_date).strftime('%Y-%m-%d')
        except:
            pass
    
    fund_context = f" in {fund_name}" if fund_name else ""
    summary = f"Funds NOT holding '{input}'{fund_context}"
    if report_date:
        summary += f" (Report Date: {report_date})"
    summary += ":\n\n" + "\n".join(f"- {name}" for name in non_holding_funds)
    return summary

@tool
def fund_holdings_table(input: str = "") -> str:
    """List top 20 holdings of a fund as a markdown table."""
    fund_name = find_closest_fund(input)
    df = find_holdings_for_fund(fund_name)
    if df.empty:
        return f"No holdings found for fund '{input}'."
    
    # Get the report date and format it as YYYY-MM-DD
    report_date = df['Report Date'].iloc[0] if not df.empty else None
    if report_date:
        try:
            report_date = pd.to_datetime(report_date).strftime('%Y-%m-%d')
        except:
            pass
    
    # Sort by market value and get total counts
    df = df.sort_values("Market Value", ascending=False)
    total_holdings = len(df)
    total_value = df['Market Value'].astype(float).sum()
    
    # Only show top 20 holdings
    df_top = df.head(20)
    top_20_value = df_top['Market Value'].astype(float).sum()
    top_20_percentage = (top_20_value / total_value) * 100
    
    # Rename Bransch_namn to Sector
    if 'Bransch_namn' in df_top.columns:
        df_top = df_top.rename(columns={'Bransch_namn': 'Sector'})
    
    # Select and reorder columns
    columns = [
        'Instrument Name', 'ISIN', 'Country Code', 'Market Value', 
        '% of Fund AUM', 'Sector', 'Currency'
    ]
    
    # Filter to only existing columns
    available_columns = [col for col in columns if col in df_top.columns]
    df_display = df_top[available_columns]
    
    # Format the DataFrame
    if 'Market Value' in df_display.columns:
        df_display['Market Value'] = df_display['Market Value'].apply(lambda x: f"{int(float(x)):,}")
    if '% of Fund AUM' in df_display.columns:
        df_display['% of Fund AUM'] = df_display['% of Fund AUM'].apply(lambda x: f"{float(x):.2f}%")
    
    # Create markdown table with right-aligned numeric columns
    table = "| " + " | ".join(available_columns) + " |\n"
    table += "|" + "|".join(["-" * len(col) for col in available_columns]) + "|\n"
    
    for _, row in df_display.iterrows():
        values = []
        for col in available_columns:
            value = row[col]
            if col in ['Market Value', '% of Fund AUM']:
                values.append(f"{value:>12}")  # Right-align numeric columns
            else:
                values.append(str(value))
        table += "| " + " | ".join(values) + " |\n"
    
    summary = f"Here are the top 20 holdings for {fund_name}:\n\n"
    if report_date:
        summary += f"*Report Date: {report_date}*\n\n"
    summary += table
    
    # Add summary section
    summary += "\n### Summary:\n"
    if total_holdings > 20:
        summary += f"- Showing top 20 holdings out of {total_holdings} total holdings."
    summary += f"- Market value of top 20 holdings: {int(top_20_value):,}\n"
    summary += f"- Total fund market value: {int(total_value):,}\n"
    summary += f"- Top 20 holdings represent: {top_20_percentage:.1f}% of total fund value\n"
    

    
    return summary

@tool
def list_all_funds(_: str = "") -> str:
    """List all available funds along with metadata such as ISIN, AUM, fees, benchmark, and risk."""
    columns = ["fund_name", "fund_company", "fund_isin", "fund_aum", "cash", "other_assets_liabilities", "active_risk", "stddev_24m", "benchmark", "Report Date"]
    df = FUND_DF[columns].drop_duplicates().sort_values("fund_name")
    
    # Format the report date
    if 'Report Date' in df.columns:
        df['Report Date'] = df['Report Date'].apply(lambda x: str(x) if pd.notnull(x) else 'N/A')
    
    return "### All Registered Funds with Metadata:\n\n" + df.to_markdown(index=False)

@tool
def lookup_sector_by_isin(isin: str) -> str:
    """Lookup sector and industry using ISIN via OpenFIGI."""
    return get_sector_and_industry_from_figi(isin)

@tool
def stop_looping(_: str = "") -> str:
    """Use this if the current tools do not work or input format is unclear."""
    return "⚠️ It seems none of the tools matched the input. Please clarify your request!"

@tool
def portfolio_risk_analysis(fund_list: str) -> str:
    """Analyze the risk level of a portfolio consisting of multiple funds."""
    try:
        # Split the fund list and clean up
        fund_names = [name.strip() for name in fund_list.split(",")]
        return calculate_portfolio_risk(fund_names)
    except Exception as e:
        return f"Error analyzing portfolio risk: {str(e)}"

@tool
def get_fund_risk_metrics(fund_name: str) -> str:
    """Get the pre-calculated risk metrics for a fund directly from the fund loader data."""
    try:
        # Try exact match first
        fund_data = FUND_DF[FUND_DF['fund_name'].str.lower() == fund_name.lower()]
        
        # If no exact match, try fuzzy matching
        if fund_data.empty:
            all_funds = FUND_DF['fund_name'].unique()
            matches = difflib.get_close_matches(fund_name, all_funds, n=1, cutoff=0.6)
            if matches:
                fund_data = FUND_DF[FUND_DF['fund_name'].str.lower() == matches[0].lower()]
        
        if fund_data.empty:
            return f"Could not find fund: {fund_name}"
        
        # Get fund metadata
        fund_metadata = fund_data.iloc[0]
        fund_name = fund_metadata['fund_name']
        
        # Safely get and convert numeric values
        stddev_24m = fund_metadata.get('stddev_24m')
        if stddev_24m is not None:
            try:
                stddev_24m = float(stddev_24m)
            except (ValueError, TypeError):
                stddev_24m = None
                
        active_risk = fund_metadata.get('active_risk')
        if active_risk is not None:
            try:
                active_risk = float(active_risk)
            except (ValueError, TypeError):
                active_risk = None
        
        # Format the response with just the raw metrics
        response = f"Risk metrics for {fund_name}:\n"
        if stddev_24m is not None:
            response += f"24-month Standard Deviation: {stddev_24m:.2f}%\n"
        if active_risk is not None:
            response += f"Active Risk: {active_risk:.2f}%"
        
        return response
    except Exception as e:
        return f"Error getting fund risk metrics: {str(e)}"

# -----------------------------
# Agent Setup
# -----------------------------

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",
    temperature=0.6,
    request_timeout=120,
    max_retries=5,
)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=4000,
    k=5
)

# Create the prompt template
prompt_template = """You are a financial advisor specializing in fund analysis and holdings. You have access to detailed fund position data and can answer questions about:
- Fund holdings and allocations
- Sector and industry exposure
- Company-specific exposure calculations
- ISIN lookups and sector/industry information
- Fund performance and characteristics
- Risk analysis and portfolio risk assessment

When analyzing risk:
- Use the portfolio_risk_analysis tool for portfolio risk questions
- Use the fund_exposure_lookup tool for individual fund risk questions
- Consider both individual fund risk and portfolio-level risk
- Provide context about what the risk levels mean
- Suggest risk-mitigation strategies when appropriate

You have access to the following tools:
{tools}

Tool names: {tool_names}

Instructions:
- For single fund exposure queries:
  * Use fund_exposure_lookup with format "Fund Name, Company Name" or "X% in Fund Name, exposure to Company Name"
- For multi-fund exposure queries:
  * Use total_exposure_to_company with format "X% in Fund A, Y% in Fund B, Company Name"
- For risk analysis:
  * Use portfolio_risk_analysis for portfolio risk questions
  * Use fund_exposure_lookup for individual fund risk questions
- For ISIN lookups, use lookup_sector_by_isin
- For fund holdings, use fund_holdings_table or top_holdings_for_fund
- For sector/industry queries, use sector_exposure_lookup
- Always provide clear, structured responses with relevant data
- Include report dates when available
- Use markdown formatting for better readability

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Conversation so far:
{chat_history}

Question: {input}
{agent_scratchpad}"""

# Define the tools
tools = [
    fund_exposure_lookup,
    total_exposure_to_company,
    largest_position_across_funds,
    top_holdings_for_fund,
    sector_exposure_lookup,
    funds_holding_company,
    funds_not_holding_company,
    fund_holdings_table,
    list_all_funds,
    lookup_sector_by_isin,
    portfolio_risk_analysis,
    get_fund_risk_metrics
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
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    return_intermediate_steps=True,
    structured=True,
    early_stopping_method="force"
)

# -----------------------------
# LangGraph-Compatible Wrapper
# -----------------------------
async def run_fund_position_agent(state: GraphState) -> GraphState:
    last_message = state["messages"][-1]["content"] if state["messages"] else state["input"]
    
    try:
        result = await agent_executor.ainvoke({
            "input": last_message
        })
        
        # Check if we got a valid response
        output = result.get("output", "")
        if "❌" in output:  # Error response
            return {
                **state,
                "messages": state["messages"] + [{"role": "assistant", "content": output}],
                "fund_output": output,
                "last_ran_agent": "fund"
            }
        
        # Check for missing API key
        if "Missing OpenFIGI API key" in output:
            error_msg = "Unable to look up sector/industry information. Please ensure the OpenFIGI API key is configured."
            return {
                **state,
                "messages": state["messages"] + [{"role": "assistant", "content": error_msg}],
                "fund_output": error_msg,
                "last_ran_agent": "fund"
            }
        
        # Check for iteration/time limit
        if "stopped due to iteration limit" in output or "stopped due to time limit" in output:
            error_msg = "I apologize, but I was unable to complete the request. Please try rephrasing your question."
            return {
                **state,
                "messages": state["messages"] + [{"role": "assistant", "content": error_msg}],
                "fund_output": error_msg,
                "last_ran_agent": "fund"
            }
        
        # Valid response
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": output}],
            "fund_output": output,
            "last_ran_agent": "fund"
        }
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"Error in fund position agent: {str(e)}")
        print(traceback.format_exc())
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": error_msg}],
            "fund_output": error_msg,
            "last_ran_agent": "fund"
        }
    
    
# -----------------------------
# Standalone Test (Optional)
# -----------------------------
async def process_user_input(message: str, thread_id: str = "test-fund") -> dict:
    state: GraphState = {
        "input": message,
        "thread_id": thread_id,
        "messages": [{"role": "user", "content": message}],
        "portfolio_output": None,
        "market_report_output": None,
        "conversational_output": None,
        "websearch_output": None,
        "stock_output": None,
        "chat_output": None,
        "fund_output": None,
        "market_charts": None,
        "user_name": None,
        "last_ran_agent": None
    }
    return await run_fund_position_agent(state)


__all__ = [
    "run_fund_position_agent",
    "agent_executor",
    "memory",
    "total_exposure_to_company",
    "fund_exposure_lookup",
    "largest_position_across_funds",
    "top_holdings_for_fund",
    "sector_exposure_lookup",
    "funds_holding_company",
    "funds_not_holding_company",
    "fund_holdings_table",
    "list_all_funds",
    "lookup_sector_by_isin",
    "portfolio_risk_analysis",
    "get_fund_risk_metrics"
]