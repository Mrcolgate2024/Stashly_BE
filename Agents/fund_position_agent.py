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
    
    # Alias match (strongest)
    for alias, full_name in FUND_ALIASES.items():
        if alias == name:
            return full_name
        if name.startswith(alias):
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
    
    for short_name, pct in fund_weights.items():
        fund_name = find_closest_fund(short_name)
        df = FUND_DF[FUND_DF["fund_name"].str.lower() == fund_name.lower()]
        matches = df[df["Instrument Name"].str.contains(company, case=False, na=False)]

        fund_exposure = matches["% of Fund AUM"].astype(float).sum()
        user_exposure = (pct / 100.0) * fund_exposure

        total_exposure += user_exposure
        rows.append((fund_name, pct, fund_exposure, user_exposure))
        
        # Collect report dates
        if not matches.empty and 'Report Date' in matches.columns:
            report_dates.update(matches['Report Date'].dropna().unique())
    
    # Format report date if available
    report_date = None
    if report_dates:
        try:
            report_date = pd.to_datetime(list(report_dates)[0]).strftime('%Y-%m-%d')
        except:
            pass

    table = "| Fund | Allocation (%) | Exposure in Fund (%) | Total Exposure (%) |\n"
    table += "|------|-----------------|------------------------|---------------------|\n"
    for name, alloc, exp_fund, exp_total in rows:
        table += f"| {name} | {alloc:.2f} | {exp_fund:.4f} | {exp_total:.4f} |\n"

    summary = f"\U0001F4CA Your total exposure to '{company}' is approximately {total_exposure:.4f}%\n\n{table}"
    if report_date:
        summary += f"\n*Report Date: {report_date}*"
    return summary

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
        weights = {}
        for p in parts[:-1]:
            m = re.match(r"(\d+(\.\d+)?)%\s+(.+)", p.strip())
            if m:
                pct = float(m.group(1))
                fund = m.group(3).strip()
                weights[fund] = pct

        return calculate_total_exposure(company, weights)
    except Exception as e:
        return f"❌ Error in exposure calculation: {str(e)}"

@tool
def fund_exposure_lookup(input: str = "") -> str:
    """Use this only for a SINGLE fund + SINGLE company. Format: '25% in LF Global, exposure to Tesla'."""
    match = re.search(r"(?i)(\d+(\.\d+)?)%.*in (.*?) .*exposure to (.*)", input)
    if match:
        ownership_pct = float(match.group(1))
        fund_name = find_closest_fund(match.group(3).strip())
        company = match.group(4).strip()
        return calculate_exposure(fund_name, company, ownership_pct)
    else:
        return (
            "❌ This tool is for estimating exposure to **one company in one fund**. "
            "Use the format like: '25% in LF Global, exposure to Tesla'. "
            "If you're looking across multiple funds, use the `total_exposure_to_company` tool instead."
        )

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

# -----------------------------
# Agent Setup
# -----------------------------

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
    stop_looping
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4000)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = PromptTemplate.from_template("""
You are a fund analyst assistant helping users understand their fund holdings, exposures, and metadata using data from the Swedish Financial Supervisory Authority (Finansinspektionen).

Your job is to pick the most relevant tool, pass in clean input, and avoid repeating the same tool or rephrasing the same question.

Available tools: {tool_names}

{tools}

IMPORTANT FORMATTING INSTRUCTIONS:
- Use `fund_exposure_lookup` only if the user asks about **one fund + one company** (e.g., "How much exposure to Tesla in LF Global at 25%?")
- Use `total_exposure_to_company` when the user mentions **multiple funds + ownership percentages**
- Use `funds_holding_company` to find who owns a company (e.g., "Who holds Apple?")
- Use `top_holdings_for_fund` or `fund_holdings_table` for holdings (e.g., "Top 10 in LF Global", or "Show all holdings in LF Europe")
- Use `sector_exposure_lookup` for broad sector queries (e.g., "How much is exposed to tech?")
- Use `lookup_sector_by_isin` when given an ISIN
- Use `list_all_funds` if asked "Which funds are available?" or similar
- If nothing applies or the question is ambiguous, use `stop_looping` to gracefully stop

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


{chat_history}
User question: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=3,
    max_execution_time=180,
    early_stopping_method="generate",
    handle_parsing_errors=True  # 👈 this is key
)

# -----------------------------
# LangGraph Entrypoint
# -----------------------------

async def run_fund_position_agent(state: dict) -> dict:
    last_message = state["messages"][-1]["content"]

    # Create isolated memory for this call
    local_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Use local memory in a temporary executor
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=local_memory,
        verbose=True,
        max_iterations=10,
        max_execution_time=180,
        handle_parsing_errors=True,
        structured=True,
        return_intermediate_steps=True,
        early_stopping_method="force",
    )

    response = await executor.ainvoke({"input": last_message})
    output = response.get("output", "No answer.")
    return {
        **state,
        "fund_output": output,
        "last_ran_agent": "fund",
        "messages": state["messages"] + [{"role": "assistant", "content": output}]
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
    "lookup_sector_by_isin"
]