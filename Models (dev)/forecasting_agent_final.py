# forecasting_agent_final.py
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import joblib
from datetime import datetime
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from fredapi import Fred
import yfinance as yf
import matplotlib.pyplot as plt
import os
from typing import TypedDict, Optional, List
from sklearn.cluster import KMeans

MODEL_PATHS = {
    "1M": "Models/lasso_macro_forecast_1M.joblib",
    "3M": "Models/lasso_macro_forecast_3M.joblib",
    "1Y": "Models/lasso_macro_forecast_1Y.joblib",
    "2Y": "Models/lasso_macro_forecast_2Y.joblib",
    "4Y": "Models/lasso_macro_forecast_4Y.joblib"
}
SCALER_PATH = "Models/final_fresh_scaler.joblib"
KMEANS_PATH = "Models/final_fresh_kmeans.joblib"
EXCEL_FALLBACK_PATH = "Knowledge_base/market_data_with_features.xlsx"

FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "gdp": "GDP",
    "inflation": "CPIAUCSL",
    "unemployment": "UNRATE",
    "gov10y": "DGS10"
}

DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

# Initialize global resources
MODELS = {k: joblib.load(MODEL_PATHS[k]) for k in MODEL_PATHS}
SCALER = joblib.load(SCALER_PATH)
KMEANS = joblib.load(KMEANS_PATH)
FRED = Fred(api_key=os.getenv("FRED_API_KEY"))
RMSE_BY_MODEL = {
    "1M": 8.32,
    "3M": 17.34,
    "1Y": 13.27,
    "2Y": 11.25,
    "4Y": 6.74
}

@tool
def get_latest_macro_data(tool_input: str = "") -> pd.DataFrame:
    """Fetch the latest macroeconomic indicators from FRED, yfinance, and Excel fallback."""
    try:
        data = {}
        for key, series in FRED_SERIES.items():
            s = FRED.get_series(series).dropna()
            data[key] = s.iloc[-1]

        cpi_series = FRED.get_series("CPIAUCSL").dropna()
        data["inflation_yoy"] = ((cpi_series.iloc[-1] - cpi_series.iloc[-13]) / cpi_series.iloc[-13]) * 100

        fx = yf.download("SEK=X", period="5d", interval="1d")
        data["usdsek"] = fx['Close'].iloc[-1]

        excel_df = pd.read_excel(EXCEL_FALLBACK_PATH, index_col=0)
        latest_data = excel_df.iloc[[-1]][[
            'Sweden inflation YOY%', 'Govt 10Y Sweden %', 'Sweden GDP YOY %',
            'Govt 10Y Europe %', 'MSCI World', 'VIX'
        ]].to_dict(orient='records')[0]
        
        data.update({
            'swed_inflation': latest_data['Sweden inflation YOY%'],
            'swed_gov10y': latest_data['Govt 10Y Sweden %'],
            'swed_gdp': latest_data['Sweden GDP YOY %'],
            'euro_gov10y': latest_data['Govt 10Y Europe %'],
            'msci_world': latest_data['MSCI World'],
            'vix': latest_data['VIX']
        })

        return pd.DataFrame([data])
    except Exception as e:
        print("Data fetch failed:", e)
        df = pd.read_excel(EXCEL_FALLBACK_PATH, index_col=0)
        return df.iloc[[-1]]

@tool
def preprocess(tool_input: str = "") -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    """
    try:
        # Get latest macro data
        macro_df = get_latest_macro_data.invoke("")
        
        # Get the feature names from the scaler
        scaler_features = SCALER.get_feature_names_out()
        
        # Create base features dictionary
        base_cols = {
            'fed_funds': 'Fed Funds rate %',
            'gdp': 'USA GDP',
            'inflation': 'USA inflation',
            'unemployment': 'USA unemployment %',
            'usdsek': 'FX SEKUSD',
            'gov10y': 'Govt 10Y US %',
            'euro_gov10y': 'Govt 10Y Europe %',
            'inflation_yoy': 'USA inflation YOY%',
            'swed_inflation': 'Sweden inflation YOY%',
            'swed_gov10y': 'Govt 10Y Sweden %',
            'swed_gdp': 'Sweden GDP YOY %',
            'msci_world': 'MSCI World',
            'vix': 'VIX'
        }
        
        # Initialize feature dictionary with base features
        feature_data = {old_name: macro_df[new_name].iloc[0] for new_name, old_name in base_cols.items()}
        
        # Create a list of all possible lags and combinations
        lags = list(range(1, 13))
        
        # Generate all feature names and initialize with zeros
        for old_name in base_cols.values():
            # Add lagged features
            for lag in lags:
                feature_data[f'{old_name}_lag{lag}'] = 0
            
            # Add double-lagged features and their percentage changes
            for lag1 in lags:
                for lag2 in lags:
                    feature_data[f'{old_name}_lag{lag1}_lag{lag2}'] = 0
                    feature_data[f'{old_name}_lag{lag1}_lag{lag2}_pct_1m'] = 0
            
            # Add percentage change features
            for lag in [1, 3, 6, 12]:
                feature_data[f'{old_name}_pct_{lag}m'] = 0
            
            # Add percentage change features for lagged values
            for lag in lags:
                feature_data[f'{old_name}_lag{lag}_pct_1m'] = 0
        
        # Add target variables (initialized to 0)
        target_features = ['MSCI_World_target', 'MSCI_World_target_lag1', 'MSCI_World_target_lag3', 'MSCI_World_target_lag6']
        for feature in target_features:
            feature_data[feature] = 0
            feature_data[f"{feature}_pct_1m"] = 0
        
        # Create DataFrame with all features at once
        features = pd.DataFrame([feature_data])
        
        # Ensure all features from the scaler are present and in the correct order
        missing_features = set(scaler_features) - set(features.columns)
        if missing_features:
            # Add missing features all at once
            features = pd.concat([features, pd.DataFrame(0, index=features.index, columns=list(missing_features))], axis=1)
        
        # Remove any extra features not in the scaler
        extra_features = set(features.columns) - set(scaler_features)
        if extra_features:
            features = features.drop(columns=list(extra_features))
        
        # Ensure features are in the same order as the scaler
        features = features[scaler_features]
        
        # Scale features
        X = SCALER.transform(features)
        X = pd.DataFrame(X, columns=features.columns)
        
        # Predict regime for single sample
        regime = KMEANS.predict(X)
        
        # Create dummy variables for regime
        regime_dummies = pd.DataFrame(0, index=X.index, columns=[f'regime_{i}' for i in range(3)])
        regime_dummies.iloc[0, regime[0]] = 1
        
        # Concatenate features and regime dummies
        X = pd.concat([X, regime_dummies], axis=1)
        
        # Ensure we have exactly 424 features
        if X.shape[1] != 424:
            # Remove regime features if they're causing the issue
            regime_cols = [f'regime_{i}' for i in range(3)]
            X = X.drop(columns=regime_cols)
        
        return X
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

@tool
def predict_market(tool_input: str = "") -> str:
    """Predict MSCI World index value for a given forecast horizon (e.g. '1M', '1Y')."""
    try:
        # Extract horizon from input, handling potential comments
        horizon = tool_input.split('#')[0].strip().strip("'").strip('"')
        if not horizon:
            return "Please provide a forecast horizon (e.g. '1M', '3M', '1Y', '2Y', '4Y')."
        
        if horizon not in MODEL_PATHS:
            return f"Horizon {horizon} not supported. Available horizons: {', '.join(MODEL_PATHS.keys())}"
        
        X = preprocess.invoke("")
        model = MODELS[horizon]
        prediction = model.predict(X)[0]
        rmse = RMSE_BY_MODEL.get(horizon, 50)
        return f"In {horizon}, MSCI World is projected to be {prediction:.0f} ± {rmse:.0f} points."
    except Exception as e:
        if "Error in preprocessing" in str(e):
            return str(e)
        return f"An error occurred while making the prediction: {str(e)}"

@tool
def current_market_regime(tool_input: str = "") -> str:
    """Determine the current market regime cluster based on latest macro data."""
    X = preprocess.invoke("")
    regime = KMEANS.predict(X)[0]
    return f"Current market regime is: Regime {regime}"

@tool
def explain_prediction(tool_input: str = "") -> str:
    """Explain the most influential macro features driving the forecast."""
    horizon = tool_input.strip("'").strip('"')  # Remove quotes if present
    model = MODELS.get(horizon)
    if not model:
        return f"Horizon '{horizon}' not supported."
    lasso = model.named_steps['lasso']
    scaler = model.named_steps['scaler']
    features = scaler.get_feature_names_out()
    coefs = lasso.coef_
    ranked = sorted(zip(features, coefs), key=lambda x: abs(x[1]), reverse=True)[:5]
    lines = []
    for name, weight in ranked:
        direction = "positively" if weight > 0 else "negatively"
        lines.append(f"{name} influenced the forecast {direction}.")
    return "Top 5 factors: " + " ".join(lines)

@tool
def compare_to_past(tool_input: str = "") -> str:
    """Compare current market conditions with a past period."""
    try:
        # Get latest macro data
        current_data = get_latest_macro_data.invoke("")
        
        # Load historical data
        historical_df = pd.read_excel(EXCEL_FALLBACK_PATH, index_col=0)
        
        # Find closest date to 2008 financial crisis (September 15, 2008)
        crisis_date = pd.to_datetime('2008-09-15')
        historical_df.index = pd.to_datetime(historical_df.index)
        crisis_data = historical_df.loc[historical_df.index.asof(crisis_date)]
        
        # Format comparison
        comparison = "Comparison with 2008 Financial Crisis:\n"
        comparison += "=====================================\n\n"
        
        # Compare key indicators
        indicators = {
            'fed_funds': ('Federal Funds Rate', '%'),
            'gdp': ('GDP', 'billion USD'),
            'inflation_yoy': ('Inflation', '%'),
            'unemployment': ('Unemployment', '%'),
            'msci_world': ('MSCI World Index', 'points'),
            'vix': ('VIX', '')
        }
        
        for col, (label, unit) in indicators.items():
            try:
                current_val = current_data[col].iloc[0]
                past_val = crisis_data[col]
                comparison += f"{label}:\n"
                comparison += f"  Current: {current_val:.2f}{unit}\n"
                comparison += f"  2008:    {past_val:.2f}{unit}\n\n"
            except:
                continue
        
        return comparison
    except Exception as e:
        return f"An error occurred while comparing to past data: {str(e)}"

@tool
def plot_forecast(tool_input: str = "") -> str:
    """Generate a line plot of MSCI World forecasts across all horizons."""
    X = preprocess.invoke("")
    values = {h: MODELS[h].predict(X)[0] for h in MODELS}
    plt.figure(figsize=(8, 4))
    plt.plot(list(values.keys()), list(values.values()), marker="o")
    plt.title("MSCI World Forecast by Horizon")
    plt.xlabel("Horizon")
    plt.ylabel("Index Level")
    plt.grid(True)
    path = "forecast_plot.png"
    plt.savefig(path)
    return os.path.abspath(path)

# --------------------
# Agent Setup
# --------------------

prompt = PromptTemplate.from_template("""You are a forecasting assistant specialized in macroeconomic data and MSCI World index prediction.

{tool_names}

{tools}

You must always think step-by-step and use this format:

Thought: Explain your reasoning.
Action: tool_name
Action Input: the input to that tool

When you have the final answer, write:
Final Answer: your complete response to the user.

---

Previous conversation:
{chat_history}

User question: {input}

{agent_scratchpad}
""")


# Define the tools
tools = [
    get_latest_macro_data,
    predict_market,
    current_market_regime,
    explain_prediction,
    compare_to_past,
    plot_forecast,
    preprocess,
]

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
    request_timeout=120,
    max_retries=5
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=10000)

# Create the agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,  # Reduce max iterations for faster responses
    return_intermediate_steps=True,
    early_stopping_method="force", # force generate if stuck
    timeout=120
)
# --------------------
# Graph State (Optional if standalone)
# --------------------

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

# --------------------
# Entrypoint for LangGraph
# --------------------

async def run_forecasting_agent(state: dict) -> dict:
    """Run the forecasting agent on the current state."""
    last_message = state["messages"][-1]["content"]
    chat_history = state.get("chat_history", [])
    fallback_needed = False
    
    try:
        # Step 1: Use agent to try answering
        response = await agent_executor.ainvoke({
            "input": last_message,
            "chat_history": chat_history
        })
        output = response.get("output", "")
        
    except Exception as e:
        output = f"An error occurred while trying to answer the question: {str(e)}"
        fallback_needed = True

    return {
        **state,
        "forecast_output": output,
        "used_fallback": fallback_needed,
        "last_ran_agent": "forecasting",
        "messages": state["messages"] + [{"role": "assistant", "content": output}]
    }

# --------------------
# Exports
# --------------------

__all__ = ['run_forecasting_agent', 'agent_executor', 'memory']