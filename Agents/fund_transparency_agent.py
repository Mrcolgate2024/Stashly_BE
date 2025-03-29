from langchain_core.messages import AIMessage
from tools.fund_xml_parser import extract_exposure_from_xml
from config.settings import Settings
import pandas as pd

def run_fund_transparency_agent(state, settings: Settings):
    try:
        user_msg = state.get("messages", [])[-1].content.lower()
        # Extract company name from prompt (e.g. "exposure to Apple")
        if "exposure to" in user_msg:
            target = user_msg.split("exposure to")[1].strip().split()[0]
        else:
            return {"messages": [AIMessage(content="Please specify a company (e.g. 'What is my exposure to Apple?')")]}

        results = extract_exposure_from_xml(settings.exposure_xml_dir, target)
        if not results:
            return {"messages": [AIMessage(content=f"No exposure found for '{target}'")]}

        df = pd.DataFrame(results)
        df = df[["fund", "isin", "company", "exposure_pct", "source"]].sort_values(by="exposure_pct", ascending=False)

        markdown = df.to_markdown(index=False)
        return {"messages": [AIMessage(content=f"Here is your exposure to **{target.title()}**:\n\n{markdown}")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error in fund transparency agent: {str(e)}")]}
