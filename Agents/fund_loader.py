import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict
from pathlib import Path
from datetime import datetime

# Path where FI XML files are stored
DATA_ROOT = Path("knowledge_base/Funddata")

# Optional: industry code mapping (can be extended)
SECTOR_CODES = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}

# Optional: English translation mapping
COLUMN_TRANSLATIONS = {
    "Instrumentnamn": "Instrument Name",
    "ISIN-kod_instrument": "ISIN",
    "Landkod_Emittent": "Country Code",
    "Valuta": "Currency",
    "Antal": "Quantity",
    "Nominellt_belopp": "Nominal Value",
    "Kurs_som_använts_vid_värdering_av_instrumentet": "Valuation Price",
    "Valutakurs_instrument": "FX Rate",
    "Marknadsvärde_instrument": "Market Value",
    "Andel_av_fondförmögenhet_instrument": "% of Fund AUM",
    "Branschkod_instrument": "Industry Code",
    "Bransch_namn_instrument": "Industry Name",
    "Tillgångsslag_enligt_LVF_5_kap": "Asset Class",
    "report_date": "Report Date"
}

def parse_date(date_str: str) -> datetime:
    """Parse date string in format YYYY-MM-DD to datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def parse_fund_xml(file_path: str) -> pd.DataFrame:
    NS = {'fi': 'http://schemas.fi.se/publika/vardepappersfonder/20200331'}
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract fund-level metadata
    fund_info = {
        "fund_name": root.findtext(".//fi:Fond_namn", default="", namespaces=NS),
        "fund_company": root.findtext(".//fi:Fondbolag_namn", default="", namespaces=NS),
        "fund_aum": root.findtext(".//fi:Fondförmögenhet", default="", namespaces=NS),
        "fund_institute_number": root.findtext(".//fi:Fond_institutnummer", default="", namespaces=NS),
        "fund_lei": root.findtext(".//fi:Fondbolag_LEI-kod", default="", namespaces=NS),
        "fund_isin": root.findtext(".//fi:Fond_ISIN-kod", default="", namespaces=NS),
        "report_date": root.findtext(".//fi:Kvartalsslut", default="", namespaces=NS),
        "cash": root.findtext(".//fi:Likvida_medel", default="", namespaces=NS),
        "other_assets_liabilities": root.findtext(".//fi:Övriga_tillgångar_och_skulder", default="", namespaces=NS),
        "active_risk": root.findtext(".//fi:Aktiv_risk", default="", namespaces=NS),
        "stddev_24m": root.findtext(".//fi:Standardavvikelse_24_månader", default="", namespaces=NS),
        "benchmark": root.findtext(".//fi:Jämförelseindex/fi:Jämförelseindex", default="", namespaces=NS)
    }

    try:
        fund_info["fund_aum"] = float(fund_info["fund_aum"].replace(" ", "").replace(",", "."))
    except:
        fund_info["fund_aum"] = None

    # Parse the report date
    fund_info["report_date"] = parse_date(fund_info["report_date"])

    instruments = []
    instrument_nodes = root.findall(".//fi:FinansielltInstrument", namespaces=NS)
    for inst in instrument_nodes:
        inst_data = fund_info.copy()
        for child in inst.iter():
            tag = child.tag.split("}")[-1]
            text = child.text.strip() if child.text else ""
            inst_data[tag] = text

        numeric_fields = [
            "Antal",
            "Kurs_som_använts_vid_värdering_av_instrumentet",
            "Valutakurs_instrument",
            "Marknadsvärde_instrument",
            "Andel_av_fondförmögenhet_instrument",
        ]
        for field in numeric_fields:
            if field in inst_data:
                try:
                    inst_data[field] = float(inst_data[field].replace(" ", "").replace(",", "."))
                except:
                    inst_data[field] = None

        sector_code = inst_data.get("Branschkod_instrument")
        inst_data["Bransch_namn"] = SECTOR_CODES.get(sector_code, "Unknown")

        instruments.append(inst_data)

    df = pd.DataFrame(instruments)
    if not df.empty:
        df.rename(columns=COLUMN_TRANSLATIONS, inplace=True, errors="ignore")
    return df

def load_all_fund_positions(data_dir: Path = DATA_ROOT) -> pd.DataFrame:
    xml_files = glob.glob(str(data_dir / "**/*.xml"), recursive=True)
    all_data = []
    for file_path in xml_files:
        try:
            df = parse_fund_xml(file_path)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

if __name__ == "__main__":
    df = load_all_fund_positions()
    print(df.head())
    print(f"Loaded {len(df)} holdings from {df['fund_name'].nunique()} funds.")
    print("\nDate range of reports:")
    print(f"Earliest report: {df['Report Date'].min()}")
    print(f"Latest report: {df['Report Date'].max()}")
