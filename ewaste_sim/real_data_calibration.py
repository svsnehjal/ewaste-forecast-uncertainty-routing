"""
Semi-real calibration references for the e-waste routing simulation.

This module DOES NOT claim that regional annual totals are campus-level
collection-point observations. It records official external reference
statistics and builds explicitly scaled synthetic network scenarios.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json

DAYS_PER_YEAR = 365.0

# CPCB / SPCB inventory estimates reported in the NGT proceeding.
TAMIL_NADU_GENERATION_MT = {
    "FY2023-24": 448878.0,
    "FY2024-25": 466578.0,
}

# TNPCB Annual Report 2023-24: industry annual returns, collected/processed.
TNPCB_COLLECTED_PROCESSED_TONNES = {
    "IT_and_telecommunication": 20111.0,
    "consumer_electrical_electronics": 9897.0,
    "other_telecom_items_and_packaging": 2361.0,
    "materials_recovered": 24911.0,
}

# MoEFCC / PIB parliamentary reply: registered recycler processing totals.
REGISTERED_RECYCLER_PROCESSED_MT = {
    "FY2023-24": 23203.636,
    "FY2024-25": 16360.815,
}

SOURCES = {
    "CPCB_Tamil_Nadu_inventory": (
        "CPCB report filed in NGT proceedings; Tamil Nadu estimated "
        "generation: 448,878 MT (FY2023-24), 466,578 MT (FY2024-25). "
        "https://www.greentribunal.gov.in/sites/default/files/news_updates/"
        "Report%20on%20behalf%20of%20CPCB%20in%20EA%20NO%2004%20of%202024%20"
        "IN%20OA%20No.%20512%20of%202018%20%28Shailesh%20Singh%20Vs.%20"
        "State%20of%20U.P.%20%26%20Ors.%29.pdf"
    ),
    "TNPCB_annual_report_2023_24": (
        "TNPCB Annual Report 2023-24; annual returns submitted by industries; "
        "quantities collected/processed and materials recovered. "
        "https://tnpcb.gov.in/PDF/Information/annualreports/English/"
        "AnnualRpt_Eng2024.pdf"
    ),
    "MoEFCC_PIB_24_July_2025": (
        "MoEFCC parliamentary reply, 24 July 2025; registered recycler "
        "processing quantity for Tamil Nadu. "
        "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2147876&lang=2&reg=48"
    ),
}

@dataclass
class NetworkScenario:
    scenario_name: str
    state_annual_generation_mt: float
    network_share: float
    network_annual_kg: float
    network_daily_kg: float
    n_collection_points: int
    median_point_daily_kg: float
    lognormal_sigma: float
    mean_point_daily_kg: float
    interpretation: str

def build_network_scenario(
    scenario_name="small_network_illustrative",
    financial_year="FY2023-24",
    network_share=1e-7,
    n_collection_points=30,
    lognormal_sigma=0.5,
):
    """
    Convert a tiny, user-editable share of the statewide estimate into a
    network-scale synthetic demand target.

    network_share is an explicit modelling assumption, NOT a measured
    fraction of Tamil Nadu waste attributable to a real site.
    """
    if financial_year not in TAMIL_NADU_GENERATION_MT:
        raise ValueError(f"Unknown financial year: {financial_year}")
    if not (0 < network_share < 1):
        raise ValueError("network_share must be between 0 and 1.")
    if n_collection_points < 1:
        raise ValueError("n_collection_points must be positive.")

    state_annual_kg = TAMIL_NADU_GENERATION_MT[financial_year] * 1000.0
    network_annual_kg = state_annual_kg * network_share
    network_daily_kg = network_annual_kg / DAYS_PER_YEAR

    # For a lognormal distribution, mean = median * exp(sigma^2 / 2).
    mean_point_daily = network_daily_kg / n_collection_points
    median_point_daily = mean_point_daily / (2.718281828459045 ** (lognormal_sigma**2 / 2.0))

    return NetworkScenario(
        scenario_name=scenario_name,
        state_annual_generation_mt=TAMIL_NADU_GENERATION_MT[financial_year],
        network_share=network_share,
        network_annual_kg=network_annual_kg,
        network_daily_kg=network_daily_kg,
        n_collection_points=n_collection_points,
        median_point_daily_kg=median_point_daily,
        lognormal_sigma=lognormal_sigma,
        mean_point_daily_kg=mean_point_daily,
        interpretation=(
            "Synthetic network-scale scenario derived using an explicit "
            "share of the statewide inventory estimate; not an observed "
            "campus, municipality, recycler, or collection-point dataset."
        ),
    )

def export_calibration(output_dir="results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Multiple scales enable sensitivity analysis; no single share is
    # presented as empirically measured.
    shares = [1e-6, 1e-5, 1e-4]
    scenarios = [
        asdict(build_network_scenario(
            scenario_name=f"share_{share:g}",
            network_share=share,
        ))
        for share in shares
    ]

    with (out / "external_reference_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["source_dataset", "financial_year", "indicator", "value", "unit", "interpretation"])
        for fy, value in TAMIL_NADU_GENERATION_MT.items():
            writer.writerow(["CPCB_Tamil_Nadu_inventory", fy, "estimated_generation", value, "MT/year",
                             "Statewide inventory estimate; not network-level collection data"])
        for category, value in TNPCB_COLLECTED_PROCESSED_TONNES.items():
            indicator = "materials_recovered" if category == "materials_recovered" else "collected_processed"
            writer.writerow(["TNPCB_Annual_Report_2023_24", "FY2023-24", f"{indicator}:{category}", value, "tonnes",
                             "Industry annual returns; do not interpret as total generation"])
        for fy, value in REGISTERED_RECYCLER_PROCESSED_MT.items():
            writer.writerow(["MoEFCC_PIB_registered_recyclers", fy, "registered_recycler_processed", value, "MT",
                             "Registered recycler processing quantity; separate reporting measure"])

    with (out / "calibrated_network_scenarios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(scenarios[0].keys()))
        writer.writeheader()
        writer.writerows(scenarios)

    metadata = {
        "sources": SOURCES,
        "method": "Statewide annual estimate multiplied by explicit illustrative network-share assumptions.",
        "caveat": (
            "No daily or location-level public data were available in these sources. "
            "The resulting network demand scenarios are synthetic and should not be "
            "described as measured local collection data."
        ),
        "shares_tested": shares,
    }
    (out / "calibration_sources_and_method.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return scenarios

if __name__ == "__main__":
    scenarios = export_calibration()
    for item in scenarios:
        print(item)
