# Semi-real calibration notes

## What is externally reported
- CPCB/SPCB Tamil Nadu e-waste generation inventory:
  448,878 MT for FY 2023-24 and 466,578 MT for FY 2024-25.
- TNPCB Annual Report 2023-24, based on industry annual returns:
  IT/telecommunication collected or processed ~20,111 tonnes;
  consumer electrical/electronics ~9,897 tonnes;
  other types ~2,361 tonnes; materials recovered ~24,911 tonnes.
- MoEFCC/PIB parliamentary reply:
  Tamil Nadu registered-recycler processing ~23,203.636 MT in FY 2023-24
  and ~16,360.815 MT in FY 2024-25.

## What is not externally reported
These sources do not provide daily demand for individual collection points,
campus-level fill levels, vehicle routes, bin capacities, or local collection
frequencies. Those variables remain synthetic assumptions.

## Calibration method
`real_data_calibration.py` scales the Tamil Nadu annual generation inventory
by three explicitly assumed network shares (1e-8, 1e-7, 1e-6), then distributes
the resulting expected daily mass across 30 synthetic points using a lognormal
distribution. The point median is calculated so that the sum of expected
point means equals the selected network daily mass.

Do not describe any selected network share as observed or measured. Report
the sensitivity range, state the scale assumption, and retain the source
definitions separately. The TNPCB collected/processed values and registered
recycler processing values are operational reference indicators, not
interchangeable generation totals.
