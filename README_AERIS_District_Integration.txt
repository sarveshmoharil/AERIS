AERIS DISTRICT-GRID INTEGRATION
================================

INSTALL
Extract this ZIP into E:\TrialShit and allow it to overwrite:
  aeris\ui\scientist_workbench.py

Adds:
  aeris\prediction\district_crop_rotation_engine.py
  run_aeris_district.py

RUN
PowerShell:
  cd E:\TrialShit
  .\.venv\Scripts\python.exe .\run_aeris_district.py

The runner reuses an existing RIE-01 forecast. If it is missing, it runs
RIE-01 only when the NASA POWER daily input exists. It then runs DCR-01
and opens the Scientist Workbench.

OUTPUTS in aeris\data\processed\district_grid:
  crop_suitability_district.csv
  rotation_candidates_district.csv
  ga_history_district.json
  district_crop_rotation_report.json

SCIENTIFIC BOUNDARIES
- Each of the 15 district management cells is evaluated independently.
- 16 crops from the existing knowledge base are screened.
- Annual crops are evaluated for Kharif, Rabi and Summer; perennial fruit
  crops are evaluated separately.
- Suitability is a heuristic environmental screening score, NOT yield
  or profit prediction.
- Cell-specific inputs are surface pH, sand/clay texture proxies and slope.
  Temperature and SMAP root-zone moisture are not cell-specific in the
  supplied district profile.
- Rainfall outlook, when present, is one Nashik climate point shared by
  all cells; its crop-demand adjustment is heuristic.
- SoilGrids raw scale factors/units require source metadata verification.
  Existing AERIS pH/sand/clay conversion conventions are used.
- Cells explicitly marked INSUFFICIENT_RASTER_EVIDENCE are blocked from
  generating rotations. PARTIAL cells may generate provisional candidates;
  approval prompts disclose evidence status and coverage.
- GA history is recorded per cell. The ranked shortlist is exhaustive
  within each season's top candidate pools.
- Legacy SRO-01 G101-G402 outputs remain separate and are not mixed with
  district-grid results.

No new package installation is required if the existing AERIS environment
already has numpy, pandas, matplotlib, shapely and pyproj.
