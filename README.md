# AERIS
### Agricultural Environmental & Rotation Intelligence System

AERIS is an agricultural intelligence system designed to make crop rotation more data-driven, geographically informed, and climate-aware.

The system combines geospatial datasets, soil characteristics, environmental parameters, and computational optimization to evaluate agricultural feasibility and generate suitable crop rotation sequences.

**Current geographic focus:** Nashik District, Maharashtra, India.

---

## Objectives

- Integrate heterogeneous agricultural and environmental datasets.
- Process geospatial raster data into structured, usable datasets.
- Evaluate agricultural suitability using multiple parameters.
- Generate and rank potential crop rotation sequences.
- Provide interpretable results to support agricultural decision-making.

## System Architecture

AERIS follows a data-processing and optimization workflow:

1. **Data acquisition** — Collect relevant soil, geographical, and environmental datasets.
2. **Geospatial processing** — Extract and process raster-based information using Python geospatial libraries.
3. **Data preparation** — Transform extracted information into structured CSV datasets for computational analysis.
4. **Feasibility evaluation** — Evaluate agricultural suitability using a hybrid feasibility engine.
5. **Rotation generation** — Generate candidate crop rotation sequences.
6. **Optimization** — Evaluate and rank generated rotations according to the system's fitness criteria.
7. **Results** — Present recommended rotations and associated suitability metrics.

## Agricultural Parameters

The system incorporates soil and environmental characteristics, including:

- Soil pH
- Cation Exchange Capacity (CEC)
- Soil texture and composition
- Climatic and environmental parameters
- Crop-specific suitability characteristics

Parameter availability and spatial coverage may vary between datasets.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and algorithms |
| NumPy | Numerical computation |
| Pandas | Data processing and CSV management |
| Rasterio | Geospatial raster processing |
| GDAL | Geospatial data handling |
| Shapely | Geometric operations |
| PyProj | Coordinate reference transformations |
| Scikit-learn | Machine learning and analytical utilities |
| Tkinter | Desktop application interface |

## Data Sources

The project uses geospatial and environmental datasets, including SoilGrids-derived soil information and relevant climatic datasets.

Raw raster data is processed and transformed into structured datasets before being used by the feasibility and optimization engines.

Please refer to individual dataset licences and attribution requirements before redistributing source data.

## Installation

Clone the repository:

```bash
git clone https://github.com/sarveshmoharil/AERIS.git
cd AERIS
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application using the project's main Python entry point.

## Results

AERIS produces crop rotation recommendations alongside quantitative evaluation metrics, including rotation fitness and mean suitability scores.

The system is designed to make its recommendations more interpretable by providing the reasoning and suitability information behind generated results.

## Project Scope and Limitations

- Current implementation is focused on Nashik District.
- Results depend on the availability, resolution, quality, and coverage of input datasets.
- Generated recommendations are computational assessments and should not be interpreted as guaranteed agricultural outcomes.
- Further validation using field-level observations and agricultural expertise is necessary.

## Demonstration

A demonstration video showcasing the AERIS application and workflow will be linked here.

**Video:** [Add demonstration video link]

## Future Development

- Expansion to additional geographical regions.
- Integration of higher-resolution agricultural datasets.
- Improved climate-aware crop suitability modelling.
- Further validation against real-world agricultural observations.
- Additional optimization and analytical capabilities.

## Licence

The source code is distributed under the MIT License, subject to the terms in the LICENSE file.

Third-party datasets and dependencies remain subject to their respective licences.

---

**AERIS — Towards data-driven and climate-aware agriculture.**
