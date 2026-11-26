# Data and attribution

Copy the course workbook to `data/raw/AmesHousing.xls` before training. The publisher also hosts the [original workbook](https://jse.amstat.org/v19n3/decock/AmesHousing.xls) and [data documentation](https://jse.amstat.org/v19n3/decock/DataDocumentation.txt). Cite Dean De Cock (2011), *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*, Journal of Statistics Education 19(3), [DOI](https://doi.org/10.1080/10691898.2011.11889627).

An explicit redistribution license for the supplied original workbook was not verified. The raw data, filtered record export and observation-level generated files are excluded from public Git. The app uses local records for comparable sales and the explorer. The software license covers project code.

| Feature | Source field | Unit or meaning |
|---|---|---|
| overall_quality | Overall Qual | Overall materials and finish, 1–10 |
| living_area | Gr Liv Area | Above-grade finished living area, ft² |
| basement_area | Total Bsmt SF | Basement area, ft² |
| garage_area | Garage Area | Garage area, ft² |
| year_built | Year Built | Original construction year |
| year_remodeled | Year Remod/Add | Remodel/addition year, otherwise original build convention |
| first_floor_area | 1st Flr SF | First-floor area, ft² |
| lot_area | Lot Area | Lot area, ft² |
| full_baths | Full Bath | Above-grade full bathrooms |
| total_rooms | TotRms AbvGrd | Above-grade rooms excluding bathrooms |
| neighborhood | Neighborhood | Ames neighborhood code |
| house_style | House Style | Dwelling style code |
| building_type | Bldg Type | Dwelling type code |
| sale_price | SalePrice | Target, nominal USD |
| parcel | PID | Parcel grouping key |
| sale_year / sale_month | Yr Sold / Mo Sold | Chronological split keys |
| original_row | Workbook order | One-based row lineage excluding header |

Results include two missing feature cells, no exact duplicate rows and no repeated parcel identifiers. Date and area inconsistencies are audited. Source value flags remain visible rather than being silently rewritten.

The supplied workbook SHA-256 is `1f6bc021c651b2c34c50f0e97829f638d2945cf7b11592fddd7c4defb95d90d2`. Place an authorized local copy of the course report at `data/raw/MAT326_Final_Report.pdf` to enable the download button in the research view.
