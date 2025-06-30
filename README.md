# St. Cloud Housing Shortage Analysis

**Understanding Housing Markets Through Data: A Deep Dive into Methodology and Policy Implications**

---

## What This Project Is About

This repository contains a comprehensive analysis of housing shortages in the St. Cloud, Minnesota metropolitan area. But it's really about something bigger: **how the methodological choices researchers make can completely change policy conclusions**.

We implemented the National Association of Home Builders (NAHB) vacancy rate methodology, exactly as described by the Brookings Institute, to answer a seemingly simple question: *How much does including the Sherburne County portion of St. Cloud change our housing shortage estimates?*

What we discovered was much more interesting than the answer to that question.

## The Big Discovery

**Including pandemic years in your "normal" baseline changes shortage estimates by 25-35%.** 

Think about that for a second. The same data, the same methodology, but different assumptions about what's "normal" can mean the difference between recommending 450 new housing units or 688 new housing units. That's a $60-120 million difference in housing policy.

## Key Findings

### 📊 Methodological Impact
- **NAHB Official Method** (includes 2020-2021): 688 unit shortage
- **Pre-Pandemic Method** (2013-2019 only): ~550 unit shortage  
- **Conservative Method** (2015-2019 only): ~450 unit shortage

### 🗺️ Geographic Insights
- **St. Cloud MSA**: 797 units (1.0% of housing stock)
- **St. Cloud City**: 688 units (2.4% of housing stock)
- **College town effect**: Cities have different vacancy patterns than surrounding counties

### 🔍 Research Transparency
- **184 individual datasets** collected and preserved
- **Complete API audit trail** for full replication
- **Multiple baseline approaches** tested and documented

## What Makes This Research Different

Most housing studies give you a single number—"X units shortage"—without telling you that this number could easily be 30% different depending on methodological choices that sound technical but have huge policy implications.

We preserved everything:
- Every raw API response (184 JSON files)
- Every processing step (CSV outputs)
- Every methodological decision (detailed logs)
- Every chart and visualization

**Anyone can replicate our exact analysis or modify our assumptions to see how results change.**

## Project Structure

```
├── raw_data/                    # 184 Census API responses (JSON format)
├── processed_data/              # Cleaned datasets (CSV format)  
├── methodology_logs/            # Complete research audit trail
├── st_cloud_analysis.py         # Main analysis script
├── housing_analysis_log_*.log   # Execution logs with timestamps
├── st_cloud_housing_analysis.png # Charts and visualizations
└── *.html                       # Interactive dashboards
```

## The Research Question

**Original question**: How does including the Sherburne County portion of St. Cloud affect housing shortage estimates?

**What we actually learned**: The methodology you choose matters way more than most people realize, and housing researchers need to be more transparent about these choices.

## Methodology: How We Did This

### Data Collection
- **Source**: U.S. Census Bureau American Community Survey (ACS) 5-Year Estimates
- **Years**: 2009-2022 (184 separate API calls)
- **Geography**: 5 different geographic definitions of "St. Cloud area"
- **Variables**: Complete housing tenure and vacancy status data

### The NAHB Approach
The logic is straightforward: compare current vacancy rates to historical "normal" rates. When vacancy is too low, you need more housing.

But here's the catch: **What's "normal"?**

We tested three different approaches:
1. **NAHB Official** (2009-2021): Includes pandemic years
2. **Zandi Style** (2013-2019): Stable growth period only  
3. **Academic Conservative** (2015-2019): Most recent stable period

### Why This Matters for Policy
If you're a city planner, the difference between these approaches could mean recommending 238 fewer housing units. That's not just an academic distinction—it's millions of dollars in development decisions.

## Results That Matter

| Geography | Housing Shortage | % of Stock | Key Insight |
|-----------|-----------------|------------|-------------|
| **St. Cloud MSA** | 797 units | 1.0% | Official metro definition |
| **St. Cloud City** | 688 units | 2.4% | College town effect visible |
| **Sherburne County** | 232 units | 0.7% | Adjacent area impact |

### The College Town Effect
St. Cloud city shows higher shortage rates than surrounding counties, likely because college towns have historically higher rental vacancy due to student housing turnover. This suggests **one-size-fits-all methodologies might miss important local market dynamics**.

## How to Use This Research

### For Researchers
```bash
# Install requirements
pip install pandas numpy requests matplotlib

# Get free Census API key
# https://api.census.gov/data/key_signup.html

# Run the analysis
python st_cloud_analysis.py
```

All raw data and methodology logs are preserved, so you can:
- Replicate our exact analysis
- Test different baseline periods
- Apply the methodology to other cities
- Modify geographic boundaries

### For Policy Makers
When you see housing shortage estimates, ask:
1. What baseline period defined "normal" vacancy rates?
2. Are seasonal/vacation homes excluded?
3. How were geographic boundaries chosen?
4. How much do the estimates change with different methodological choices?

### For Students
This project demonstrates:
- How to collect and validate large datasets systematically
- Why methodological transparency matters in policy research
- How to implement academic methodologies with complete documentation
- The importance of testing sensitivity to key assumptions

## What's Next: The Other Three Methods

The Brookings Institute identified four different approaches to measuring housing shortages. We've completed Method 1 (vacancy rates). Coming next:

### Method 2: Under-Building Analysis
Compare recent construction rates to historical norms. Are we building houses as fast as we used to?

### Method 3: Latent Demand Analysis  
Count people who want their own housing but can't afford it (living with relatives, overcrowded conditions).

### Method 4: Regulatory Constraint Analysis
Estimate how much more housing would exist without zoning and regulatory barriers.

Each method asks a different question and likely gives different answers. The comparison will reveal how much housing shortage estimates depend on which aspect of the market you focus on.

## The Bigger Picture

Housing policy affects everyone, but most people don't realize how much the "shortage" numbers they hear in the news depend on methodological choices that researchers rarely discuss openly.

Our research suggests that **methodology transparency should be standard practice in housing policy research**. The stakes are too high—and the methodological impacts too large—for these choices to remain buried in technical appendices.

## Technical Details

- **Programming Language**: Python 3.11+
- **Key Libraries**: pandas, numpy, requests, matplotlib
- **Data Source**: Census Bureau API (public, no authentication required)
- **Analysis Framework**: Vacancy rate methodology per NAHB/Brookings specification
- **Reproducibility**: Complete audit trail with timestamped logs

## Academic Context

This work implements methodologies described in:
- Wessel, David. "Where do the estimates of a 'housing shortage' come from?" *Brookings Institute*, 2024
- National Association of Home Builders housing shortage methodology
- Zandi, Mark. *Moody's Analytics* equilibrium vacancy rate approach

## Contact & Collaboration

This research is part of an ongoing economics project examining housing market measurement methodologies. 

**Interested in collaboration or have questions about methodology?** Feel free to open an issue or reach out.

---

*All data collection followed Census Bureau API terms of service. Raw data is preserved for transparency but respects data provider guidelines.*
