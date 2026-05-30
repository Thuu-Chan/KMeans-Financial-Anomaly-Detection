# Financial Anomaly Detection in Vietnamese FMCG Firms Using K-Means

## Overview

This project investigates financial reporting anomalies among Vietnamese FMCG firms using an unsupervised machine learning approach.

Rather than directly predicting fraud, the study focuses on identifying firms whose financial characteristics deviate from industry norms. To achieve this, a K-Means clustering framework is integrated with an expert-guided anomaly scoring model, where risk weights are informed by prior financial reporting fraud research, accounting theory, tax-related incentives, economic cycle considerations, and the evolving financial reporting environment in Vietnam. The objective is to demonstrate how machine learning can be enhanced by domain expertise to support early detection of financial reporting risks.

## Dataset

- 80 Vietnamese FMCG firms observed over a two-year period (2023–2024)
- Total observations: 160 firm-year records
- Financial statement data from publicly available sources

Variables used:

- ROA
- Revenue Growth
- Accruals
- Tax Ratio
- Receivable Ratio

## Methodology

The research framework consists of four main stages:

1. **Knowledge-Guided Feature Selection**  
   Variables were selected through a systematic literature review (SLR) following PRISMA principles and informed by prior financial reporting fraud studies.

2. **Data Collection and Preprocessing**  
   Financial statement data from 80 Vietnamese FMCG firms (160 firm-year observations) over 2023–2024 were collected and transformed using Robust Scaling to preserve economically meaningful outliers.

3. **Unsupervised Learning and Risk Assessment**  
   K-Means clustering and Silhouette Analysis were employed to identify latent risk structures. An expert-guided anomaly scoring framework was then applied to quantify financial reporting risk.

4. **Validation and Case Analysis**  
   Statistical tests (Kruskal–Wallis and ANOVA) and case-based analysis were conducted to evaluate the robustness and practical relevance of the findings.

## Results

The model classifies firms into three groups:

- Normal
- Watch
- High Risk

The results suggest that firms in the High Risk group exhibit significantly different financial characteristics compared to normal firms.

## Repository Structure

data/ – dataset

notebooks/ – source code

results/ – outputs and visualizations

paper/ – full research paper

## Author

### Đặng Trần Hà Thu

Project Lead & Main Researcher

- Research design
- Data collection and preprocessing
- Feature engineering
- Machine learning implementation
- Statistical analysis
- Visualization
- Research writing

## Acknowledgements

### Nguyễn Khánh Ngọc

Research Support

- Literature review assistance
- Citation management
- Manuscript formatting and editing
## Research Paper

See:

paper/Research_Report_Financial_Anomaly_Detection.pdf
