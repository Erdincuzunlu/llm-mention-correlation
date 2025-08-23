# llm-mention-correlation
Correlation study on ChatGPT brand mentions &amp; Wikipedia presence


# LLM Mention Correlation Study

This project is a small demonstration of how to analyze factors that may influence whether ChatGPT mentions certain brands.  
Specifically, we tested whether having a Wikipedia page correlates with being mentioned in ChatGPT responses.

## Dataset
- **brands.csv**: Contains a list of sample brands and categories (e.g., laptops, earbuds).
- Each brand is paired with a generated prompt (e.g., *"Is Apple a good laptop brand?"*).

## Steps in Analysis
1. **Prompt Generation**  
   Created simple prompts for each brand and category.  
   Example: `"Is Apple a good laptop brand?"`.

2. **ChatGPT Responses (Simulated)**  
   Collected sample responses (in this demo, some were manually filled).  
   - `Mentioned = 1` if the brand name appears in the response.  
   - `Mentioned = 0` otherwise.

3. **Wikipedia Check**  
   Used the `wikipedia` Python package to check if each brand has a Wikipedia page.  
   - `HasWiki = 1` if a page was found.  
   - `HasWiki = 0` otherwise.

4. **Correlation Analysis**  
   Built a contingency table (Mentioned × HasWiki).  
   Applied Chi-square test and computed the **Phi coefficient** (effect size).  

## Example Output
— Contingency Table (HasWiki x Mentioned) —
Mentioned  0  1
HasWiki
1          8  3

Chi-square: 0.0000 | p-value: 1.000 | dof: 0
Phi coefficient (effect size): 0.0000


**Interpretation**:  
In this small demo dataset, no statistically significant association was detected between having a Wikipedia page and being mentioned by ChatGPT.

## Requirements
- Python 3.9+
- pandas
- numpy
- scipy
- wikipedia

Install dependencies:
```bash
pip install pandas numpy scipy wikipedia

How to Run

python Example.py
