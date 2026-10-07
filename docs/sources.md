# RAG source documents

The Nutrition Plan chat retrieves from 14 clinical guideline and journal PDFs. They are copyrighted (or licensed for non-commercial distribution only), so they are **not** included in this repository.

To reproduce the index:

1. Download each document below from its publisher into a local `pdfs/` folder at the repo root (git-ignored). The original filenames are listed so the indexed `source` metadata matches; any filename works.
2. Start Milvus and the Streamlit app (see the root README).
3. In the **Index PDFs** tab, upload the files from `pdfs/`, assign a paper type, and click **Start Indexing**.

Titles below were read from the PDF metadata or first page. DOIs are given only where they are printed in the PDF; otherwise search the title on the publisher site listed.

| # | Original filename | Document | Where to get it |
|---|---|---|---|
| 1 | `1_obsesity_management.pdf` | Tsigos C, et al. *Management of Obesity in Adults: European Clinical Practice Guidelines.* Obesity Facts 2008;1:106-116 | https://doi.org/10.1159/000126822 |
| 2 | `2_obesity_management.pdf` | World Health Organization. *Health service delivery framework for prevention and management of obesity* (2023) | WHO publications (iris.who.int), search the title |
| 3 | `3_joslin-clinical-nutrition-guideline-for-overweight and-obese-adults.pdf` | Joslin Diabetes Center. *Clinical Nutrition Guideline for Overweight and Obese Adults with Type 2 Diabetes, Prediabetes or Those at High Risk* (Chapter 2) | joslin.org, clinical guidelines page |
| 4 | `4_medical-nutritional-therapy-for-the-patient-with-diabetes.pdf` | Gray A. *Nutritional Recommendations for Individuals with Diabetes* (updated May 2015). Endotext | NCBI Bookshelf (Endotext), search the title |
| 5 | `5_2015-OMTF-European-Guidelines-for-Obesity-Management.pdf` | Yumuk V, et al. *European Guidelines for Obesity Management in Adults.* Obesity Facts 2015;8:402-424 (CC BY-NC 3.0) | https://doi.org/10.1159/000442721 |
| 6 | `6_American-Association-of-Clinical-Endocrinology-Con.pdf` | *American Association of Clinical Endocrinology Consensus Statement: Algorithm for the Evaluation and Treatment of Adults with Obesity/Adiposity-Based Chronic Disease, 2025 Update.* Endocrine Practice | Endocrine Practice (endocrinepractice.org), search the title |
| 7 | `7_Different-Glucose-Variability-in-Women-With-Polycy.pdf` | *Different Glucose Variability in Women With Polycystic Ovary Syndrome With and Without Insulin Resistance: A Pilot Study.* Endocrine Practice | Endocrine Practice, search the title |
| 8 | `8_Editorial-Board_eprac.pdf` | Endocrine Practice, editorial board page | Endocrine Practice issue front matter |
| 9 | `9_Hypocalcemia-Post-Total-Thyroidectomy--A-Ten-Year,.pdf` | *Hypocalcemia Post Total Thyroidectomy: A Ten-Year, Single Institution Experience With a Parathyroid Hormone-Guided Calcium and Calcitriol Supplementation Protocol.* Endocrine Practice | Endocrine Practice, search the title |
| 10 | `10_Impact-of-Vitamin-D-and-Calcium-on-Falls-and-Fract.pdf` | *Impact of Vitamin D and Calcium on Falls and Fractures in Older Adults.* Endocrine Practice | Endocrine Practice, search the title |
| 11 | `11_Progestogen-Experience-Among-Transgender-Women-and.pdf` | *Progestogen Experience Among Transgender Women and Gender Diverse Adults Assigned Male at Birth in the United States.* Endocrine Practice | Endocrine Practice, search the title |
| 12 | `12_Table-of-contents_eprac.pdf` | Endocrine Practice, table of contents page | Endocrine Practice issue front matter |
| 13 | `13_obesity_management.pdf` | Wharton S, Lau DCW, Vallis M, Sharma AM, et al. *Obesity in adults: a clinical practice guideline.* CMAJ (2020) | CMAJ (cmaj.ca), search the title |
| 14 | `14_15_OBE-CPG_2025-Guideline_final_20251105.pdf` | U.S. Department of Veterans Affairs / Department of Defense. *VA/DoD Clinical Practice Guideline for the Management of Adult Overweight and Obesity* (2025) | healthquality.va.gov, obesity guideline page |

Items 6 to 12 appear to come from the same Endocrine Practice issue that carried the 2025 AACE obesity algorithm; items 8 and 12 are front matter and contribute little to retrieval.
