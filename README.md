# ACSE-Supermarket-Recommender-System-Baby-Products-
This project builds a targeted recommender to increase Huggies penetration among shoppers who currently buy competitor baby brands at a U.S. grocer.
Using 393,416 baby-category transactions (2017–2020), we engineer customer-level RFM, brand diversity, and loyalty flags, then compare three approaches—Item-Based Collaborative Filtering (cosine), Association Rules (Apriori, brand level), and SVD (matrix factorization)—on a forward-looking, chronological split (Train: Jan-2017–Jul-2019; Test: Aug-2019–Dec-2020).

Why it matters: the system operationalizes promotion targeting (discount tiers by recommendation confidence) to convert competitor-brand shoppers, backed by interpretable market-basket insights (e.g., wipes as “gateway” SKUs).

Headline results (offline, Top-K on Test):
	•	Item-CF chosen for deployment framing: AUC ≈ 0.70, Precision@5 ≈ 6.4%, HitRate@5 ≈ 18%, best business-relevant relevance among models.
	•	Apriori: highly interpretable but sparse coverage (near-zero Top-K recall).
	•	SVD: competitive but less explainable for marketers than Item-CF.

What’s here
	•	Data Understanding Sampled 2.ipynb – store-aware sampling, cleaning, product normalization.
	•	RS_Final_A2_update.ipynb – feature engineering, Item-CF/Apriori/SVD, Top-K metrics (P@K, R@K, F1@K, NDCG@K, HitRate@K), ROC-AUC, and a simple promo-lift simulation.
	•	Executive reports summarizing EDA, modeling choices, results, and deployment plan.

Stack
Python (pandas, NumPy), scikit-learn (metrics), mlxtend (Apriori), matplotlib; reproducible, notebook-driven workflow.

TL;DR
A production-lean, explainable Item-CF recommender that targets non-Huggies baby shoppers with optimized promotions, validated on a time-split test to mimic real deployment.
