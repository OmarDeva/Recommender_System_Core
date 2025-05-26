# Recommender_System_Core
Core functions for comparing Model A and Model B in the paper “Fairness Through Normative Ethics: A Rule Utilitarian Perspective on Recommender Systems.” Includes image-based similarity (CNN), knowledge based filtering, collaborative filtering, and differential privacy controls.


## 💡 Key Files

| File                     | Description                                                                                       |
|--------------------------|---------------------------------------------------------------------------------------------------|
| `core_recommender_app.py` | Code snippets illustrating core recommendation logic: CNN-based similarity, knowledge based filtering, collaborative filtering, differential privacy control, and user filtering parameters (age, sex, ε). Not a complete Streamlit app. |
| `Collaborative_Filtering_train.py`             | Script for training the collaborative filtering model (SVD) based on both explicit and implicit signals. |
| `feature_extraction.py`   | Script for extracting visual features from product images using ResNet50 (for similarity calculations). |
