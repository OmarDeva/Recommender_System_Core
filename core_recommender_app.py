# UI Controls
st.sidebar.header("User Settings")
user_id = st.sidebar.text_input("User ID")
user_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)
user_sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
use_age_rec = st.sidebar.checkbox("Use Age-Based Recommendations", value=True)
use_sex_rec = st.sidebar.checkbox("Use Sex-Based Recommendations", value=True)
privacy_level = st.sidebar.slider("Privacy Level", 1, 10, 5)


disable_age_rec = not use_age_rec
disable_sex_rec = not use_sex_rec
epsilon = np.interp(privacy_level, [1, 10], [10, 0.5])

# Differential Privacy Function
def apply_dp_to_cf_scores(cf_scores, epsilon):
    noise = np.random.laplace(loc=0, scale=1/epsilon, size=cf_scores.shape)
    return cf_scores + noise

# CNN + CF Recommendation Function From Kaggle Data cited in the manuscript "https://www.kaggle.com/code/vikashrajluhaniwal/building-visual-similarity-based-recommendation"
def get_similar_products_cnn(product_id, num_results, fashion_df, user_sex, user_age, user_id,
                             disable_age_rec=False, disable_sex_rec=False, epsilon=1.0):

    # Generate a unique cache key
    cache_key = f"similar_products_{product_id}_{user_sex}_{user_age}_{disable_age_rec}_{epsilon}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        # Download ResNet features
        if disable_sex_rec:
            gender_filter = "All"
        else:
            gender_filter = "Men" if user_sex == "Male" else "Women"

        if gender_filter == "Men":
            features_data = download_blob_to_memory_cached('GCP_bucket', 'Example:Men_ResNet50_features.npy')
            product_ids_data = download_blob_to_memory_cached('GCP_bucket', 'Example:Men_ResNet50_feature_product_ids.npy')
        elif gender_filter == "Women":
            features_data = download_blob_to_memory_cached('GCP_bucket', 'Example:Women_ResNet50_features.npy')
            product_ids_data = download_blob_to_memory_cached('GCP_bucket', 'Example:Women_ResNet50_feature_product_ids.npy')
        else:
            men_features_data = download_blob_to_memory_cached('GCP_bucket', 'Example:Men_ResNet50_features.npy')
            men_product_ids_data = download_blob_to_memory_cached('GCP_bucket', 'Example:Men_ResNet50_feature_product_ids.npy')
            women_features_data = download_blob_to_memory_cached('GCP_bucket', 'Example:Women_ResNet50_features.npy')
            women_product_ids_data = download_blob_to_memory_cached('GCP_bucket', 'Example:Women_ResNet50_feature_product_ids.npy')

            men_features = np.load(BytesIO(men_features_data))
            women_features = np.load(BytesIO(women_features_data))
            men_product_ids = [str(pid) for pid in np.load(BytesIO(men_product_ids_data))]
            women_product_ids = [str(pid) for pid in np.load(BytesIO(women_product_ids_data))]

            extracted_features = np.vstack((men_features, women_features))
            Productids = men_product_ids + women_product_ids
        if gender_filter in ["Men", "Women"]:
            extracted_features = np.load(BytesIO(features_data))
            Productids = [str(pid) for pid in np.load(BytesIO(product_ids_data))]

        extracted_features = normalize(extracted_features)
        doc_id = Productids.index(product_id)
        query_feature = extracted_features[doc_id].reshape(1, -1)
        pairwise_dist = pairwise_distances(extracted_features, query_feature).flatten()

        product_distances = pd.DataFrame({'ProductId': Productids, 'Distance': pairwise_dist})
        product_distances = product_distances[product_distances['ProductId'] != product_id]
        product_details = product_distances.merge(fashion_df, on='ProductId', how='left')

        input_product_row = fashion_df[fashion_df['ProductId'] == product_id]
        if input_product_row.empty:
            return pd.DataFrame()
        input_product_type = input_product_row['ProductType'].values[0]
        product_details = product_details[product_details['ProductType'] == input_product_type]

        # Age-based color filtering conceptualized from literature as demonstrated in the manuscript
        if not disable_age_rec:
            age_ranges = {
                'Men': {
                    'Navy Blue': (18, 44), 'Grey': (45, 90), 'Brown': (45, 90),
                    'Black': (18, 55), 'Olive': (25, 55), 'Blue': (18, 44),
                    'White': (18, 44), 'Green': (45, 90), 'Red': (45, 90),
                    'Yellow': (18, 34),
                },
                'Women': {
                    'Pink': (18, 25), 'Turquoise Blue': (18, 44), 'Magenta': (18, 44),
                    'Purple': (18, 44), 'Lavender': (25, 55), 'Peach': (18, 34),
                    'Maroon': (45, 90), 'Grey': (45, 90), 'Brown': (45, 90),
                    'Black': (20, 50), 'White': (18, 30), 'Red': (45, 90),
                }
            }
            if gender_filter == "All":
                preferred_colors = []
                for g in ['Men', 'Women']:
                    preferred_colors.extend([c for c, (min_a, max_a) in age_ranges[g].items() if min_a <= user_age <= max_a])
            else:
                preferred_colors = [c for c, (min_a, max_a) in age_ranges[gender_filter].items() if min_a <= user_age <= max_a]

            if preferred_colors:
                product_details = product_details[product_details['Colour'].isin(preferred_colors)]

        if product_details.empty:
            return pd.DataFrame()

        product_details = product_details.sort_values('Distance').head(num_results)

 
        # Collaborative Filtering Re-Ranking

        if cf_model:
            cf_predictions = []
            for pid in product_details['ProductId'].unique():
                pred = cf_model.predict(str(user_id), str(pid))
                cf_predictions.append((pid, pred.est))

            cf_df = pd.DataFrame(cf_predictions, columns=['ProductId', 'cf_score'])
            cf_df['cf_score'] = apply_dp_to_cf_scores(cf_df['cf_score'], epsilon)
            product_details = product_details.merge(cf_df, on='ProductId', how='left')
            mean_score = product_details['cf_score'].mean()
            product_details['cf_score'] = product_details['cf_score'].fillna(mean_score)
            product_details = product_details.sort_values(by='cf_score', ascending=False)

        st.session_state[cache_key] = product_details
        return product_details

    except Exception as e:
        st.error(f"Error retrieving recommendations: {e}")
        return pd.DataFrame()