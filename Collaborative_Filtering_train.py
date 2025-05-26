        #Collaborative Filtering and Training

from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

          reader = Reader(rating_scale=(merged_df['combined_score'].min(), merged_df['combined_score'].max()))
          surprise_data = Dataset.load_from_df(merged_df[['user_id', 'ProductId', 'combined_score']], reader)

          trainset = surprise_data.build_full_trainset()
          model = SVD()
          model.fit(trainset)

          cross_validate(model, surprise_data, cv=5, verbose=True)


          merged_df['combined_score'] = (
          0.55 * merged_df['rating'].fillna(0) +
          0.1 * merged_df['click_count'].fillna(0) +
          0.1 * merged_df['view_count'].fillna(0) +
          0.05 * merged_df['time_spent'].fillna(0) +
          0.2 * merged_df['product_purchase_count'].fillna(0)
)