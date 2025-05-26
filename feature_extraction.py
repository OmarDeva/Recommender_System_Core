#This code snippet is taken from Kaggle "https://www.kaggle.com/code/vikashrajluhaniwal/building-visual-similarity-based-recommendation"
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

def extract_features(image_dir, output_prefix, num_samples):
    img_width, img_height = 224, 224
    batch_size = 1
    datagen = ImageDataGenerator(rescale=1. / 255)

    model = ResNet50(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        image_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    itemcodes = [filename.split("/")[1].split(".")[0] for filename in generator.filenames]

    features = model.predict(generator, num_samples // batch_size)
    features = features.reshape((num_samples, -1))

    np.save(f'./{output_prefix}_ResNet50_features.npy', features)
    np.save(f'./{output_prefix}_ResNet50_feature_product_ids.npy', np.array(itemcodes))

    print(f"{output_prefix} features extracted and saved.")

# Example usage:
if __name__ == "__main__":
    start = datetime.now()

    # Men
    extract_features(
        image_dir="/path/to/Men/Images",   # <- replace with your actual image path
        output_prefix="Men",
        num_samples=811
    )

    # Women
    extract_features(
        image_dir="/path/to/Women/Images",  # <- replace with your actual image path
        output_prefix="Women",
        num_samples=837  # Replace with actual count
    )

    print("Total time taken:", datetime.now() - start)
