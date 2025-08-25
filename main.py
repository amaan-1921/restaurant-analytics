from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import folium
from folium.plugins import HeatMap
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load and preprocess dataset
df = pd.read_csv("data/Dataset.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ============================================================================
# GLOBAL DATA PREPROCESSING (Run this FIRST before all tasks)
# ============================================================================

def global_data_preprocessing(df):
    """Global preprocessing that all tasks will use"""
    logger.info("Starting global data preprocessing")
    
    # Basic cleaning and filling missing values
    df_global = df.copy()
    
    # Fill missing values intelligently
    df_global['cuisines'] = df_global['cuisines'].fillna('Unknown')
    df_global['average_cost_for_two'] = df_global['average_cost_for_two'].fillna(
        df_global.groupby('city')['average_cost_for_two'].transform('median')
    )
    df_global['price_range'] = df_global['price_range'].fillna(df_global['price_range'].mode()[0])
    df_global['aggregate_rating'] = df_global['aggregate_rating'].fillna(
        df_global.groupby('city')['aggregate_rating'].transform('mean')
    )
    df_global['votes'] = df_global['votes'].fillna(0)
    df_global['has_online_delivery'] = df_global['has_online_delivery'].fillna('No')
    df_global['city'] = df_global['city'].fillna('Unknown')
    df_global['locality'] = df_global['locality'].fillna('Unknown')
    df_global['latitude'] = df_global['latitude'].fillna(0)
    df_global['longitude'] = df_global['longitude'].fillna(0)
    
    # Create primary_cuisine for ALL tasks to use
    df_global['primary_cuisine'] = df_global['cuisines'].apply(
        lambda x: x.split(',')[0].strip() if isinstance(x, str) and x.strip() and x.lower() != 'unknown' else 'Unknown'
    )
    
    # Group rare cuisines together
    cuisine_counts = df_global['primary_cuisine'].value_counts()
    rare_cuisines = cuisine_counts[cuisine_counts < 10].index
    df_global['primary_cuisine'] = df_global['primary_cuisine'].apply(
        lambda x: 'Other' if x in rare_cuisines else x
    )
    
    # Remove rows with missing target variable for all tasks
    df_global = df_global.dropna(subset=['aggregate_rating'])
    
    logger.info(f"Global preprocessing completed. Shape: {df_global.shape}")
    logger.info(f"Primary cuisines: {df_global['primary_cuisine'].value_counts().head(10).to_dict()}")
    
    return df_global

# Apply global preprocessing
df = global_data_preprocessing(df)

# ============================================================================
# TASK 1: IMPROVED RATING PREDICTION
# ============================================================================

def preprocess_cuisines_task1(df):
    """Additional cuisine preprocessing for Task 1"""
    df['cuisine_count'] = df['cuisines'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) and x.strip() and x.lower() != 'unknown' else 1
    )
    return df

def create_price_features(df):
    """Create more meaningful price-related features"""
    df['price_per_person'] = df['average_cost_for_two'] / 2
    
    # Price category based on quartiles
    df['price_category'] = pd.qcut(df['average_cost_for_two'], 
                                   q=4, 
                                   labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
    return df

def setup_improved_task1():
    """Improved Task 1 with better feature engineering"""
    global model_task1, preprocessor_task1
    
    logger.info("Starting improved Task 1 setup")
    
    # Use the globally preprocessed dataframe
    df_task1 = df.copy()
    
    # Additional Task 1 specific preprocessing
    df_task1 = preprocess_cuisines_task1(df_task1)
    df_task1 = create_price_features(df_task1)
    
    # Create interaction features
    df_task1['votes_per_rating'] = df_task1['votes'] / (df_task1['aggregate_rating'] + 0.1)
    
    # Define features for modeling
    categorical_features = ['primary_cuisine', 'city', 'price_category', 'has_online_delivery']
    numerical_features = [
        'average_cost_for_two', 'price_per_person', 'votes', 
        'cuisine_count', 'votes_per_rating'
    ]
    
    # Keep only rows where we have all required features
    required_cols = categorical_features + numerical_features + ['aggregate_rating']
    df_processed = df_task1.dropna(subset=required_cols)
    
    logger.info(f"Task 1 data shape: {df_processed.shape}")
    
    # Create preprocessor
    preprocessor_task1 = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])
    
    # Prepare features and target
    X = df_processed[categorical_features + numerical_features]
    y = df_processed['aggregate_rating']
    
    # Split data for proper evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit preprocessor and transform data
    X_train_transformed = preprocessor_task1.fit_transform(X_train)
    X_test_transformed = preprocessor_task1.transform(X_test)
    
    # Train model
    model_task1 = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model_task1.fit(X_train_transformed, y_train)
    
    # Evaluate model
    y_test_pred = model_task1.predict(X_test_transformed)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("=== Improved Task 1 Model Performance ===")
    print(f"Test - MAE: {test_mae:.3f}, RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_task1, 'models/improved_rf_rating_model.pkl')
    joblib.dump(preprocessor_task1, 'models/improved_preprocessor_task1.pkl')

# Initialize improved Task 1
setup_improved_task1()

# ============================================================================
# TASK 2: RECOMMENDATION SYSTEM (Updated to use global df)
# ============================================================================

def setup_task2_preprocessing():
    global ohe_task2, scaler_task2
    try:
        # Check if pre-trained models exist
        ohe_path = "models/ohe_task2.pkl"
        scaler_path = "models/scaler_task2.pkl"
        if os.path.exists(ohe_path) and os.path.exists(scaler_path):
            ohe_task2 = joblib.load(ohe_path)
            scaler_task2 = joblib.load(scaler_path)
            print("Loaded existing Task 2 preprocessors.")
        else:
            # Fit new preprocessors using the global df (which now has primary_cuisine)
            ohe_task2 = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ohe_task2.fit(df[["primary_cuisine"]])
            scaler_task2 = StandardScaler()
            scaler_task2.fit(df[["average_cost_for_two", "price_range", "aggregate_rating"]])
            joblib.dump(ohe_task2, ohe_path)
            joblib.dump(scaler_task2, scaler_path)
            print("Saved new Task 2 preprocessors.")
    except Exception as e:
        print(f"Error in Task 2 preprocessing: {str(e)}")
        raise

setup_task2_preprocessing()

# ============================================================================
# TASK 3: CUISINE CLASSIFICATION (Updated to use global df)
# ============================================================================

def setup_task3_preprocessing():
    global classifier_task3, preprocessor_task3, cuisine_classes
    try:
        # Now primary_cuisine exists in the global df
        logger.info("Starting Task 3 preprocessing")
        model_path = "models/classifier_task3.pkl"
        preprocessor_path = "models/preprocessor_task3.pkl"
        
        # Clear existing models to force retraining
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(preprocessor_path):
            os.remove(preprocessor_path)
        
        X = df[["average_cost_for_two", "price_range", "aggregate_rating", "city"]]
        y = df["primary_cuisine"]
        
        preprocessor_task3 = ColumnTransformer([
            ("num", StandardScaler(), ["average_cost_for_two", "price_range", "aggregate_rating"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"])
        ])
        
        X_transformed = preprocessor_task3.fit_transform(X)
        cuisine_classes = y.unique().tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
        
        classifier_task3 = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier_task3.fit(X_train, y_train)
        
        y_pred = classifier_task3.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0)
        }
        
        with open("models/task3_metrics.txt", "w") as f:
            f.write(f"Accuracy: {metrics['accuracy']:.2f}\nPrecision: {metrics['precision']:.2f}\nRecall: {metrics['recall']:.2f}")
        
        joblib.dump(classifier_task3, model_path)
        joblib.dump(preprocessor_task3, preprocessor_path)
        logger.info("Saved new Task 3 classifier and preprocessor.")
    except Exception as e:
        logger.error(f"Error in Task 3 preprocessing: {str(e)}")
        raise

setup_task3_preprocessing()

# ============================================================================
# TASK 4: LOCATION ANALYSIS (Keep as is, but use global df)
# ============================================================================

# Task 4: Preprocessing
m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=5)
heat_data = [[row["latitude"], row["longitude"]] for _, row in df.iterrows() if row["latitude"] != 0 and row["longitude"] != 0]
HeatMap(heat_data).add_to(m)
m.save("visuals/restaurant_heatmap.html")

city_stats = df.groupby("city").agg({
    "restaurant_name": "count",
    "aggregate_rating": "mean",
    "average_cost_for_two": "mean"
}).rename(columns={"restaurant_name": "count"}).sort_values("count", ascending=False)

locality_stats = df.groupby(["city", "locality"]).agg({
    "restaurant_name": "count",
    "aggregate_rating": "mean"
}).rename(columns={"restaurant_name": "count"}).sort_values("count", ascending=False)

top_cities = city_stats.head(5).index
cuisine_by_city = df[df["city"].isin(top_cities)][["city", "cuisines"]].copy()
cuisine_by_city["cuisines"] = cuisine_by_city["cuisines"].str.split(",")
cuisine_by_city = cuisine_by_city.explode("cuisines")
cuisine_counts = cuisine_by_city.groupby(["city", "cuisines"]).size().unstack().fillna(0)

plt.figure(figsize=(12, 8))
sns.heatmap(cuisine_counts, cmap="YlGnBu")
plt.title("Top Cuisines in Top 5 Cities")
plt.xlabel("Cuisine")
plt.ylabel("City")
plt.savefig("visuals/cuisine_heatmap_task4.png")
plt.close()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

from typing import Optional

# Keep the OLD input model for backward compatibility (what your frontend sends)
class RatingInput(BaseModel):
    average_cost_for_two: float
    price_range: int
    cuisines: str  # This is the full cuisines string from your frontend
    city: str
    locality: str
    votes: int
    has_online_delivery: str

# Other existing models (keep these as they are)
class CuisinePredictionInput(BaseModel):
    average_cost_for_two: float
    price_range: int
    aggregate_rating: float
    city: str

class LocationAnalysisInput(BaseModel):
    city: str
    cuisine: str
    min_rating: float

class RecommendationInput(BaseModel):
    cuisine: str
    price_range: int
    min_rating: float
    city: str
    
def convert_old_to_new_format(old_input: RatingInput) -> dict:
    """Convert old input format to new format"""
    # Extract primary cuisine from the cuisines string
    primary_cuisine = old_input.cuisines.split(',')[0].strip() if old_input.cuisines else 'Unknown'
    
    # Group rare cuisines (match what we did in global preprocessing)
    cuisine_counts = df['primary_cuisine'].value_counts()
    rare_cuisines = cuisine_counts[cuisine_counts < 10].index
    if primary_cuisine in rare_cuisines:
        primary_cuisine = 'Other'
    
    return {
        'primary_cuisine': primary_cuisine,
        'average_cost_for_two': old_input.average_cost_for_two,
        'price_range': old_input.price_range,
        'city': old_input.city,
        'has_online_delivery': old_input.has_online_delivery,
        'votes': old_input.votes
    }

def create_prediction_features_from_dict(data: dict) -> dict:
    """Create engineered features for prediction from dictionary"""
    # Add engineered features
    data['price_per_person'] = data['average_cost_for_two'] / 2
    data['cuisine_count'] = 1  # Single cuisine
    data['votes_per_rating'] = data['votes'] / 4.0  # Reasonable assumption
    
    # Price category based on cost (use quartiles from the actual data)
    try:
        if data['average_cost_for_two'] <= df['average_cost_for_two'].quantile(0.25):
            data['price_category'] = 'Budget'
        elif data['average_cost_for_two'] <= df['average_cost_for_two'].quantile(0.5):
            data['price_category'] = 'Mid-Range'
        elif data['average_cost_for_two'] <= df['average_cost_for_two'].quantile(0.75):
            data['price_category'] = 'Premium'
        else:
            data['price_category'] = 'Luxury'
    except:
        # Fallback to fixed thresholds if quantile calculation fails
        if data['average_cost_for_two'] <= 400:
            data['price_category'] = 'Budget'
        elif data['average_cost_for_two'] <= 800:
            data['price_category'] = 'Mid-Range'
        elif data['average_cost_for_two'] <= 1500:
            data['price_category'] = 'Premium'
        else:
            data['price_category'] = 'Luxury'
    
    return data


# ============================================================================
# API ENDPOINTS
# ============================================================================

def create_prediction_features(input_data):
    """Create engineered features for prediction"""
    data = input_data.dict()
    
    # Add engineered features
    data['price_per_person'] = data['average_cost_for_two'] / 2
    data['cuisine_count'] = 1  # Single cuisine
    data['votes_per_rating'] = data['votes'] / 4.0  # Reasonable assumption
    
    # Price category
    if data['average_cost_for_two'] <= 400:
        data['price_category'] = 'Budget'
    elif data['average_cost_for_two'] <= 800:
        data['price_category'] = 'Mid-Range'
    elif data['average_cost_for_two'] <= 1500:
        data['price_category'] = 'Premium'
    else:
        data['price_category'] = 'Luxury'
    
    return data

@app.post("/predict_rating")
async def predict_rating(input_data: RatingInput):  # Uses the OLD format for compatibility
    try:
        # Convert old format to new format
        new_format_data = convert_old_to_new_format(input_data)
        
        # Create features
        features = create_prediction_features_from_dict(new_format_data)
        
        print(f"Original input: {input_data.dict()}")
        print(f"Converted features: {features}")
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Select required features in correct order
        categorical_features = ['primary_cuisine', 'city', 'price_category', 'has_online_delivery']
        numerical_features = [
            'average_cost_for_two', 'price_per_person', 'votes', 
            'cuisine_count', 'votes_per_rating'
        ]
        
        # Ensure all required columns exist
        for col in categorical_features + numerical_features:
            if col not in input_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        input_df = input_df[categorical_features + numerical_features]
        
        print(f"DataFrame for prediction: {input_df.to_dict()}")
        
        # Transform and predict
        input_transformed = preprocessor_task1.transform(input_df)
        prediction = model_task1.predict(input_transformed)[0]
        
        # Ensure prediction is within valid range
        prediction = max(0.0, min(5.0, prediction))
        
        return {
            "predicted_rating": round(prediction, 2)
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============================================================================
# ADD THESE DEBUGGING ENDPOINTS AT THE END OF YOUR FILE (BEFORE if __name__)
# ============================================================================

@app.get("/predict_rating/schema")
async def get_prediction_schema():
    """Returns the expected input schema for debugging"""
    return {
        "expected_input": {
            "average_cost_for_two": "float (e.g., 1200.0)",
            "price_range": "int (1-4)",
            "cuisines": "string (e.g., 'Italian, Continental')",
            "city": "string (e.g., 'Mumbai')",
            "locality": "string (e.g., 'Bandra')",
            "votes": "int (e.g., 100)",
            "has_online_delivery": "string ('Yes' or 'No')"
        },
        "example": {
            "average_cost_for_two": 1200.0,
            "price_range": 3,
            "cuisines": "Italian, Continental",
            "city": "Mumbai",
            "locality": "Bandra",
            "votes": 150,
            "has_online_delivery": "Yes"
        }
    }

@app.post("/test_prediction")
async def test_prediction():
    """Test endpoint with hardcoded values"""
    test_input = RatingInput(
        average_cost_for_two=1200.0,
        price_range=3,
        cuisines="Italian, Continental",
        city="Mumbai",
        locality="Bandra",
        votes=150,
        has_online_delivery="Yes"
    )
    
    return await predict_rating(test_input)
# Keep all your other endpoints exactly as they were:
@app.post("/recommend")
async def recommend(input: RecommendationInput):
    try:
        filtered_df = df.copy()
        if input.city != "All":
            filtered_df = filtered_df[filtered_df["city"] == input.city]
        filtered_df = filtered_df[filtered_df["aggregate_rating"] >= input.min_rating]
        
        # primary_cuisine already exists in filtered_df from global preprocessing
        if filtered_df.empty:
            return []
            
        input_df = pd.DataFrame({
            "primary_cuisine": [input.cuisine],
            "average_cost_for_two": [filtered_df["average_cost_for_two"].median()],
            "price_range": [input.price_range],
            "aggregate_rating": [input.min_rating]
        })
        
        cuisine_vec = ohe_task2.transform(input_df[["primary_cuisine"]])
        num_vec = scaler_task2.transform(input_df[["average_cost_for_two", "price_range", "aggregate_rating"]])
        input_vec = np.hstack([cuisine_vec, num_vec])
        
        filtered_cuisine_encoded = ohe_task2.transform(filtered_df[["primary_cuisine"]])
        filtered_numerical_features = scaler_task2.transform(filtered_df[["average_cost_for_two", "price_range", "aggregate_rating"]])
        filtered_X = np.hstack([filtered_cuisine_encoded, filtered_numerical_features])
        
        scores = cosine_similarity(input_vec, filtered_X).flatten()
        filtered_df["similarity_score"] = scores
        
        results = filtered_df.sort_values(by=["aggregate_rating", "similarity_score"], ascending=False).head(5)
        results = results[["restaurant_name", "cuisines", "average_cost_for_two", "price_range", "aggregate_rating", "city"]].reset_index(drop=True)
        results["aggregate_rating"] = results["aggregate_rating"].astype(float)
        results.to_csv("data/recommendations.csv", index=False)
        return results.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in recommendation: {str(e)}")


@app.post("/predict-cuisine")
async def predict_cuisine(input: CuisinePredictionInput):
    try:
        logger.info(f"Received cuisine prediction request: {input.dict()}")
        input_df = pd.DataFrame({
            "average_cost_for_two": [input.average_cost_for_two],
            "price_range": [input.price_range],
            "aggregate_rating": [input.aggregate_rating],
            "city": [input.city]
        })
        input_transformed = preprocessor_task3.transform(input_df)
        pred_proba = classifier_task3.predict_proba(input_transformed)[0]
        top_indices = np.argsort(pred_proba)[-3:][::-1]
        top_cuisines = [cuisine_classes[i] for i in top_indices]
        top_probabilities = pred_proba[top_indices]
        results = pd.DataFrame({
            "cuisine": top_cuisines,
            "probability": top_probabilities
        })
        results.to_csv("data/cuisine_predictions_task3.csv", index=False)
        plt.figure(figsize=(8, 6))
        sns.barplot(x="probability", y="cuisine", data=results, color="skyblue")
        plt.title(f"Top Predicted Cuisines for Restaurant in {input.city}")
        plt.xlabel("Probability")
        plt.ylabel("Cuisine")
        plt.tight_layout()
        plot_path = "visuals/cuisine_prediction_task3.png"
        plt.savefig(plot_path)
        plt.close()
        with open("models/task3_metrics.txt", "r") as f:
            metrics_text = f.read()
        metrics = {
            "accuracy": float(metrics_text.split("\n")[0].split(": ")[1]),
            "precision": float(metrics_text.split("\n")[1].split(": ")[1]),
            "recall": float(metrics_text.split("\n")[2].split(": ")[1])
        }
        logger.info("Cuisine prediction completed successfully.")
        return {
            "top_cuisines": results.to_dict(orient="records"),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error in cuisine prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in cuisine prediction: {str(e)}")
    
    
@app.post("/location-analysis")
async def location_analysis(input: LocationAnalysisInput):
    try:
        logger.info(f"Received location analysis request: {input.dict()}")
        filtered_df = df.copy()
        if input.city != "All":
            filtered_df = filtered_df[filtered_df["city"] == input.city]
        if input.cuisine != "All":
            filtered_df = filtered_df[filtered_df["primary_cuisine"] == input.cuisine]
        filtered_df = filtered_df[filtered_df["aggregate_rating"] >= input.min_rating]
        if filtered_df.empty:
            return {"heatmap_path": "", "stats": {}, "top_restaurants": []}
        # Create heatmap
        center_lat = filtered_df["latitude"].mean()
        center_lon = filtered_df["longitude"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        heat_data = [[row.latitude, row.longitude, row.aggregate_rating] for row in filtered_df.itertuples()]
        HeatMap(heat_data, radius=15).add_to(m)
        # Add markers for top 5 restaurants
        top_restaurants = filtered_df.sort_values(by="aggregate_rating", ascending=False).head(5)
        for row in top_restaurants.itertuples():
            folium.Marker(
                location=[row.latitude, row.longitude],
                popup=f"{row.restaurant_name}: {row.aggregate_rating:.2f} ★",
                icon=folium.Icon(color="blue")
            ).add_to(m)
        heatmap_path = "visuals/restaurant_heatmap_filtered.html"
        m.save(heatmap_path)
        # Calculate stats
        stats = {
            "restaurant_count": len(filtered_df),
            "avg_rating": float(filtered_df["aggregate_rating"].mean()),
            "avg_cost_for_two": float(filtered_df["average_cost_for_two"].mean())
        }
        # Save filtered data
        filtered_df[["restaurant_name", "primary_cuisine", "aggregate_rating", "average_cost_for_two", "city", "latitude", "longitude"]].to_csv("data/location_analysis.csv", index=False)
        top_restaurants_data = top_restaurants[["restaurant_name", "primary_cuisine", "aggregate_rating", "average_cost_for_two"]].to_dict(orient="records")
        logger.info("Location analysis completed successfully.")
        return {
            "heatmap_path": heatmap_path,
            "stats": stats,
            "top_restaurants": top_restaurants_data
        }
    except Exception as e:
        logger.error(f"Error in location analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in location analysis: {str(e)}")