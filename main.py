from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
df = df.fillna({
    "cuisines": "Unknown",
    "average_cost_for_two": df["average_cost_for_two"].median(),
    "price_range": df["price_range"].mode()[0],
    "aggregate_rating": df["aggregate_rating"].mean(),
    "city": "Unknown",
    "locality": "Unknown",
    "latitude": 0,
    "longitude": 0
})
df_cleaned = df.drop(columns=[
    "restaurant_id", "address", "locality_verbose", "rating_color", "rating_text", "switch_to_order_menu"
])

# Task 1: Preprocessing and Model
cat_cols = ["price_range", "cuisines", "city", "locality", "has_online_delivery"]
num_cols = ["average_cost_for_two", "votes"]
df["primary_cuisine"] = df["cuisines"].apply(
    lambda x: x.split(",")[0].strip() if isinstance(x, str) and x.strip() and x.lower() != "unknown" else np.nan
)
df = df.dropna(subset=["primary_cuisine"])
logger.info(f"Primary cuisine created (after dropping NaN): {df['primary_cuisine'].value_counts().head(10).to_dict()}")
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols)
])
X_task1 = df_cleaned[cat_cols + num_cols]
y_task1 = df_cleaned["aggregate_rating"]
X_task1_transformed = preprocessor.fit_transform(X_task1)
model_task1 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model_task1.fit(X_task1_transformed, y_task1)
joblib.dump(model_task1, "models/rf_rating_model.pkl")
joblib.dump(preprocessor, "models/preprocessor_task1.pkl")
# Generate feature importance plot
feature_names = num_cols + preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
importances = pd.Series(model_task1.feature_importances_, index=feature_names).sort_values(ascending=False).head(5)
plt.figure(figsize=(10, 6))
importances.plot(kind="bar")
plt.title("Top 5 Feature Importances (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("visuals/feature_importance_task1.png")

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
            # Fit new preprocessors
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


# Task 3: Cuisine Classification Preprocessing
def setup_task3_preprocessing():
    global classifier_task3, preprocessor_task3, cuisine_classes
    try:
        if "primary_cuisine" not in df.columns:
            raise ValueError("primary_cuisine column missing in DataFrame")
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

# Pydantic models
class RatingInput(BaseModel):
    average_cost_for_two: float
    price_range: int
    cuisines: str
    city: str
    locality: str
    votes: int
    has_online_delivery: str

class CuisinePredictionInput(BaseModel):
    average_cost_for_two: float
    price_range: int
    aggregate_rating: float
    city: str

class LocationAnalysisInput(BaseModel):
    city: str
    cuisine: str
    min_rating: float

# Endpoints
@app.post("/predict_rating")
async def predict_rating(input: RatingInput):
    input_df = pd.DataFrame([input.dict()])
    input_transformed = preprocessor.transform(input_df)
    prediction = model_task1.predict(input_transformed)[0]
    return {"predicted_rating": round(prediction, 2)}


class RecommendationInput(BaseModel):
    cuisine: str
    price_range: int
    min_rating: float
    city: str

@app.post("/recommend")
async def recommend(input: RecommendationInput):
    try:
        filtered_df = df.copy()
        if input.city != "All":
            filtered_df = filtered_df[filtered_df["city"] == input.city]
        filtered_df = filtered_df[filtered_df["aggregate_rating"] >= input.min_rating]
        filtered_df["primary_cuisine"] = filtered_df["cuisines"].apply(lambda x: x.split(",")[0].strip() if isinstance(x, str) and x.strip() and x.lower() != "unknown" else np.nan)
        filtered_df = filtered_df.dropna(subset=["primary_cuisine"])

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
        results["aggregate_rating"] = results["aggregate_rating"].astype(float)  # Ensure float type
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