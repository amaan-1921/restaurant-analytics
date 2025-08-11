import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import folium
import streamlit.components.v1
import branca
from folium import Map
import streamlit_folium as st_folium
import io
from io import BytesIO

st.set_page_config(page_title="Cognifyz Restaurant Analytics", layout="wide")
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
body { background-color: #f3f4f6; }
.stTab { background-color: #ffffff; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "heatmap_file" not in st.session_state:
    st.session_state.heatmap_file = None
if "city_stats" not in st.session_state:
    st.session_state.city_stats = None
if "selected_city_task1" not in st.session_state:
    st.session_state.selected_city_task1 = None
if "locality_options_task1" not in st.session_state:
    st.session_state.locality_options_task1 = ["Unknown"]
if "selected_city_task3" not in st.session_state:
    st.session_state.selected_city_task3 = None
if "locality_options_task3" not in st.session_state:
    st.session_state.locality_options_task3 = ["Unknown"]

# Load dataset for dropdowns
df = pd.read_csv("data/Dataset.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.fillna({"city": "Unknown", "locality": "Unknown"})
df["primary_cuisine"] = df["cuisines"].apply(lambda x: x.split(",")[0].strip() if isinstance(x, str) and x.strip() else "Unknown")

# Create city-to-locality mapping
city_locality_map = df.groupby("city")["locality"].unique().apply(sorted).to_dict()
city_options = ["All"] + sorted(df["city"].unique().tolist())
cuisine_options = ["All"] + sorted(df["cuisines"].fillna("Unknown").unique().tolist())

# Sidebar
st.sidebar.title("Cognifyz Internship")
st.sidebar.markdown("**Mohammed Amaan Thayyil | Restaurant Analytics**")

# Tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Home", "Rating Prediction", "Resturant Recommendation", "Cuisine Classification", "Location Analysis"])

# Home Tab
with tab0:
    st.header("Cognifyz Machine Learning Internship")
    st.markdown("""
    **Mohammed Amaan Thayyil**  
    This dashboard presents four machine learning tasks developed during my Cognifyz internship, leveraging a restaurant dataset to deliver actionable insights for customers, restaurant owners, and platforms.

    - **Task 1: Rating Prediction**  
      **Functionality**: Uses a Random Forest model to predict a restaurant's rating (0–5) based on features like cost, price range, cuisine, city, locality, votes, and online delivery. Users input these features to view the predicted rating, star visualization, feature importance plot, and download results.  
      **Usefulness**: Helps customers estimate restaurant quality before visiting and assists owners in understanding factors driving high ratings for better business decisions.

    - **Task 2: Restaurant Recommendation**  
      **Functionality**: Recommends top 5 restaurants based on user-selected cuisine, price range, minimum rating, and city, using cosine similarity. Displays a sorted list with justifications, star ratings, city context (average rating and cost comparisons), a rating distribution plot, and downloadable CSV/plot.  
      **Usefulness**: Guides customers to find restaurants matching their preferences and helps owners identify competitive offerings in their market.

    - **Task 3: Cuisine Classification**  
      **Functionality**: Predicts a restaurant’s top 3 likely cuisines using a Random Forest Classifier based on cost, price range, rating, and city. Shows probabilities, model performance (accuracy, precision, recall), a bar chart, and downloadable CSV/plot.  
      **Usefulness**: Assists owners in positioning new restaurants by predicting market-fit cuisines and helps customers understand cuisine trends in specific cities.

    - **Task 4: Location Analysis**  
      **Functionality**: Visualizes restaurant density with an interactive Folium heatmap, filtered by city, primary cuisine, and minimum rating. Displays area stats (restaurant count, average rating, cost), top 5 restaurants with clickable markers, and downloadable CSV/HTML.  
      **Usefulness**: Helps customers find dining hubs with high-rated or budget-friendly options and aids owners in choosing optimal locations by identifying dense or underserved areas.

    """)

# Task 1: Rating Prediction
with tab1:
    st.header("Predict Restaurant Rating")
    st.markdown("""
    **Purpose**: Predict the rating (0–5) for a hypothetical restaurant based on its features, such as location, cost, and cuisine.  
    Use this to plan a new restaurant, optimize an existing one, or estimate customer satisfaction in a specific locality.
    """)
    # City selection and locality update outside the form
    city = st.selectbox("City", city_options[1:], key="city_task1")
    if st.button("Update Localities", key="update_locality_task1"):
        st.session_state.selected_city_task1 = city
        st.session_state.locality_options_task1 = city_locality_map.get(city, ["Unknown"])
    locality = st.selectbox("Locality", st.session_state.locality_options_task1, key="locality_task1")
    with st.form("rating_form"):
        col1, col2 = st.columns(2)
        with col1:
            restaurant_name = st.text_input("Restaurant Name (Optional)", value="My Restaurant")
            cost = st.number_input("Average Cost for Two", min_value=0.0, value=1000.0)
            votes = st.number_input("Votes", min_value=0, value=100)
        with col2:
            cuisines = st.selectbox("Cuisines", cuisine_options[1:])
            price_range = st.selectbox("Price Range", [1, 2, 3, 4])
            online_delivery = st.selectbox("Has Online Delivery", ["Yes", "No"])
        submitted = st.form_submit_button("Predict")
        if submitted:
            payload = {
                "average_cost_for_two": cost,
                "price_range": price_range,
                "cuisines": cuisines,
                "city": city,
                "locality": locality,
                "votes": votes,
                "has_online_delivery": online_delivery
            }
            response = requests.post("http://localhost:8000/predict_rating", json=payload)
            if response.status_code == 200:
                predicted_rating = response.json()['predicted_rating']
                # Star rating visualization
                stars = "★" * int(predicted_rating) + "☆" * (5 - int(predicted_rating))
                st.success(f"Predicted Rating for {restaurant_name} in {locality}, {city}: {predicted_rating:.2f} {stars}")
                # Interpretation
                if predicted_rating >= 4.0:
                    interpretation = "This rating suggests a high-quality dining experience, likely driven by high votes or popular cuisine."
                elif predicted_rating >= 3.0:
                    interpretation = "This rating indicates a satisfactory experience, with potential to improve through more votes or better pricing."
                else:
                    interpretation = "This rating suggests areas for improvement, such as enhancing service or reducing costs."
                st.markdown(f"**Interpretation**: {interpretation}")
                # City/locality context
                city_avg = df[df["city"] == city]["aggregate_rating"].mean()
                locality_avg = df[df["locality"] == locality]["aggregate_rating"].mean() if locality in df["locality"].values else city_avg
                st.markdown(f"**Context**: Compared to {city}'s average rating of {city_avg:.2f}, this prediction is {'above' if predicted_rating > city_avg else 'below or equal to'} the city average.")
                if locality != "Unknown":
                    st.markdown(f"For {locality}, the average rating is {locality_avg:.2f}, making this prediction {'above' if predicted_rating > locality_avg else 'below or equal to'} the locality average.")
            else:
                st.error(f"Error predicting rating: {response.text}")
    # Feature importance plot
    import os
    image_path = "visuals/feature_importance_task1.png"  # Corrected from .txt
    if os.path.exists(image_path):
        st.image(image_path, caption="Feature Importance (Key factors influencing the rating)")
        with open(image_path, "rb") as f:
            st.download_button("Download Feature Importance", f, "feature_importance_task1.png")
    else:
        st.warning("Feature importance plot not found. Ensure it is generated in the visuals/ directory.")
        
# Task 2: Restaurant Recommendation
with tab2:
    st.header("Restaurant Recommendation")
    st.markdown("""
    **Purpose**: Find the top 5 restaurants matching your preferred cuisine, price range, minimum rating, and city.  
    Use this to discover dining options or analyze competitors in a specific location.
    """)
    with st.form("recommend_form"):
        col1, col2 = st.columns(2)
        with col1:
            cuisine = st.selectbox("Preferred Cuisine", cuisine_options[1:], index=cuisine_options.index("North Indian")-1)
            price_range = st.slider("Price Range (1=Low to 4=High)", 1, 4, 2)
        with col2:
            min_rating = st.slider("Minimum Rating (0-5)", 0.0, 5.0, 3.5, step=0.1)
            city = st.selectbox("City", city_options, key="city_task2")
        submitted = st.form_submit_button("Get Recommendations")
        if submitted:
            payload = {
                "cuisine": cuisine,
                "price_range": price_range,
                "min_rating": min_rating,
                "city": city
            }
            response = requests.post("http://localhost:8000/recommend", json=payload)
            if response.status_code == 200:
                results = pd.DataFrame(response.json())
                st.session_state.recommendations = results
                if not results.empty:
                    # Sorted list with justifications using enumerate
                    st.subheader("Recommended Restaurants (Sorted by Rating)")
                    for i, row in enumerate(results.itertuples(), 1):
                        try:
                            rating = float(row.aggregate_rating)  # Ensure float
                            stars = "★" * int(round(rating)) + "☆" * (5 - int(round(rating)))
                            justification = f"Matches your {cuisine} preference with a {rating:.2f} rating and {row.price_range} price range in {city}."
                            st.markdown(f"**{i}. {row.restaurant_name} ({rating:.2f} {stars})**: {justification}")
                        except (ValueError, TypeError):
                            st.warning(f"Skipping {row.restaurant_name} due to invalid rating data.")
                            continue
                    # City context
                    city_avg_rating = df[df["city"] == city]["aggregate_rating"].mean() if city != "All" else df["aggregate_rating"].mean()
                    city_avg_price = df[df["city"] == city]["average_cost_for_two"].mean() if city != "All" else df["average_cost_for_two"].mean()
                    top_restaurant = results.iloc[0]
                    st.markdown(f"**Context**: The average rating in {city} is {city_avg_rating:.2f}, and the average cost for two is {city_avg_price:.2f}. Your top recommendation ({top_restaurant['restaurant_name']}) {'exceeds' if float(top_restaurant['aggregate_rating']) > city_avg_rating else 'is below or equal to'} this average.")
                    # Display table
                    st.subheader("Recommendation Details")
                    st.dataframe(results)
                    # Generate and save rating distribution plot
                    plt.figure(figsize=(8, 6))
                    results["aggregate_rating"].astype(float).hist(bins=5, color="skyblue", edgecolor="black")
                    plt.title(f"Rating Distribution of Recommended Restaurants in {city}")
                    plt.xlabel("Rating")
                    plt.ylabel("Count")
                    plt.tight_layout()
                    plot_path = "visuals/recommendation_ratings_task2.png"
                    plt.savefig(plot_path)
                    plt.close()
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Rating Distribution of Recommendations")
                    else:
                        st.warning("Rating distribution plot not found. Try resubmitting.")
            else:
                st.error(f"Error getting recommendations: {response.text}")
    # Download buttons outside the form
    if st.session_state.get("recommendations") is not None:
        with open("data/recommendations.csv", "rb") as f:
            st.download_button("Download Recommendations", f, "recommendations.csv")
        if os.path.exists("visuals/recommendation_ratings_task2.png"):
            with open("visuals/recommendation_ratings_task2.png", "rb") as f:
                st.download_button("Download Rating Distribution", f, "recommendation_ratings_task2.png")
# Task 3: Cuisine Prediction
with tab3:
    st.header("Cuisine Prediction")
    st.markdown("""
    **Purpose**: Predict the likely cuisine of a restaurant based on its cost, price range, rating, and city.  
    Use this to understand market trends or position a new restaurant (e.g., high-rated restaurants in Abu Dhabi are often North Indian).
    """)
    # Debug dataset columns
    try:
        df = pd.read_csv("data/Dataset.csv")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df["primary_cuisine"] = df["cuisines"].apply(lambda x: x.split(",")[0].strip() if isinstance(x, str) and x.strip() else "Unknown")
    except Exception as e:
        st.error(f"Error loading dataset in app.py: {str(e)}")
    with st.form("cuisine_form"):
        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("City", city_options, key="city_task3")
            price_range = st.slider("Price Range (1=Low to 4=High)", 1, 4, 2)
        with col2:
            average_cost_for_two = st.number_input("Average Cost for Two", min_value=0.0, value=1000.0, step=50.0)
            aggregate_rating = st.slider("Rating (0-5)", 0.0, 5.0, 3.5, step=0.1)
        submitted = st.form_submit_button("Predict Cuisine")
        if submitted:
            payload = {
                "average_cost_for_two": average_cost_for_two,
                "price_range": price_range,
                "aggregate_rating": aggregate_rating,
                "city": city
            }
            try:
                response = requests.post("http://localhost:8000/predict-cuisine", json=payload)
                response.raise_for_status()
                result = response.json()
                st.session_state.cuisine_predictions = pd.DataFrame(result["top_cuisines"])
                st.subheader("Top Predicted Cuisines")
                for i, row in enumerate(st.session_state.cuisine_predictions.itertuples(), 1):
                    st.markdown(f"**{i}. {row.cuisine}**: {row.probability:.2%} probability")
                st.markdown(f"**Model Performance**: Accuracy: {result['metrics']['accuracy']:.2f}, Precision: {result['metrics']['precision']:.2f}, Recall: {result['metrics']['recall']:.2f}")
                plot_path = "visuals/cuisine_prediction_task3.png"
                if os.path.exists(plot_path):
                    st.image(plot_path, caption="Top Predicted Cuisines")
                else:
                    st.warning("Cuisine prediction plot not found. Try resubmitting.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {str(e)}. Ensure FastAPI server is running on http://localhost:8000.")
    if st.session_state.get("cuisine_predictions") is not None:
        with open("data/cuisine_predictions_task3.csv", "rb") as f:
            st.download_button("Download Cuisine Predictions", f, "cuisine_predictions_task3.csv")
        if os.path.exists("visuals/cuisine_prediction_task3.png"):
            with open("visuals/cuisine_prediction_task3.png", "rb") as f:
                st.download_button("Download Cuisine Plot", f, "cuisine_prediction_task3.png")

with tab4:
    st.header("Location Analysis")
    st.markdown("""
    **Purpose**: Visualize restaurant density and identify high-rated or budget-friendly dining hubs by city, cuisine, or minimum rating.  
    Use this to find dining clusters or choose optimal locations for new restaurants.
    """)
    cuisine_options = ["All"] + sorted(df["primary_cuisine"].unique().tolist())
    city_options = ["All"] + sorted(df["city"].unique().tolist())
    with st.form("location_form"):
        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("City", city_options, key="city_task4")
            cuisine = st.selectbox("Cuisine", cuisine_options, key="cuisine_task4")
        with col2:
            min_rating = st.slider("Minimum Rating (0-5)", 0.0, 5.0, 3.5, step=0.1)
        submitted = st.form_submit_button("Generate Location Analysis")
        if submitted:
            payload = {
                "city": city,
                "cuisine": cuisine,
                "min_rating": min_rating
            }
            try:
                response = requests.post("http://localhost:8000/location-analysis", json=payload)
                response.raise_for_status()
                result = response.json()
                st.session_state.location_analysis = result
                if result["heatmap_path"]:
                    st.subheader("Restaurant Density Heatmap")
                    # Load map directly
                    heatmap_path = result["heatmap_path"]
                    if os.path.exists(heatmap_path):
                        # Load HTML map and render it (cleanest way is to re-read as Folium map if possible, else embed)
                        with open(heatmap_path, "r", encoding="utf-8") as f:
                            html = f.read()
                        streamlit.components.v1.html(html, height=500)
                    else:
                        st.warning("Heatmap file not found.")                    # Display stats
                    st.markdown(f"**Area Stats**: {result['stats']['restaurant_count']} restaurants, Avg Rating: {result['stats']['avg_rating']:.2f}, Avg Cost for Two: {result['stats']['avg_cost_for_two']:.2f}")
                    # Display top restaurants
                    st.subheader("Top 5 Restaurants in Area")
                    top_df = pd.DataFrame(result["top_restaurants"])
                    if not top_df.empty:
                        st.dataframe(top_df)
                    else:
                        st.warning("No restaurants match your criteria. Try adjusting filters.")
                else:
                    st.warning("No restaurants match your criteria. Try adjusting filters (e.g., lower minimum rating).")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {str(e)}. Ensure FastAPI server is running on http://localhost:8000.")
    if st.session_state.get("location_analysis") is not None and os.path.exists("data/location_analysis.csv"):
        with open("data/location_analysis.csv", "rb") as f:
            st.download_button("Download Location Data", f, "location_analysis.csv")
        if os.path.exists("visuals/restaurant_heatmap_filtered.html"):
            with open("visuals/restaurant_heatmap_filtered.html", "rb") as f:
                st.download_button("Download Heatmap", f, "restaurant_heatmap_filtered.html")