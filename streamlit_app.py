import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-score {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model, scaler, and feature names
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('best_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        return model, scaler, feature_names
    except FileNotFoundError as e:
        return None, None, None

model, scaler, feature_names = load_model()

if model is None:
    st.error("""
    **Model files not found!**
    
    Please run the following code in your Jupyter notebook first:
    ```python
    import pickle
    pickle.dump(best_rf, open('best_model.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    pickle.dump(X_train_scaled.columns.tolist(), open('feature_names.pkl', 'wb'))
    ```
    Then place the .pkl files in the same folder as this streamlit_app.py file.
    """)
    st.stop()

# Main content area - Input Form
st.header("Enter Student Information")

# Create three columns for organized input
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    age = st.slider("Age", 16, 25, 19)
    gender = st.selectbox("Gender", ["Male", "Female"])
    grade_level = st.selectbox("Grade Level", ["1st Year", "2nd Year", "3rd Year", "4th Year"])

with col2:
    st.subheader("Study Factors")
    study_hours = st.slider("Study Hours per Day", 0.0, 10.0, 3.5, 0.5)
    sleep_hours = st.slider("Sleep Hours per Day", 4.0, 12.0, 7.0, 0.5)
    social_media_hours = st.slider("Social Media Hours per Day", 0.0, 10.0, 2.5, 0.5)
    attendance_pct = st.slider("Attendance Percentage", 0, 100, 85)

with col3:
    st.subheader("AI Usage")
    uses_ai = st.selectbox("Uses AI Tools?", ["Yes", "No"])
    
    if uses_ai == "Yes":
        ai_usage_time = st.slider("AI Usage Time (minutes/day)", 0, 180, 90)
        ai_tools = st.selectbox("Primary AI Tool", ["ChatGPT", "Gemini", "Claude", "Other", "Multiple"])
        ai_purpose = st.selectbox("AI Usage Purpose", 
            ["Research", "Exam Prep", "Homework Help", "Content Generation", "Concept Explanation"])
        ai_dependency = st.slider("AI Dependency Score (1-10)", 1, 10, 6)
        ai_content_pct = st.slider("AI-Generated Content %", 0, 100, 40)
    else:
        ai_usage_time = 0
        ai_tools = "None"
        ai_purpose = "None"
        ai_dependency = 0
        ai_content_pct = 0

# Additional factors
st.markdown("---")
st.subheader("Additional Academic Factors")
col4, col5, col6 = st.columns(3)

with col4:
    concept_understanding = st.slider("Concept Understanding Score (1-10)", 1, 10, 7)
    study_consistency = st.slider("Study Consistency Index (1-10)", 1, 10, 6)

with col5:
    class_participation = st.slider("Class Participation Score (1-10)", 1, 10, 7)
    improvement_rate = st.slider("Improvement Rate (%)", 0, 100, 15)

with col6:
    tutoring_hours = st.slider("Tutoring Hours per Week", 0.0, 10.0, 2.0, 0.5)
    ai_prompts_week = st.slider("AI Prompts per Week", 0, 200, 50) if uses_ai == "Yes" else 0
    ai_ethics_score = st.slider("AI Ethics Awareness (1-10)", 1, 10, 8) if uses_ai == "Yes" else 0

# Predict button
st.markdown("---")
col_predict1, col_predict2, col_predict3 = st.columns([1, 2, 1])

with col_predict2:
    predict_button = st.button("PREDICT FINAL SCORE", use_container_width=True, type="primary")

if predict_button:
    # Create input dataframe (match the exact features used in training)
    # NOTE: You need to adjust this based on your actual feature names
    input_data = {
        'age': age,
        'study_hours_per_day': study_hours,
        'uses_ai': 1 if uses_ai == "Yes" else 0,
        'ai_usage_time_minutes': ai_usage_time,
        'ai_dependency_score': ai_dependency,
        'ai_generated_content_percentage': ai_content_pct,
        'ai_prompts_per_week': ai_prompts_week,
        'ai_ethics_score': ai_ethics_score,
        'attendance_percentage': attendance_pct,
        'concept_understanding_score': concept_understanding,
        'study_consistency_index': study_consistency,
        'improvement_rate': improvement_rate,
        'sleep_hours': sleep_hours,
        'social_media_hours': social_media_hours,
        'tutoring_hours': tutoring_hours,
        'class_participation_score': class_participation,
        # Add engineered features
        'ai_to_study_ratio': ai_usage_time / (study_hours * 60 + 1),
        'total_learning_hours': study_hours + (ai_usage_time / 60),
        'work_life_balance': sleep_hours - social_media_hours,
    }
    
    # Add categorical features (one-hot encoded)
    input_df = pd.DataFrame([input_data])
    
    # Add categorical variables
    input_df['gender'] = gender
    input_df['grade_level'] = grade_level
    input_df['ai_tools_used'] = ai_tools
    input_df['ai_usage_purpose'] = ai_purpose
    
    # One-hot encode (match training encoding)
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    
    # Ensure all columns from training are present (align with feature_names)
    # Add missing columns with value 0
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Remove any extra columns not in training
    input_encoded = input_encoded[feature_names]
    
    # Scale the input using the fitted scaler
    input_scaled = scaler.transform(input_encoded)
    
    # Convert back to DataFrame with feature names to avoid sklearn warning
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
    
    try:
        # Make prediction using scaled input with feature names
        prediction = model.predict(input_scaled_df)[0]
        
        # Display prediction
        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: #1f77b4; margin-bottom: 0.5rem;">PREDICTED FINAL SCORE</h2>
            <p class="prediction-score">{prediction:.1f} / 100</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance category
        if prediction >= 70:
            category = "High Performer"
            color = "green"
            message = "Excellent! Keep up the great work!"
        elif prediction >= 50:
            category = "Average Performer"
            color = "orange"
            message = "Good effort! Consider improving study consistency and AI usage patterns."
        else:
            category = "At-Risk Student"
            color = "red"
            message = "Warning: Intervention recommended. Focus on fundamentals and reduce AI dependency."
        
        st.markdown(f"### Performance Category: {category}")
        st.markdown(f"**Expected deviation:** ±10.2 points")
        st.info(message)
        
        # Recommendations based on input
        st.markdown("---")
        st.subheader("Personalized Recommendations")
        
        recommendations = []
        
        if concept_understanding < 6:
            recommendations.append("**Focus on strengthening concept understanding** - This is the #1 predictor of success!")
        
        if ai_dependency > 7 and prediction < 60:
            recommendations.append("**Reduce AI dependency** - Over-reliance on AI may be hindering deep learning")
        
        if ai_purpose == "Homework Help" and uses_ai == "Yes":
            recommendations.append("**Change AI usage to Research/Exam Prep** - Students who use AI for research score ~15 points higher")
        
        if study_hours < 3:
            recommendations.append("**Increase study time** - Aim for at least 3-4 hours daily study")
        
        if attendance_pct < 80:
            recommendations.append("**Improve attendance** - Class attendance correlates with better performance")
        
        if sleep_hours < 6:
            recommendations.append("**Get more sleep** - Sleep deprivation affects academic performance")
        
        if study_consistency < 6:
            recommendations.append("**Build consistent study habits** - Consistency matters more than cramming")
        
        if not recommendations:
            recommendations.append("**Keep it up!** Your study patterns look great. Maintain this balance!")
        
        for rec in recommendations:
            st.write(rec)
        
        # Show feature importance context
        st.markdown("---")
        st.subheader("What Matters Most?")
        st.write("""
        Based on our analysis of 1000 students:
        1. **Concept Understanding** (Most important - 0.43 correlation)
        2. **Study Consistency** (0.28 correlation)
        3. **Class Participation** (0.24 correlation)
        4. **AI Usage Purpose** (Research > Exam Prep > Homework Help)
        5. **AI-Generated Content %** (Negative correlation - less is better)
        """)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("""
        **Troubleshooting tips:**
        1. Ensure model was trained with same feature names
        2. Check that all encoded features match training data
        3. Verify pickle files are not corrupted
        4. Consider saving and loading feature column names from training
        """)
