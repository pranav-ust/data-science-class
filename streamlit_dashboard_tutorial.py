"""
Simple Regression Analysis Dashboard with Streamlit
==================================================

This dashboard demonstrates regression analysis using the Iris dataset.
Students can select features and see linear regression in action.

Author: Educational Tutorial
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Configure the page
st.set_page_config(
    page_title="Simple Regression Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load Iris dataset
@st.cache_data
def load_data():
    """Load the Iris dataset"""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target_names[data.target]
    return df, data.feature_names

df, feature_names = load_data()

# Title
st.title("📈 Simple Regression Analysis Dashboard")
st.markdown("**Using the Iris Dataset to learn regression!**")

# Show dataset info
st.subheader("📁 Iris Dataset Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples", len(df))
with col2:
    st.metric("Features", len(feature_names))
with col3:
    st.metric("Species", df['species'].nunique())

# Sidebar controls
st.sidebar.header("🎛️ Regression Controls")

# Select target (Y) and predictor (X) variables
target_feature = st.sidebar.selectbox(
    "Select Target Variable (Y):",
    feature_names,
    index=0
)

predictor_feature = st.sidebar.selectbox(
    "Select Predictor Variable (X):",
    [f for f in feature_names if f != target_feature],
    index=0
)

# Show scatter plot with regression line
st.subheader(f"📊 Regression: {target_feature} vs {predictor_feature}")

# Prepare data for regression
X = df[[predictor_feature]]
y = df[target_feature]

# Fit regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Create regression line data - fix the feature names issue
x_range = np.linspace(df[predictor_feature].min(), df[predictor_feature].max(), 100)
# Create DataFrame with proper column name for prediction
x_range_df = pd.DataFrame({predictor_feature: x_range})
y_line = model.predict(x_range_df)

# Add regression line to dataframe for plotting
line_df = pd.DataFrame({
    predictor_feature: x_range,
    target_feature: y_line,
    'species': 'Regression Line'
})

# Combine original data with regression line
plot_df = pd.concat([df, line_df], ignore_index=True)

# Create scatter plot with regression line
fig = px.scatter(
    plot_df,
    x=predictor_feature,
    y=target_feature,
    color='species',
    title=f"Linear Regression: {target_feature} vs {predictor_feature}",
    color_discrete_map={'Regression Line': 'red'}
)

# Update the regression line to be a line, not points
fig.data[-1].mode = 'lines'
fig.data[-1].line.width = 3

st.plotly_chart(fig, use_container_width=True)

# Display metrics
st.subheader("📈 Regression Results")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("R² Score", f"{r2:.3f}")
with col2:
    st.metric("RMSE", f"{rmse:.3f}")
with col3:
    st.metric("Slope", f"{model.coef_[0]:.3f}")
with col4:
    st.metric("Intercept", f"{model.intercept_:.3f}")

# Regression equation
st.subheader("📝 Regression Equation")
equation = f"**{target_feature} = {model.intercept_:.3f} + {model.coef_[0]:.3f} × {predictor_feature}**"
st.markdown(equation)

# Interpretation
st.subheader("🎯 Interpretation")
if r2 >= 0.7:
    strength = "**strong**"
    color = "green"
elif r2 >= 0.5:
    strength = "**moderate**"
    color = "orange"
elif r2 >= 0.3:
    strength = "**weak**"
    color = "orange"
else:
    strength = "**very weak**"
    color = "red"

st.markdown(f"The regression shows a {strength} relationship (R² = {r2:.3f}) between {predictor_feature} and {target_feature}.")

direction = "increases" if model.coef_[0] > 0 else "decreases"
st.write(f"For every 1 unit increase in {predictor_feature}, {target_feature} {direction} by {abs(model.coef_[0]):.3f} units on average.")

# Multiple regression section
st.subheader("🔄 Multiple Regression")
st.markdown("Select multiple predictors to see how they together predict the target:")

# Multiple predictor selection
available_predictors = [f for f in feature_names if f != target_feature]
selected_predictors = st.multiselect(
    "Select Multiple Predictors:",
    available_predictors,
    default=available_predictors[:2]
)

if len(selected_predictors) >= 2:
    # Multiple regression
    X_multi = df[selected_predictors]
    y_multi = df[target_feature]
    
    model_multi = LinearRegression()
    model_multi.fit(X_multi, y_multi)
    y_pred_multi = model_multi.predict(X_multi)
    
    r2_multi = r2_score(y_multi, y_pred_multi)
    rmse_multi = np.sqrt(mean_squared_error(y_multi, y_pred_multi))
    
    # Actual vs Predicted plot
    fig_multi = px.scatter(
        x=y_multi,
        y=y_pred_multi,
        color=df['species'],
        title=f"Multiple Regression: Actual vs Predicted {target_feature}",
        labels={'x': f'Actual {target_feature}', 'y': f'Predicted {target_feature}'}
    )
    
    # Add perfect prediction line
    min_val = min(y_multi.min(), y_pred_multi.min())
    max_val = max(y_multi.max(), y_pred_multi.max())
    fig_multi.add_scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    )
    
    st.plotly_chart(fig_multi, use_container_width=True)
    
    # Multiple regression metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Multi R² Score", f"{r2_multi:.3f}")
    with col2:
        st.metric("Multi RMSE", f"{rmse_multi:.3f}")
    with col3:
        improvement = r2_multi - r2
        st.metric("R² Improvement", f"{improvement:.3f}")
    
    # Feature coefficients
    st.subheader("📊 Feature Importance")
    coef_df = pd.DataFrame({
        'Feature': selected_predictors,
        'Coefficient': model_multi.coef_
    })
    
    fig_coef = px.bar(
        coef_df,
        x='Feature',
        y='Coefficient',
        title="Feature Coefficients",
        color='Coefficient',
        color_continuous_scale='RdBu'
    )
    
    st.plotly_chart(fig_coef, use_container_width=True)
    
    # Multiple regression equation
    st.subheader("📝 Multiple Regression Equation")
    equation_parts = [f"{model_multi.intercept_:.3f}"]
    for feature, coef in zip(selected_predictors, model_multi.coef_):
        sign = "+" if coef >= 0 else "-"
        equation_parts.append(f" {sign} {abs(coef):.3f}×{feature}")
    
    equation_multi = f"**{target_feature} = " + "".join(equation_parts) + "**"
    st.markdown(equation_multi)

# Educational notes
st.markdown("---")
st.markdown("""

**Simple Linear Regression:**
- Shows relationship between TWO variables
- R² closer to 1 = stronger relationship
- Slope shows how much Y changes when X increases by 1

**Multiple Linear Regression:**
- Uses MULTIPLE variables to predict target
- Can improve prediction accuracy
- Each coefficient shows the effect of that specific feature

**Try This:**
1. Change the target and predictor variables
2. See how R² changes with different combinations
3. Compare simple vs multiple regression performance
4. Notice which features are most important
""")

# Show sample data
if st.checkbox("Show Sample Data"):
    st.subheader("📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True) 