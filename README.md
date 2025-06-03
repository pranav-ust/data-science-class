# 📊 Streamlit Dashboard Tutorial

Welcome to the **Interactive Dashboard Tutorial** using Streamlit! This project is designed to teach students the fundamentals of building interactive web dashboards with Python.

## 🎯 Learning Objectives

By the end of this tutorial, students will understand:

1. **Basic Streamlit Components**: How to use widgets, layouts, and displays
2. **Data Visualization**: Creating interactive charts and plots
3. **User Interface Design**: Building intuitive dashboard layouts
4. **Data Analysis**: Performing exploratory data analysis through interactive tools
5. **Code Organization**: Structuring a Streamlit application

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic knowledge of Python and pandas

### Installation

1. **Clone or download this project**
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv streamlit_env
   source streamlit_env/bin/activate  # On Windows: streamlit_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit pyplot
   ```

4. **Run the dashboard**:
   ```bash
   streamlit run streamlit_dashboard_tutorial.py
   ```

5. **Open your browser** to `http://localhost:8501`


### Key Streamlit Concepts Demonstrated

| Component | Purpose | Code Example |
|-----------|---------|--------------|
| `st.selectbox()` | Dropdown selection | Dataset and feature selection |
| `st.multiselect()` | Multiple selections | Feature selection for analysis |
| `st.slider()` | Range selection | Number of features to display |
| `st.columns()` | Layout organization | Side-by-side visualizations |
| `st.tabs()` | Content organization | Different analysis sections |
| `st.metric()` | Key performance indicators | Dataset statistics |
| `@st.cache_data` | Performance optimization | Data loading caching |

