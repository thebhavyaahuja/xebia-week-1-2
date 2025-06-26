import numpy as np
import pandas as pd
# You can install Streamlit using pip in your environment by running the following command:
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import io

st.set_page_config(
    page_title="Data Visualization",
    page_icon=":bar_chart:",
    layout="wide"
)
st.title("Data Visualization with Streamlit")

# Load the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())


    st.subheader("stats")
    data_info = st.radio("Show Data Information", ("Yes", "No"), index=0)
    data_info = True if data_info == "Yes" else False
    if data_info:
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
    missing_values = st.radio("Show Missing Values", ("yes", "no"), index=0)
    missing_values = True if missing_values == "yes" else False
    if missing_values:
        st.write(df.isnull().sum())
    data_stats = st.radio("Show Data Statistics", ("Yes", "No"), index=0)
    data_stats = True if data_stats == "Yes" else False
    if data_stats:
        st.write(df.describe())


    #visualization
    numeric_cols = df.select_dtypes(include = ["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include = ["object"]).columns.tolist()
    st.write("Numeric Columns:", numeric_cols)
    st.write("Categorical Columns:", categorical_cols)


    # uni variate analysis
    st.subheader("Count plot")
    selected_col = st.selectbox("Select a column for numerical plot", numeric_cols)
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=selected_col, ax=ax)
    st.pyplot(fig)

    #count plot for categorical
    st.subheader("count plot")
    selected_cat_col = st.selectbox("Select a column for categorical plot", categorical_cols)
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=selected_cat_col, ax=ax)
    st.pyplot(fig)

    # box plot for numerical
    st.subheader("Box plot")
    selected_num_col = st.selectbox("Select a column for box plot", numeric_cols)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=selected_num_col, ax=ax)
    st.pyplot(fig)

    # hist plot
    st.subheader("Histogram")
    selected_hist_col = st.selectbox("Select a column for histogram", numeric_cols)
    fig, ax = plt.subplots()        
    sns.histplot(data=df, x=selected_hist_col, ax=ax, kde=True)
    st.pyplot(fig)

    # bi variate analysis
    st.subheader("Bi-variate Analysis :categorical vs Numerical")
    num_cols = st.selectbox("Select a numerical column", numeric_cols)
    cat_cols = st.selectbox("Select a categorical column", categorical_cols)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=cat_cols, y=num_cols, ax=ax)
    st.pyplot(fig)

    # scatter plot
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis column for scatter plot", numeric_cols)
    y_axis = st.selectbox("Select Y-axis column for scatter plot", categorical_cols)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)

    # multi variate analysis
    st.subheader("Multi-variate Analysis using heatmap")
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='magma', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for heatmap.")

    #pair plot
    st.subheader("Pair Plot")
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots()
        sns.pairplot(df[numeric_cols])
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.warning("Not enough numeric columns for pair plot.")

else:
    st.warning("Please upload a CSV file to visualize the data.")

st.subheader("Data Columns")
st.subheader(df.columns.tolist())