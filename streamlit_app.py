# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:22:44 2024

@author: Aria Abdi
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, expon

st.set_page_config(
    page_title="DataDash • Visualize and Fit Your Data",
    page_icon="📊",
    layout="wide"
)

st.image("thumbnail.png", use_column_width=True)
st.title("📊 DataDash")
st.markdown("_A fast, interactive data visualization and curve-fitting app._")

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["X", "Y"])

if "point_color" not in st.session_state:
    st.session_state.point_color = "#0000FF"

if "line_color" not in st.session_state:
    st.session_state.line_color = "#FF0000"
    
st.sidebar.title("Navigation")
data_input_method = st.sidebar.radio(
    "Choose how to input data: ", 
    ("Enter Manually", "Upload File"))

st.sidebar.markdown("### Tips for Using the App")
st.sidebar.write(
    """
    - To input data manually, enter X and Y values as comma-separated lists.
    - For file upload, ensure the file has two columns (X and Y).
    - Choose a curve type to fit your data.
    - Note the reset feature may not work with file uploads, to clear data from this method, simply delete the uploaded file through the UI. 
    """
)
st.sidebar.markdown("---")
def reset_session_state():
    st.session_state.clear() 

    st.session_state.data = pd.DataFrame(columns=["X", "Y"])
    st.session_state.point_color = "#0000FF"
    st.session_state.line_color = "#FF0000"
    st.session_state.raw_data = None
    st.session_state.x_column = None
    st.session_state.y_column = None
    st.session_state.stats_x = None  
    st.session_state.bins = 10      
    st.session_state.hist_color = "#FFCC00" 
    st.session_state.fit_type = None 
    st.session_state.rmse = None
    st.session_state.selected_column = None
    

if st.sidebar.button("Reset Application"):
    reset_session_state()
    st.sidebar.success("Application reset! You may start again.")

st.title("Final Project App")

st.header("Step 1: Input Data")
if data_input_method == "Enter Manually":
    st.subheader("Manual Data Entry")
    st.write("Enter your data as comma-separated values (e.g., 1,2,3 for X and 4,5,6 for Y).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_input = st.text_area("X Values (comma-separated)", placeholder="1,2,3")
    with col2:
        y_input = st.text_area("Y Values (comma-separated)", placeholder="4,5,6")
    
    if st.button("Submit Data"):
        try:    
            x_values = [float(x.strip()) for x in x_input.split(",")]
            y_values = [float(y.strip()) for y in y_input.split(",")]

            if len(x_values) != len(y_values):
                st.error("The number of X and Y values must match.")
            else:
                st.session_state.data = pd.DataFrame({"X": x_values, "Y": y_values})
                st.session_state.raw_data = st.session_state.data.copy()
                st.success("Data entered successfully!")
                st.write("Preview of your data:")
                st.dataframe(st.session_state.data)
       
        except ValueError:
            st.error("Invalid input. Ensure all values are numeric and separated by commas.")

elif data_input_method == "Upload File":
    st.subheader("File Upload")
    st.write("Upload a CSV file with two columns (X and Y). The file can have optional headers")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file, header=0)
            elif uploaded_file.name.endswith(".xlsx"):
                data = pd.read_excel(uploaded_file, header=0)
                            
            if data.shape[1] < 2:
                st.error("The uploaded file must have exactly two columns.")
            else:
                st.session_state.raw_data = data
                
                st.write("Select the columns to use for X and Y:")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_column = st.selectbox("Select X column:", options=data.columns, key="x_column")
                with col2:
                    y_column = st.selectbox("Select Y column:", options=data.columns, key="y_column")
                
                if x_column and y_column:
                    filtered_data = data[[x_column, y_column]]
                    filtered_data.columns = ["X", "Y"]
                    
                    filtered_data = filtered_data.apply(pd.to_numeric, errors="coerce")
                    filtered_data = filtered_data.dropna()
                    
                    if filtered_data.empty:
                        st.error("The selected columns contain no valid data.")
                    else: 
                        st.session_state.data = filtered_data
                        st.success("File uploaded and processed successfully!")
                        st.write("Preview of your data:")
                        st.dataframe(st.session_state.data)
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")


def linear(x, a, b):
    return a * x + b

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def logarithmic(x, a, b):
    return a * np.log(x) + b

def power(x, a, b):
    return a * np.power(x, b)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def sinusoidal(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

st.header("Step 2: Visualize Data and Fit Curve")
if "data" in st.session_state and not st.session_state.data.empty:
    st.write("This section will display the uploaded or manually entered data as a graph.")

    with st.expander("Graph Customization", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            point_color = st.color_picker("Pick a color for the data points", "#0000FF", key="point_color")
        with col2:
            line_color = st.color_picker("Pick a color for the curve", "#FF0000", key="line_color")
        with col3:
            marker_size = st.slider("Select marker size:", 5, 30, value=10)

        col1, col2 = st.columns(2)
        with col1:
            line_style = st.selectbox("Select line style for curve:", ["-", "--", "-.", ":"])
        with col2:
            show_grid = st.checkbox("Show Grid", value=True)

    st.subheader("Curve Fitting Options")
    curve_type = st.selectbox(
        "Select Curve Type:",
        ["Linear", "Polynomial", "Exponential", "Logarithmic", "Power", "Moving Average", "Logistic", "Sinusoidal"]
    )
    rmse = {}
    max_error = {}
    r_squared = {}
    
    x = pd.to_numeric(st.session_state.data["X"], errors="coerce")
    y = pd.to_numeric(st.session_state.data["Y"], errors="coerce")
    if x.isnull().any() or y.isnull().any():
        st.error("Data contains non-numeric or invalid values.")
    else:
        if curve_type == "Polynomial":
            degree = st.slider("Select the polynomial degree (N):", min_value=1, max_value=10, value=1, step=1)
            coeffs = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coeffs)
            y_fitted = polynomial(x)
            equation = " + ".join(
                [f"{coef:.4f}x^{degree - i}" for i, coef in enumerate(coeffs)]
            )
            rmse["Polynomial"] = np.sqrt(np.mean((y - y_fitted) ** 2))
            max_error["Polynomial"] = np.max(np.abs(y - y_fitted))
            ss_res = np.sum((y - y_fitted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared["Polynomial"] = 1 - (ss_res / ss_tot)

        elif curve_type == "Linear":
            popt, _ = curve_fit(linear, x, y)
            y_fitted = linear(x, *popt)
            equation = f"{popt[0]:.4f}x + {popt[1]:.4f}"
            rmse["Linear"] = np.sqrt(np.mean((y - y_fitted) ** 2))
            max_error["Linear"] = np.max(np.abs(y - y_fitted))
            ss_res = np.sum((y - y_fitted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared["Linear"] = 1 - (ss_res / ss_tot)

        elif curve_type == "Exponential":
            try:
                popt, _ = curve_fit(exponential, x, y, maxfev=10000)
                y_fitted = exponential(x, *popt)
                equation = f"{popt[0]:.4f}e^({popt[1]:.4f}x) + {popt[2]:.4f}"
                rmse["Exponential"] = np.sqrt(np.mean((y - y_fitted) ** 2))
                max_error["Exponential"] = np.max(np.abs(y - y_fitted))
                ss_res = np.sum((y - y_fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared["Exponential"] = 1 - (ss_res / ss_tot)
            except RuntimeError:
                st.warning("Exponential fit failed, skipping...")
                popt = [0, 0, 0]  # Safe defaults

        elif curve_type == "Logarithmic":
            try:
                if np.any(x <= 0):
                    raise ValueError("Logarithmic fit requires all X values to be greater than 0.")
                popt, _ = curve_fit(logarithmic, x, y)
                y_fitted = logarithmic(x, *popt)
                equation = f"{popt[0]:.4f}ln(x) + {popt[1]:.4f}"
                rmse["Logarithmic"] = np.sqrt(np.mean((y - y_fitted) ** 2))
                max_error["Logarithmic"] = np.max(np.abs(y - y_fitted))
                ss_res = np.sum((y - y_fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared["Logarithmic"] = 1 - (ss_res / ss_tot)
            except (RuntimeError, ValueError) as e:
                st.warning(f"Logarithmic fit failed: {e}")
                popt = [0, 0]
                
        elif curve_type == "Power":
            try:
                if np.any(x <= 0): 
                    raise ValueError("Power fit requires all X values to be greater than 0.")
                popt, _ = curve_fit(power, x, y)
                y_fitted = power(x, *popt)
                equation = f"{popt[0]:.4f}x^{popt[1]:.4f}"
                rmse["Power"] = np.sqrt(np.mean((y - y_fitted) ** 2))
                max_error["Power"] = np.max(np.abs(y - y_fitted))
                ss_res = np.sum((y - y_fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared["Power"] = 1 - (ss_res / ss_tot)
            except (RuntimeError, ValueError) as e:
                st.warning(f"Power fit failed: {e}")
                popt = [0, 0] 
       
        elif curve_type == "Moving Average":
            try:
                window_size = st.slider("Select the moving average window size:", 1, len(x), 3)
                y_fitted = moving_average(y, window_size)
                x_fitted = x[:len(y_fitted)] 
                equation = "Moving Average (no specific equation)"
                rmse["Moving Average"] = np.sqrt(np.mean((y - y_fitted) ** 2))
                max_error["Moving Average"] = np.max(np.abs(y - y_fitted))
                ss_res = np.sum((y - y_fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared["Moving Average"] = 1 - (ss_res / ss_tot)
            except Exception as e:
                st.warning(f"Moving Average fit failed: {e}")
                y_fitted = []
        
        elif curve_type == "Logistic":
            try:
                popt, _ = curve_fit(logistic, x, y, maxfev=10000)
                y_fitted = logistic(x, *popt)
                equation = f"{popt[0]:.4f} / (1 + e^(-{popt[1]:.4f}(x - {popt[2]:.4f})))"
                rmse["Logistic"] = np.sqrt(np.mean((y - y_fitted) ** 2))
                max_error["Logistic"] = np.max(np.abs(y - y_fitted))
                ss_res = np.sum((y - y_fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared["Logistic"] = 1 - (ss_res / ss_tot)
            except RuntimeError:
                st.warning("Logistic fit failed, skipping...")
                popt = [0, 0, 0]
       
        elif curve_type == "Sinusoidal":
            try:
                popt, _ = curve_fit(sinusoidal, x, y, maxfev=10000)
                y_fitted = sinusoidal(x, *popt)
                equation = f"{popt[0]:.4f}sin({popt[1]:.4f}x + {popt[2]:.4f}) + {popt[3]:.4f}"
                rmse["Sinusoidal"] = np.sqrt(np.mean((y - y_fitted) ** 2))
                max_error["Sinusoidal"] = np.max(np.abs(y - y_fitted))
                ss_res = np.sum((y - y_fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared["Sinusoidal"] = 1 - (ss_res / ss_tot)
            except RuntimeError:
                st.warning("Sinusoidal fit failed, skipping...")
                popt = [0, 0, 0, 0]
            

        st.subheader("Graph Output")
        fig, ax = plt.subplots()

        x_fine = np.linspace(min(x), max(x), 500)

        if curve_type == "Polynomial":
            y_fitted_fine = polynomial(x_fine)
        elif curve_type == "Linear":
            y_fitted_fine = linear(x_fine, *popt)
        elif curve_type == "Exponential":
            y_fitted_fine = exponential(x_fine, *popt)
        elif curve_type == "Logarithmic":
            y_fitted_fine = logarithmic(x_fine, *popt)
        elif curve_type == "Power":
            y_fitted_fine = power(x_fine, *popt)
        elif curve_type == "Moving Average":
            if len(y) < window_size:
                st.error("Window size cannot exceed the data size.")
                y_fitted_fine = []
            else: 
                y_fitted_fine = moving_average(y, window_size)
                x_fine = x[:len(y_fitted_fine)]
        elif curve_type == "Logistic":
            y_fitted_fine = logistic(x_fine, *popt)
        elif curve_type == "Sinusoidal":
            y_fitted_fine = sinusoidal(x_fine, *popt)

        ax.scatter(x, y, color=point_color, s=marker_size, label="Data Points")
        ax.plot(x_fine, y_fitted_fine, color=line_color, linestyle=line_style, label=f"{curve_type} Fit")
        if show_grid:
            ax.grid(True)
        ax.set_title("Data Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Fitting Parameters and Equation")
        st.write(f"**Selected Curve Type:** {curve_type}")
        st.write(f"**Root Mean Square Error (RMSE):** {rmse[curve_type]:.4f}")
        st.write(f"**Maximum Absolute Error:** {max_error[curve_type]:.4f}")
        st.write(f"**Coefficient of Determination (R²):** {r_squared[curve_type]:.4f}")
        st.write("**Equation of the Curve:**")
        st.latex(equation)

else:
    st.write("No data available. Please input or upload data to proceed.")
    
st.header("Step 3: Fit Statistical Distributions to Histogram")

if "raw_data" in st.session_state and st.session_state.raw_data is not None and not st.session_state.raw_data.empty:
    column_options = st.session_state.raw_data.columns
    selected_column = st.selectbox("Select column for histogram:", column_options)

    if selected_column not in st.session_state.raw_data.columns:
        st.error(f"The selected column '{selected_column}' does not exist. Please check your uploaded file.")
    else:
        data = pd.to_numeric(st.session_state.raw_data[selected_column], errors="coerce").dropna()

        if data.empty:
            st.error("No valid numeric data available for the histogram.")
        else:
            st.subheader("Histogram Options")
            bins = st.slider("Number of bins:", min_value=5, max_value=50, value=10)
            hist_color = st.color_picker("Pick a color for the histogram", "#FFCC00")

            st.subheader("Histogram and Fitted Distributions")
            fig, ax = plt.subplots()
            counts, bin_edges, _ = ax.hist(data, bins=bins, color=hist_color, alpha=0.6, label="Histogram", density=True)

            st.subheader("Fit Options")
            fit_type = st.selectbox("Select a distribution to fit:", ["Normal", "Exponential"])

            if fit_type == "Normal":
                mu, std = norm.fit(data)
                x = np.linspace(min(data), max(data), 500)
                pdf = norm.pdf(x, mu, std)
                ax.plot(x, pdf, color="blue", lw=2, label=f"Normal Fit (μ={mu:.2f}, σ={std:.2f})")

                y_hist = counts
                y_fit = norm.pdf((bin_edges[:-1] + bin_edges[1:]) / 2, mu, std)
                rmse = np.sqrt(np.mean((y_hist - y_fit) ** 2))

            elif fit_type == "Exponential":
                loc, scale = expon.fit(data, floc=0) 
                x = np.linspace(min(data), max(data), 500)
                pdf = expon.pdf(x, loc=loc, scale=scale)
                ax.plot(x, pdf, color="green", lw=2, label=f"Exponential Fit (λ={1/scale:.2f})")

                y_hist = counts
                y_fit = expon.pdf((bin_edges[:-1] + bin_edges[1:]) / 2, loc=loc, scale=scale)
                rmse = np.sqrt(np.mean((y_hist - y_fit) ** 2))

            ax.set_title("Histogram with Fitted Distribution")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Density")
            ax.legend()

            st.pyplot(fig)

            st.subheader("Fitting Parameters and Metrics")
            if fit_type == "Normal":
                st.write(f"**Mean (μ):** {mu:.4f}")
                st.write(f"**Standard Deviation (σ):** {std:.4f}")
            elif fit_type == "Exponential":
                st.write(f"**Rate Parameter (λ):** {1/scale:.4f}")
                st.write(f"**Scale Parameter:** {scale:.4f}")

            st.write(f"**Root Mean Square Error (RMSE):** {rmse:.4f}")

else:
    st.write("No data available. Please input or upload data to proceed.")

