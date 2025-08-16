import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Stock Prediction", "About Us", "Contact Us"])

if page == "Stock Prediction":
    st.title("Stock Prediction App")

    selected_stock = st.text_input("Enter Stock Symbol :")

    if not selected_stock:
        st.warning("Please enter a stock symbol to proceed.")
    else:
        n_years = st.slider("Years of prediction:", 1, 4)
        period = n_years * 365

        @st.cache_data
        def load_data(ticker):
            """Fetch historical data for the given stock ticker."""
            try:
                data = yf.download(ticker, start=START, end=TODAY)
                data.reset_index(inplace=True)  # Ensure Date is a column
                data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                return data
            except Exception as e:
                st.error(f"Error loading data for {ticker}: {e}")
                return pd.DataFrame()

        # Load data and update status
        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        if data.empty:
            st.error("No data found for the entered stock symbol.")
        else:
            st.subheader('Raw data')
            st.write(data.tail())

            def plot_raw_data():
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
                fig.update_layout(
                    title_text="Time Series Data",
                    xaxis_rangeslider_visible=True,
                    xaxis=dict(showgrid=True, gridcolor='LightGrey', gridwidth=0.75),
                    yaxis=dict(showgrid=True, gridcolor='LightGrey', gridwidth=0.5),
                    width=800,
                    height=600
                )
                st.plotly_chart(fig)

            plot_raw_data()

            df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
            train_data = df_train[:-period]
            test_data = df_train[-period:]

            m = Prophet()
            m.fit(train_data)

            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            st.subheader('Forecast data')
            st.write(forecast.tail())

            st.write("Forecast Data")
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.write("Forecast Components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)

            test_pred = forecast[['ds', 'yhat']].tail(period)
            test_actual = test_data.reset_index(drop=True)
            comparison = pd.merge(test_pred, test_actual, how='inner', left_on='ds', right_on='ds')

            st.subheader('Prediction vs Actual')
            st.write(comparison)

            mae = mean_absolute_error(comparison['y'], comparison['yhat'])
            mse = mean_squared_error(comparison['y'], comparison['yhat'])
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(comparison['y'], comparison['yhat'])

            avg_prediction = comparison['yhat'].mean()
            avg_actual = comparison['y'].mean()
            accuracy_percentage = (100 - (mape * 100))

            st.subheader('Accuracy Metrics')
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
            st.write(f"Average Prediction: {avg_prediction:.2f}")
            st.write(f"Average Actual: {avg_actual:.2f}")
            st.write(f"Accuracy: {accuracy_percentage:.2f}%")

elif page == "About Us":
    st.title("About Us")
    st.write("""
    **About Us**

    Welcome to the **Stock Prediction App**, a cutting-edge platform that leverages the power of **Artificial Intelligence (AI)** and **Machine Learning (ML)** to provide intelligent insights into the stock market. Our app is designed to empower users with data-driven predictions, enhancing decision-making and improving the accuracy of stock analysis.

    **Our Vision**
    - Bridge the gap between academia and industry by creating practical AI-driven solutions.
    - Showcase the transformative power of modern technology in real-world applications.

    **Key Features**
    1. **Historical Data Analysis**: Access and visualize historical stock data to identify trends.
    2. **AI-Powered Forecasting**: Use advanced models like **Facebook Prophet** for accurate stock price predictions.
    3. **Evaluation Metrics**: Validate predictions with metrics such as **MAE**, **MSE**, **RMSE**, and **MAPE**.
    4. **Interactive Visualization**: Explore trends with dynamic and interactive charts using **Plotly**.
    5. **User-Friendly Interface**: Built with **Streamlit** for an intuitive and easy-to-use experience.

    **Why Choose Us?**
    - **Innovation**: Showcases cutting-edge AI techniques for stock forecasting.
    - **Educational Focus**: Reflects our commitment to learning and advancing **Artificial Intelligence** research.
    - **Real-World Impact**: Demonstrates AI’s role in providing insights for financial technology (**FinTech**).
    - **Collaboration and Excellence**: Developed with the guidance of esteemed mentors like **Dr. Mallikarjuna Reddy HOD OF ARTIFICIAL INTELLIGENCE ANURAG UNIVERSITY,HYDERABAD**, highlighting academic and technological collaboration.

    **Our Technology Stack**
    - **Programming**: Python for robust computations.
    - **Frameworks and Libraries**:
        - **Streamlit**: For interactive web interfaces.
        - **Facebook Prophet**: For time-series forecasting.
        - **Plotly**: For interactive data visualizations.
        - **Scikit-learn**: For statistical evaluation metrics.
    - **Data Source**: Real-time data from **Yahoo Finance API**.

    **Disclaimer**
    - This app is intended for **educational purposes** only.
    - It should not be considered financial advice. Always consult with a financial advisor before making investment decisions.
    """)


elif page == "Contact Us":
    st.title("Contact Us")
    st.write("""
    We'd love to hear from you! 
    
    **Email:** 22eg107a01@anurag.edu.in  
    **Address:** Anurag University, Hyderabad  
    """)
