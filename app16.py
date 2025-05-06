import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

st.set_page_config(page_title="Tabular Data Deep Learning", layout="wide")
st.title("🤖 Deep Learning on Tabular Data")

# 1. Upload dataset
st.sidebar.header("Step 1: Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # 2. Select features and target
    st.sidebar.header("Step 2: Choose Variables")
    all_columns = df.columns.tolist()
    target_col = st.sidebar.selectbox("Select Dependent (Target) Variable", all_columns)
    feature_cols = st.sidebar.multiselect("Select Independent Variables", [col for col in all_columns if col != target_col])

    if target_col and feature_cols:
        # 3. Choose split ratios
        st.sidebar.header("Step 3: Data Split Ratio")
        train_size = st.sidebar.slider("Training Set %", 0.1, 0.9, 0.7)
        val_size = st.sidebar.slider("Validation Set %", 0.05, 0.4, 0.15)
        test_size = 1.0 - train_size - val_size

        # 4. Choose algorithm/model (for simplicity, we'll vary layer size here)
        st.sidebar.header("Step 4: Choose Model Depth")
        num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
        units_per_layer = st.sidebar.slider("Neurons per Layer", 4, 128, 16)

        if st.sidebar.button("Train Model"):
            X = df[feature_cols].values
            y = df[target_col].values

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train/val/test split
            X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, train_size=train_size, random_state=42)
            relative_val_size = val_size / (val_size + test_size)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-relative_val_size, random_state=42)

            # Build model
            model = Sequential()
            model.add(Dense(units_per_layer, activation='relu', input_shape=(X.shape[1],)))
            for _ in range(num_layers - 1):
                model.add(Dense(units_per_layer, activation='relu'))
            model.add(Dense(1, activation='linear' if np.issubdtype(y.dtype, np.number) else 'sigmoid'))
            model.compile(optimizer='adam', loss='mse' if np.issubdtype(y.dtype, np.number) else 'binary_crossentropy', metrics=['mae' if np.issubdtype(y.dtype, np.number) else 'accuracy'])

            # Train model
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=0)

            # Predict and evaluate
            y_pred = model.predict(X_test).flatten()
            st.subheader("📊 Model Evaluation on Test Set")
            if np.issubdtype(y.dtype, np.number):
                st.write("**Mean Squared Error:**", mean_squared_error(y_test, y_pred))
                st.write("**R² Score:**", r2_score(y_test, y_pred))
            else:
                st.write("**Accuracy:**", accuracy_score(y_test, np.round(y_pred)))

            # Display results
            st.write("### 📈 Loss & Metric over Epochs")
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].plot(history.history['loss'], label='Train Loss')
            ax[0].plot(history.history['val_loss'], label='Val Loss')
            ax[0].legend()
            ax[0].set_title("Loss Curve")

            metric_key = 'mae' if 'mae' in history.history else 'accuracy'
            ax[1].plot(history.history[metric_key], label='Train Metric')
            ax[1].plot(history.history[f'val_{metric_key}'], label='Val Metric')
            ax[1].legend()
            ax[1].set_title("Metric Curve")
            st.pyplot(fig)
