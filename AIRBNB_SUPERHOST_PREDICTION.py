import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Airbnb Superhost Prediction",
    page_icon="🏠",
    layout="wide"
)

# Judul aplikasi
st.title("🏠 Airbnb Superhost Prediction & Data Analysis")

# Sidebar untuk upload file dan navigasi
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file", 
    type=['csv'],
    help="Upload file CSV dengan data Airbnb"
)

# Info tentang format data yang diharapkan
with st.sidebar.expander("ℹ️ Format Data yang Diharapkan"):
    st.markdown("""
    **Kolom yang diharapkan:**
    - `host_is_superhost`: Target variable (0/1 atau True/False)
    - `review_scores_*`: Review scores (numerik)
    - `host_response_time`: Response time host
    - `price` atau `monthly_price`: Harga listing
    - `property_type`: Tipe properti
    - `neighbourhood_*`: Info lokasi/neighborhood
    - Dan kolom lainnya...
    
    **Format:**
    - File CSV dengan header
    - Encoding: UTF-8
    - Separator: koma (,)
    """)

st.sidebar.markdown("---")
st.sidebar.header("📍 Navigasi")
page = st.sidebar.radio("Pilih halaman:", [
    "📊 Data Overview", 
    "🔍 Exploratory Data Analysis", 
    "🤖 Model Prediction",
    "📈 Model Evaluation", 
    "📌 Tentang Project"
])

@st.cache_data
def load_uploaded_data(uploaded_file):
    """Load data dari file yang diupload"""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data
def generate_sample_data():
    """Generate sample data jika file tidak tersedia"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'host_is_superhost': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'review_scores_cleanliness': np.random.normal(4.5, 0.5, n_samples),
        'review_scores_location': np.random.normal(4.3, 0.6, n_samples),
        'review_scores_communication': np.random.normal(4.6, 0.4, n_samples),
        'review_scores_checkin': np.random.normal(4.4, 0.5, n_samples),
        'review_scores_accuracy': np.random.normal(4.5, 0.4, n_samples),
        'host_response_time': np.random.choice(['within an hour', 'within a few hours', 'within a day', 'a few days or more'], n_samples),
        'price': np.random.lognormal(4.5, 0.8, n_samples),
        'monthly_price': np.random.lognormal(6, 0.5, n_samples),
        'require_guest_phone_verification': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'host_identity_verified': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'property_type': np.random.choice(['Apartment', 'House', 'Condominium', 'Townhouse', 'Loft'], n_samples),
        'neighbourhood_cleansed': np.random.choice(['Capitol Hill', 'Belltown', 'Queen Anne', 'Fremont', 'Ballard'], n_samples),
        'amenities_count': np.random.poisson(15, n_samples),
        'accommodates': np.random.randint(1, 8, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples),
        'bedrooms': np.random.randint(1, 5, n_samples),
        'beds': np.random.randint(1, 6, n_samples),
        'minimum_nights': np.random.choice([1, 2, 3, 7, 30], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'availability_365': np.random.randint(0, 366, n_samples)
    }
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocessing sederhana untuk data"""
    df = df.copy()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0] if len(df[categorical_cols].mode()) > 0 else 'Unknown')
    
    return df

def prepare_features_for_ml(df, target_col='host_is_superhost'):
    """Persiapkan features untuk machine learning"""
    if target_col not in df.columns:
        return None, None, "Target column tidak ditemukan"
    
    # Pisahkan features dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert boolean target ke numeric jika perlu
    if y.dtype == 'bool':
        y = y.astype(int)
    elif y.dtype == 'object':
        if set(y.unique()).issubset({'True', 'False', 'true', 'false'}):
            y = y.map({'True': 1, 'true': 1, 'False': 0, 'false': 0})
        else:
            le = LabelEncoder()
            y = le.fit_transform(y)
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X, y, None

# Load data
if uploaded_file is not None:
    df, error = load_uploaded_data(uploaded_file)
    if error:
        st.error(f"Error loading file: {error}")
        df = generate_sample_data()
        st.info("Menggunakan data sampel untuk demonstrasi.")
    else:
        st.success(f"✅ Data berhasil diupload! ({len(df)} baris, {len(df.columns)} kolom)")
        df = preprocess_data(df)
else:
    st.info("📤 Upload file CSV untuk mulai analisis, atau gunakan data sampel di bawah.")
    df = generate_sample_data()

# Halaman Data Overview
if page == "📊 Data Overview":
    st.header("📊 Overview Dataset")
    
    if df.empty:
        st.warning("Data tidak tersedia.")
    else:
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", len(df))
        with col2:
            st.metric("📈 Total Features", len(df.columns))
        with col3:
            missing_values = df.isnull().sum().sum()
            st.metric("❓ Missing Values", missing_values)
        with col4:
            if "host_is_superhost" in df.columns:
                superhost_pct = (df["host_is_superhost"].sum() / len(df) * 100)
                st.metric("⭐ Superhost %", f"{superhost_pct:.1f}%")
        
        # Sample data
        st.subheader("🔍 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data types
        st.subheader("📋 Informasi Kolom")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Basic statistics
        st.subheader("📊 Statistik Deskriptif")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

# Halaman EDA
elif page == "🔍 Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    if df.empty:
        st.warning("Data tidak tersedia.")
    else:
        # Target variable analysis
        if "host_is_superhost" in df.columns:
            st.subheader("🎯 Analisis Target Variable")
            
            col1, col2 = st.columns(2)
            
            with col1:
                superhost_counts = df["host_is_superhost"].value_counts()
                fig = px.pie(
                    values=superhost_counts.values,
                    names=['Regular Host', 'Superhost'],
                    title="Distribusi Status Host",
                    color_discrete_sequence=['#ff7f0e', '#1f77b4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig_bar = px.bar(
                    x=['Regular Host', 'Superhost'],
                    y=superhost_counts.values,
                    title="Jumlah Host berdasarkan Status",
                    labels={'x': 'Status Host', 'y': 'Jumlah'},
                    color=['Regular Host', 'Superhost'],
                    color_discrete_sequence=['#ff7f0e', '#1f77b4']
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Analisis Korelasi")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Correlation heatmap
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation with target
            if "host_is_superhost" in numeric_cols:
                target_corr = corr_matrix["host_is_superhost"].drop("host_is_superhost").sort_values(ascending=True)
                
                fig = px.bar(
                    x=target_corr.values,
                    y=target_corr.index,
                    orientation='h',
                    title="Korelasi Features dengan Status Superhost",
                    labels={'x': 'Korelasi', 'y': 'Features'},
                    color=target_corr.values,
                    color_continuous_scale='RdBu'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribution analysis
        st.subheader("📈 Analisis Distribusi")
        
        # Select columns for analysis
        available_cols = [col for col in df.columns if col != 'host_is_superhost']
        selected_cols = st.multiselect(
            "Pilih kolom untuk analisis distribusi:",
            available_cols,
            default=available_cols[:4] if len(available_cols) >= 4 else available_cols
        )
        
        if selected_cols:
            # Numeric columns - histograms
            numeric_selected = [col for col in selected_cols if df[col].dtype in ['int64', 'float64']]
            if numeric_selected:
                st.markdown("**Distribusi Variabel Numerik:**")
                cols = st.columns(2)
                for i, col in enumerate(numeric_selected):
                    with cols[i % 2]:
                        fig = px.histogram(
                            df, x=col,
                            title=f"Distribusi {col.replace('_', ' ').title()}",
                            nbins=30,
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Categorical columns - bar charts
            categorical_selected = [col for col in selected_cols if df[col].dtype == 'object']
            if categorical_selected:
                st.markdown("**Distribusi Variabel Kategorikal:**")
                cols = st.columns(2)
                for i, col in enumerate(categorical_selected):
                    with cols[i % 2]:
                        value_counts = df[col].value_counts().head(10)
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f"Top 10 {col.replace('_', ' ').title()}",
                            labels={'x': col, 'y': 'Count'}
                        )
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)

# Halaman Model Prediction
elif page == "🤖 Model Prediction":
    st.header("🤖 Model Prediction")
    
    if df.empty:
        st.warning("Data tidak tersedia.")
    else:
        st.subheader("🔮 Prediksi Status Superhost")
        
        # Check if target column exists
        target_options = [col for col in df.columns if 'superhost' in col.lower() or 'target' in col.lower()]
        
        if not target_options:
            st.error("Target column tidak dietemukan. Pastikan dataset memiliki kolom 'host_is_superhost' atau kolom target lainnya.")
        else:
            target_col = st.selectbox("Pilih target column:", target_options, index=0)
            
            # Prepare data for ML
            X, y, error = prepare_features_for_ml(df, target_col)
            
            if error:
                st.error(f"Error preparing data: {error}")
            else:
                # Train model
                if st.button("🚀 Train Model"):
                    with st.spinner("Training model..."):
                        try:
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            
                            # Train Random Forest
                            rf_model = RandomForestClassifier(
                                n_estimators=100,
                                random_state=42,
                                max_depth=10
                            )
                            rf_model.fit(X_train, y_train)
                            
                            # Predictions
                            y_pred = rf_model.predict(X_test)
                            
                            # Metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            # Store in session state
                            st.session_state.model = rf_model
                            st.session_state.feature_names = X.columns.tolist()
                            st.session_state.metrics = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            }
                            
                            # Display results
                            st.success("✅ Model berhasil ditraining!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🎯 Accuracy", f"{accuracy:.3f}")
                            with col2:
                                st.metric("🧠 Precision", f"{precision:.3f}")
                            with col3:
                                st.metric("❤️ Recall", f"{recall:.3f}")
                            with col4:
                                st.metric("⚡ F1-Score", f"{f1:.3f}")
                            
                            # Confusion Matrix
                            cm = confusion_matrix(y_test, y_pred)
                            fig = px.imshow(
                                cm,
                                text_auto=True,
                                aspect="auto",
                                title="Confusion Matrix",
                                labels=dict(x="Predicted", y="Actual", color="Count")
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature Importance
                            importance = rf_model.feature_importances_
                            feature_imp = pd.DataFrame({
                                'feature': X.columns,
                                'importance': importance
                            }).sort_values('importance', ascending=True).tail(15)
                            
                            fig = px.bar(
                                feature_imp,
                                x='importance',
                                y='feature',
                                orientation='h',
                                title="Top 15 Feature Importance"
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                
                # Prediction interface
                if 'model' in st.session_state:
                    st.subheader("🔍 Prediksi Individual")
                    st.write("Masukkan nilai untuk melakukan prediksi:")
                    
                    # Create input form
                    with st.form("prediction_form"):
                        input_data = {}
                        
                        # Create inputs for top features
                        important_features = st.session_state.feature_names[:10]  # Top 10 features
                        
                        cols = st.columns(2)
                        for i, feature in enumerate(important_features):
                            with cols[i % 2]:
                                if feature in df.columns:
                                    if df[feature].dtype in ['int64', 'float64']:
                                        min_val = float(df[feature].min())
                                        max_val = float(df[feature].max())
                                        mean_val = float(df[feature].mean())
                                        input_data[feature] = st.number_input(
                                            f"{feature.replace('_', ' ').title()}",
                                            min_value=min_val,
                                            max_value=max_val,
                                            value=mean_val,
                                            key=f"input_{feature}"
                                        )
                                    else:
                                        unique_values = df[feature].unique()
                                        input_data[feature] = st.selectbox(
                                            f"{feature.replace('_', ' ').title()}",
                                            unique_values,
                                            key=f"input_{feature}"
                                        )
                        
                        # Fill remaining features with median/mode
                        for feature in st.session_state.feature_names:
                            if feature not in input_data:
                                if feature in df.columns:
                                    if df[feature].dtype in ['int64', 'float64']:
                                        input_data[feature] = df[feature].median()
                                    else:
                                        input_data[feature] = df[feature].mode().iloc[0] if len(df[feature].mode()) > 0 else 0
                                else:
                                    input_data[feature] = 0
                        
                        if st.form_submit_button("🎯 Prediksi"):
                            try:
                                # Prepare input
                                input_df = pd.DataFrame([input_data])
                                
                                # Encode categorical if needed
                                for col in input_df.columns:
                                    if input_df[col].dtype == 'object':
                                        le = LabelEncoder()
                                        if col in df.columns:
                                            le.fit(df[col].astype(str))
                                            input_df[col] = le.transform(input_df[col].astype(str))
                                        else:
                                            input_df[col] = 0
                                
                                # Make prediction
                                prediction = st.session_state.model.predict(input_df)[0]
                                probability = st.session_state.model.predict_proba(input_df)[0]
                                
                                # Display result
                                if prediction == 1:
                                    st.success(f"🌟 **Prediksi: SUPERHOST** (Confidence: {probability[1]:.2%})")
                                else:
                                    st.info(f"👤 **Prediksi: Regular Host** (Confidence: {probability[0]:.2%})")
                                
                                # Show probability distribution
                                prob_df = pd.DataFrame({
                                    'Status': ['Regular Host', 'Superhost'],
                                    'Probability': probability
                                })
                                
                                fig = px.bar(
                                    prob_df,
                                    x='Status',
                                    y='Probability',
                                    title="Probability Distribution",
                                    color='Probability',
                                    color_continuous_scale='viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error making prediction: {str(e)}")

# Halaman Model Evaluation (default example)
elif page == "📈 Model Evaluation":
    st.header("📈 Model Evaluation Example")
    
    st.markdown("""
    Halaman ini menampilkan contoh evaluasi model dengan berbagai algoritma machine learning.
    Untuk evaluasi pada data Anda, gunakan halaman **Model Prediction**.
    """)
    
    # Model Performance Summary
    st.subheader("🏆 Hasil Evaluasi Model (Contoh)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🥇 Model Terbaik: Random Forest Classifier
        
        **Metrik Performa:**
        - 🎯 **Akurasi:** 85.2%
        - 🧠 **Precision:** 81.4%
        - ❤️ **Recall:** 80.6%
        - 🔍 **F1-Score:** 81.0%
        - 📊 **AUC Score:** 0.89
        """)
    
    with col2:
        # Example Confusion Matrix
        confusion_matrix_example = np.array([[650, 120], [95, 285]])
        
        fig = px.imshow(
            confusion_matrix_example,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix Example - Random Forest",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Regular Host', 'Superhost'],
            y=['Regular Host', 'Superhost']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance Example
    st.subheader("📊 Feature Importance Example")
    
    feature_importance = {
        'review_scores_cleanliness': 0.21,
        'review_scores_location': 0.18,
        'host_response_time': 0.15,
        'monthly_price': 0.12,
        'review_scores_communication': 0.10,
        'require_guest_phone_verification': 0.08,
        'host_identity_verified': 0.07,
        'property_type': 0.05,
        'neighbourhood_cleansed': 0.02,
        'amenities_count': 0.02,
    }
    
    imp_df = pd.DataFrame.from_dict(
        feature_importance, 
        orient="index", 
        columns=["Importance"]
    ).sort_values("Importance", ascending=True)
    
    fig = px.bar(
        x=imp_df["Importance"],
        y=imp_df.index,
        orientation='h',
        title="Top 10 Most Important Features",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=imp_df["Importance"],
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Comparison Example
    st.subheader("🔍 Perbandingan Model")
    
    model_comparison = {
        'Model': ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'Decision Tree'],
        'Accuracy': [0.852, 0.841, 0.798, 0.785, 0.756],
        'Precision': [0.814, 0.808, 0.752, 0.741, 0.698],
        'Recall': [0.806, 0.795, 0.743, 0.728, 0.701],
        'F1-Score': [0.810, 0.801, 0.747, 0.734, 0.699]
    }
    
    comparison_df = pd.DataFrame(model_comparison)
    
    fig = px.bar(
        comparison_df,
        x='Model',
        y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        title="Perbandingan Metrik Performa Model",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

# Halaman Tentang Project
elif page == "📌 Tentang Project":
    st.header("📌 Tentang Proyek")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Airbnb Superhost Prediction & Analysis Platform
        
        ### 👥 Tim Data & Analisis:
        - **Daniel Andrew Siahaan**
        - **Raihan**
        - **Bunga Lestari**
        
        ### 🎯 Tujuan Platform:
        Platform ini memungkinkan Anda untuk:
        1. **Upload dataset Airbnb** dalam format CSV
        2. **Analisis data** secara mendalam dengan visualisasi interaktif
        3. **Training model ML** untuk prediksi status Superhost
        4. **Prediksi individual** berdasarkan karakteristik listing
        
        ### 🔬 Fitur Utama:
        - **📊 Data Overview**: Statistik dan informasi dasar dataset
        - **🔍 EDA**: Analisis eksploratori dengan berbagai visualisasi
        - **🤖 Model Prediction**: Training dan prediksi dengan Random Forest
        - **📈 Model Evaluation**: Contoh evaluasi model komprehensif
        
        ### 💡 Cara Penggunaan:
        1. Upload file CSV Anda di sidebar
        2. Eksplorasi data melalui halaman EDA
        3. Train model di halaman Model Prediction
        4. Lakukan prediksi individual
        """)
    
    with col2:
        st.markdown("""
        ### 🛠 Teknologi:
        
        **Data Science:**
        - 🐍 Python
        - 🐼 Pandas
        - 📊 NumPy
        - 🤖 Scikit-Learn
        
        **Visualization:**
        - 📊 Plotly
        - 🎨 Streamlit
        
        **Machine Learning:**
        - 🌲 Random Forest
        - 📈 Classification Metrics
        - 🔍 Feature Importance
        """)
    
    st.markdown("---")
    
    # Upload instructions
    st.subheader("📋 Panduan Upload Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Format yang Didukung:**
        - File CSV dengan encoding UTF-8
        - Separator koma (,)
        - Header row (baris pertama berisi nama kolom)
        - Kolom target: `host_is_superhost` (boolean/binary)
        """)
    
    with col2:
        st.markdown("""
        **💡 Tips untuk Hasil Terbaik:**
        - Pastikan data sudah dibersihkan
        - Minimal 100 baris data untuk training yang baik
        - Sertakan review scores dan karakteristik host
        - Balance antara Superhost dan Regular Host
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Business Value:
    
    - **📈 Untuk Host**: Memahami faktor kunci menjadi Superhost
    - **🏢 Untuk Platform**: Insight tentang karakteristik host berkualitas
    - **👥 Untuk Guest**: Prediksi kualitas host sebelum booking
    - **📊 Untuk Analyst**: Tool analysis yang mudah digunakan
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2024 Airbnb Superhost Prediction Platform | Upload CSV & Start Analyzing! 🚀"
    "</div>", 
    unsafe_allow_html=True
)
