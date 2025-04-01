import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Import để tạo heatmap Plotly
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import io
import time # Để thêm hiệu ứng chờ, progress bar

# --- Cấu hình trang ---
st.set_page_config(
    page_title="🌈✨ ML App Siêu Cấp Vip Pro ✨🌈",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="auto" # Để sidebar tự động hoặc 'expanded'
)

# --- CSS Siêu Cấp ---
# Font Awesome CDN
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>', unsafe_allow_html=True)
# Google Fonts (Ví dụ: Montserrat)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

custom_css = """
<style>
    /* --- Font chữ --- */
    html, body, [class*="st-"], button, input, select, textarea {
        font-family: 'Montserrat', sans-serif;
    }

    /* --- Background Gradient --- */
    .stApp {
        /* background-image: linear-gradient(to right top, #6d327c, #485DA6, #00a1ba, #00BF98, #36C486); */
         background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
         background-size: 400% 400%;
         animation: gradientBG 15s ease infinite;
         color: #FFFFFF; /* Đổi màu chữ mặc định thành trắng cho dễ đọc trên nền gradient */
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* --- Headers --- */
    h1, h2, h3 {
        /* font-weight: 700; */
        /* text-shadow: 2px 2px 4px rgba(0,0,0,0.2); */
         color: #ffffff; /* Màu trắng nổi bật trên nền gradient */
         font-weight: 700; /* Chữ đậm hơn */
         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* Đổ bóng nhẹ */
    }
     h1 {
         font-size: 2.8em; /* Tăng kích thước tiêu đề chính */
         text-align: center;
         margin-bottom: 20px;
     }
     h2 { font-size: 2em; border-bottom: 2px solid rgba(255,255,255,0.5); padding-bottom: 5px; margin-top: 30px;}
     h3 { font-size: 1.5em; color: #f0f0f0;}


    /* --- Nút bấm nổi bật --- */
    .stButton>button {
        border: none; /* Bỏ viền mặc định */
        border-radius: 25px; /* Bo tròn nhiều hơn */
        padding: 12px 28px; /* Tăng padding */
        font-size: 16px;
        font-weight: 600; /* Đậm vừa */
        color: white;
        background-image: linear-gradient(to right, #fc5c7d, #6a82fb, #fc5c7d); /* Gradient cho nút */
        background-size: 200% auto; /* Kích thước gradient để tạo hiệu ứng chuyển động */
        transition: 0.5s; /* Hiệu ứng chuyển màu mượt */
        box-shadow: 0 4px 15px 0 rgba(116, 79, 168, 0.75); /* Đổ bóng */
        margin-top: 10px; /* Khoảng cách trên */
    }
    .stButton>button:hover {
        background-position: right center; /* Chuyển gradient khi hover */
        color: #fff;
        text-decoration: none;
        box-shadow: 0 6px 20px 0 rgba(116, 79, 168, 0.9); /* Bóng đậm hơn khi hover */
        transform: translateY(-2px); /* Nâng nút lên nhẹ */
    }
     .stButton>button:active {
        transform: translateY(0px); /* Hạ nút xuống khi nhấn */
        box-shadow: 0 4px 15px 0 rgba(116, 79, 168, 0.75);
     }

    /* --- Styling cho Selectbox, Multiselect, NumberInput --- */
    /* (Việc style sâu các widget của Streamlit có thể phức tạp và dễ bị lỗi khi cập nhật) */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stMultiSelect"] > div {
        background-color: rgba(255, 255, 255, 0.1); /* Nền hơi trong suốt */
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
     div[data-baseweb="select"] > div:hover,
     div[data-baseweb="input"] > div:hover,
     div[data-testid="stMultiSelect"] > div:hover {
         border: 1px solid rgba(255, 255, 255, 0.7);
     }
    /* Màu chữ cho input/select */
     /* .stTextInput input, .stNumberInput input, div[data-baseweb="select"] { color: #FFFFFF !important; } */


    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px; /* Khoảng cách lớn hơn giữa các tab */
        justify-content: center; /* Căn giữa các tab */
        border-bottom: 2px solid rgba(255,255,255,0.2); /* Đường kẻ dưới tab list */
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px 10px 0 0; /* Bo góc trên */
        gap: 10px;
        padding: 15px 25px;
        color: rgba(255, 255, 255, 0.7); /* Màu chữ tab không active nhạt hơn */
        font-weight: 600;
        font-size: 1.1em;
        transition: all 0.3s ease;
        border: none; /* Bỏ viền mặc định */
    }
    .stTabs [aria-selected="true"] {
        background-image: linear-gradient(to top, rgba(255,255,255,0.15), rgba(255,255,255,0.0)); /* Gradient nhẹ cho tab active */
        color: white !important; /* Màu chữ trắng rõ ràng */
        border-bottom: 3px solid #FFFFFF; /* Đường kẻ dưới tab active */
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }

    /* --- Card Style cho Expander --- */
    .stExpander {
        border: none; /* Bỏ viền mặc định */
        background-color: rgba(255, 255, 255, 0.1); /* Nền mờ */
        backdrop-filter: blur(5px); /* Hiệu ứng kính mờ */
        border-radius: 10px; /* Bo góc */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px; /* Khoảng cách dưới */
        overflow: hidden; /* Đảm bảo nội dung không tràn ra ngoài */
    }
    .stExpander header {
        font-weight: 600;
        color: #FFFFFF; /* Chữ tiêu đề expander màu trắng */
        background-color: rgba(0, 0, 0, 0.1); /* Nền nhẹ cho header expander */
        padding: 10px 15px !important;
        border-radius: 10px 10px 0 0;
    }
     .stExpander header:hover {
         background-color: rgba(0, 0, 0, 0.2);
     }
    .stExpander > div[role="button"] > div > svg {
         fill: #FFFFFF; /* Màu icon mũi tên trắng */
     }
     .streamlit-expanderContent {
         padding: 15px; /* Padding cho nội dung bên trong expander */
         color: #e0e0e0; /* Màu chữ nội dung bên trong */
     }

    /* --- Styling cho DataFrame --- */
     .stDataFrame {
         border: 1px solid rgba(255, 255, 255, 0.2);
         border-radius: 8px;
         /* background-color: rgba(0, 0, 0, 0.1); */
     }
     /* Header của dataframe */
     /* .stDataFrame th {
         background-color: rgba(255, 255, 255, 0.15);
         color: white;
         font-weight: bold;
     } */
     /* Cell của dataframe */
     /* .stDataFrame td {
         color: #e0e0e0;
     } */


    /* --- Metrics --- */
     div[data-testid="stMetric"] {
         background-color: rgba(255, 255, 255, 0.1);
         border: 1px solid rgba(255, 255, 255, 0.3);
         padding: 15px 20px;
         border-radius: 10px;
         box-shadow: 0 4px 10px rgba(0,0,0,0.1);
         text-align: center;
     }
     div[data-testid="stMetricLabel"] {
         font-weight: bold;
         color: #f0f0f0; /* Màu label */
     }
     div[data-testid="stMetricValue"] {
         font-size: 2.2em; /* Giá trị to hơn */
         font-weight: 700;
         color: #FFFFFF; /* Giá trị màu trắng */
     }
     div[data-testid="stMetricDelta"] { /* Chỉ số thay đổi (delta) */
        font-weight: normal;
        color: #a0a4b8 !important;
     }
     div[data-testid="stMetric"] > div > div:last-child > div { /* Phần caption (ghi chú) */
         color: #cccccc !important;
         font-style: italic;
     }

    /* --- Input Forms --- */
     div[data-testid="stForm"] {
         border: 1px dashed rgba(255, 255, 255, 0.3);
         border-radius: 10px;
         padding: 20px;
         background-color: rgba(0, 0, 0, 0.1);
     }


     /* --- Sidebar (nếu dùng) --- */
     section[data-testid="stSidebar"] > div:first-child {
         background-image: linear-gradient(to bottom, #485DA6, #00a1ba); /* Gradient cho sidebar */
     }

    /* === CSS HEADER & Loại bỏ viền lạ === */

    /* Loại bỏ margin/padding/border của body để tránh viền lạ */
    body {
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        /* overflow-x: hidden; */ /* Thử thêm nếu có thanh cuộn ngang lạ */
    }

    /* Target thanh header chính */
    header[data-testid="stHeader"] {
        background-image: none !important;
        background-color: #0E1117 !important; /* Giữ màu nền tối */
        border: none !important; /* Xóa border */
        margin: 0 !important; /* Xóa margin */
        padding: 0 !important; /* Xóa padding */
        box-shadow: none !important; /* Xóa shadow nếu có */
    }

    /* Target dải màu trang trí */
    div[data-testid="stDecoration"] {
        background-image: none !important;
        background-color: #0E1117 !important; /* Đồng bộ màu nền tối */
        border: none !important; /* Xóa border */
        margin: 0 !important; /* Xóa margin */
        padding: 0 !important; /* Xóa padding */
        box-shadow: none !important; /* Xóa shadow nếu có */
        height: 5px !important; /* Đặt chiều cao cố định nhỏ cho nó, hoặc thử display: none */
        /* display: none !important; */ /* Bỏ comment để ẩn hoàn toàn nếu cần */
    }

    /* Đảm bảo các icon/link trên toolbar vẫn màu trắng */
     header [data-testid="stToolbar"] button svg,
     header [data-testid="stToolbar"] button span,
     div[data-testid="stMainMenu"] button svg,
     header [data-testid="stToolbar"] a svg,
     header [data-testid="stToolbar"] a {
         fill: #FFFFFF !important;
         color: #FFFFFF !important;
     }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Các hàm trợ giúp (Giữ nguyên logic, chỉ thay đổi plot) ---
@st.cache_data # Sử dụng cache_data cho data loading
def load_data(uploaded_file):
    """Tải dữ liệu từ tệp CSV được tải lên."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        # Trả về lỗi để có thể hiển thị trong luồng chính
        return f"Lỗi khi đọc tệp CSV: {e}"

# Hàm tiền xử lý (Giữ nguyên logic cốt lõi)
def preprocess_data(df, target_column, selected_features):
    """Tiền xử lý dữ liệu: chọn cột, xử lý NaN, one-hot encode, scale."""
    if not selected_features:
        return "Vui lòng chọn ít nhất một feature.", None, None, None, None, None, None, None # Thêm None cho imputer và features
    if not target_column:
         return "Vui lòng chọn cột target.", None, None, None, None, None, None, None
    if target_column not in df.columns:
        return f"Cột target '{target_column}' không tồn tại trong dữ liệu.", None, None, None, None, None, None, None
    if not all(feature in df.columns for feature in selected_features):
         missing_features = [f for f in selected_features if f not in df.columns]
         return f"Các feature sau không tồn tại: {', '.join(missing_features)}.", None, None, None, None, None, None, None

    df_subset = df[selected_features + [target_column]].copy()

    # Kiểm tra xem target có phải dạng số không (nếu dùng cho hồi quy)
    if df_subset[target_column].dtype not in [np.number]:
        is_binary_after_potential_get_dummies = False
        if not is_binary_after_potential_get_dummies:
            return f"Cột target '{target_column}' phải là dạng số cho mô hình hồi quy.", None, None, None, None, None, None, None


    numeric_cols = df_subset.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_subset.select_dtypes(exclude=np.number).columns.tolist()

    # Xác định cột gốc trước khi xử lý NaN
    original_numeric_features = [col for col in numeric_cols if col != target_column and col in selected_features]
    original_categorical_features = [col for col in categorical_cols if col != target_column and col in selected_features] # Chỉ lấy các cột được chọn

    # Xử lý NaN cho target trước tiên
    df_subset.dropna(subset=[target_column], inplace=True)
    if df_subset.empty:
        return "Không còn dữ liệu sau khi loại bỏ NaN ở cột target.", None, None, None, None, None, None, None

    # --- SỬA LỖI IMUTER ---
    # Khởi tạo imputers
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # FIT imputer LUÔN LUÔN trên các cột đã chọn (nếu có) để đảm bảo nó được fitted
    # Ngay cả khi không có NaN trong tập dữ liệu hiện tại, nó cần được fit để sử dụng sau này
    if original_numeric_features:
        numeric_imputer.fit(df_subset[original_numeric_features])
        # Chỉ TRANSFORM nếu thực sự có NaN
        if df_subset[original_numeric_features].isnull().any().any():
            df_subset[original_numeric_features] = numeric_imputer.transform(df_subset[original_numeric_features])
    else:
        # Nếu không có cột số nào được chọn, fit imputer trên một mảng trống để tránh lỗi NotFittedError
        # Mặc dù nó sẽ không làm gì, nhưng nó sẽ ở trạng thái "fitted"
        numeric_imputer.fit(np.empty((0, 0)))


    if original_categorical_features:
        categorical_imputer.fit(df_subset[original_categorical_features])
        # Chỉ TRANSFORM nếu thực sự có NaN
        if df_subset[original_categorical_features].isnull().any().any():
            df_subset[original_categorical_features] = categorical_imputer.transform(df_subset[original_categorical_features])
    else:
         # Fit trên mảng trống nếu không có cột category
        categorical_imputer.fit(np.empty((0, 0), dtype=object))
    # --- KẾT THÚC SỬA LỖI ---


    # One-Hot Encoding cho các cột category gốc ĐÃ CHỌN
    df_processed = pd.get_dummies(df_subset, columns=original_categorical_features, drop_first=True)

    # Tách X, y sau khi encoding
    if target_column in original_categorical_features:
        st.warning(f"Cột target '{target_column}' là dạng category. Đang cố gắng tìm cột sau one-hot encoding.")
        possible_target_cols = [col for col in df_processed.columns if col.startswith(target_column + "_")]
        if len(possible_target_cols) > 0:
             target_column_processed = possible_target_cols[0]
             st.info(f"Sử dụng '{target_column_processed}' làm target sau encoding.")
             y = df_processed[target_column_processed]
             cols_to_drop = [target_column] + possible_target_cols
             X = df_processed.drop(columns=cols_to_drop, errors='ignore')
        else:
             return f"Không thể xác định cột target '{target_column}' sau khi mã hóa.", None, None, None, None, None, None, None
    elif target_column in df_processed.columns:
        y = df_processed[target_column]
        X = df_processed.drop(target_column, axis=1)
    else:
         return f"Không tìm thấy cột target '{target_column}' trong dữ liệu đã xử lý.", None, None, None, None, None, None, None


    feature_names_processed = X.columns.tolist() # Lấy tên các cột feature sau khi xử lý

    if X.empty or len(feature_names_processed) == 0:
        # Trả về đủ các giá trị None
        return "Không còn feature nào sau khi tiền xử lý.", None, None, None, numeric_imputer, categorical_imputer, original_numeric_features, original_categorical_features

    # Chuẩn hóa chỉ các cột features (X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Trả về các đối tượng đã được FIT (scaler, imputers)
    return X_scaled, y, feature_names_processed, scaler, numeric_imputer, categorical_imputer, original_numeric_features, original_categorical_features


# Hàm huấn luyện (Thêm progress bar)
def train_model(X_train, y_train, model_name, params):
    """Huấn luyện mô hình đã chọn với các tham số."""
    model = None
    model_display_name = model_name # Giữ tên gốc để hiển thị

    try:
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest Regressor":
            # Lấy tham số với giá trị mặc định an toàn
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', None)
            model = RandomForestRegressor(
                n_estimators=n_estimators if n_estimators > 0 else 100, # Đảm bảo > 0
                max_depth=max_depth if max_depth is None or max_depth >= 0 else None, # None hoặc >= 0
                random_state=42,
                n_jobs=-1
            )
        elif model_name == "Support Vector Regressor (SVR)":
             # Lấy tham số với giá trị mặc định an toàn
            C = params.get('C', 1.0)
            epsilon = params.get('epsilon', 0.1)
            kernel = params.get('kernel', 'rbf')
            model = SVR(
                C=C if C > 0 else 1.0, # Đảm bảo > 0
                epsilon=epsilon if epsilon > 0 else 0.1, # Đảm bảo > 0
                kernel=kernel if kernel in ['rbf', 'linear', 'poly', 'sigmoid'] else 'rbf'
            )
        else:
            st.error(f"Tên mô hình không hợp lệ: {model_name}")
            return None, "Mô hình không hợp lệ"

        # --- Thêm Progress Bar ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"⏳ Bắt đầu huấn luyện {model_display_name}...")

        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        # Cập nhật progress bar và status
        for i in range(100):
            time.sleep(0.005) # Giả lập thời gian xử lý nhỏ
            progress_bar.progress(i + 1)

        status_text.success(f"✅ Hoàn thành huấn luyện {model_display_name} trong {training_time:.2f} giây!")
        time.sleep(1) # Chờ 1 giây để user thấy thông báo success
        status_text.empty() # Xóa thông báo text
        progress_bar.empty() # Xóa progress bar

        return model, None # Trả về model và không có lỗi

    except Exception as e:
        error_message = f"Lỗi trong quá trình huấn luyện mô hình {model_display_name}: {e}"
        st.error(error_message)
        # Đảm bảo xóa progress bar và status text nếu lỗi xảy ra giữa chừng
        if 'progress_bar' in locals() and progress_bar is not None: progress_bar.empty()
        if 'status_text' in locals() and status_text is not None: status_text.empty()
        return None, error_message # Trả về không có model và thông báo lỗi

# --- Khởi tạo Session State (Thêm các mục cần thiết) ---
if 'model' not in st.session_state: st.session_state.model = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'numeric_imputer' not in st.session_state: st.session_state.numeric_imputer = None
if 'categorical_imputer' not in st.session_state: st.session_state.categorical_imputer = None
if 'feature_names_processed' not in st.session_state: st.session_state.feature_names_processed = None # Tên feature SAU tiền xử lý
if 'target_column' not in st.session_state: st.session_state.target_column = None
if 'numeric_cols_original' not in st.session_state: st.session_state.numeric_cols_original = None # Tên feature số GỐC
if 'categorical_cols_original' not in st.session_state: st.session_state.categorical_cols_original = None # Tên feature category GỐC
if 'df_loaded' not in st.session_state: st.session_state.df_loaded = False
if 'df' not in st.session_state: st.session_state.df = None # DataFrame gốc
if 'uploaded_filename' not in st.session_state: st.session_state.uploaded_filename = None
if 'preprocessing_error' not in st.session_state: st.session_state.preprocessing_error = None # Lưu lỗi tiền xử lý
if 'training_error' not in st.session_state: st.session_state.training_error = None # Lưu lỗi huấn luyện
if 'last_prediction' not in st.session_state: st.session_state.last_prediction = None # Lưu kết quả dự đoán cuối

# --- Giao diện người dùng với Tabs và Style Mới ---
st.title("🚀 Ứng dụng Machine Learning Siêu Cấp Vip Pro 🚀")
st.markdown("---", unsafe_allow_html=True) # Dùng HTML để đường kẻ có thể được style nếu muốn

# Loại bỏ thẻ <i> khỏi tên tab, có thể dùng emoji thay thế
tab1, tab2, tab3 = st.tabs([
    "**📊 Tải & Khám phá Dữ liệu**", # Thay icon bằng emoji hoặc bỏ đi
    "**⚙️ Huấn luyện Mô hình**",
    "**✨ Dự đoán Kết quả**"
])

# == Tab 1: Tải dữ liệu và EDA ==
with tab1:
    upload_container = st.container()
    with upload_container:
        st.markdown("### <i class='fas fa-file-csv'></i> 1. Tải lên tệp CSV của bạn", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Kéo thả hoặc chọn tệp...",
            type="csv", # Chỉ chấp nhận file .csv
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        # Chỉ load lại nếu chưa load hoặc tên file khác
        if not st.session_state.df_loaded or st.session_state.uploaded_filename != uploaded_file.name:
            with st.spinner("🔄 Đang tải và kiểm tra tệp..."):
                # Reset trạng thái trước khi load file mới
                st.session_state.df = None
                st.session_state.df_loaded = False
                st.session_state.model = None
                st.session_state.scaler = None
                st.session_state.numeric_imputer = None
                st.session_state.categorical_imputer = None
                st.session_state.feature_names_processed = None
                st.session_state.numeric_cols_original = None
                st.session_state.categorical_cols_original = None
                st.session_state.target_column = None
                st.session_state.preprocessing_error = None
                st.session_state.training_error = None
                st.session_state.last_prediction = None
                st.session_state.uploaded_filename = uploaded_file.name

                result = load_data(uploaded_file)
                if isinstance(result, pd.DataFrame):
                    st.session_state.df = result
                    st.session_state.df_loaded = True
                    st.success(f"✅ Đã tải thành công tệp: **{st.session_state.uploaded_filename}**")
                    # Không cần rerun ở đây nữa, sẽ tự chạy xuống dưới
                else: # Nếu load_data trả về lỗi (string)
                    st.error(result) # Hiển thị lỗi load_data
                    st.session_state.uploaded_filename = None # Reset tên file nếu lỗi
                    st.session_state.df_loaded = False
            # st.rerun() # Không cần rerun, để code chạy tiếp xuống phần EDA nếu thành công

    # Chỉ hiển thị EDA nếu đã load thành công
    if st.session_state.df_loaded and st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("### <i class='fas fa-search'></i> 2. Khám phá Dữ liệu (EDA)", unsafe_allow_html=True)

        with st.expander("📊 Xem trước dữ liệu (5 dòng đầu)", expanded=False):
            st.dataframe(df.head())

        col_info1, col_info2, col_info3 = st.columns([1,1,1])
        with col_info1:
            with st.expander("ℹ️ Thông tin chung", expanded=False):
                st.write(f"**Dòng:** {df.shape[0]}, **Cột:** {df.shape[1]}")
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.code(s, language=None) # language=None để không cố highlight cú pháp
        with col_info2:
            with st.expander("🔢 Thống kê mô tả (Số)", expanded=False):
                try:
                    st.dataframe(df.describe(include=np.number))
                except Exception:
                    st.info("Không có cột số.")
        with col_info3:
            with st.expander("❓ Giá trị thiếu (NaN)", expanded=False):
                missing_values = df.isnull().sum()
                missing_df = pd.DataFrame({'Cột': missing_values.index, 'Số lượng NaN': missing_values.values})
                missing_df = missing_df[missing_df['Số lượng NaN'] > 0].sort_values(by='Số lượng NaN', ascending=False)
                if not missing_df.empty:
                    st.dataframe(missing_df)
                    st.warning(f"Tìm thấy **{missing_df['Số lượng NaN'].sum()}** giá trị thiếu.")
                else:
                    st.success("🎉 Không có giá trị thiếu!")

        st.markdown("### <i class='fas fa-chart-bar'></i> 3. Trực quan hóa Dữ liệu", unsafe_allow_html=True)
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

        if not numeric_columns and not categorical_columns:
             st.warning("⚠️ Không có cột dữ liệu nào để vẽ biểu đồ.")
        else:
            col_plot_select, col_plot_display = st.columns([0.3, 0.7], gap="large")

            with col_plot_select:
                plot_options = []
                if numeric_columns:
                    plot_options.extend(["Histogram", "Box Plot", "Heatmap Tương quan"])
                if len(numeric_columns) >= 2:
                     plot_options.append("Scatter Plot")
                # Thêm các plot cho category nếu muốn (ví dụ: Count Plot)
                # if categorical_columns:
                #     plot_options.append("Count Plot")

                if not plot_options:
                     st.info("Không đủ loại cột để vẽ biểu đồ.")
                else:
                    plot_type = st.selectbox(
                        "Chọn loại biểu đồ:",
                        plot_options,
                        key="plot_type_select"
                    )

                    # Chọn cột cho từng loại plot
                    selected_col_hist_box = None
                    selected_col_scatter_x = None
                    selected_col_scatter_y = None
                    selected_col_scatter_hue = None

                    if plot_type in ["Histogram", "Box Plot"] and numeric_columns:
                        selected_col_hist_box = st.selectbox("Chọn cột số:", numeric_columns, key="hist_box_col")
                    elif plot_type == "Scatter Plot" and len(numeric_columns) >= 2:
                        selected_col_scatter_x = st.selectbox("Chọn cột X:", numeric_columns, key="scatter_x", index=0)
                        col2_options = [col for col in numeric_columns if col != selected_col_scatter_x]
                        if col2_options: # Đảm bảo có cột Y để chọn
                             selected_col_scatter_y = st.selectbox("Chọn cột Y:", col2_options, key="scatter_y", index=min(1, len(col2_options)-1))
                             # Cho phép chọn cột category hoặc numeric để tô màu
                             hue_options = [None] + numeric_columns + categorical_columns
                             # Loại bỏ các cột đã chọn cho X, Y khỏi hue options nếu chúng tồn tại
                             hue_options = [opt for opt in hue_options if opt != selected_col_scatter_x and opt != selected_col_scatter_y]
                             selected_col_scatter_hue = st.selectbox("Phân màu theo (Tùy chọn):", hue_options, key="scatter_hue", index=0) # Mặc định là None
                        else:
                             st.warning("Cần ít nhất 2 cột số khác nhau.")
                             plot_type = None # Vô hiệu hóa plot nếu không đủ cột
                    # Heatmap không cần chọn cột thêm

            with col_plot_display:
                # Tạo và hiển thị biểu đồ Plotly
                try: # Bọc trong try-except để bắt lỗi vẽ biểu đồ
                    if plot_type == "Histogram" and selected_col_hist_box:
                        fig = px.histogram(
                            df,
                            x=selected_col_hist_box,
                            marginal="box", # Thêm box plot ở trên
                            title=f"<b>Phân phối của {selected_col_hist_box}</b>",
                            opacity=0.8,
                            color_discrete_sequence=px.colors.qualitative.Vivid # Bảng màu sặc sỡ
                        )
                        fig.update_layout(bargap=0.1, title_x=0.5, xaxis_title=selected_col_hist_box, yaxis_title="Số lượng")
                        st.plotly_chart(fig, use_container_width=True)

                    elif plot_type == "Box Plot" and selected_col_hist_box:
                        fig = px.box(
                            df,
                            y=selected_col_hist_box,
                            title=f"<b>Biểu đồ Box Plot của {selected_col_hist_box}</b>",
                            points="all", # Hiển thị tất cả các điểm
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_layout(title_x=0.5)
                        st.plotly_chart(fig, use_container_width=True)

                    elif plot_type == "Scatter Plot" and selected_col_scatter_x and selected_col_scatter_y:
                        fig = px.scatter(
                            df,
                            x=selected_col_scatter_x,
                            y=selected_col_scatter_y,
                            color=selected_col_scatter_hue,
                            title=f"<b>Mối quan hệ giữa {selected_col_scatter_x} và {selected_col_scatter_y}</b>",
                            opacity=0.7,
                            color_continuous_scale=px.colors.sequential.Viridis if selected_col_scatter_hue and df[selected_col_scatter_hue].dtype in [np.int64, np.float64] else None, # Scale màu nếu cột màu là số
                            color_discrete_sequence=px.colors.qualitative.Bold if selected_col_scatter_hue and df[selected_col_scatter_hue].dtype not in [np.int64, np.float64] else px.colors.qualitative.Plotly # Scale màu nếu cột màu là category
                        )
                        fig.update_layout(title_x=0.5, xaxis_title=selected_col_scatter_x, yaxis_title=selected_col_scatter_y)
                        st.plotly_chart(fig, use_container_width=True)

                    elif plot_type == "Heatmap Tương quan":
                        if len(numeric_columns) > 1:
                            # Chỉ tính corr trên cột số, xử lý NaN trước khi tính corr
                            corr = df[numeric_columns].dropna().corr()
                            fig = go.Figure(data=go.Heatmap(
                                    z=corr.values,
                                    x=corr.columns,
                                    y=corr.columns,
                                    colorscale='RdBu', # Bảng màu đỏ-xanh dương
                                    zmin=-1, zmax=1, # Chuẩn hóa thang màu
                                    text=corr.values, # Hiển thị giá trị
                                    texttemplate="%{text:.2f}", # Định dạng giá trị
                                    hoverongaps = False))
                            fig.update_layout(
                                title='<b>Ma trận Tương quan Heatmap (Cột số)</b>',
                                xaxis_tickangle=-45,
                                title_x=0.5,
                                height=max(400, len(numeric_columns) * 30) # Điều chỉnh chiều cao
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Giá trị từ -1 đến 1. Gần 1: Tương quan dương mạnh, Gần -1: Tương quan âm mạnh, Gần 0: Ít tương quan tuyến tính.")
                        else:
                             st.warning("Cần ít nhất 2 cột số để vẽ heatmap tương quan.")
                except Exception as plot_error:
                     st.error(f"Lỗi khi vẽ biểu đồ: {plot_error}")


    elif not uploaded_file: # Chỉ hiển thị nếu chưa upload file gì
        st.info("👋 Chào mừng! Hãy tải lên một tệp CSV để bắt đầu hành trình khám phá dữ liệu và xây dựng mô hình.")
        # Có thể thêm Lottie animation ở đây
        # try:
        #     import requests
        #     from streamlit_lottie import st_lottie
        #     # Ví dụ: animation từ lottiefiles.com
        #     # response = requests.get("URL_TO_LOTTIE_JSON")
        #     # lottie_json = response.json()
        #     # st_lottie(lottie_json, speed=1, reverse=False, loop=True, quality="low", height=300, key="lottie_hello")
        # except ImportError:
        #      st.write("Cài đặt streamlit-lottie để hiển thị animation: pip install streamlit-lottie")
        # except Exception as e:
        #      st.write(f"Không thể tải animation: {e}")

# == Tab 2: Huấn luyện Mô hình ==
with tab2:
    st.markdown("### <i class='fas fa-tasks'></i> 3. Chọn Features, Target & Mô hình", unsafe_allow_html=True)

    if st.session_state.df_loaded and st.session_state.df is not None:
        df = st.session_state.df

        col1_setup, col2_setup = st.columns(2, gap="large")

        with col1_setup:
            st.markdown("#### <i class='fas fa-bullseye'></i> Features & Target", unsafe_allow_html=True)
            all_columns = df.columns.tolist()
            # Cho phép chọn bất kỳ cột số nào làm target
            potential_target_cols = df.select_dtypes(include=np.number).columns.tolist()

            if not potential_target_cols:
                st.error("⛔ Không tìm thấy cột số nào trong dữ liệu để làm biến mục tiêu (Target) cho mô hình hồi quy.")
                # st.stop() # Không dừng hẳn, cho phép user quay lại tải file khác
            else:
                # Khôi phục lựa chọn cũ nếu có
                target_index = 0
                if st.session_state.target_column and st.session_state.target_column in potential_target_cols:
                    target_index = potential_target_cols.index(st.session_state.target_column)

                selected_target = st.selectbox(
                    "🎯 Chọn cột mục tiêu (Target - phải là số):",
                     potential_target_cols,
                     index=target_index,
                     key="target_select"
                 )
                available_features = [col for col in all_columns if col != selected_target]

                # Khôi phục lựa chọn feature cũ nếu có (cả số và category)
                default_features = []
                if st.session_state.numeric_cols_original or st.session_state.categorical_cols_original:
                     default_features = [f for f in (st.session_state.numeric_cols_original or []) + (st.session_state.categorical_cols_original or []) if f in available_features]
                else: # Nếu chưa có gì thì mặc định chọn tất cả trừ target
                     default_features = available_features


                selected_features_original = st.multiselect(
                    "✨ Chọn các cột đặc trưng (Features):",
                    available_features,
                    default=default_features,
                    key="feature_select"
                )

        with col2_setup:
            st.markdown("#### <i class='fas fa-robot'></i> Thuật toán & Tham số", unsafe_allow_html=True)
            model_options = ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor (SVR)"]
            selected_model_name = st.selectbox("🤖 Chọn thuật toán:", model_options, key="model_select")

            # Sử dụng expander cho siêu tham số
            with st.expander(f"🛠️ Tinh chỉnh siêu tham số cho {selected_model_name}", expanded=False):
                params = {}
                if selected_model_name == "Random Forest Regressor":
                    params['n_estimators'] = st.slider("Số cây (n_estimators):", 10, 1000, 100, 10, key="rf_n_estimators", help="Số lượng cây trong rừng.") # Giảm min
                    max_depth_input = st.number_input("Độ sâu tối đa (max_depth, 0=không giới hạn):", min_value=0, value=0, step=1, key="rf_max_depth", help="Độ sâu tối đa của mỗi cây. 0 nghĩa là không giới hạn.")
                    params['max_depth'] = None if max_depth_input == 0 else max_depth_input
                elif selected_model_name == "Support Vector Regressor (SVR)":
                    params['C'] = st.slider("Tham số C (Regularization):", 0.01, 100.0, 1.0, 0.01, format="%.2f", key="svr_c", help="Nghịch đảo của cường độ điều chuẩn. Giá trị nhỏ hơn tương ứng với điều chuẩn mạnh hơn.")
                    params['epsilon'] = st.slider("Epsilon:", 0.01, 1.0, 0.1, 0.01, format="%.2f", key="svr_epsilon", help="Xác định biên độ mà không có hình phạt nào được liên kết trong hàm mất mát.")
                    params['kernel'] = st.radio("Kernel:", ['rbf', 'linear', 'poly', 'sigmoid'], index=0, key="svr_kernel", horizontal=True, help="Loại kernel được sử dụng trong thuật toán.")
                else:
                    st.info("Linear Regression cơ bản không yêu cầu tinh chỉnh siêu tham số phức tạp.")

        st.markdown("---")

        # Đặt nút huấn luyện ở giữa
        col_button_spacer1, col_button, col_button_spacer2 = st.columns([1, 1.5, 1])
        with col_button:
            if st.button("⚡️ Huấn luyện Mô hình Ngay! ⚡️", key="train_button_main", use_container_width=True):
                # Reset lỗi cũ trước khi huấn luyện
                st.session_state.preprocessing_error = None
                st.session_state.training_error = None
                st.session_state.model = None # Reset model cũ

                if not selected_features_original:
                    st.warning("⚠️ Vui lòng chọn ít nhất một Feature.")
                elif not selected_target:
                    st.warning("⚠️ Vui lòng chọn Target.")
                else:
                    with st.spinner("⚙️ Đang tiền xử lý dữ liệu..."):
                        # Gọi hàm tiền xử lý đã sửa
                        preprocess_result = preprocess_data(df.copy(), selected_target, selected_features_original)

                        # Kiểm tra kết quả tiền xử lý
                        if isinstance(preprocess_result[0], str): # Nếu phần tử đầu là string -> lỗi
                            st.session_state.preprocessing_error = preprocess_result[0]
                            st.error(f"Lỗi tiền xử lý: {st.session_state.preprocessing_error}")
                             # Đảm bảo reset các state khác nếu lỗi xảy ra ở đây
                            st.session_state.scaler = None
                            st.session_state.numeric_imputer = None
                            st.session_state.categorical_imputer = None
                            st.session_state.feature_names_processed = None
                            st.session_state.numeric_cols_original = None
                            st.session_state.categorical_cols_original = None
                        else:
                            # Giải nén kết quả thành công
                            X, y, feature_names_proc, scaler, num_imputer, cat_imputer, num_orig, cat_orig = preprocess_result
                            st.session_state.preprocessing_error = None # Không có lỗi

                            # Lưu các thành phần đã FIT vào session_state
                            st.session_state.scaler = scaler
                            st.session_state.numeric_imputer = num_imputer # Đã được fit
                            st.session_state.categorical_imputer = cat_imputer # Đã được fit
                            st.session_state.feature_names_processed = feature_names_proc
                            st.session_state.target_column = selected_target
                            st.session_state.numeric_cols_original = num_orig
                            st.session_state.categorical_cols_original = cat_orig

                            # Chia dữ liệu
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Tăng test size
                            st.write(f"📊 Kích thước tập Train: {X_train.shape} | Test: {X_test.shape}")
                            st.write(f"📈 Số lượng features sau tiền xử lý: {X_train.shape[1]}")
                            # st.write(f"Tên features sau xử lý: {st.session_state.feature_names_processed}") # Debug

                    # Chỉ huấn luyện nếu tiền xử lý thành công
                    if st.session_state.preprocessing_error is None:
                         # Huấn luyện (hàm này đã có spinner và progress)
                         model, training_err = train_model(X_train, y_train, selected_model_name, params)

                         st.session_state.training_error = training_err
                         st.session_state.model = model # Lưu model (có thể là None nếu lỗi)

                         if st.session_state.model:
                             # st.success(f"✅ Huấn luyện mô hình {selected_model_name} thành công!")
                             st.balloons() # Thêm hiệu ứng bóng bay khi thành công!

                             st.markdown(f"#### <i class='fas fa-check-circle'></i> Kết quả Đánh giá trên tập Test ({selected_model_name})", unsafe_allow_html=True)
                             y_pred = st.session_state.model.predict(X_test)
                             mse = mean_squared_error(y_test, y_pred)
                             r2 = r2_score(y_test, y_pred)
                             rmse = np.sqrt(mse)

                             res_col1, res_col2 = st.columns(2)
                             res_col1.metric(
                                 label="📉 RMSE (Root Mean Squared Error)",
                                 value=f"{rmse:.4f}",
                                 delta=None,
                                 help="Căn bậc hai của MSE, cùng đơn vị với target. Càng nhỏ càng tốt."
                             )
                             res_col2.metric(
                                 label="📈 R² Score",
                                 value=f"{r2:.4f}",
                                 help="Hệ số xác định, đo lường mức độ biến thiên của target được giải thích bởi mô hình. Càng gần 1 càng tốt (tối đa là 1)."
                             )

                             # Plotly so sánh thực tế vs dự đoán
                             with st.expander("🔍 Xem biểu đồ so sánh Thực tế vs. Dự đoán", expanded=True): # Mở rộng mặc định
                                 comparison_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_pred})
                                 # Giới hạn số điểm vẽ nếu quá nhiều để tránh chậm
                                 if len(comparison_df) > 1000:
                                      comparison_df_sample = comparison_df.sample(1000, random_state=42)
                                      plot_title = '<b>So sánh Giá trị Thực tế và Dự đoán (1000 điểm mẫu)</b>'
                                 else:
                                      comparison_df_sample = comparison_df
                                      plot_title = '<b>So sánh Giá trị Thực tế và Dự đoán</b>'

                                 fig_comp = px.scatter(
                                     comparison_df_sample, x='Thực tế', y='Dự đoán',
                                     title=plot_title,
                                     opacity=0.6,
                                     trendline='ols', # Thêm đường hồi quy tuyến tính OLS
                                     trendline_color_override='red',
                                     labels={'Thực tế': f'Giá trị Thực tế ({selected_target})', 'Dự đoán': f'Giá trị Dự đoán ({selected_target})'}
                                 )
                                 # Thêm đường y=x để so sánh
                                 min_val = min(y_test.min(), y_pred.min())
                                 max_val = max(y_test.max(), y_pred.max())
                                 fig_comp.add_shape(type='line', line=dict(dash='dash', color='white', width=2),
                                                    x0=min_val, y0=min_val, x1=max_val, y1=max_val)
                                 fig_comp.update_layout(title_x=0.5)
                                 st.plotly_chart(fig_comp, use_container_width=True)
                                 st.caption("Đường nét đứt màu trắng là đường y=x (dự đoán hoàn hảo). Đường màu đỏ là đường xu hướng OLS.")

                         # else: # Lỗi đã được hiển thị trong hàm train_model
                         #     # st.error("❌ Huấn luyện thất bại. Vui lòng kiểm tra thông báo lỗi ở trên.")
                         #     pass

    elif st.session_state.df is None and not st.session_state.df_loaded: # Nếu chưa tải file thành công
         st.warning("☝️ Vui lòng tải dữ liệu hợp lệ ở Tab 1 trước khi huấn luyện.")
    # Không cần else nếu df_loaded=True nhưng df=None (đã có lỗi ở Tab 1)


# == Tab 3: Dự đoán ==
with tab3:
    st.markdown("### <i class='fas fa-hat-wizard'></i> 4. Dự đoán Giá trị Mới", unsafe_allow_html=True)

    # Chỉ hiển thị form nếu đã huấn luyện model thành công và có đủ thông tin cần thiết
    if (st.session_state.model is not None and
        st.session_state.scaler is not None and
        st.session_state.numeric_imputer is not None and # Cần imputer
        st.session_state.categorical_imputer is not None and # Cần imputer
        st.session_state.feature_names_processed is not None and
        st.session_state.numeric_cols_original is not None and
        st.session_state.categorical_cols_original is not None):

        required_original_features = st.session_state.numeric_cols_original + st.session_state.categorical_cols_original
        st.info(f"👇 Nhập giá trị cho các đặc trưng **gốc** sau đây để dự đoán **{st.session_state.target_column}**:")
        # st.write(f"({', '.join(required_original_features)})") # Ghi chú các cột cần nhập

        with st.form(key="prediction_form_styled"):
            input_data_raw = {} # Lưu giá trị người dùng nhập
            input_cols_for_df = {} # Dùng để tạo DataFrame đầu vào

            # Chia cột linh hoạt hơn cho các widget input
            total_orig_cols = len(required_original_features)
            num_widget_cols = min(total_orig_cols, 3) # Tối đa 3 cột input trên 1 hàng
            if num_widget_cols <= 0:
                 st.warning("Không có feature nào được chọn để huấn luyện.")
            else:
                widget_cols = st.columns(num_widget_cols)
                current_col_index = 0

                # Input số
                for col in st.session_state.numeric_cols_original:
                    with widget_cols[current_col_index % num_widget_cols]:
                         # Sử dụng giá trị trung bình từ imputer làm giá trị mặc định
                         default_val_num = 0.0
                         try:
                             # Tìm index của cột trong imputer
                              col_idx_imputer = st.session_state.numeric_cols_original.index(col)
                              if st.session_state.numeric_imputer and hasattr(st.session_state.numeric_imputer, 'statistics_'):
                                  default_val_num = float(st.session_state.numeric_imputer.statistics_[col_idx_imputer])
                         except Exception: # Bỏ qua lỗi nếu không tìm thấy hoặc imputer chưa fit
                              pass
                         input_data_raw[col] = st.number_input(
                             f"{col} (Số)",
                             value=default_val_num,
                             format="%.5f", # Cho phép nhập số thập phân
                             key=f"input_{col}",
                             help=f"Nhập giá trị số cho {col}"
                         )
                         input_cols_for_df[col] = [input_data_raw[col]] # Đặt trong list để tạo df
                    current_col_index += 1

                # Input category
                df_predict_source = st.session_state.df # Lấy df gốc để tìm unique values
                for col in st.session_state.categorical_cols_original:
                    with widget_cols[current_col_index % num_widget_cols]:
                         unique_vals = [""] + sorted(df_predict_source[col].dropna().unique().astype(str).tolist())
                         # default_val_cat = "" # Mặc định trống
                         # Lấy giá trị phổ biến nhất làm mặc định nếu có thể
                         default_val_cat = ""
                         try:
                             col_idx_cat_imputer = st.session_state.categorical_cols_original.index(col)
                             if st.session_state.categorical_imputer and hasattr(st.session_state.categorical_imputer, 'statistics_'):
                                  imputed_val = st.session_state.categorical_imputer.statistics_[col_idx_cat_imputer]
                                  if imputed_val in unique_vals: # Chỉ dùng nếu giá trị đó có trong list
                                       default_val_cat = imputed_val
                         except Exception:
                              pass

                         input_data_raw[col] = st.selectbox(
                             f"{col} (Category)",
                             options=unique_vals,
                             index=unique_vals.index(default_val_cat) if default_val_cat in unique_vals else 0, # Chọn mặc định nếu tìm thấy, nếu không chọn ""
                             key=f"input_{col}",
                             help=f"Chọn một giá trị cho {col}. Để trống nếu không biết (sẽ được xử lý)."
                         )
                         # Nếu người dùng chọn "", coi như là NaN để imputer xử lý
                         input_cols_for_df[col] = [np.nan if input_data_raw[col] == "" else input_data_raw[col]]
                    current_col_index += 1

                # --- Nút Submit ---
                submitted = st.form_submit_button("🔮 Dự đoán Ngay!", use_container_width=True)

                if submitted:
                    st.session_state.last_prediction = None # Reset dự đoán cũ
                    with st.spinner("🧠 Đang xử lý và dự đoán..."):
                        try:
                            # 1. Tạo DataFrame từ input (đúng thứ tự cột gốc)
                            input_df = pd.DataFrame(input_cols_for_df, index=[0])
                            input_df = input_df[required_original_features] # Đảm bảo đúng thứ tự

                            # 2. Xử lý NaN bằng imputer đã fit (không cần fit lại)
                            numeric_features_predict = [col for col in st.session_state.numeric_cols_original if col in input_df.columns]
                            categorical_features_predict = [col for col in st.session_state.categorical_cols_original if col in input_df.columns]

                            if numeric_features_predict and st.session_state.numeric_imputer:
                                input_df[numeric_features_predict] = st.session_state.numeric_imputer.transform(input_df[numeric_features_predict])
                            if categorical_features_predict and st.session_state.categorical_imputer:
                                input_df[categorical_features_predict] = st.session_state.categorical_imputer.transform(input_df[categorical_features_predict])

                            # 3. One-Hot Encoding (phải giống hệt lúc train)
                            input_df_processed = pd.get_dummies(input_df, columns=st.session_state.categorical_cols_original, drop_first=True)

                            # 4. Căn chỉnh cột với feature_names_processed (thêm cột thiếu, xóa cột thừa)
                            # Các cột có trong tập train nhưng thiếu trong input -> thêm vào và gán giá trị 0 (vì đã drop_first=True)
                            missing_cols = set(st.session_state.feature_names_processed) - set(input_df_processed.columns)
                            for c in missing_cols:
                                input_df_processed[c] = 0
                            # Các cột có trong input nhưng không có trong tập train -> xóa đi
                            extra_cols = set(input_df_processed.columns) - set(st.session_state.feature_names_processed)
                            input_df_processed = input_df_processed.drop(columns=list(extra_cols))

                            # Đảm bảo thứ tự cột giống hệt lúc train
                            input_df_processed = input_df_processed[st.session_state.feature_names_processed]

                            # 5. Chuẩn hóa bằng scaler đã fit
                            input_scaled = st.session_state.scaler.transform(input_df_processed)

                            # 6. Dự đoán
                            prediction = st.session_state.model.predict(input_scaled)
                            st.session_state.last_prediction = prediction[0] # Lấy giá trị đầu tiên

                            # 7. Hiển thị kết quả
                            st.success(f"✨ **Kết quả Dự đoán cho {st.session_state.target_column}:**")
                            # Sử dụng metric để hiển thị đẹp hơn
                            st.metric(label=f"Giá trị dự đoán ({st.session_state.target_column})", value=f"{st.session_state.last_prediction:,.2f}") # Định dạng số

                        except Exception as e:
                            st.error(f"Lỗi khi thực hiện dự đoán: {e}")
                            st.exception(e) # In traceback để debug

    elif st.session_state.model is None and st.session_state.df_loaded: # Nếu đã load data nhưng chưa train
         st.warning("⏳ Vui lòng huấn luyện mô hình ở Tab 2 trước khi dự đoán.")
    elif not st.session_state.df_loaded: # Nếu chưa load data
         st.warning("☝️ Vui lòng tải dữ liệu ở Tab 1 và huấn luyện mô hình ở Tab 2 trước.")

    # Hiển thị lại kết quả dự đoán cuối cùng nếu có (ngoài form)
    # if st.session_state.last_prediction is not None and not submitted: # Chỉ hiển thị nếu không phải vừa submit
    #      st.info(f"Kết quả dự đoán lần cuối: {st.session_state.last_prediction:,.2f}")
