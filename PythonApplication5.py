import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Không cần nữa nếu dùng Plotly hoàn toàn
# import seaborn as sns # Không cần nữa nếu dùng Plotly hoàn toàn
import plotly.express as px # Import Plotly Express
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


</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Các hàm trợ giúp (Giữ nguyên logic, chỉ thay đổi plot) ---
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc tệp CSV: {e}")
        return None

# Hàm tiền xử lý (Giữ nguyên logic)
def preprocess_data(df, target_column, selected_features):
    df_subset = df[selected_features + [target_column]].copy()

    numeric_cols = df_subset.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_subset.select_dtypes(exclude=np.number).columns.tolist()
    original_numeric_features = [col for col in numeric_cols if col != target_column]
    original_categorical_features = categorical_cols.copy()

    # Xử lý NaN
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
        df_subset.dropna(subset=[target_column], inplace=True)
    else:
        if target_column in categorical_cols:
             categorical_cols.remove(target_column)
             df_subset.dropna(subset=[target_column], inplace=True)


    if numeric_cols:
        num_imputer = SimpleImputer(strategy='mean')
        df_subset[numeric_cols] = num_imputer.fit_transform(df_subset[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_subset[categorical_cols] = cat_imputer.fit_transform(df_subset[categorical_cols])

    # One-Hot Encoding
    df_processed = pd.get_dummies(df_subset, columns=categorical_cols, drop_first=True)

    # Tách X, y
    if target_column not in df_processed.columns:
         possible_target_cols = [col for col in df_processed.columns if col.startswith(target_column)]
         if len(possible_target_cols) == 1:
             target_column_processed = possible_target_cols[0]
             y = df_processed[target_column_processed]
             X = df_processed.drop(target_column_processed, axis=1)
         else:
              st.error("Không thể xác định cột target sau khi mã hóa.")
              return None, None, None, None, None, None
    else:
        y = df_processed[target_column]
        X = df_processed.drop(target_column, axis=1)


    feature_names_processed = X.columns.tolist()

    # Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names_processed, scaler, original_numeric_features, original_categorical_features


# Hàm huấn luyện (Thêm progress bar)
def train_model(X_train, y_train, model_name, params):
    model = None
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "Support Vector Regressor (SVR)":
        model = SVR(
            C=params.get('C', 1.0),
            epsilon=params.get('epsilon', 0.1),
            kernel=params.get('kernel', 'rbf')
        )
    else:
        st.error("Mô hình không hợp lệ!")
        return None

    try:
        # --- Thêm Progress Bar ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Bắt đầu huấn luyện...")

        # Giả lập quá trình huấn luyện (thay bằng model.fit thật)
        # model.fit(X_train, y_train) # Thay thế dòng này bằng vòng lặp nếu muốn cập nhật progress bar chi tiết hơn
        # Hoặc đơn giản là chạy fit và cập nhật progress sau khi xong
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        # Cập nhật progress bar và status
        for i in range(100):
            time.sleep(0.01) # Giả lập thời gian xử lý nhỏ
            progress_bar.progress(i + 1)
        status_text.success(f"Hoàn thành huấn luyện trong {end_time - start_time:.2f} giây!")
        time.sleep(1) # Chờ 1 giây để user thấy thông báo success
        status_text.empty() # Xóa thông báo text
        progress_bar.empty() # Xóa progress bar

        return model
    except Exception as e:
        st.error(f"Lỗi trong quá trình huấn luyện mô hình: {e}")
        if 'progress_bar' in locals(): progress_bar.empty() # Đảm bảo xóa progress bar nếu lỗi
        if 'status_text' in locals(): status_text.empty()
        return None


# --- Khởi tạo Session State (Giữ nguyên) ---
if 'model' not in st.session_state: st.session_state.model = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'feature_names' not in st.session_state: st.session_state.feature_names = None
if 'target_column' not in st.session_state: st.session_state.target_column = None
if 'numeric_cols_original' not in st.session_state: st.session_state.numeric_cols_original = None
if 'categorical_cols_original' not in st.session_state: st.session_state.categorical_cols_original = None
if 'df_loaded' not in st.session_state: st.session_state.df_loaded = False
if 'df' not in st.session_state: st.session_state.df = None
if 'uploaded_filename' not in st.session_state: st.session_state.uploaded_filename = None

# --- Giao diện người dùng với Tabs và Style Mới ---
st.title("🚀 Ứng dụng Machine Learning Siêu Cấp Vip Pro 🚀")
st.markdown("---", unsafe_allow_html=True) # Dùng HTML để đường kẻ có thể được style nếu muốn

tab1, tab2, tab3 = st.tabs([
    "**<i class='fas fa-upload'></i> Tải & Khám phá Dữ liệu**",
    "**<i class='fas fa-cogs'></i> Huấn luyện Mô hình**",
    "**<i class='fas fa-magic-wand-sparkles'></i> Dự đoán Kết quả**"
])

# == Tab 1: Tải dữ liệu và EDA ==
with tab1:
    # st.header("1. Tải lên tập dữ liệu (.csv)")
    upload_container = st.container() # Container để style nếu cần
    with upload_container:
        st.markdown("### <i class='fas fa-file-csv'></i> 1. Tải lên tệp CSV của bạn")
        uploaded_file = st.file_uploader(
            "Kéo thả hoặc chọn tệp...",
            type="csv",
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        if not st.session_state.df_loaded or st.session_state.uploaded_filename != uploaded_file.name:
            with st.spinner("🔄 Đang tải và xử lý tệp..."):
                st.session_state.df = load_data(uploaded_file)
                st.session_state.df_loaded = True
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.model = None # Reset model khi load file mới
            st.rerun()

    if st.session_state.df_loaded and st.session_state.df is not None:
        st.success(f"✅ Đã tải thành công tệp: **{st.session_state.uploaded_filename}**")
        df = st.session_state.df

        st.markdown("### <i class='fas fa-search'></i> 2. Khám phá Dữ liệu (EDA)")

        with st.expander("📊 Xem trước dữ liệu (5 dòng đầu)", expanded=False):
            st.dataframe(df.head())

        # Sử dụng columns để bố trí thông tin
        col_info1, col_info2, col_info3 = st.columns([1,1,1])
        with col_info1:
            with st.expander("ℹ️ Thông tin chung", expanded=False):
                 st.write(f"**Dòng:** {df.shape[0]}, **Cột:** {df.shape[1]}")
                 buffer = io.StringIO()
                 df.info(buf=buffer)
                 s = buffer.getvalue()
                 st.text(s)
        with col_info2:
            with st.expander("🔢 Thống kê mô tả", expanded=False):
                 try:
                      st.dataframe(df.describe())
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

        st.markdown("### <i class='fas fa-chart-bar'></i> 3. Trực quan hóa Dữ liệu")
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_columns:
            st.warning("⚠️ Không có cột dữ liệu dạng số để vẽ biểu đồ.")
        else:
            # Sử dụng columns cho selectbox gọn hơn
            col_plot_select, col_plot_display = st.columns([0.3, 0.7])

            with col_plot_select:
                plot_type = st.selectbox(
                    "Chọn loại biểu đồ:",
                    ["Histogram", "Box Plot", "Scatter Plot", "Heatmap Tương quan"],
                    key="plot_type_select"
                )

                # Chọn cột cho từng loại plot
                selected_col_hist_box = None
                selected_col_scatter_x = None
                selected_col_scatter_y = None
                selected_col_scatter_hue = None

                if plot_type in ["Histogram", "Box Plot"]:
                    selected_col_hist_box = st.selectbox("Chọn cột số:", numeric_columns, key="hist_box_col")
                elif plot_type == "Scatter Plot":
                    selected_col_scatter_x = st.selectbox("Chọn cột X:", numeric_columns, key="scatter_x")
                    col2_options = [col for col in numeric_columns if col != selected_col_scatter_x]
                    if len(numeric_columns) > 1 and col2_options:
                         selected_col_scatter_y = st.selectbox("Chọn cột Y:", col2_options, key="scatter_y")
                         hue_options = [None] + df.select_dtypes(exclude=np.number).columns.tolist()
                         selected_col_scatter_hue = st.selectbox("Phân màu theo (Tùy chọn):", hue_options, key="scatter_hue")
                    else:
                        st.warning("Cần ít nhất 2 cột số khác nhau.")
                # Heatmap không cần chọn cột thêm

            with col_plot_display:
                # Tạo và hiển thị biểu đồ Plotly
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
                         corr = df[numeric_columns].corr()
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
                             title='<b>Ma trận Tương quan Heatmap</b>',
                             xaxis_tickangle=-45,
                             title_x=0.5,
                             height=600 # Tăng chiều cao heatmap
                         )
                         st.plotly_chart(fig, use_container_width=True)
                         st.caption("Giá trị từ -1 đến 1. Gần 1: Tương quan dương mạnh, Gần -1: Tương quan âm mạnh, Gần 0: Ít tương quan tuyến tính.")
                     else:
                          st.warning("Cần ít nhất 2 cột số để vẽ heatmap tương quan.")

    else:
         st.info("👋 Chào mừng! Hãy tải lên một tệp CSV để bắt đầu hành trình khám phá dữ liệu và xây dựng mô hình.")
         # Có thể thêm Lottie animation ở đây cho đẹp mắt khi chưa có dữ liệu
         # import streamlit_lottie
         # from streamlit_lottie import st_lottie
         # try:
         #      # Ví dụ: animation từ lottiefiles.com
         #      # response = requests.get("URL_TO_LOTTIE_JSON")
         #      # lottie_json = response.json()
         #      # st_lottie(lottie_json, speed=1, reverse=False, loop=True, quality="low", height=300, key="lottie_hello")
         #      pass
         # except:
         #      st.write("Không thể tải animation.")

# == Tab 2: Huấn luyện Mô hình ==
with tab2:
    st.markdown("### <i class='fas fa-tasks'></i> 3. Chọn Features, Target & Mô hình")

    if st.session_state.df_loaded and st.session_state.df is not None:
        df = st.session_state.df

        col1_setup, col2_setup = st.columns(2, gap="large")

        with col1_setup:
            st.markdown("#### <i class='fas fa-bullseye'></i> Features & Target")
            all_columns = df.columns.tolist()
            potential_target_cols = df.select_dtypes(include=np.number).columns.tolist()

            if not potential_target_cols:
                 st.error("⛔ Không tìm thấy cột số phù hợp để làm biến mục tiêu.")
                 st.stop()

            selected_target = st.selectbox("🎯 Chọn cột mục tiêu (Target - số):", potential_target_cols, key="target_select")
            available_features = [col for col in all_columns if col != selected_target]
            selected_features = st.multiselect("✨ Chọn các cột đặc trưng (Features):", available_features, default=available_features, key="feature_select")

        with col2_setup:
            st.markdown("#### <i class='fas fa-robot'></i> Thuật toán & Tham số")
            model_options = ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor (SVR)"]
            selected_model = st.selectbox("🤖 Chọn thuật toán:", model_options, key="model_select")

            # Sử dụng expander cho siêu tham số
            with st.expander(f"🛠️ Tinh chỉnh siêu tham số cho {selected_model}", expanded=False):
                params = {}
                if selected_model == "Random Forest Regressor":
                    params['n_estimators'] = st.slider("Số cây (n_estimators):", 50, 1000, 150, 10, key="rf_n_estimators")
                    max_depth_input = st.number_input("Độ sâu tối đa (max_depth, 0=none):", min_value=0, value=10, step=1, key="rf_max_depth")
                    params['max_depth'] = None if max_depth_input == 0 else max_depth_input
                elif selected_model == "Support Vector Regressor (SVR)":
                    params['C'] = st.slider("Tham số C:", 0.1, 20.0, 1.5, 0.1, key="svr_c")
                    params['epsilon'] = st.slider("Epsilon:", 0.05, 1.0, 0.15, 0.01, key="svr_epsilon")
                    params['kernel'] = st.radio("Kernel:", ['rbf', 'linear', 'poly', 'sigmoid'], index=0, key="svr_kernel", horizontal=True)
                else:
                    st.info("Linear Regression cơ bản không cần tinh chỉnh nhiều.")

        st.markdown("---")

        # Đặt nút huấn luyện ở giữa
        col_button_spacer1, col_button, col_button_spacer2 = st.columns([1, 1.5, 1])
        with col_button:
            if st.button("⚡️<i class='fas fa-bolt'></i> Huấn luyện Mô hình Ngay! ⚡️", key="train_button_main"):
                if not selected_features:
                    st.warning("⚠️ Vui lòng chọn ít nhất một Feature.")
                elif not selected_target:
                     st.warning("⚠️ Vui lòng chọn Target.")
                else:
                    # Logic huấn luyện (có progress bar trong hàm train_model)
                    X, y, feature_names_proc, scaler, num_orig, cat_orig = preprocess_data(df.copy(), selected_target, selected_features)

                    if X is not None:
                        st.session_state.scaler = scaler
                        st.session_state.feature_names = feature_names_proc
                        st.session_state.target_column = selected_target
                        st.session_state.numeric_cols_original = num_orig
                        st.session_state.categorical_cols_original = cat_orig

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Tăng test size
                        st.write(f"📊 Kích thước tập Train: {X_train.shape} | Test: {X_test.shape}")

                        # Huấn luyện (hàm này đã có spinner và progress)
                        model = train_model(X_train, y_train, selected_model, params)

                        if model:
                            st.session_state.model = model
                            st.balloons() # Thêm hiệu ứng bóng bay khi thành công!

                            st.markdown(f"#### <i class='fas fa-check-circle'></i> Kết quả Đánh giá ({selected_model})")
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)

                            res_col1, res_col2 = st.columns(2)
                            res_col1.metric(
                                label="📉 Mean Squared Error (MSE)",
                                value=f"{mse:.4f}",
                                delta=None, # Có thể tính delta so với lần chạy trước nếu lưu lại
                                help="Sai số bình phương trung bình, càng nhỏ càng tốt."
                             )
                            res_col2.metric(
                                label="📈 R² Score",
                                value=f"{r2:.4f}",
                                help="Hệ số xác định, càng gần 1 càng tốt (tối đa là 1)."
                            )

                            # Plotly so sánh thực tế vs dự đoán
                            with st.expander("🔍 Xem biểu đồ so sánh Thực tế vs. Dự đoán", expanded=False):
                                comparison_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_pred})
                                fig_comp = px.scatter(
                                    comparison_df, x='Thực tế', y='Dự đoán',
                                    title='<b>So sánh Giá trị Thực tế và Dự đoán</b>',
                                    opacity=0.6,
                                    trendline='ols', # Thêm đường hồi quy tuyến tính OLS
                                    trendline_color_override='red'
                                )
                                # Thêm đường y=x để so sánh
                                fig_comp.add_shape(type='line', line=dict(dash='dash', color='white'),
                                                   x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                                fig_comp.update_layout(title_x=0.5)
                                st.plotly_chart(fig_comp, use_container_width=True)
                        else:
                             st.error("❌ Huấn luyện thất bại.")
                             st.session_state.model = None
    else:
         st.warning("☝️ Vui lòng tải dữ liệu ở Tab 1 trước.")


# == Tab 3: Dự đoán ==
with tab3:
    st.markdown("### <i class='fas fa-hat-wizard'></i> 4. Dự đoán Giá trị Mới")

    if st.session_state.model is not None and st.session_state.scaler is not None and st.session_state.feature_names is not None:

        required_original_features = st.session_state.numeric_cols_original + st.session_state.categorical_cols_original
        st.info(f"👇 Nhập giá trị cho các đặc trưng sau: `{', '.join(required_original_features)}`")

        with st.form(key="prediction_form_styled"):
            input_data = {}
            input_df_cols = {}

            # Chia cột linh hoạt hơn
            total_orig_cols = len(required_original_features)
            num_widget_cols = min(total_orig_cols, 4) # Tối đa 4 cột
            widget_cols = st.columns(num_widget_cols)
            current_col_index = 0

            # Input số
            for col in st.session_state.numeric_cols_original:
                with widget_cols[current_col_index % num_widget_cols]:
                    input_data[col] = st.number_input(f"{col} <i class='fas fa-hashtag'></i>", value=0.0, format="%.5f", key=f"input_{col}", unsafe_allow_html=True)
                    input_df_cols[col] = [input_data[col]]
                current_col_index += 1

            # Input category
            df_predict_source = st.session_state.df
            for col in st.session_state.categorical_cols_original:
                 with widget_cols[current_col_index % num_widget_cols]:
                      unique_vals = [""] + df_predict_source[col].unique().astype(str).tolist()
                      unique_vals = [val for val in unique_vals if pd.notna(val)]
                      input_data[col] = st.selectbox(f"{col} <i class='fas fa-tag'></i>", options=unique_vals, index=0, key=f"input_{col}", help=f"Chọn một giá trị cho {col}", unsafe_allow_html=True)
                      input_df_cols[col] = [np.nan if input_data[col] == "" else input_data[col]]
                 current_col_index += 1

            # Nút Submit
            col_form_button_spacer1, col_form_button, col_form_button_spacer2 = st.columns([1,1,1])
            with col_form_button:
                 submit_button = st.form_submit_button(label="<i class='fas fa-paper-plane'></i> Gửi Dự đoán")

        if submit_button:
             with st.spinner("🧙‍♂️ Đang thực hiện phép màu dự đoán..."):
                 try:
                     input_df = pd.DataFrame(input_df_cols)
                     # st.write("Input DataFrame:") # Debug
                     # st.dataframe(input_df)

                     # Xử lý NaN đơn giản cho input
                     input_df[st.session_state.numeric_cols_original] = input_df[st.session_state.numeric_cols_original].fillna(0)
                     input_df[st.session_state.categorical_cols_original] = input_df[st.session_state.categorical_cols_original].fillna(df_predict_source[st.session_state.categorical_cols_original].mode().iloc[0])

                     # One-Hot Encode
                     input_df_processed = pd.get_dummies(input_df, columns=st.session_state.categorical_cols_original, drop_first=True)
                     # st.write("Input after dummies:") # Debug
                     # st.dataframe(input_df_processed)

                     # Align columns
                     input_df_aligned = input_df_processed.reindex(columns=st.session_state.feature_names, fill_value=0)
                     # st.write("Input after align:") # Debug
                     # st.dataframe(input_df_aligned)

                     # Scale
                     input_scaled = st.session_state.scaler.transform(input_df_aligned)

                     # Predict
                     prediction = st.session_state.model.predict(input_scaled)

                 except Exception as e:
                      st.error(f"❌ Lỗi khi dự đoán: {e}")
                      st.exception(e) # In đầy đủ traceback để debug
                      prediction = None # Đặt prediction là None nếu có lỗi

             # Hiển thị kết quả nếu không có lỗi
             if 'prediction' in locals() and prediction is not None:
                  st.markdown("---")
                  st.markdown("### <i class='fas fa-bullseye-arrow'></i> Kết quả Dự đoán:")
                  # Sử dụng st.metric để hiển thị đẹp hơn
                  st.metric(label=f"✨ Giá trị dự đoán cho '{st.session_state.target_column}' ✨", value=f"{prediction[0]:,.4f}")
                  st.success("🎉 Dự đoán thành công!")
                  # Thêm hiệu ứng tuyết rơi :D
                  st.snow()


    elif not st.session_state.df_loaded:
         st.warning("☝️ Vui lòng tải dữ liệu ở Tab 1 trước.")
    else:
        st.warning("⏳ Vui lòng huấn luyện mô hình ở Tab 2 trước.")


# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: rgba(255,255,255,0.7);'> Made with <i class='fas fa-heart' style='color: red;'></i> & <i class='fas fa-brain' style='color: pink;'></i> by Gemini & You @ 2025</div>", unsafe_allow_html=True)