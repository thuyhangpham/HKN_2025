import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import requests 
import io       
import os       

# --- 1. Tải và xử lý dữ liệu ---

@st.cache_data
def load_data_and_model():
    
    # Sử dụng URL download trực tiếp (không dùng gdown)
    # Đây là link direct download cho tệp True_News.csv của bạn
    # Streamlit Cloud có thể tải tệp này dễ dàng hơn
    file_id = '1IPXh27wrlhgdqx2GogVCpvxr5C1I7iNS'
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    
    output = 'True_News.csv'

    try:
        # Sử dụng requests để tải nội dung tệp trực tiếp vào bộ nhớ
        st.info("Đang tải dữ liệu từ Google Drive. Vui lòng chờ...")
        response = requests.get(download_url)
        response.raise_for_status() # Bắt lỗi nếu request không thành công
        
        # Đọc nội dung CSV trực tiếp từ bộ nhớ (BytesIO)
        df = pd.read_csv(io.BytesIO(response.content))
        st.success("Đã tải và đọc dữ liệu thành công!")

    except requests.exceptions.RequestException as e:
        st.error(f"LỖI TẢI TỆP (Requests): Không thể tải dữ liệu từ Google Drive.")
        st.error("Vui lòng kiểm tra lại ID tệp hoặc quyền truy cập.")
        return None, None
    except Exception as e:
        st.error(f"LỖI KHÁC: {e}")
        return None, None

    # Đổi tên cột (giống mã cũ)
    if 'Title' in df.columns:
        df = df.rename(columns={'Title':'title'})
    elif 'headline' in df.columns:
        df = df.rename(columns={'headline':'title'})

    if 'Category' in df.columns:
        df = df.rename(columns={'Category':'category'})
    elif 'subject' in df.columns:
        df = df.rename(columns={'subject':'category'})

    # Xử lý dữ liệu 
    df = df[['category', 'title']].dropna().head(1000)
    df['entities'] = df['title'].apply(lambda x: ' '.join(x.split()[:3]))

    def clean_text(text):
        return str(text).replace(":", "").replace("|", "").replace("\n", " ")

    df['title_clean'] = df['title'].apply(clean_text)
    df['entities_clean'] = df['entities'].apply(clean_text)

    # "Huấn luyện" mô hình khuyến nghị
    vectorizer = CountVectorizer()
    X_features = vectorizer.fit_transform(df['title_clean'] + ' ' + df['entities_clean'])
    X_embeddings = X_features.toarray()
    similarity_matrix = cosine_similarity(X_embeddings)
    
    return df, similarity_matrix

# --- 2. Hàm khuyến nghị ---

def get_recommendations(article_index, similarity_matrix, df, top_k=5):
    sim_scores = similarity_matrix[article_index]
    sim_indices = np.argsort(sim_scores)[::-1]
    sim_indices = sim_indices[sim_indices != article_index]  # loại bỏ chính nó
    top_indices = sim_indices[:top_k]

    rec_list = []
    for idx in top_indices:
        rec_list.append({
            'recommended_title': df.iloc[idx]['title'],
            'recommended_category': df.iloc[idx]['category'],
            'similarity_score': sim_scores[idx]
        })
    return pd.DataFrame(rec_list)

# --- 3. Xây dựng giao diện ứng dụng Streamlit ---

st.set_page_config(page_title="Hệ thống Khuyến nghị", layout="wide")
st.title(" Hệ thống Khuyến nghị Bài báo (Content-Based)")

# Tải dữ liệu và mô hình
# Thêm st.spinner để hiển thị trạng thái đang tải
with st.spinner('Đang khởi tạo và tải dữ liệu...'):
    df, similarity_matrix = load_data_and_model()

# Nếu tải dữ liệu thành công thì mới hiển thị phần còn lại
if df is not None and similarity_matrix is not None:
    
    st.header(" 1. Chọn bài báo để tìm nội dung tương tự")
    
    # Tạo một dropdown (selectbox)
    all_titles = df['title']
    selected_title = st.selectbox(
        label="Chọn một tiêu đề từ danh sách:",
        options=all_titles,
        label_visibility="collapsed"
    )

    # Lấy index và thông tin bài báo đã chọn
    selected_index = df[df['title'] == selected_title].index[0]
    original_article = df.iloc[selected_index]

    st.divider()

    col1, col2 = st.columns(2)

    # Cột 1: Hiển thị thông tin bài báo gốc
    with col1:
        st.subheader(" Bài báo gốc đã chọn")
        with st.container(border=True):
            st.markdown(f"**Tiêu đề:** {original_article['title']}")
            st.markdown(f"**Chủ đề:** {original_article['category']}")
    
    # Cột 2: Hiển thị các bài báo được khuyến nghị
    with col2:
        st.subheader(" 2. Các bài báo được khuyến nghị")
        
        # Gọi hàm khuyến nghị
        df_recommend = get_recommendations(selected_index, similarity_matrix, df, top_k=5)
        
        # Hiển thị kết quả dạng bảng
        st.dataframe(
            df_recommend,
            use_container_width=True,
            column_config={
                "similarity_score": st.column_config.ProgressColumn(
                    "Độ tương đồng",
                    format="%.2f",
                    min_val=0,
                    max_val=1,
                )
            },
            hide_index=True
        )
