"""
Streamlit Web Application for Vietnamese Hallucination Detection
Simple and user-friendly interface
"""
import streamlit as st
import requests
import json
from typing import Dict, Optional
import time

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Vietnamese Hallucination Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #FAFAFA; /* Light text color */
    }
    .no-hallucination {
        background-color: #2E7D32; /* Darker Green */
        border-left: 5px solid #4CAF50;
    }
    .intrinsic-hallucination {
        background-color: #C62828; /* Darker Red */
        border-left: 5px solid #F44336;
    }
    .extrinsic-hallucination {
        background-color: #FF8F00; /* Darker Amber */
        border-left: 5px solid #FF9800;
    }
    .metric-card {
        background-color: #333333; /* Dark Gray */
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: #FAFAFA; /* Light text color */
    }
    .example-box {
        background-color: #424242; /* Medium-Dark Gray */
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# API CLIENT
# ==============================================================================
class HallucinationDetectorClient:
    """Client to call Modal inference API"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def predict(
        self,
        context: str,
        prompt: str,
        response: str,
        prompt_type: Optional[str] = None,
        return_probabilities: bool = True
    ) -> Dict:
        """Call prediction API"""
        
        payload = {
            "context": context,
            "prompt": prompt,
            "response": response,
            "prompt_type": prompt_type,
            "return_probabilities": return_probabilities
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Error: {str(e)}")
            return None

# ==============================================================================
# EXAMPLE DATA
# ==============================================================================
EXAMPLES = {
    "Không có ảo giác (No Hallucination)": {
        "context": "Hà Nội là thủ đô của Việt Nam từ năm 1010. Thành phố nằm ở đồng bằng sông Hồng, có diện tích 3.344 km² và dân số khoảng 8 triệu người. Hà Nội nổi tiếng với nhiều di tích lịch sử như Văn Miếu, Hoàng thành Thăng Long.",
        "prompt": "Hà Nội là thủ đô của nước nào?",
        "response": "Hà Nội là thủ đô của Việt Nam.",
        "prompt_type": "factual"
    },
    "Ảo giác nội tại (Intrinsic)": {
        "context": "Hà Nội là thủ đô của Việt Nam từ năm 1010. TP. Hồ Chí Minh là thành phố lớn nhất Việt Nam về dân số và kinh tế.",
        "prompt": "Thành phố nào là thủ đô của Việt Nam?",
        "response": "TP. Hồ Chí Minh là thủ đô của Việt Nam.",
        "prompt_type": "factual"
    },
    "Ảo giác ngoại tại (Extrinsic)": {
        "context": "Hà Nội là thủ đô của Việt Nam. Thành phố có nhiều di tích lịch sử.",
        "prompt": "Dân số Hà Nội là bao nhiêu?",
        "response": "Hà Nội có dân số khoảng 8 triệu người vào năm 2023.",
        "prompt_type": "factual"
    },
    "Thông tin về ẩm thực": {
        "context": "Phở là món ăn truyền thống của Việt Nam, có nguồn gốc từ Bắc Bộ. Phở được làm từ bánh phở, nước dùng hầm từ xương, thịt bò hoặc gà, và các gia vị như hành, ngò, hạt tiêu.",
        "prompt": "Phở được làm từ những nguyên liệu gì?",
        "response": "Phở được làm từ bánh phở, nước dùng từ xương, thịt (bò hoặc gà), hành, ngò và các gia vị.",
        "prompt_type": "factual"
    },
}

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    # Header
    st.markdown('<p class="main-header">🔍 Vietnamese Hallucination Detector</p>', unsafe_allow_html=True)
    st.markdown("### Phát hiện ảo giác (hallucination) trong câu trả lời của mô hình ngôn ngữ")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # API URL
        api_url = st.text_input(
            "API URL",
            value="https://nhuttruongg--vihallu-inference-fastapi-app.modal.run",  # Replace with your Modal URL
            help="URL của Modal inference endpoint"
        )
        
        # Prompt type
        prompt_type = st.selectbox(
            "Loại câu hỏi",
            ["factual", "noisy", "adversarial", None],
            index=0,
            help="Loại câu hỏi: factual (sự thật), noisy (nhiễu), adversarial (đối kháng)"
        )
        
        st.divider()
        
        # Information
        st.header("ℹ️ Thông tin")
        st.markdown("""
        **Ba loại ảo giác:**
        
        - **No (Không)**: Câu trả lời nhất quán với ngữ cảnh
        - **Intrinsic (Nội tại)**: Câu trả lời mâu thuẫn với ngữ cảnh
        - **Extrinsic (Ngoại tại)**: Câu trả lời có thông tin ngoài ngữ cảnh
        
        **Model**: VinAllama-7B + LoRA  
        **Task**: 3-class classification
        """)
        
        st.divider()
        
        # Statistics
        if 'prediction_count' in st.session_state:
            st.header("📊 Thống kê")
            st.metric("Số lần dự đoán", st.session_state.prediction_count)
            st.metric("Thời gian TB", f"{st.session_state.avg_time:.2f}s")
    
    # Initialize session state
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
        st.session_state.total_time = 0
        st.session_state.avg_time = 0
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Dự đoán", "📚 Ví dụ", "📖 Hướng dẫn"])
    
    # ===== TAB 1: PREDICTION =====
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Đầu vào")
            
            context = st.text_area(
                "Ngữ cảnh (Context)",
                height=150,
                placeholder="Nhập thông tin nền tảng, ngữ cảnh...",
                help="Văn bản chứa thông tin nền"
            )
            
            prompt = st.text_input(
                "Câu hỏi (Prompt)",
                placeholder="Nhập câu hỏi...",
                help="Câu hỏi được đặt ra"
            )
            
            response = st.text_area(
                "Câu trả lời (Response)",
                height=100,
                placeholder="Nhập câu trả lời cần kiểm tra...",
                help="Câu trả lời của mô hình cần đánh giá"
            )
            
            col_btn1, col_btn2 = st.columns([1, 1])
            
            with col_btn1:
                predict_btn = st.button("🔍 Phát hiện", type="primary", use_container_width=True)
            
            with col_btn2:
                clear_btn = st.button("🗑️ Xóa", use_container_width=True)
            
            if clear_btn:
                st.rerun()
        
        with col2:
            st.subheader("🎯 Kết quả")
            
            if predict_btn:
                if not context or not prompt or not response:
                    st.error("⚠️ Vui lòng điền đầy đủ thông tin!")
                else:
                    with st.spinner("🔄 Đang phân tích..."):
                        # Call API
                        client = HallucinationDetectorClient(api_url)
                        
                        start_time = time.time()
                        result = client.predict(
                            context=context,
                            prompt=prompt,
                            response=response,
                            prompt_type=prompt_type if prompt_type != "None" else None,
                            return_probabilities=True
                        )
                        elapsed_time = time.time() - start_time
                        
                        if result:
                            # Update statistics
                            st.session_state.prediction_count += 1
                            st.session_state.total_time += elapsed_time
                            st.session_state.avg_time = st.session_state.total_time / st.session_state.prediction_count
                            
                            # Display result
                            label = result['label']
                            confidence = result.get('confidence', 0) * 100
                            
                            # Result box with color coding
                            if label == "no":
                                box_class = "no-hallucination"
                                emoji = "✅"
                                title = "Không phát hiện ảo giác"
                            elif label == "intrinsic":
                                box_class = "intrinsic-hallucination"
                                emoji = "⚠️"
                                title = "Ảo giác nội tại"
                            else:
                                box_class = "extrinsic-hallucination"
                                emoji = "⚠️"
                                title = "Ảo giác ngoại tại"
                            
                            st.markdown(f"""
                            <div class="result-box {box_class}">
                                <h3>{emoji} {title}</h3>
                                <p><strong>Độ tin cậy:</strong> {confidence:.1f}%</p>
                                <p>{result.get('explanation', '')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Metrics
                            col_m1, col_m2, col_m3 = st.columns(3)
                            
                            with col_m1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>{label.upper()}</h4>
                                    <p>Nhãn dự đoán</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_m2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>{confidence:.1f}%</h4>
                                    <p>Độ tin cậy</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_m3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>{elapsed_time:.2f}s</h4>
                                    <p>Thời gian xử lý</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Probability distribution
                            if result.get('probabilities'):
                                st.subheader("📊 Phân bố xác suất")
                                
                                probs = result['probabilities']
                                
                                import pandas as pd
                                df_probs = pd.DataFrame({
                                    'Nhãn': ['No', 'Intrinsic', 'Extrinsic'],
                                    'Xác suất': [
                                        probs['no'] * 100,
                                        probs['intrinsic'] * 100,
                                        probs['extrinsic'] * 100
                                    ]
                                })
                                
                                st.bar_chart(df_probs.set_index('Nhãn'))
                                
                                # Detailed probabilities
                                with st.expander("Chi tiết xác suất"):
                                    for label_name, prob in probs.items():
                                        st.progress(prob, text=f"{label_name}: {prob*100:.2f}%")
    
    # ===== TAB 2: EXAMPLES =====
    with tab2:
        st.subheader("📚 Các ví dụ mẫu")
        st.markdown("Click vào ví dụ để tải vào form dự đoán")
        
        for example_name, example_data in EXAMPLES.items():
            with st.expander(f"📄 {example_name}"):
                st.markdown(f"**Ngữ cảnh:** {example_data['context']}")
                st.markdown(f"**Câu hỏi:** {example_data['prompt']}")
                st.markdown(f"**Câu trả lời:** {example_data['response']}")
                
                if st.button(f"Tải ví dụ này", key=f"load_{example_name}"):
                    st.session_state.loaded_example = example_data
                    st.rerun()
        
        # Load example if selected
        if 'loaded_example' in st.session_state:
            st.success("✅ Đã tải ví dụ! Chuyển sang tab 'Dự đoán' để thử nghiệm.")
    
    # ===== TAB 3: GUIDE =====
    with tab3:
        st.subheader("📖 Hướng dẫn sử dụng")
        
        st.markdown("""
        ### Cách sử dụng
        
        1. **Cấu hình API**: Nhập URL của Modal inference endpoint vào sidebar
        2. **Nhập dữ liệu**:
           - **Ngữ cảnh**: Thông tin nền, văn bản tham khảo
           - **Câu hỏi**: Câu hỏi được đặt ra cho mô hình
           - **Câu trả lời**: Câu trả lời của mô hình cần kiểm tra
        3. **Chọn loại câu hỏi** (tùy chọn): factual, noisy, hoặc adversarial
        4. **Click "Phát hiện"** để phân tích
        
        ### Giải thích kết quả
        
        #### ✅ No (Không có ảo giác)
        - Câu trả lời **nhất quán** với thông tin trong ngữ cảnh
        - Không có thông tin sai lệch hoặc thừa
        - Đây là câu trả lời mong muốn
        
        #### ⚠️ Intrinsic (Ảo giác nội tại)
        - Câu trả lời **mâu thuẫn** với ngữ cảnh
        - Thông tin trong câu trả lời **trái ngược** với sự thật trong ngữ cảnh
        - Ví dụ: Ngữ cảnh nói "A là B" nhưng trả lời "A là C"
        
        #### ⚠️ Extrinsic (Ảo giác ngoại tại)
        - Câu trả lời chứa thông tin **không có** trong ngữ cảnh
        - Mô hình "tưởng tượng" ra thông tin mới
        - Thông tin có thể đúng trong thực tế nhưng không có trong ngữ cảnh
        
        ### Ví dụ minh họa
        
        **No Hallucination:**
        - Context: "Hà Nội là thủ đô Việt Nam"
        - Prompt: "Thủ đô VN là gì?"
        - Response: "Hà Nội là thủ đô Việt Nam" ✅
        
        **Intrinsic:**
        - Context: "Hà Nội là thủ đô Việt Nam"
        - Prompt: "Thủ đô VN là gì?"
        - Response: "Sài Gòn là thủ đô Việt Nam" ❌
        
        **Extrinsic:**
        - Context: "Hà Nội là một thành phố"
        - Prompt: "Dân số Hà Nội?"
        - Response: "Hà Nội có 8 triệu dân" ⚠️ (không có trong context)
        
        ### Lưu ý
        
        - Model được train trên dataset tiếng Việt
        - Độ chính xác phụ thuộc vào chất lượng ngữ cảnh
        - Với câu trả lời phức tạp, hãy xem xét xác suất các nhãn
        """)

if __name__ == "__main__":
    main()