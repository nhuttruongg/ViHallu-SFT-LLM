import streamlit as st
import requests

st.set_page_config(page_title="Vietnamese Hallucination Detection", layout="wide")

st.title("🕵️ Vietnamese Hallucination Detection")
st.markdown("This app uses a fine-tuned model deployed on Modal to detect hallucinations in Vietnamese text.")

api_url = "https://nhuttruongg--vihallu-inference-fastapi-app.modal.run/predict/"

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        context = st.text_area("📝 Context", "Hà Nội là thủ đô của Việt Nam từ năm 1010. Thành phố có diện tích 3.344 km².", height=200)
    with col2:
        prompt = st.text_input("❓ Prompt", "Thủ đô của Việt Nam là gì?")
        response = st.text_area("💬 Response to evaluate", "Thủ đô của Việt Nam là Hà Nội.", height=125)

    submitted = st.form_submit_button("🔎 Predict")

if submitted:
    if not context or not prompt or not response:
        st.error("Please fill in all fields.")
    else:
        payload = {
            "context": context,
            "prompt": prompt,
            "response": response,
        }
        with st.spinner("🔍 Analyzing..."):
            try:
                res = requests.post(api_url, json=payload, timeout=60)
                res.raise_for_status()
                result = res.json()

                st.success("✅ Analysis Complete!")

                label = result.get("label", "N/A").upper()
                explanation = result.get("explanation", "No explanation available.")
                confidence = result.get("confidence")

                if label == "NO":
                    st.markdown(f"### <span style='color:green;'>{label}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"### <span style='color:orange;'>{label}</span>", unsafe_allow_html=True)

                st.info(explanation)

                if confidence is not None:
                    st.progress(confidence, text=f"Confidence: {confidence:.2%}")

                with st.expander("Raw JSON Response"):
                    st.json(result)

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
