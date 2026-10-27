import streamlit as st
from joblib import load
from pathlib import Path

st.set_page_config(
  page_title="Cat Toxicity Classifier",
  layout="centered"
)

st.markdown(
"""
<style>
.stMainBlockContainer {
  max-width: 650px;
  min-weight: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
h1, p {
  text-align: center;
}
div.stButton {
  display: flex;
  justify-content: center;
}
</style>
""",
unsafe_allow_html=True
)

def change_background(color):
  st.markdown(
    f"""
    <style>
    .stApp {{
      background-color: {color};
    }}
    </style>
    """,
  unsafe_allow_html=True
  )

@st.cache_resource
def load_model():
  model_path = Path(__file__).parent / "models" / "tuned_classifier.joblib"
  return load(model_path)

model = load_model()

st.title("Is It Toxic to My Cat?")

st.write(
  "Enter the name of a plant, food or household item "
  "to see whether or not it's predicted toxic to cats."
)

item = st.text_input(
  "Enter an item:",
  placeholder="e.g. chocolate, lily etc."
)

left, center, right = st.columns([1, 2, 1])

with center:
  check_button = st.button(
    "Check toxicity",
    use_container_width=True
  )

if check_button:
  if not item.strip():
    st.warning("Please enter an item.")

  else:
    cleaned_item = item.lower().strip()
    prediction = model.predict([cleaned_item])[0]

    if prediction == "toxic":
      change_background("#FFCCCC")
      st.markdown(
        """
        <h1 style="text-align: center; color: #8B0000; margin_top: 50px;">
          TOXIC
        </h1>
        """,
        unsafe_allow_html=True
      )

    elif prediction == "non-toxic":
      change_background("#D4EDDA")
      st.markdown(
        """
        <h1 style="text-align: center; color: #1B5E20; margin_top: 50px;">
          NON-TOXIC
        </h1>
        """,
        unsafe_allow_html=True
      )

st.divider()

st.caption(
  "This is only for an educational machine-learning project! "
  "Predictions may be incorrect so do not rely on this tool "
  "to decide whether or not something is safe for a cat."
)