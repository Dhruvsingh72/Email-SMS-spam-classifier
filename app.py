import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # for stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # for stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

#  Fixed loading of model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- UI Customization ---
# Page config
st.set_page_config(page_title="📩 Email/SMS Spam Classifier-DSC",layout="centered")

# Custom CSS for colorful theme
st.markdown("""
    <style>
    .main {
            padding: 2rem;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            margin-top: 25px;
        }
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 12px;
        border-radius: 10px;
        border: 2px solid #6a11cb;
    }
    .stButton>button {
        background: linear-gradient(45deg, #6a11cb, #2575fc);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #2575fc, #6a11cb);
        transform: scale(1.05);
    }
    .spam {
        background-color: #ffcccc;
        color: #a80000;
    }
    .ham {
        background-color: #ccffcc;
        color: #006400;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📩 Email / SMS Spam Classifier")
st.write(" ")
st.markdown("#### 🔍 Detect whether a message is **Spam** or **Not Spam**")
st.write(" ")
#st.set_page_config(page_title="📩 Spam Classifier", layout="wide")

# Sidebar
st.sidebar.header("🔧 Settings")
show_examples = st.sidebar.checkbox("Show Example Messages", value=True)

# Input box
input_sms = st.text_area("✏️ Enter your message below:")

# Example messages
if show_examples:
    st.write("💡 Example: *Congratulations! You’ve won a $500 Amazon gift card. Claim it here [Link].*")

if st.button("🚀 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before classifying.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown(f"<div class='result-box spam'>🚫 Spam</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box ham'>✅ Not Spam</div>", unsafe_allow_html=True)



