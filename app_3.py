# """
# BERT-Based Hate Speech Detection Streamlit App
# Uses state-of-the-art transformer model for maximum accuracy
# """

# import streamlit as st
# import torch
# import numpy as np
# import pandas as pd
# import joblib
# from transformers import BertTokenizer, BertForSequenceClassification

# # Configure page
# st.set_page_config(
#     page_title="LGBTQ+ Hate Speech Detection",
#     page_icon="🏳️‍🌈",
#     layout="wide"
# )

# # Styling (same as before)
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
#     }
    
#     .header-container {
#         background: transparent;
#         padding: 2rem;
#         margin-bottom: 2rem;
#         text-align: center;
#         border-bottom: 2px solid rgba(255, 255, 255, 0.1);
#     }
    
#     .main-header {
#         font-size: 2.8rem;
#         font-weight: 800;
#         color: #ffffff;
#         margin-bottom: 0.5rem;
#         text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
#     }
    
#     .sub-header {
#         font-size: 1.1rem;
#         color: #b8b9c4;
#         font-weight: 400;
#     }
    
#     .detector-section {
#         background-color: #0f3460;
#         padding: 2rem;
#         border-radius: 15px;
#         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
#         border: 1px solid rgba(59, 130, 246, 0.2);
#     }
    
#     .detector-header {
#         font-size: 1.8rem;
#         font-weight: 700;
#         color: #ffffff;
#         margin-bottom: 1.5rem;
#         text-align: center;
#     }
    
#     .bert-badge {
#         display: inline-block;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 0.5rem 1rem;
#         border-radius: 20px;
#         font-weight: 600;
#         font-size: 0.9rem;
#         margin-left: 1rem;
#     }
    
#     .detector-section label {
#         color: #e0e0e0 !important;
#         font-weight: 500 !important;
#     }
    
#     .stTextArea textarea {
#         background-color: #1e1e30 !important;
#         color: #ffffff !important;
#         border: 1px solid #3b82f6 !important;
#         border-radius: 10px !important;
#         font-size: 1rem !important;
#     }
    
#     .stButton button {
#         font-size: 1.1rem !important;
#         font-weight: 600 !important;
#         padding: 0.75rem 2rem !important;
#         border-radius: 10px !important;
#         background-color: #3b82f6 !important;
#         border: none !important;
#         color: white !important;
#     }
    
#     .stButton button:hover {
#         background-color: #2563eb !important;
#     }
    
#     .streamlit-expanderHeader {
#         background-color: #0f3460 !important;
#         border-radius: 10px !important;
#         font-weight: 600 !important;
#         color: #ffffff !important;
#     }
    
#     div[data-testid="stExpander"] {
#         background-color: #0f3460 !important;
#         border: 1px solid rgba(59, 130, 246, 0.3) !important;
#         border-radius: 10px !important;
#     }
    
#     .stMarkdown h3, .stMarkdown h4 {
#         color: #ffffff !important;
#     }
    
#     .stMarkdown p, .stMarkdown li {
#         color: #d1d5db !important;
#     }
    
#     .stSelectbox > div > div, .stTextInput > div > div > input {
#         background-color: #1e1e30 !important;
#         color: #ffffff !important;
#         border-color: #3b82f6 !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Load models
# @st.cache_resource
# def load_models():
#     try:
#         # Check device
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load config
#         config = joblib.load('models/bert_model_config.pkl')
        
#         # Load tokenizer
#         tokenizer = BertTokenizer.from_pretrained('models/bert_tokenizer')
        
#         # Load model
#         model = BertForSequenceClassification.from_pretrained(
#             config['model_name'],
#             num_labels=config['num_labels']
#         )
#         model.load_state_dict(torch.load('models/bert_model_best.pt', map_location=device))
#         model.to(device)
#         model.eval()
        
#         return model, tokenizer, config, device
    
#     except FileNotFoundError as e:
#         st.error(f" Model files not found! Please run train_bert_model.py first.\nMissing: {e.filename}")
#         st.stop()
#     except Exception as e:
#         st.error(f" Error loading model: {str(e)}")
#         st.stop()

# # Show loading message
# with st.spinner("Loading BERT model... (this may take 10-30 seconds)"):
#     model, tokenizer, config, device = load_models()

# st.success(f"✓ Model loaded successfully! Using: {device}")

# def predict_bert(text, model, tokenizer, device, max_length=128):
#     """Make prediction using BERT model"""
#     if not text.strip():
#         return 0, [0.5, 0.5]
    
#     # Tokenize
#     encoding = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=max_length,
#         padding='max_length',
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='pt'
#     )
    
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
    
#     # Predict
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
#         prediction = np.argmax(probs)
    
#     return prediction, probs

# def main():
#     # Header
#     st.markdown("""
#     <div class="header-container">
#         <div class="main-header">
#             🏳️‍🌈 LGBTQ+ Hate Speech Detection System
            
#         </div>
        
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Layout
#     col1, col2 = st.columns([1, 3])
    
#     # Left sidebar
#     with col1:
#         with st.expander("🤖 About BERT", expanded=False):
#             st.markdown(f"""
#             **Why BERT is Better:**
            
#             BERT (Bidirectional Encoder Representations from Transformers) understands language context much better than traditional models.
            
#             **Advantages:**
#             - ✨ **Contextual Understanding**: Reads text both left-to-right and right-to-left
#             - ✨ **Pre-trained Knowledge**: Trained on millions of texts
#             - ✨ **Better Accuracy**: 90-95% vs 85% with Random Forest
#             - ✨ **Handles Nuance**: Understands sarcasm, negations, subtle hate
            
#             **Your Model Stats:**
#             - Accuracy: {config.get('test_accuracy', 0)*100:.1f}%
#             - F1-Score: {config.get('test_f1', 0):.3f}
#             - Model: {config.get('model_name', 'bert-base-uncased')}
#             - Parameters: ~110 million
#             """)
        
#         with st.expander("📊 Performance Comparison", expanded=False):
#             st.markdown("""
#             **Model Performance:**
            
#             | Model | Accuracy | F1-Score |
#             |-------|----------|----------|
#             | Random Forest | ~85% | 0.80 |
#             | **BERT (This)** | **~92%** | **0.90** |
#             | Human Annotator | ~95% | 0.93 |
            
#             **What BERT Does Better:**
            
#             1. **Context**: Understands "i am disgusted that i am trans" as self-directed hate
#             2. **Negations**: Gets "not hate" vs "hate" correctly
#             3. **Sarcasm**: Can detect subtle, sarcastic hate speech
#             4. **Relationships**: Understands word relationships and dependencies
#             5. **Rare Words**: Handles slang and new terms better
            
#             **Typical Errors:**
#             - Ambiguous statements without clear context
#             - Very short texts (1-2 words)
#             - Mixed languages
#             - Heavy sarcasm in supportive contexts
#             """)
        
#         st.markdown("---")
#         st.markdown("### 🔍 Analysis Options")
#         option = st.radio(
#             "Choose analysis type:",
#             ["Single Text Analysis", "File Upload"],
#             label_visibility="collapsed"
#         )
    
#     # Main content
#     with col2:
#         st.markdown('<div class="detector-section">', unsafe_allow_html=True)
#         st.markdown('<div class="detector-header">BERT Hate Speech Detector</div>', unsafe_allow_html=True)
        
#         if option == "Single Text Analysis":
#             # Quick test examples
#             st.markdown("#### 💡 Quick Test Examples")
#             example_col1, example_col2, example_col3, example_col4 = st.columns(4)
            
#             with example_col1:
#                 if st.button("Test: Self-Hate", use_container_width=True):
#                     st.session_state.test_text = "i am disgusted that i am trans"
            
#             with example_col2:
#                 if st.button("Test: Direct Hate", use_container_width=True):
#                     st.session_state.test_text = "lgbt people are disgusting"
            
#             with example_col3:
#                 if st.button("Test: Support", use_container_width=True):
#                     st.session_state.test_text = "proud to be who i am"
            
#             with example_col4:
#                 if st.button("Clear", use_container_width=True):
#                     st.session_state.test_text = ""
            
#             # Text input
#             user_input = st.text_area(
#                 "Enter text to analyze:",
#                 height=200,
#                 value=st.session_state.get('test_text', ''),
#                 placeholder="Type or paste text here...\n\nBERT will analyze the full context and meaning of your text.",
#                 help="BERT model analyzes up to 128 tokens of text"
#             )
            
#             # Analyze button
#             if st.button(" Analyze with BERT", type="primary", use_container_width=True):
#                 if user_input.strip():
#                     with st.spinner("BERT is analyzing... (may take 1-3 seconds)"):
#                         prediction, probabilities = predict_bert(
#                             user_input, 
#                             model, 
#                             tokenizer, 
#                             device,
#                             max_length=config['max_length']
#                         )
                    
#                     st.markdown("---")
                    
#                     # Results
#                     col_result1, col_result2 = st.columns([2, 1])
                    
#                     with col_result1:
#                         st.subheader("Analysis Result:")
                        
#                         if prediction == 0:
#                             st.success("✅ **Non-Hate Speech**")
#                             st.info("BERT analyzed the context and determined this text is unlikely to contain hate speech targeting the LGBTQ+ community.")
#                         else:
#                             st.error("⚠️ **Hate Speech Detected**")
#                             st.warning("BERT's contextual analysis indicates this text may contain hateful or offensive content targeting the LGBTQ+ community.")
                    
#                     with col_result2:
#                         st.subheader("Confidence:")
#                         st.metric(
#                             label="Non-Hate",
#                             value=f"{probabilities[0]*100:.1f}%",
#                             delta=None
#                         )
#                         st.metric(
#                             label="Hate",
#                             value=f"{probabilities[1]*100:.1f}%",
#                             delta=None
#                         )
                    
#                     # Confidence visualization
#                     st.markdown("#### Confidence Breakdown")
                    
#                     confidence_df = pd.DataFrame({
#                         'Category': ['Non-Hate Speech', 'Hate Speech'],
#                         'Confidence': [probabilities[0]*100, probabilities[1]*100]
#                     })
                    
#                     st.bar_chart(confidence_df.set_index('Category'))
                    
#                     # Interpretation with BERT-specific insights
#                     confidence_level = max(probabilities)
                    
#                     if confidence_level < 0.6:
#                         st.info("ℹ **Low Confidence** (< 60%): BERT is uncertain. The text may be ambiguous, sarcastic, or require more context.")
#                     elif confidence_level < 0.8:
#                         st.info("**Moderate Confidence** (60-80%): BERT has reasonable certainty based on contextual patterns.")
#                     elif confidence_level < 0.95:
#                         st.success("**High Confidence** (80-95%): BERT is very certain based on strong contextual signals.")
#                     else:
#                         st.success(" **Very High Confidence** (> 95%): BERT detected clear and unambiguous patterns.")
                    
#                     # Show what BERT "sees"
#                     with st.expander(" What BERT Analyzed"):
#                         st.markdown("**Original Text:**")
#                         st.code(user_input)
                        
#                         # Tokenize and show tokens
#                         tokens = tokenizer.tokenize(user_input)
#                         st.markdown(f"**Tokens ({len(tokens)} tokens):**")
#                         st.code(" | ".join(tokens[:30]) + ("..." if len(tokens) > 30 else ""))
                        
#                         st.markdown("**How BERT Processes:**")
#                         st.markdown("""
#                         1.  Splits text into subword tokens
#                         2.  Adds special tokens [CLS] and [SEP]
#                         3.  Creates contextual embeddings (768-dimensional vectors)
#                         4.  Reads text bidirectionally (left-to-right AND right-to-left)
#                         5.  Applies attention mechanism to understand relationships
#                         6.  Classifies based on learned hate speech patterns
#                         """)
                        
#                         # Show approximate processing time
#                         st.info(f" Processing time: ~1-3 seconds\n Model size: ~400 MB in memory")
#                 else:
#                     st.warning(" Please enter some text to analyze")
        
#         elif option == "File Upload":
#             st.markdown("####  Batch Analysis with BERT")
#             st.info("Upload a CSV file with a 'text' column. BERT will analyze each entry.")
            
#             uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
            
#             if uploaded_file:
#                 try:
#                     df = pd.read_csv(uploaded_file)
                    
#                     if 'text' not in df.columns:
#                         st.error(" CSV must have a 'text' column")
#                     else:
#                         st.write(" **Preview:**")
#                         st.dataframe(df.head(3), use_container_width=True)
                        
#                         # Warning for large files
#                         if len(df) > 100:
#                             st.warning(f" Large file detected ({len(df)} rows). Processing may take several minutes.")
                        
#                         if st.button("🔍 Analyze with BERT", use_container_width=True):
#                             predictions = []
#                             confidences = []
                            
#                             # Progress bar
#                             progress_bar = st.progress(0)
#                             status_text = st.empty()
                            
#                             for idx, text in enumerate(df['text']):
#                                 status_text.text(f"Analyzing {idx+1}/{len(df)}...")
                                
#                                 pred, probs = predict_bert(
#                                     str(text), 
#                                     model, 
#                                     tokenizer, 
#                                     device,
#                                     max_length=config['max_length']
#                                 )
                                
#                                 predictions.append(pred)
#                                 confidences.append(probs[pred])
                                
#                                 progress_bar.progress((idx + 1) / len(df))
                            
#                             status_text.empty()
#                             progress_bar.empty()
                            
#                             df['Prediction'] = predictions
#                             df['Result'] = df['Prediction'].map({0: 'Non-Hate', 1: 'Hate'})
#                             df['Confidence'] = [f"{c*100:.1f}%" for c in confidences]
                            
#                             st.success(" BERT analysis complete!")
                            
#                             # Summary statistics
#                             col1, col2, col3, col4 = st.columns(4)
#                             with col1:
#                                 st.metric("Total Analyzed", len(df))
#                             with col2:
#                                 hate_count = sum(predictions)
#                                 st.metric("Hate Speech", hate_count)
#                             with col3:
#                                 non_hate_count = len(df) - hate_count
#                                 st.metric("Non-Hate", non_hate_count)
#                             with col4:
#                                 avg_conf = np.mean(confidences) * 100
#                                 st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                            
#                             # Results table
#                             st.write("📊 **Results:**")
#                             st.dataframe(df[['text', 'Result', 'Confidence']], use_container_width=True)
                            
#                             # Download button
#                             csv = df.to_csv(index=False)
#                             st.download_button(
#                                 " Download Results",
#                                 csv,
#                                 "bert_analysis_results.csv",
#                                 "text/csv",
#                                 use_container_width=True
#                             )
#                 except Exception as e:
#                     st.error(f" Error processing file: {str(e)}")
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         f"""
#         <div style='text-align: center; color: #b8b9c4; padding: 1rem;'>
#         <p> <strong>BERT Model</strong> - {config.get('model_name', 'bert-base-uncased')}</p>
#         <p style='font-size: 0.9rem;'>Accuracy: {config.get('test_accuracy', 0)*100:.1f}% | F1-Score: {config.get('test_f1', 0):.3f} | {device}</p>
#         <p style='font-size: 0.85rem; margin-top: 0.5rem;'>This tool uses state-of-the-art AI for educational and moderation purposes.</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# if __name__ == "__main__":
#     main()

"""
BERT-Based Hate Speech Detection Streamlit App
Uses state-of-the-art transformer model for maximum accuracy
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

# Configure page
st.set_page_config(
    page_title="LGBTQ+ Hate Speech Detection",
    page_icon="🏳️‍🌈",
    layout="wide"
)

# Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .header-container {
        background: transparent;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .detector-section {
        background-color: #0f3460;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .detector-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .detector-section label {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }
    
    .stTextArea textarea {
        background-color: #1e1e30 !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }
    
    .stButton button {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
        background-color: #3b82f6 !important;
        border: none !important;
        color: white !important;
    }
    
    .stButton button:hover {
        background-color: #2563eb !important;
    }
    
    .streamlit-expanderHeader {
        background-color: #0f3460 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    
    div[data-testid="stExpander"] {
        background-color: #0f3460 !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
    }
    
    .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: #d1d5db !important;
    }
    
    .stSelectbox > div > div, .stTextInput > div > div > input {
        background-color: #1e1e30 !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load models
@st.cache_resource
def load_models():
    try:
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        config = joblib.load('models/bert_model_config.pkl')
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained('models/bert_tokenizer')
        
        # Load model
        model = BertForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=config['num_labels']
        )
        model.load_state_dict(torch.load('models/bert_model_best.pt', map_location=device))
        model.to(device)
        model.eval()
        
        return model, tokenizer, config, device
    
    except FileNotFoundError as e:
        st.error(f"Model files not found! Please run train_bert_model.py first.\nMissing: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Show loading message
with st.spinner("Loading BERT model... (this may take 10-30 seconds)"):
    model, tokenizer, config, device = load_models()

st.success(f"Model loaded successfully! Using: {device}")

def predict_bert(text, model, tokenizer, device, max_length=128):
    """Make prediction using BERT model"""
    if not text.strip():
        return 0, [0.5, 0.5]
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = np.argmax(probs)
    
    return prediction, probs

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="main-header">
            LGBTQ+ Hate Speech Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout
    col1, col2 = st.columns([1, 3])
    
    # Left sidebar
    with col1:
        with st.expander("About This Application", expanded=False):
            st.markdown("""
            **Purpose:**
            
            This application uses advanced artificial intelligence to detect hate speech targeting the LGBTQ+ community. It's designed to help moderators, researchers, and community managers identify potentially harmful content.
            
            **How It Works:**
            
            The system analyzes text input and classifies it as either hate speech or non-hate speech based on patterns learned from thousands of examples. It provides confidence scores to help users understand how certain the model is about its predictions.
            
            **Use Cases:**
            - Social media content moderation
            - Research on online hate speech
            - Educational purposes
            - Community safety monitoring
            
            **Important Notes:**
            - This tool is an aid, not a replacement for human judgment
            - Results should be reviewed in context
            - The model may not catch all nuanced cases
            - Use responsibly and ethically
            """)
        
        with st.expander("About BERT", expanded=False):
            st.markdown(f"""
            **What is BERT?**
            
            BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art natural language processing model developed by Google. It revolutionized how machines understand human language.
            
            **Key Features:**
            
            - **Contextual Understanding**: BERT reads text in both directions (left-to-right and right-to-left) to understand context better than traditional models
            
            - **Pre-trained Knowledge**: Trained on millions of texts, giving it deep understanding of language patterns
            
            - **Transfer Learning**: Can be fine-tuned for specific tasks like hate speech detection
            
            - **Attention Mechanism**: Focuses on important words and their relationships in a sentence
            
            **Technical Specifications:**
            - Model: {config.get('model_name', 'bert-base-uncased')}
            - Parameters: ~110 million
            - Accuracy: {config.get('test_accuracy', 0)*100:.1f}%
            - F1-Score: {config.get('test_f1', 0):.3f}
            
            **Why BERT for Hate Speech?**
            
            BERT excels at understanding nuanced language, including sarcasm, negations, and context-dependent meanings - all crucial for accurately detecting hate speech.
            """)
        
        st.markdown("---")
        st.markdown("### Analysis Options")
        option = st.radio(
            "Choose analysis type:",
            ["Single Text Analysis", "File Upload"],
            label_visibility="collapsed"
        )
    
    # # Main content
    # with col2:
    #     st.markdown('<div class="detector-section">', unsafe_allow_html=True)
    #     st.markdown('<div class="detector-header">BERT Hate Speech Detector</div>', unsafe_allow_html=True)
        
    #     if option == "Single Text Analysis":
    #         # Quick test examples
    #         st.markdown("#### Quick Test Examples")
    #         example_col1, example_col2, example_col3, example_col4 = st.columns(4)
            
    #         with example_col1:
    #             if st.button("Test: Self-Hate", use_container_width=True):
    #                 st.session_state.test_text = "i am disgusted that i am trans"
            
    #         with example_col2:
    #             if st.button("Test: Direct Hate", use_container_width=True):
    #                 st.session_state.test_text = "lgbt people are disgusting"
            
    #         with example_col3:
    #             if st.button("Test: Support", use_container_width=True):
    #                 st.session_state.test_text = "proud to be who i am"
            
    #         with example_col4:
    #             if st.button("Clear", use_container_width=True):
    #                 st.session_state.test_text = ""
    # Main content
# Main content
    with col2:
        st.markdown('<div class="detector-section">', unsafe_allow_html=True)
        st.markdown('<div class="detector-header">BERT Hate Speech Detector</div>', unsafe_allow_html=True)
        
        if option == "Single Text Analysis":
            # Text input
            user_input = st.text_area(
                "Enter text to analyze:",
                height=300,
                value=st.session_state.get('test_text', ''),
                placeholder="Type or paste text here...\n\nBERT will analyze the full context and meaning of your text.",
                help="BERT model analyzes up to 128 tokens of text"
            )
            
            # Analyze button
            if st.button("Analyze with BERT", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("BERT is analyzing... (may take 1-3 seconds)"):
                        prediction, probabilities = predict_bert(
                            user_input, 
                            model, 
                            tokenizer, 
                            device,
                            max_length=config['max_length']
                        )
                    
                    st.markdown("---")
                    
                    # Results
                    col_result1, col_result2 = st.columns([2, 1])
                    
                    with col_result1:
                        st.subheader("Analysis Result:")
                        
                        if prediction == 0:
                            st.success("Non-Hate Speech")
                            st.info("BERT analyzed the context and determined this text is unlikely to contain hate speech targeting the LGBTQ+ community.")
                        else:
                            st.error("Hate Speech Detected")
                            st.warning("BERT's contextual analysis indicates this text may contain hateful or offensive content targeting the LGBTQ+ community.")
                    
                    with col_result2:
                        st.subheader("Confidence:")
                        st.metric(
                            label="Non-Hate",
                            value=f"{probabilities[0]*100:.1f}%",
                            delta=None
                        )
                        st.metric(
                            label="Hate",
                            value=f"{probabilities[1]*100:.1f}%",
                            delta=None
                        )
                    
                    # Interpretation
                    confidence_level = max(probabilities)
                    
                    if confidence_level < 0.6:
                        st.info("Low Confidence (< 60%): BERT is uncertain. The text may be ambiguous, sarcastic, or require more context.")
                    elif confidence_level < 0.8:
                        st.info("Moderate Confidence (60-80%): BERT has reasonable certainty based on contextual patterns.")
                    elif confidence_level < 0.95:
                        st.success("High Confidence (80-95%): BERT is very certain based on strong contextual signals.")
                    else:
                        st.success("Very High Confidence (> 95%): BERT detected clear and unambiguous patterns.")
                    
                    # Show what BERT "sees"
                    with st.expander("What BERT Analyzed"):
                        st.markdown("**Original Text:**")
                        st.code(user_input)
                        
                        # Tokenize and show tokens
                        tokens = tokenizer.tokenize(user_input)
                        st.markdown(f"**Tokens ({len(tokens)} tokens):**")
                        st.code(" | ".join(tokens[:30]) + ("..." if len(tokens) > 30 else ""))
                        
                        st.markdown("**How BERT Processes:**")
                        st.markdown("""
                        1. Splits text into subword tokens
                        2. Adds special tokens [CLS] and [SEP]
                        3. Creates contextual embeddings (768-dimensional vectors)
                        4. Reads text bidirectionally (left-to-right AND right-to-left)
                        5. Applies attention mechanism to understand relationships
                        6. Classifies based on learned hate speech patterns
                        """)
                        
                        st.info(f"Processing time: ~1-3 seconds | Model size: ~400 MB in memory")
                else:
                    st.warning("Please enter some text to analyze")
        
        elif option == "File Upload":
            st.markdown("#### Batch Analysis with BERT")
            st.info("Upload a CSV file with a 'text' column. BERT will analyze each entry.")
            
            uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' not in df.columns:
                        st.error("CSV must have a 'text' column")
                    else:
                        st.write("**Preview:**")
                        st.dataframe(df.head(3), use_container_width=True)
                        
                        # Warning for large files
                        if len(df) > 100:
                            st.warning(f"Large file detected ({len(df)} rows). Processing may take several minutes.")
                        
                        if st.button("Analyze with BERT", use_container_width=True):
                            predictions = []
                            confidences = []
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, text in enumerate(df['text']):
                                status_text.text(f"Analyzing {idx+1}/{len(df)}...")
                                
                                pred, probs = predict_bert(
                                    str(text), 
                                    model, 
                                    tokenizer, 
                                    device,
                                    max_length=config['max_length']
                                )
                                
                                predictions.append(pred)
                                confidences.append(probs[pred])
                                
                                progress_bar.progress((idx + 1) / len(df))
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            df['Prediction'] = predictions
                            df['Result'] = df['Prediction'].map({0: 'Non-Hate', 1: 'Hate'})
                            df['Confidence'] = [f"{c*100:.1f}%" for c in confidences]
                            
                            st.success("BERT analysis complete!")
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Analyzed", len(df))
                            with col2:
                                hate_count = sum(predictions)
                                st.metric("Hate Speech", hate_count)
                            with col3:
                                non_hate_count = len(df) - hate_count
                                st.metric("Non-Hate", non_hate_count)
                            with col4:
                                avg_conf = np.mean(confidences) * 100
                                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                            
                            # Results table
                            st.write("**Results:**")
                            st.dataframe(df[['text', 'Result', 'Confidence']], use_container_width=True)
                            
                            # Download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "Download Results",
                                csv,
                                "bert_analysis_results.csv",
                                "text/csv",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #b8b9c4; padding: 1rem;'>
        <p><strong>BERT Model</strong> - {config.get('model_name', 'bert-base-uncased')}</p>
        <p style='font-size: 0.9rem;'>Accuracy: {config.get('test_accuracy', 0)*100:.1f}% | F1-Score: {config.get('test_f1', 0):.3f} | {device}</p>
        <p style='font-size: 0.85rem; margin-top: 0.5rem;'>This tool uses state-of-the-art AI for educational and moderation purposes.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()