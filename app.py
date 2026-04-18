import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# إعداد الصفحة
st.set_page_config(
    page_title="الكود المصري لتصميم المستشفيات",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 المساعد الذكي: الكود المصري لتصميم المستشفيات والمنشآت الصحية")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("[Get a Google API key](https://aistudio.google.com/app/apikey)")
    api_key = st.text_input("Google API Key", type="password")
    
    st.markdown("---")
    selected_model = st.selectbox("Google Model", ["gemini-3-flash-preview", "gemini-1.5-flash", "gemini-1.5-pro"])
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
    
    system_prompt = st.text_area("System Prompt", 
        value="أنت مساعد ذكي هندسي متخصص في أكواد البناء والمعايير التصميمية. يجب أن تكون إجابتك باللغة العربية، واضحة، دقيقة، ومنسقة بنقاط إذا لزم الأمر، وتعتمد فقط على المعلومات الموجودة في النص المرفق. إذا لم تكن الإجابة موجودة في النص، قل بوضوح 'عذراً، لا أستطيع إيجاد إجابة لهذا السؤال في الكود المرفق'.", 
        height=150)
    
    st.markdown("Retrieval is fixed to K = 4.")
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    process_btn = st.button("Process PDFs", use_container_width=True)
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    if st.button("Clear PDFs + Chat", use_container_width=True):
        st.session_state.messages = []
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]
        st.rerun()

# --- Process Logic ---
if process_btn:
    if not api_key:
        st.sidebar.error("Please enter your API Key!")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one PDF!")
    else:
        with st.spinner("⏳ جاري معالجة الملفات وبناء قاعدة البيانات... (قد تستغرق بعض الوقت لتفادي حدود الـ API)"):
            try:
                all_documents = []
                # Save uploaded files to temp files so PyPDFLoader can read them
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    all_documents.extend(docs)
                    os.remove(tmp_file_path) # clean up
                
                # Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
                chunks = text_splitter.split_documents(all_documents)
                
                # Create Vector Store
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-001",
                    google_api_key=api_key,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                vectorstore = Chroma(
                    collection_name="dynamic_pdfs",
                    embedding_function=embeddings
                )
                
                import time
                batch_size = 10
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                for i in range(0, len(chunks), batch_size):
                    status_text.text(f"⏳ جاري تجهيز الدفعة {i//batch_size + 1} لتفادي حدود API جوجل...")
                    batch = chunks[i:i+batch_size]
                    
                    # Try to add documents with a simple retry mechanism
                    retries = 3
                    while retries > 0:
                        try:
                            vectorstore.add_documents(batch)
                            break
                        except Exception as e:
                            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                                retries -= 1
                                if retries == 0:
                                    raise e
                                status_text.text(f"⏳ تم الوصول للحد الأقصى مؤقتاً، ننتظر 30 ثانية...")
                                time.sleep(30)
                            else:
                                raise e
                    
                    progress_bar.progress(min((i + batch_size) / len(chunks), 1.0))
                    
                    if i + batch_size < len(chunks):
                        time.sleep(12) # Wait 12 seconds between batches
                        
                status_text.empty()
                progress_bar.empty()
                
                st.session_state.vectorstore = vectorstore
                st.sidebar.success("✅ تم معالجة الملفات بنجاح! يمكنك الآن طرح الأسئلة.")
            except Exception as e:
                st.sidebar.error(f"حدث خطأ أثناء المعالجة: {e}")

# --- Chat Interface ---
if "vectorstore" not in st.session_state:
    st.info("👈 يرجى إدخال الـ API Key ورفع ملفات الـ PDF ثم الضغط على 'Process PDFs' من القائمة الجانبية للبدء.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("اكتب سؤالك هنا..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("⏳ جاري البحث والتفكير..."):
                try:
                    query_embeddings = GoogleGenerativeAIEmbeddings(
                        model="gemini-embedding-001",
                        google_api_key=api_key,
                        task_type="RETRIEVAL_QUERY"
                    )
                    query_vector = query_embeddings.embed_documents([prompt])[0]
                    
                    docs = st.session_state.vectorstore.similarity_search_by_vector(query_vector, k=4)
                    context = "\n\n".join([f"--- صفحة {doc.metadata.get('page', 'غير معروف')} ---\n{doc.page_content}" for doc in docs])
                    
                    llm = ChatGoogleGenerativeAI(
                        model=selected_model,
                        google_api_key=api_key,
                        temperature=temperature
                    )
                    
                    full_prompt = f"{system_prompt}\n\nالسؤال: {prompt}\n\nالنص المقتبس (Context):\n{context}\n\nالإجابة:"
                    
                    response = llm.invoke(full_prompt)
                    
                    if isinstance(response.content, list):
                        answer = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in response.content]).strip()
                    else:
                        answer = response.content.strip()
                        
                    st.markdown(answer)
                    
                    with st.expander("📚 المصادر المرجعية (Source Documents)"):
                        for i, doc in enumerate(docs, 1):
                            page = doc.metadata.get('page', 'غير معروف')
                            st.markdown(f"**المصدر {i} (صفحة {page}):**")
                            st.markdown(f"```text\n{doc.page_content[:350]}...\n```")
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"حدث خطأ أثناء توليد الإجابة: {e}")
