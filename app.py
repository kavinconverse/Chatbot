import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers,Replicate
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import tempfile

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
os.environ['REPLICATE_API_TOKEN'] = 'r8_41NjdjY0j8nRmUV5GOXibTrftt5k3Wk0c2AIn'

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'Question' not in st.session_state:
        st.session_state['Question'] = ['Hey! 👋']
    if 'Generated' not in st.session_state:
        st.session_state['Generated'] = ['Hello!,Ask me anything 🤗']

def conversation_chat(query,chain,chat_history):
    result = chain({'question':query,'chat_history':chat_history})
    chat_history.append((query,result['answer']))
    return result['answer']

def display_chat_history(chain):
    query_container = st.container()
    reply_container = st.container()

    with query_container:
        with st.form(key = 'my form',clear_on_submit=True):
            user_input = st.text_input('question',placeholder='Ask about documents',key = 'input')
            submit_button = st.form_submit_button(label='Submit')

            if submit_button and user_input:
                with st.spinner('Generating Response...'):
                    output = conversation_chat(user_input,chain,st.session_state['chat_history'])

                st.session_state['Question'].append(user_input)
                st.session_state['Generated'].append(output)

        if st.session_state['Generated']:
            with reply_container:
                for i in range(len(st.session_state['Generated'])):
                    message(st.session_state['Question'][i],is_user=True,key = str(i)+'_user',avatar_style='thumbs')
                    message(st.session_state['Generated'][i],  key=str(i), avatar_style='fun-emoji')


def conversational_chain(vectorstore):
    llm = Replicate(
        streaming = True,
        model = 'meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00',
        callbacks = [StreamingStdOutCallbackHandler()],
        input = {'temperature':0.01,'max_length':500,'top_p':1}
    )
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                  chain_type='stuff',
                                                  retriever= vectorstore.as_retriever(search_kwargs={'k':2}),
                                                  memory=memory
                                                  )
    return chain
def main():
    initialize_session_state()

    st.title('Multi-Documents ChatBot using LLaMA2')
    st.sidebar.title('Documnet Processing')
    uploaded_files = st.sidebar.file_uploader('Upload your files',accept_multiple_files=True)

    if uploaded_files:
        raw_text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

        #load diff files
        loader = None
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx' or file_extension == '.doc':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        if loader:
            raw_text.extend(loader.load())
            os.remove(temp_file_path)

        #st.write(text)
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(raw_text)

    # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Create vector store
        vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)

    # Create the chain object
        chain = conversational_chain(vectorstore)

        display_chat_history(chain)

if __name__ == '__main__':
    main()



