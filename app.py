import os
import json
import streamlit as st
import asyncio
import nest_asyncio
from typing import List, Dict, Any, Optional
from datetime import timedelta
from dotenv import load_dotenv
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.vector_search import VectorQuery, VectorSearch
import couchbase.search as search
from openai import OpenAI

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Load environment variables
load_dotenv()

class CouchbaseRAG:
    def __init__(self, cluster=None):
        """Initialize with an existing Couchbase cluster connection or create a new one"""
        # Connection details from environment variables
        connection_string = os.getenv("COUCHBASE_CONNECTION_STRING", "couchbases://your-cluster-here?tls_verify=no_verify")
        username = os.getenv("COUCHBASE_USERNAME", "your-username")
        password = os.getenv("COUCHBASE_PASSWORD", "your-password")
        
        # Connect to Couchbase if no cluster provided
        if cluster is None:
            auth = PasswordAuthenticator(username, password)
            options = ClusterOptions(auth)
            self.cluster = Cluster(connection_string, options)
        else:
            self.cluster = cluster
            
        self.bucket_name = "customer_stories"
        self.scope_name = "stories"
        self.collection_name = "docs"
        self.search_index_name = "textembedding"
        
        # Initialize bucket, scope and collection
        self.bucket = self.cluster.bucket(self.bucket_name)
        self.scope = self.bucket.scope(self.scope_name)
        self.collection = self.scope.collection(self.collection_name)
        
        # OpenAI setup
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"))
        
        # Nebius setup
        self.nebius_api_key = os.getenv("NEBIUS_API_KEY", "your-nebius-api-key-here")
        self.nebius_base_url = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the input text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            raise

    async def get_relevant_documents(self, embedding: List[float]) -> List[Dict[str, Any]]:
        """Perform vector search to find relevant documents"""
        try:
            num_results = 10
            vector_field = "embedding"
            
            search_req = search.SearchRequest.create(search.MatchNoneQuery()).with_vector_search(
                VectorSearch.from_vector_query(
                    VectorQuery(vector_field, embedding, num_candidates=num_results)
                )
            )
            
            search_options = SearchOptions(timeout=timedelta(seconds=5.0))
            result = self.scope.search(self.search_index_name, search_req, search_options)
            
            rows = list(result.rows())
            if not rows:
                return []
            
            documents = []
            for row in rows[:5]:
                try:
                    doc_result = self.collection.get(row.id, timeout=timedelta(seconds=2.0))
                    content = doc_result.value
                    if "embedding" in content:
                        del content["embedding"]
                    documents.append({
                        "content": content,
                        "score": row.score,
                        "id": row.id
                    })
                except Exception as err:
                    st.warning(f"Error fetching document {row.id}: {str(err)}")
            
            return documents
        except Exception as e:
            st.error(f"Error in vector search: {str(e)}")
            raise

    async def generate_rag_response(self, messages: List[Dict[str, str]], last_message: str) -> Dict[str, Any]:
        """Generate RAG response using Nebius API"""
        try:
            embedding = self.generate_embedding(last_message)
            documents = await self.get_relevant_documents(embedding)
            
            prompt = f"""
            You are a helpful AI assistant. Use the following context to answer the question.
            If you don't know the answer, say so. Don't make up answers.
            If the question is unrelated to the context, say you're tuned for context-specific questions only.
            <context>
            {''.join([json.dumps(doc["content"]) for doc in documents])}
            </context>
            User Query: {last_message}
            """
            
            nebius_client = OpenAI(
                base_url=self.nebius_base_url,
                api_key=self.nebius_api_key
            )
            
            result = nebius_client.chat.completions.create(
                temperature=0.6,
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=[*messages[:-1], {"role": "user", "content": prompt}],
                stream=True
            )
            
            image_url = None
            try:
                image_response = await nebius_client.images.generate(
                    model="black-forest-labs/flux-schnell",
                    prompt=f"Create an image of {last_message}",
                    response_format="url",
                    size="256x256",
                    quality="standard",
                    n=1
                )
                image_url = image_response.data[0].url
            except Exception:
                pass
            
            return {"text_stream": result, "image_url": image_url}
        except Exception as e:
            st.error(f"Error in RAG response: {str(e)}")
            raise

class StreamlitRAGApp:
    def __init__(self):
        """Initialize the Streamlit app with CouchbaseRAG"""
        self.rag = CouchbaseRAG()
        
        st.set_page_config(
            page_title="Customer Stories RAG Assistant",
            page_icon="📖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
            <style>
            .stChatMessage {
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                background-color: #f5f5f5;
            }
            .user-message {
                background-color: #e3f2fd;
            }
            </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with options and information"""
        with st.sidebar:
            st.title("RAG Assistant Settings")
            st.markdown("---")
            
            st.subheader("About")
            st.info("""
                AI-powered assistant using RAG to answer questions based on 
                customer stories in Couchbase.
            """)
            
            st.subheader("Configuration")
            self.temperature = st.slider("Response Temperature", 0.0, 1.0, 0.6)
            self.max_results = st.number_input("Max Results", 1, 20, 5)
            
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.experimental_rerun()
            
            st.markdown("---")
            st.caption("Built with Streamlit & Couchbase")

    async def process_query(self, messages: List[Dict[str, str]], query: str) -> tuple:
        """Process the user query and return response and image URL"""
        try:
            response = await self.rag.generate_rag_response(messages, query)
            full_response = ""
            for chunk in response["text_stream"]:
                token = chunk.choices[0].delta.content
                if token:
                    full_response += token
            return full_response, response.get("image_url")
        except Exception as e:
            return f"Error processing query: {str(e)}", None

    def render_main_content(self):
        """Render the main content area"""
        st.title("📖 Customer Stories RAG Assistant")
        st.markdown("Ask questions about customer stories and get AI-powered responses!")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        with st.form(key="query_form", clear_on_submit=True):
            query = st.text_input("Enter your question:", placeholder="e.g., What was the reduction percentage for Oracle infrastructure?")
            submit_button = st.form_submit_button(label="Ask")
        
        if submit_button and query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.spinner("Generating response..."):
                loop = asyncio.get_event_loop()
                response_text, image_url = loop.run_until_complete(
                    self.process_query(st.session_state.messages, query)
                )
                st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(
                    message["role"],
                    avatar="👤" if message["role"] == "user" else "🤖"
                ):
                    st.markdown(
                        f'<div class="stChatMessage {"user-message" if message["role"] == "user" else ""}">'
                        f'{message["content"]}</div>',
                        unsafe_allow_html=True
                    )
        
        if submit_button and image_url:
            st.markdown("---")
            st.subheader("Generated Image")
            st.image(image_url, caption=f"Generated image for: {query}", width=300)

    def run(self):
        """Run the Streamlit app"""
        self.render_sidebar()
        self.render_main_content()

if __name__ == "__main__":
    app = StreamlitRAGApp()
    app.run()
