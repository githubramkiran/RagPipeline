import io
import logging
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(".env")

# Configure the app
st.set_page_config(page_title="SQL and Plot Workflow", page_icon="📊", layout="wide")

# Hide default Streamlit elements
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add title
st.title("📊 ChatBot")

# API endpoint configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:4000")
print(API_BASE_URL)
if not API_BASE_URL:
    st.error("Error: API_BASE_URL environment variable is not set. Please set it to your API endpoint URL.")
    st.stop()


def submit_query(query):
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": query,"thread_id":"thread_id"},
            timeout=10,  # Add timeout to prevent hanging
            headers={"accept": "application/json","Content-Type": "application/json"}
        )
        response.raise_for_status()  # Raise error for bad status codes
        print(response.json())
        st.text_area("response:",response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error submitting query: {e}")
        raise Exception(f"Failed to submit query: {e}")



# Create the main query input
query = st.text_area(
    "Enter your query",
    placeholder="start chatting.",
    height=100,
)
print(query)
# Add a submit button
if st.button("Submit"):
    if query:
        try:
            # Simple status message placeholder
            status_placeholder = st.empty()
            status_placeholder.info("Processing your query... Please wait.")

            # Submit the query
            result = submit_query(query)

        except Exception as e:
            st.error("An error occurred. Please try again.")
            logger.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query first.")

# Add some helpful examples in the sidebar
st.sidebar.header("Example Queries")
st.sidebar.markdown(
    """
- Show me the cost trends by model over the last week
- Compare usage patterns across the top 5 models
- List the top 5 most active users by request count in the last 30 days
- Show token usage by model for the past month
"""
)

# Add information about the project
st.sidebar.header("About")
st.sidebar.markdown(
    """
This dashboard provides a user-friendly interface to the SQL and Plot Workflow API.
It allows you to:
- Submit natural language queries
- Track query processing in real-time
- View generated visualizations
- Download results
"""
)