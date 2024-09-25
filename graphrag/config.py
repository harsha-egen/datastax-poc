import logging
import os
import coloredlogs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

# Initialize embeddings and LLM using OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Initialize Astra connection using Cassio
astra_db_id = os.getenv("ASTRA_DB_DATABASE_ID")
astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
astra_endpoint = os.getenv("ASTRA_DB_ENDPOINT")
if not all([astra_db_id, astra_token, astra_endpoint]):
    raise ValueError("Astra DB credentials must be set.")