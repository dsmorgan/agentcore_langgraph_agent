"""
Tool initialization module for LangGraph agent.
Handles setup of Tavily and Wikipedia search tools.
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)


def load_tavily_api_key() -> str | None:
    """
    Load Tavily API key from multiple sources in priority order:
    1. Environment variable (TAVILY_API_KEY)
    2. Local config.py file
    3. .env file
    4. AWS Secrets Manager

    Returns:
        API key string if found, None otherwise
    """

    # Check environment variable first
    if "TAVILY_API_KEY" in os.environ:
        logger.info("Loaded TAVILY_API_KEY from environment variable")
        return os.environ["TAVILY_API_KEY"]

    # Try importing from local config.py
    try:
        from config import TAVILY_API_KEY
        logger.info("Loaded TAVILY_API_KEY from config.py")
        os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
        return TAVILY_API_KEY
    except (ImportError, AttributeError):
        logger.debug("Could not load from config.py")

    # Try .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if "TAVILY_API_KEY" in os.environ:
            logger.info("Loaded TAVILY_API_KEY from .env file")
            return os.environ["TAVILY_API_KEY"]
    except ImportError:
        logger.debug("dotenv not available")

    # Try AWS Secrets Manager
    try:
        import boto3
        import json
        secrets_client = boto3.client('secretsmanager', region_name='us-east-1')
        secret = secrets_client.get_secret_value(SecretId='tavily-api-key')
        api_key = json.loads(secret['SecretString']).get('api_key')
        if api_key:
            logger.info("Loaded TAVILY_API_KEY from AWS Secrets Manager")
            os.environ['TAVILY_API_KEY'] = api_key
            return api_key
    except Exception as e:
        logger.debug(f"Could not load from AWS Secrets Manager: {e}")

    logger.warning("TAVILY_API_KEY not found in any source")
    return None


def load_search_tools() -> List:
    """
    Load and initialize available search tools (Tavily and Wikipedia).

    Returns:
        List of initialized tool objects
    """
    tools = []

    # Load Tavily search tool
    tavily_api_key = load_tavily_api_key()
    if tavily_api_key:
        try:
            from langchain_tavily import TavilySearch
            tavily_search = TavilySearch(
                max_results=5,
                search_depth="basic",
                api_key=tavily_api_key
            )
            tools.append(tavily_search)
            logger.info("Tavily search tool loaded successfully")
        except ImportError as e:
            logger.warning(f"Could not import TavilySearch: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Tavily search tool: {e}")
    else:
        logger.info("Skipping Tavily search tool - no API key available")

    # Load Wikipedia search tool
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        wikipedia_wrapper = WikipediaAPIWrapper()
        wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
        tools.append(wikipedia_tool)
        logger.info("Wikipedia search tool loaded successfully")
    except ImportError as e:
        logger.warning(f"Could not import Wikipedia tool: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize Wikipedia search tool: {e}")

    if not tools:
        logger.warning("No search tools available - agent will operate without tools")

    return tools
