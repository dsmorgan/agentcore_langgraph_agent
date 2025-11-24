# agentcore_langgraph_agent
Experimentation using Langraph and AWS Agentcore to build a basic chatbot agent

## Prerequisites
- Python 3.12+
- uv
- AWS account with Bedrock/Bedrock Agentcore access

## Setup
```bash
# Install uv if you don't have it already
pip install uv

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  

# Install requirements
uv pip install -r requirements.txt

# Configure your agent for deployment
agentcore configure

# Deploy the agent into AWS, use --local flag for local development
agentcore launch
```

During configure, you'll be prompted for:
- AWS region
- A deployment name
- Other deployment settings
    - Entrypoint Selection: agent.py
    - agent name: agent
    - dependency file: requirements.txt
    - Deployment type: Container
    - auto-create execution role
    - auto-create ECR repository
    - default IAM authorization
    - default request header configuration


## Tools
- Wikipedia
- Tavily

For local development, you can store the Tavily API key in .env but for AWS, that won't work. In that case you can either use AWS secrets manager, or store it as an environmental variable in the Agent Runtime, under advanced configuration.




