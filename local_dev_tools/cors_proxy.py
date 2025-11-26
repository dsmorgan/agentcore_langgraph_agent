#!/usr/bin/env python3
"""
Combined server for local AgentCore testing.
Serves the test UI and acts as a CORS proxy to localhost:8080.
Supports both local and AWS AgentCore deployments with IAM authentication.
"""

from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
import requests
import os
import yaml
import boto3
import logging
import sys
import json
import uuid
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

AGENT_URL = "http://localhost:8080"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache for boto3 clients
_dataplane_client = None


def load_aws_config():
    """Auto-detect AWS configuration from .bedrock_agentcore.yaml"""
    config = {
        'region': os.environ.get('AWS_REGION', 'us-east-1'),
        'agent_id': None,
        'service': 'bedrock-agentcore-runtime'
    }

    yaml_path = os.path.join(os.path.dirname(CURRENT_DIR), '.bedrock_agentcore.yaml')
    logger.info(f"Looking for AWS config at: {yaml_path}")

    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Navigate through YAML structure
            agent_config = yaml_config.get('agents', {}).get('agent', {})
            aws_config = agent_config.get('aws', {})
            bedrock_config = agent_config.get('bedrock_agentcore', {})

            if 'region' in aws_config:
                config['region'] = aws_config['region']
            if 'agent_id' in bedrock_config:
                config['agent_id'] = bedrock_config['agent_id']

            logger.info(f"Loaded AWS config - Region: {config['region']}, Agent ID: {config['agent_id']}")
        except Exception as e:
            logger.error(f"Could not load AWS config from YAML: {e}")
    else:
        logger.warning(f"AWS config file not found at {yaml_path}")

    return config


def get_bedrock_agentcore_client(region=None):
    """Get boto3 bedrock-agentcore dataplane client."""
    global _dataplane_client

    if _dataplane_client is not None:
        return _dataplane_client

    logger.info("Creating boto3 bedrock-agentcore dataplane client...")
    aws_config = load_aws_config()
    region_name = region or aws_config['region']

    try:
        session = boto3.Session()
        _dataplane_client = session.client('bedrock-agentcore', region_name=region_name)
        logger.info(f"Successfully created bedrock-agentcore client for region: {region_name}")
        return _dataplane_client
    except Exception as e:
        logger.error(f"Failed to create bedrock-agentcore client: {e}")
        raise RuntimeError(f"Failed to create AWS client: {str(e)}") from e


@app.route('/')
def index():
    """Serve the test UI at the root."""
    return send_from_directory(CURRENT_DIR, 'test_ui.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve any other static files from the current directory."""
    try:
        return send_from_directory(CURRENT_DIR, path)
    except:
        return "File not found", 404


def _invoke_local(payload):
    """Invoke local AgentCore server."""
    logger.info(f"[LOCAL MODE] Invoking local agent at {AGENT_URL}/invocations")
    logger.info(f"[LOCAL MODE] Payload: {payload}")

    try:
        agent_response = requests.post(
            f"{AGENT_URL}/invocations",
            json=payload,
            stream=True,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        logger.info(f"[LOCAL MODE] Response status: {agent_response.status_code}")
        logger.info(f"[LOCAL MODE] Response headers: {dict(agent_response.headers)}")

        # Stream the response back with CORS headers
        def generate():
            chunk_count = 0
            total_bytes = 0
            start_time = time.time()
            first_chunk_time = None

            for chunk in agent_response.iter_content(chunk_size=1024):
                if chunk:
                    if chunk_count == 0:
                        first_chunk_time = time.time()
                        logger.info(f"[LOCAL MODE] First chunk received after {first_chunk_time - start_time:.3f}s")

                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk

            duration = time.time() - start_time
            logger.info(f"[LOCAL MODE] Streaming complete: {chunk_count} chunks, {total_bytes} bytes in {duration:.2f}s")
            if first_chunk_time:
                logger.info(f"[LOCAL MODE] Time to first chunk: {first_chunk_time - start_time:.3f}s")
                logger.info(f"[LOCAL MODE] Avg chunk size: {total_bytes / chunk_count:.0f} bytes")

        return Response(
            generate(),
            status=agent_response.status_code,
            headers={
                'Content-Type': agent_response.headers.get('Content-Type', 'application/json'),
                'Access-Control-Allow-Origin': '*'
            }
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[LOCAL MODE] Connection error: {e}")
        return {'error': f'Cannot connect to local agent at {AGENT_URL}. Is it running?'}, 503
    except requests.exceptions.Timeout as e:
        logger.error(f"[LOCAL MODE] Timeout: {e}")
        return {'error': 'Request to local agent timed out'}, 504
    except Exception as e:
        logger.error(f"[LOCAL MODE] Unexpected error: {e}", exc_info=True)
        return {'error': f'Local invocation failed: {str(e)}'}, 500


def _invoke_aws(payload):
    """Invoke AWS AgentCore Runtime using boto3 client."""
    logger.info("[AWS MODE] Starting AWS invocation...")
    aws_config = load_aws_config()

    if not aws_config['agent_id']:
        logger.error("[AWS MODE] Agent ID not found in configuration")
        return {'error': 'Agent ID not found in .bedrock_agentcore.yaml'}, 500

    region = aws_config['region']
    agent_id = aws_config['agent_id']

    # Construct agent ARN from the config
    # The agent ARN format is: arn:aws:bedrock-agentcore:region:account:runtime/agent_id
    # We need to get the full ARN from the YAML
    yaml_path = os.path.join(os.path.dirname(CURRENT_DIR), '.bedrock_agentcore.yaml')
    try:
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        agent_arn = yaml_config.get('agents', {}).get('agent', {}).get('bedrock_agentcore', {}).get('agent_arn')

        if not agent_arn:
            logger.error("[AWS MODE] Agent ARN not found in .bedrock_agentcore.yaml")
            return {'error': 'Agent ARN not found in configuration'}, 500

        logger.info(f"[AWS MODE] Agent ARN: {agent_arn}")
    except Exception as e:
        logger.error(f"[AWS MODE] Failed to read agent ARN: {e}")
        return {'error': f'Failed to read agent configuration: {str(e)}'}, 500

    logger.info(f"[AWS MODE] Payload: {payload}")

    try:
        client = get_bedrock_agentcore_client(region)

        # Generate session ID if not provided (must be at least 33 characters)
        session_id = payload.get('session_id', '') or str(uuid.uuid4())
        logger.info(f"[AWS MODE] Session ID: {session_id}")

        # Prepare the request parameters
        request_params = {
            'agentRuntimeArn': agent_arn,
            'qualifier': 'DEFAULT',  # Use DEFAULT endpoint
            'runtimeSessionId': session_id,
            'payload': json.dumps(payload)
        }

        # Add actor_id as runtimeUserId if present
        if 'actor_id' in payload:
            request_params['runtimeUserId'] = payload['actor_id']

        logger.info(f"[AWS MODE] Invoking agent runtime with boto3...")
        response = client.invoke_agent_runtime(**request_params)

        logger.info(f"[AWS MODE] Response metadata: {response.get('ResponseMetadata', {})}")

        # Handle the response
        content_type = response.get('contentType', 'application/json')
        logger.info(f"[AWS MODE] Response content type: {content_type}")

        if 'text/event-stream' in content_type:
            # Streaming response
            logger.info("[AWS MODE] Handling streaming response")

            def generate():
                chunk_count = 0
                total_bytes = 0
                start_time = time.time()
                first_chunk_time = None

                try:
                    # Get the EventStream from response
                    event_stream = response.get('response')

                    # Diagnostic logging
                    logger.info(f"[AWS MODE] EventStream type: {type(event_stream)}")
                    logger.info(f"[AWS MODE] EventStream attributes: {[x for x in dir(event_stream) if not x.startswith('_')][:10]}")

                    # Use iter_chunks for StreamingBody to get smaller, more frequent chunks (like local mode)
                    if hasattr(event_stream, 'iter_chunks'):
                        logger.info("[AWS MODE] Using iter_chunks(1024) for optimal streaming")
                        chunk_iterator = event_stream.iter_chunks(chunk_size=1024)
                    else:
                        chunk_iterator = event_stream

                    # Iterate over events as they arrive
                    for event in chunk_iterator:
                        if chunk_count == 0:
                            first_chunk_time = time.time()
                            logger.info(f"[AWS MODE] First chunk received after {first_chunk_time - start_time:.3f}s")
                            logger.info(f"[AWS MODE] First event type: {type(event)}")
                            if isinstance(event, dict):
                                logger.info(f"[AWS MODE] First event keys: {list(event.keys())}")

                        # Process the event
                        data = None
                        if isinstance(event, bytes):
                            data = event
                        elif isinstance(event, dict):
                            # Try to extract bytes from dict structure
                            if 'chunk' in event and isinstance(event['chunk'], bytes):
                                data = event['chunk']
                            elif 'bytes' in event:
                                data = event['bytes']
                            else:
                                # Fallback: encode entire event as JSON
                                data = json.dumps(event).encode('utf-8') + b'\n'
                        else:
                            data = json.dumps(event).encode('utf-8') + b'\n'

                        if data:
                            chunk_count += 1
                            total_bytes += len(data)
                            yield data

                    duration = time.time() - start_time
                    logger.info(f"[AWS MODE] Streaming complete: {chunk_count} chunks, {total_bytes} bytes in {duration:.2f}s")
                    if first_chunk_time:
                        logger.info(f"[AWS MODE] Time to first chunk: {first_chunk_time - start_time:.3f}s")
                        logger.info(f"[AWS MODE] Avg chunk size: {total_bytes / chunk_count:.0f} bytes")

                except Exception as e:
                    logger.error(f"[AWS MODE] Error streaming response: {e}", exc_info=True)
                    yield json.dumps({'error': str(e)}).encode('utf-8')

            return Response(
                generate(),
                status=200,
                headers={
                    'Content-Type': content_type,
                    'Access-Control-Allow-Origin': '*'
                }
            )
        else:
            # Non-streaming response
            logger.info("[AWS MODE] Handling non-streaming response")
            result_data = response.get('response', [])

            # Process the response data
            if isinstance(result_data, list):
                # Combine list items into a single response
                combined_response = []
                for item in result_data:
                    if isinstance(item, bytes):
                        combined_response.append(item.decode('utf-8'))
                    else:
                        combined_response.append(str(item))
                result = ''.join(combined_response)
            else:
                result = str(result_data)

            return Response(
                result,
                status=200,
                headers={
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            )

    except Exception as e:
        logger.error(f"[AWS MODE] Unexpected error: {e}", exc_info=True)
        return {'error': f'AWS invocation failed: {str(e)}'}, 500


@app.route('/invocations', methods=['POST', 'OPTIONS'])
def proxy_invocations():
    """Unified endpoint - routes based on X-Target-Environment header."""
    if request.method == 'OPTIONS':
        # Handle preflight request
        logger.info("Handling CORS preflight request")
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Target-Environment'
        return response

    # Forward POST request based on target environment
    try:
        payload = request.get_json()
        target_env = request.headers.get('X-Target-Environment', 'local').lower()

        logger.info("="*60)
        logger.info(f"Received invocation request")
        logger.info(f"Target Environment: {target_env.upper()}")
        logger.info(f"Headers: {dict(request.headers)}")
        logger.info("="*60)

        if target_env == 'aws':
            return _invoke_aws(payload)
        else:
            return _invoke_local(payload)

    except Exception as e:
        logger.error(f"Error in proxy_invocations: {e}", exc_info=True)
        return {'error': str(e)}, 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AgentCore Test Server Starting...")
    print("="*60)
    print("Server running at: http://localhost:9000")
    print("Test UI available at: http://localhost:9000/")
    print("Proxying agent requests to:")
    print("  - Local: http://localhost:8080")

    # Try to load AWS config
    aws_config = load_aws_config()
    if aws_config['agent_id']:
        print(f"  - AWS: {aws_config['region']}/{aws_config['agent_id']}")
    else:
        print("  - AWS: Not configured (no .bedrock_agentcore.yaml found)")

    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=9000, debug=False)
