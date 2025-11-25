#!/usr/bin/env python3
"""
Combined server for local AgentCore testing.
Serves the test UI and acts as a CORS proxy to localhost:8080.
"""

from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

AGENT_URL = "http://localhost:8080"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


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


@app.route('/invocations', methods=['POST', 'OPTIONS'])
def proxy_invocations():
    """Proxy requests to the AgentCore local server with CORS headers."""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    # Forward POST request to AgentCore server
    try:
        payload = request.get_json()

        # Forward to actual agent
        agent_response = requests.post(
            f"{AGENT_URL}/invocations",
            json=payload,
            stream=True,
            headers={'Content-Type': 'application/json'}
        )

        # Stream the response back with CORS headers
        def generate():
            for chunk in agent_response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

        return Response(
            generate(),
            status=agent_response.status_code,
            headers={
                'Content-Type': agent_response.headers.get('Content-Type', 'application/json'),
                'Access-Control-Allow-Origin': '*'
            }
        )

    except Exception as e:
        return {'error': str(e)}, 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AgentCore Test Server Starting...")
    print("="*60)
    print("Server running at: http://localhost:9000")
    print("Test UI available at: http://localhost:9000/")
    print("Proxying agent requests to: http://localhost:8080")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=9000, debug=False)
