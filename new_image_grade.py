import dash
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
import base64
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import torchvision.models.densenet as densenet
import torch.nn.functional as F
from datetime import datetime
import pandas as pd
import requests
import json
import time
from typing import List, Dict
import threading

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Load model
with torch.serialization.safe_globals([densenet.DenseNet]):
    model = torch.load(r"DenseNet169model.pth", map_location=torch.device('cpu'), weights_only=False)
model.eval()

# Globals
global_img = {'imgs': []}
session_history = []
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
labels = ["Grade-I", "Grade-II", "Grade-III", "Grade-IV"]

PROMPT_TEMPLATES = {
    "system": """[INSTRUCTIONS]
You are a medical AI assistant for tumor grading. Follow these guidelines:

1. **When asked about the developer**:
"This application was developed by Muhammad Umar Attique, an Artificial Intelligence student at Islamia University of Bahawalpur, Pakistan. 

Key details about the project:
- Developed as a research initiative in medical AI
- Uses DenseNet169 deep learning architecture
- Specifically designed to assist pathologists with tumor grading (Grade I-IV)
- Created to bridge AI technology with clinical practice"

2. **For greetings**:
"Hello! I'm an AI assistant specialized in tumor pathology. How may I assist you with medical image analysis today?"

3. **Technical inquiries**:
"The system uses:
✓ DenseNet169 - a convolutional neural network
✓ Pretrained on histopathological images
✓ Fine-tuned for tumor grading accuracy
✓ Provides Grade I-IV classification with confidence scores"

4. **Application features**:
"Medical Tumor Grading Assistant capabilities:
- AI-powered histopathology analysis
- Grading according to WHO standards
- Case documentation tools
- Research-grade accuracy metrics"

5. **Medical questions**:
[Provide concise professional answers about tumor grading]

6. **Off-topic questions**:
"I specialize in histopathological image analysis. Would you like information about tumor grading standards or how to use this diagnostic tool?"
"""
}

# --- Add custom CSS for the blinking dot ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Medical Tumor Grading Assistant</title>
        {%favicon%}
        {%css%}
        <style>
            .blinking-dot {
                height: 10px;
                width: 10px;
                background-color: #007bff;
                border-radius: 50%;
                display: inline-block;
                margin-left: 5px;
                animation: blink 1s infinite;
            }
            @keyframes blink {
                0% { opacity: 1; }
                50% { opacity: 0; }
                100% { opacity: 1; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = dbc.Container([
    # Hidden store for handling enter key
    dcc.Store(id='enter-key-pressed', data=False),
    # Store to trigger scroll (you might not need this if clientside auto-scroll works fine)
    dcc.Store(id='scroll-trigger', data=0),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("\U0001F916 ChatBot Assistant", className="text-light mb-3"),
                html.Div(
                    id='chat-response',
                    style={
                        'height': 'calc(100vh - 200px)',
                        'overflowY': 'auto',
                        'padding': '10px',
                        'backgroundColor': '#343a40',
                        'borderRadius': '5px',
                        'color': 'white',
                        'marginBottom': '15px',
                        'scrollBehavior': 'smooth'  # fixed typo here
                    }
                ),
                # New chat input block with embedded send icon
                html.Div([
                    html.Div([
                        dcc.Textarea(
                            id='user-input',
                            placeholder='Type your message here...',
                            style={
                                'width': '100%',
                                'height': '80px',
                                'resize': 'none',
                                'padding': '10px 40px 10px 10px',  # space for the icon
                                'borderRadius': '5px',
                                'border': '1px solid #495057',
                                'backgroundColor': '#343a40',
                                'color': 'white',
                                'position': 'relative'
                            }
                        ),
                        html.Span(
                            '\u27A4',  # Send icon character
                            id='send-icon',
                            n_clicks=0,
                            style={
                                'cursor': 'pointer',
                                'fontSize': '20px',
                                'position': 'absolute',
                                'right': '10px',
                                'bottom': '10px',
                                'color': '#6c757d',
                                'zIndex': '10'
                            }
                        ),
                        dbc.Tooltip(id="send-tooltip", target="send-icon", placement="top")
                    ], style={'position': 'relative'}),
                ], style={'marginBottom': '15px'})
            ], style={
                'height': '100vh',
                'overflow': 'hidden',
                'display': 'flex',
                'flexDirection': 'column',
                'padding': '20px'
            })
        ], width=3, style={'backgroundColor': '#2b2b3b'}),
        dbc.Col([
            html.Br(),
            html.H2("\U0001F9E0 Medical Tumor Image Grading System", className="text-center text-light"),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div(['\U0001F4E4 Drag and Drop or ', html.A('Select Images')]),
                        style={
                            'width': '100%',
                            'height': '100px',
                            'lineHeight': '100px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'backgroundColor': '#2c3e50',
                            'color': 'white'
                        },
                        multiple=True
                    ),
                    html.Div(id='output-image', style={'marginTop': 20, 'textAlign': 'center'}),
                    dbc.Button("\U0001F50D Predict", id='predict-btn', color='info', className='w-100 mt-3'),
                    dbc.Spinner(html.Div(id="prediction-output"), color="light", type="border", spinner_class_name="mt-3")
                ], width=12)
            ]),
            html.Hr(),
            html.H4("\U0001F4CA Session History", className="text-light"),
            html.Div(id='history-table', className="text-light mt-3"),
            dbc.Button("\U0001F4E5 Download CSV", id="download-btn", color='success', className="mt-3"),
            dcc.Download(id="download-data")
        ], width=9, style={'backgroundColor': '#1e1e3f', 'padding': '20px', 'height': '100vh', 'overflowY': 'auto'})
    ])
], fluid=True, style={'backgroundColor': '#1e1e3f', 'padding': '0'})

# Store for streaming responses
response_store = {'text': '', 'complete': False}

# Add interval component for streaming updates
app.layout.children.append(dcc.Interval(id='interval-component', interval=50, n_intervals=0, disabled=True))

# JavaScript for handling Enter key and auto-scroll
app.clientside_callback(
    """
    function(value, children) {
        // Handle Enter key press in the textarea
        const textarea = document.getElementById('user-input');
        if (textarea) {
            textarea.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    document.getElementById('send-icon').click();
                }
            });
        }
        // Auto-scroll functionality
        const chatContainer = document.getElementById('chat-response');
        if (chatContainer) {
            setTimeout(() => {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }, 50);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('enter-key-pressed', 'data'),
    [Input('user-input', 'value'),
     Input('chat-response', 'children')],
    prevent_initial_call=True
)

@app.callback(
    Output('scroll-trigger', 'data', allow_duplicate=True),
    Input('chat-response', 'children'),
    prevent_initial_call=True
)
def trigger_initial_scroll(children):
    return True

# Tooltip callback for send icon
@app.callback(
    Output('send-tooltip', 'children'),
    Input('send-icon', 'n_clicks')
)
def update_send_tooltip(n_clicks):
    if response_store.get("streaming", False):
        return "Stop"
    return "Send"

# Callback for handling user messages
@app.callback(
    [Output('chat-response', 'children', allow_duplicate=True),
     Output('user-input', 'value'),
     Output('send-icon', 'style'),
     Output('interval-component', 'disabled')],
    Input('send-icon', 'n_clicks'),
    [State('user-input', 'value'),
     State('chat-response', 'children')],
    prevent_initial_call=True
)
def handle_user_message(n_clicks, query, current_chat):
    if n_clicks and query:
        if not response_store.get('streaming', False):
            # Start a new response stream
            response_store['streaming'] = True
            response_store['stop'] = False
            response_store['text'] = ''
            response_store['complete'] = False
            send_icon_style = {
                'cursor': 'not-allowed',
                'fontSize': '20px',
                'position': 'absolute',
                'right': '20px',
                'bottom': '10px',
                'color': '#adb5bd'
            }
            user_message = html.Div(
                f"👤: {query}",
                style={
                    'padding': '8px 12px',
                    'marginBottom': '10px',
                    'backgroundColor': '#495057',
                    'borderRadius': '5px',
                    'color': 'white'
                }
            )
            bot_response = html.Div(
                children=[
                    html.Span("🤖 : ", style={'marginRight': '6px'}),
                    # Initially show "Processing..." and then the blinking dot until complete
                    html.Span(id='streaming-response', children="Processing...")
                ],
                style={
                    'padding': '8px 12px',
                    'marginBottom': '10px',
                    'backgroundColor': '#343a40',
                    'borderRadius': '5px',
                    'color': 'white',
                    'border': '1px solid #495057'
                }
            )
            updated_chat = (current_chat or []) + [user_message, bot_response]
            response_store['text'] = ''
            threading.Thread(target=stream_llm_response, args=(query,), daemon=True).start()
            return updated_chat, '', send_icon_style, False
        else:
            # Stop the ongoing response stream
            response_store['stop'] = True
            response_store['complete'] = True
            response_store['streaming'] = False
            send_icon_style = {
                'cursor': 'pointer',
                'fontSize': '20px',
                'position': 'absolute',
                'right': '20px',
                'bottom': '10px',
                'color': '#6c757d'
            }
            return dash.no_update, dash.no_update, send_icon_style, True
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback to update streaming response text (with blinking dot)
@app.callback(
    Output('streaming-response', 'children'),
    Output('interval-component', 'disabled', allow_duplicate=True),
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=True
)
def update_streaming_response(n):
    # If not complete, show the current text plus a blinking blue dot
    if not response_store['complete']:
        return [response_store['text'], html.Span("", className="blinking-dot")], False
    else:
        return response_store['text'], True

def stream_llm_response(prompt):
    full_response = ""
    try:
        # Build full prompt using the system instructions
        full_prompt = f"""
        {PROMPT_TEMPLATES["system"]}
        User: {prompt}
        Assistant:"""
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma:2b",
            "prompt": full_prompt,
            "stream": True
        }, stream=True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if response_store.get('stop', False):
                    response.close()  # Explicitly close connection when stopping
                    break
                if line:
                    decoded_line = line.decode('utf-8')
                    json_response = json.loads(decoded_line)
                    if not json_response.get("done", False):
                        chunk = json_response.get("response", "")
                        full_response += chunk
                        response_store['text'] = full_response
                        time.sleep(0.05)
            response_store['complete'] = True
        else:
            response_store['text'] = f"Error: Failed to connect (Status: {response.status_code})"
            response_store['complete'] = True
    except Exception as e:
        response_store['text'] = f"Error: {str(e)}"
        response_store['complete'] = True
    response_store['streaming'] = False 

# Callback for generating LLM response (alternative approach)
@app.callback(
    [Output('send-icon', 'style', allow_duplicate=True),
     Output('interval-component', 'disabled', allow_duplicate=True)],
    Input('send-icon', 'n_clicks'),
    State('user-input', 'value'),
    prevent_initial_call=True
)
def get_llm_response(n_clicks, query):
    if n_clicks and query and not response_store.get('streaming', False):
        try:
            query_lower = query.lower()
            if any(kw in query_lower for kw in ["about this application", "about this app", "what can you do", "who are you"]):
                prompt = PROMPT_TEMPLATES["system"]
            else:
                prompt = PROMPT_TEMPLATES["system"] + f"\nUser: {query}"
            threading.Thread(target=stream_llm_response, args=(prompt,), daemon=True).start()
            send_icon_style = {
                'cursor': 'not-allowed',
                'fontSize': '20px',
                'position': 'absolute',
                'right': '20px',
                'bottom': '10px',
                'color': '#adb5bd'
            }
            return send_icon_style, False
        except Exception as e:
            response_store['text'] = f"Error: {str(e)}"
            response_store['complete'] = True
            return dash.no_update, True
    return dash.no_update, dash.no_update

# Callback for displaying uploaded images
@app.callback(
    Output('output-image', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def display_images(contents, filenames):
    if contents and filenames:
        global_img['imgs'] = []
        images = []
        for content, name in zip(contents, filenames):
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            img = Image.open(io.BytesIO(decoded)).convert("RGB")
            global_img['imgs'].append(img)
            images.append(html.Div([
                html.Img(src=content, style={'maxWidth': '100%', 'maxHeight': '200px'}),
                html.P(name, className='text-light')
            ], style={'textAlign': 'center', 'padding': '10px'}))
        return dbc.Row([dbc.Col(img, width=4) for img in images])
    return None

# Prediction callback for medical image grading
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks')
)
def predict_batch(n_clicks):
    if n_clicks and global_img['imgs']:
        results = []
        for i, img in enumerate(global_img['imgs']):
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output[0], dim=0)
                top_probs, top_indices = torch.topk(probabilities, 3)
            session_history.append({
                "Filename": f"Image_{i+1}.png",
                "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Top1": labels[top_indices[0].item()],
                "Top1_Conf": f"{top_probs[0].item() * 100:.2f}%",
                "Top2": labels[top_indices[1].item()],
                "Top2_Conf": f"{top_probs[1].item() * 100:.2f}%",
                "Top3": labels[top_indices[2].item()],
                "Top3_Conf": f"{top_probs[2].item() * 100:.2f}%"
            })
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            prediction_list = [
                html.Li(f"{j+1}. {labels[top_indices[j].item()]} ({top_probs[j].item() * 100:.2f}%)")
                for j in range(3)
            ]
            results.append(dbc.Col([
                html.Img(src=f'data:image/png;base64,{img_b64}', style={'maxWidth': '100%', 'height': '200px'}),
                html.Div([
                    html.H6("Top-3 Predictions:", className="text-center text-light mt-2"),
                    html.Ul(prediction_list, className="text-light")
                ], className="mt-2")
            ], width=4, style={'textAlign': 'center'}))
        return html.Div([
            html.H5("Batch Prediction Results", className="text-center text-light mb-4"),
            dbc.Row(results, justify="center")
        ])
    return None

# History table callback
@app.callback(
    Output('history-table', 'children'),
    Input('predict-btn', 'n_clicks')
)
def update_history_table(n_clicks):
    if session_history:
        header = [html.Th(col) for col in session_history[0].keys()]
        rows = [html.Tr([html.Td(row[col]) for col in row]) for row in session_history]
        return dbc.Table([html.Thead(html.Tr(header)), html.Tbody(rows)], bordered=True, dark=True, hover=True, responsive=True, striped=True)
    else:
        return html.Div("No predictions yet.", className="text-light")

# Download callback for session history
@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_csv(n_clicks):
    if session_history:
        df = pd.DataFrame(session_history)
        return dcc.send_data_frame(df.to_csv, filename="predictions_session.csv", index=False)
    return None

if __name__ == '__main__':
    app.run_server(debug=True)
