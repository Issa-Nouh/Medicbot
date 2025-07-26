import os
import json
import uuid
import sys
import re
from openai import OpenAI
from neo4j import GraphDatabase, basic_auth
from datetime import datetime
import warnings
import gradio as gr 
from utils import send_to_neo4j, chat_with_kg
from pyvis.network import Network
import tempfile
import html as _html
import html as _html
import tempfile
from pyvis.network import Network
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI

print("--- Render Env Check ---")
print(f"OPENAI_API_KEY Set: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
print(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME')}")
print(f"NEO4J_PASSWORD Set: {bool(os.getenv('NEO4J_PASSWORD'))}")
print(f"NEO4J_DATABASE: {os.getenv('NEO4J_DATABASE')}")
print(f"PORT: {os.getenv('PORT', 'Not Set (using default)')}")
print(f"PYTHON_VERSION (reported by system): {sys.version}")
print("--- End Render Env Check ---")

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")


def chat_interface_fn(message, history):
    """Gradio fn for chat: calls chat_with_kg, returns response."""
    if not client: return "Error: Backend OpenAI Client not configured."
    if not NEO4J_PASSWORD: return "Error: Backend Neo4j connection not configured."

    print(f"Gradio Chat Received: '{message}'")
    response = chat_with_kg(message)
    print(f"Gradio Chat Sending: '{response[:100]}...'") 
    return response

def process_medical_record_file_for_blocks(uploaded_file):
    """
    Gradio fn for file upload: reads file, calls send_to_neo4j,
    logs details, returns simple "Done!".
    """
    if not client:
        print("[Error] Backend OpenAI Client not configured.")
        return "Done!"
    if not NEO4J_PASSWORD:
        print("[Error] Backend Neo4j connection not configured.")
        return "Done!"
    if uploaded_file is None:
        print("[Info] No file provided for upload.")
        return "Done!"

    try:
        file_path = uploaded_file.name
        print(f"Gradio Upload: Processing file {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            print("[Info] Uploaded file is empty.")
            return "Done!"

        print("Gradio Upload: Attempting send_to_neo4j...")
        result_id = send_to_neo4j(content)

        if result_id and "Data written" in result_id:
             print(f"Gradio Upload: Partial Success - {result_id}")
        elif result_id:
            print(f"Gradio Upload: Success - Patient ID {result_id}")
        else:
            print("Gradio Upload: Failed or no confirmation.")

        return "Done!"

    except Exception as e:
        print(f"Error in Gradio file processing block: {e}")
        return "Done!"

def graph_snapshot():
    """
    Pulls up to 100 nodes+rels from Neo4j, renders via PyVis (non-notebook mode),
    and returns an <iframe> that embeds the full HTML so its scripts actually run.
    """
    net = Network(height="600px", width="100%", notebook=False)
    
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(
            "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100"
        )
        for record in result:
            n = record["n"]
            m = record["m"]
            r = record["r"]
            net.add_node(n.id, label=":".join(n.labels), title=str(dict(n)))
            net.add_node(m.id, label=":".join(m.labels), title=str(dict(m)))
            net.add_edge(n.id, m.id, label=r.type)
    
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    net.write_html(tmp.name, open_browser=False, notebook=False)
    
    with open(tmp.name, "r", encoding="utf-8") as f:
        full_page = f.read()
    
    srcdoc = _html.escape(full_page)
    return (
        f'<iframe srcdoc="{srcdoc}" '
        'style="border:none; width:100%; height:600px;"></iframe>'
    )
    
if not client:
    print("CRITICAL: Cannot launch Gradio UI - OpenAI client failed.")
    sys.exit(1)
if not NEO4J_PASSWORD:
    print("CRITICAL: Cannot launch Gradio UI - NEO4J_PASSWORD not set.")
    sys.exit(1)

print("Configuration checks passed. Defining Gradio interface...")
print(f"Neo4j Target: {NEO4J_URI} (DB: {NEO4J_DATABASE})")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Hospital Knowledge Graph Interface
        Use the tabs below to either upload new patient records or chat with the existing knowledge graph.
        """
    )

    with gr.Tabs():
        # --- CHAT TAB ---
        with gr.TabItem("Chat with KG"):
            gr.ChatInterface(
                fn=chat_interface_fn, 
                title="Chat with Hospital Knowledge Graph",
                description="Ask questions about the patient data stored in the Neo4j knowledge graph.",
                examples=[ 
                    "Can you list the names of the patients in the database?",
                    "What is the condition of Johnathan Smith?",
                    "Are there patients that share the same condition(s)?",
                ],
                 chatbot=gr.Chatbot(height=450), 
                 # submit_btn="Ask KG", # Customize button text (optional)
                 # retry_btn="Retry", # Customize button text (optional)
                 # undo_btn="Undo", # Customize button text (optional)
                 # clear_btn="Clear Chat" # Customize button text (optional)
            )
            # gr.Markdown("Enter your question about patients, conditions, medications, etc., based on the data previously uploaded. The system will query the graph and generate a response.")
            
        # --- UPLOAD TAB ---
        with gr.TabItem("Upload Record"):
            with gr.Row(): 
                with gr.Column(scale=1): # Column for input
                    file_input = gr.File(label="Upload Medical Record (.txt)", file_types=['.txt'])
                    upload_button = gr.Button("Process Uploaded File") # Explicit button
                with gr.Column(scale=2): # Column for output
                    upload_status_output = gr.Textbox(label="Processing Status", lines=5, interactive=False) # Use Textbox for more detail
            gr.Markdown("Upload a `.txt` file containing an unstructured patient record. Click 'Process Uploaded File'. The system will attempt to extract data and store it in Neo4j. Status and any generated/found Patient ID will appear above.")
            upload_button.click(
                fn=process_medical_record_file_for_blocks,
                inputs=[file_input],
                outputs=[upload_status_output] # Output to the Textbox
            )
        
        with gr.TabItem("Graph Snapshot"):
            graph_html = gr.HTML(
                "<i>Click “Load Graph” to see a snapshot of your Neo4j KG.</i>"
            )
            load_button = gr.Button("Load Graph Snapshot")
            load_button.click(
                fn=graph_snapshot,
                inputs=[],
                outputs=[graph_html]
            )
            
            
app = demo.app

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860))
    )
