import os
import json
import uuid
import sys 
from dotenv import load_dotenv
import openai
from openai import OpenAI
from neo4j import GraphDatabase, basic_auth 
from datetime import datetime
import warnings 

print("--- Render Env Check ---")
print(f"OPENAI_API_KEY Set: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
print(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME')}")
print(f"NEO4J_PASSWORD Set: {bool(os.getenv('NEO4J_PASSWORD'))}")
print(f"NEO4J_DATABASE: {os.getenv('NEO4J_DATABASE')}")
print(f"PORT: {os.getenv('PORT', 'Not Set (using default)')}")
print(f"PYTHON_VERSION (reported by system): {sys.version}")
print("--- End Render Env Check ---")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

def extract_medical_data_from_text(text_prompt: str) -> dict | None:
    """
    Uses GPT-4o-mini to extract structured medical data from text based on a predefined schema.

    Args:
        text_prompt: The unstructured medical record text.

    Returns:
        A dictionary containing the extracted data, or None if extraction fails.
    """
    system_prompt = """
You are a medical data extraction assistant. Your task is to read the provided unstructured medical record text and extract key information about the patient, their conditions, medications, allergies, procedures, and reported symptoms.

Format your output STRICTLY as a JSON object containing the following keys:
- "patient": An object with "name", "dateOfBirth" (try YYYY-MM-DD format, otherwise as text), and "sex". If a Medical Record Number (MRN) or other unique patient identifier is explicitly mentioned, include it as "extractedId".
- "conditions": A list of objects, each with "name" (normalized title case) and optionally "diagnosisDate" (YYYY-MM-DD).
- "medications": A list of objects, each with "name" (normalized title case) and optionally "dosage", "frequency", and "startDate" (YYYY-MM-DD).
- "allergies": A list of objects, each with "allergen" (normalized title case) and optionally "reaction".
- "procedures": A list of objects, each with "name" (normalized title case) and optionally "procedureDate" (YYYY-MM-DD).
- "symptoms": A list of objects, each with "name" (normalized title case) and optionally "reportDate" (YYYY-MM-DD), "severity".

If information for a category is not present, provide an empty list (e.g., "allergies": []).
If specific properties like dates or dosage are not mentioned for an item, omit them from that item's object.
Use standard medical terminology where appropriate (e.g., 'Hypertension' instead of 'high blood pressure').
Ensure the output is a single, valid JSON object. Do not include any explanations or text outside the JSON structure.
    """
    user_prompt = f"""
Medical Record Text:
---
{text_prompt}
---

JSON Output:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, 
            response_format={"type": "json_object"} 
        )
        response_content = response.choices[0].message.content

        if not response_content:
            print("Error: LLM returned an empty response.")
            return None

        extracted_data = json.loads(response_content)

        if isinstance(extracted_data, dict) and "patient" in extracted_data:
            return extracted_data
        else:
            print(f"Error: LLM output was not the expected JSON structure.\nReceived: {response_content}")
            return None

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"LLM Raw Output: {response_content}")
        return None
    except Exception as e:
        print(f"An error occurred during LLM communication: {e}")
        return None

def generate_cypher_query(extracted_data: dict) -> tuple[str, dict] | None:
    """
    Generates a parameterized Cypher query to MERGE patient data into Neo4j.

    Args:
        extracted_data: The structured data extracted by the LLM.

    Returns:
        A tuple (cypher_query_string, parameters_dict), or None if generation fails.
    """
    if not extracted_data or not isinstance(extracted_data.get("patient"), dict):
        print("Error: Invalid or missing patient data for Cypher generation.")
        return None

    patient_info = extracted_data.get("patient", {})
    conditions = extracted_data.get("conditions", [])
    medications = extracted_data.get("medications", [])
    allergies = extracted_data.get("allergies", [])
    procedures = extracted_data.get("procedures", [])
    symptoms = extracted_data.get("symptoms", [])

    if "extractedId" in patient_info and patient_info["extractedId"]:
        patient_id = patient_info["extractedId"]
        id_property = "mrn" 
        print(f"Using extracted ID ({id_property}): {patient_id}")
    else:
        patient_id = str(uuid.uuid4())
        id_property = "patientId" 
        print(f"Warning: No 'extractedId' found. Generated new {id_property}: {patient_id}")

    cypher_parts = []
    params = {'patient_id': patient_id} 

    cypher_parts.append(f"MERGE (p:Patient {{{id_property}: $patient_id}})")

    patient_props = {k: v for k, v in patient_info.items() if k != "extractedId" and v is not None}
    if id_property == "patientId":
        patient_props['patientId'] = patient_id

    params['patient_props'] = patient_props

    cypher_parts.append("ON CREATE SET p = $patient_props, p.createdAt = timestamp()")
    cypher_parts.append("ON MATCH SET p += $patient_props, p.lastUpdatedAt = timestamp()")

    def add_related_nodes(data_list, node_label, rel_type, node_key_prop, rel_props_keys):
        """Generates Cypher clauses for related nodes and relationships."""
        if not isinstance(data_list, list) or not data_list:
            return

        list_param_name = f"{node_label.lower()}_list"

        valid_items = [item for item in data_list if isinstance(item, dict) and node_key_prop in item and item[node_key_prop]]
        if not valid_items:
            # print(f"Debug: No valid items found for {node_label} to process.")
            return 

        params[list_param_name] = valid_items

        cypher_parts.append(f"WITH p") 
        cypher_parts.append(f"UNWIND ${list_param_name} AS item")
        cypher_parts.append(f"MERGE (n:{node_label} {{{node_key_prop}: item.{node_key_prop}}})")

        rel_prop_parts = []
        for key in rel_props_keys:
            if key != node_key_prop:
                rel_prop_parts.append(f"{key}: item.{key}")

        rel_props_str = ""
        if rel_prop_parts:
            rel_props_str = " {" + ", ".join(rel_prop_parts) + "}"

        cypher_parts.append(f"MERGE (p)-[r:{rel_type}]->(n)")
        if rel_props_str:
            cypher_parts.append(f"ON CREATE SET r = apoc.map.clean({rel_props_str}, [], [null, ''])") 
            cypher_parts.append(f"ON MATCH SET r += apoc.map.clean({rel_props_str}, [], [null, ''])")

    add_related_nodes(conditions, "Condition", "HAS_CONDITION", "name", ["diagnosisDate"])
    add_related_nodes(medications, "Medication", "TAKES_MEDICATION", "name", ["dosage", "frequency", "startDate"])
    add_related_nodes(allergies, "Allergy", "HAS_ALLERGY", "allergen", ["reaction"]) 
    add_related_nodes(procedures, "Procedure", "UNDERWENT_PROCEDURE", "name", ["procedureDate"])
    add_related_nodes(symptoms, "Symptom", "REPORTS_SYMPTOM", "name", ["reportDate", "severity"])

    cypher_parts.append("WITH DISTINCT p")

    cypher_parts.append(f"RETURN p.{id_property} AS patientId") 

    final_cypher = "\n".join(cypher_parts)

    return final_cypher, params

def execute_neo4j_query(query: str, parameters: dict) -> str | None:
    """
    Executes a Cypher query with parameters against the configured Neo4j database.

    Args:
        query: The Cypher query string.
        parameters: A dictionary of parameters for the query.

    Returns:
        The patient ID returned by the query upon successful execution,
        or None if an error occurs or the ID is not returned.
    """
    if not NEO4J_PASSWORD:
        print("CRITICAL Error: NEO4J_PASSWORD environment variable not set. Cannot connect to Neo4j.")
        return None

    driver = None 
    try:
        auth = basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD)
        driver = GraphDatabase.driver(NEO4J_URI, auth=auth)
        driver.verify_connectivity()
        print(f"Successfully connected to Neo4j at {NEO4J_URI} (database: '{NEO4J_DATABASE}').")

        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, parameters)
            record = result.single()
            if record and "patientId" in record:
                patient_id_returned = record["patientId"]
                print(f"Successfully executed query. Patient ID: {patient_id_returned}")
                return patient_id_returned
            elif record:
                print(f"Warning: Query executed but did not return the expected 'patientId' key in the record: {record}")
                return None 
            else:
                print("Warning: Query executed but returned no records.")
                summary = result.consume()
                if summary.counters.nodes_created > 0 or summary.counters.relationships_created > 0:
                     print("Note: Database writes may have occurred despite no patientId returned.")
                return None


    except Exception as e:
        print(f"Error executing Cypher query against Neo4j: {e}")
        # print("Failing Query:\n", query)
        # print("Failing Parameters:\n", json.dumps(parameters, indent=2, default=str))
        return None
    finally:
        if driver:
            driver.close()
            # print("Neo4j driver closed.")

def send_to_neo4j(prompt: str) -> str | None:
    """
    End-to-end processing of a medical record string:
    1. Extracts structured data using LLM.
    2. Generates a Neo4j Cypher query.
    3. Executes the query against the configured Neo4j database.

    Args:
        prompt: The unstructured medical record text.

    Returns:
        The patient ID stored or found in Neo4j upon success, otherwise None.
    """
    print("--- Step 1: Extracting data using LLM ---")
    extracted_data = extract_medical_data_from_text(prompt)

    if not extracted_data:
        print("--- Extraction failed. Aborting. ---")
        return None

    print("--- Step 1: Extraction Successful ---")

    print("\n--- Step 2: Generating Cypher Query ---")
    cypher_result = generate_cypher_query(extracted_data)

    if not cypher_result:
        print("--- Cypher query generation failed. Aborting. ---")
        return None

    cypher_query, params = cypher_result
    print("--- Step 2: Cypher Query Generation Successful ---")

    print("\n--- Step 3: Executing Query in Neo4j ---")
    patient_id_result = execute_neo4j_query(cypher_query, params)

    if patient_id_result:
        print(f"--- Step 3: Successfully processed and stored/updated data for Patient ID: {patient_id_result} ---")
        return patient_id_result
    else:
        print("--- Step 3: Failed to execute query successfully in Neo4j or retrieve Patient ID. ---")
        return None

import os
import json
import sys
import re 
from openai import OpenAI
from neo4j import GraphDatabase, basic_auth

def generate_cypher_for_prompt(prompt: str) -> str | None:
    """
    Uses LLM (gpt-4o-mini) to generate a Cypher query from a natural language prompt,
    considering the defined KG schema with improved instructions.
    """
    schema_description = """
    Knowledge Graph Schema:
    Nodes:
    - Patient (Properties: name<String>, mrn<String>, patientId<String>, dateOfBirth<String>, sex<String>)
    - Condition (Properties: name<String>)
    - Medication (Properties: name<String>)
    - Allergy (Properties: allergen<String>)
    - Procedure (Properties: name<String>)
    - Symptom (Properties: name<String>)

    Relationships:
    - (Patient)-[r:HAS_CONDITION]->(Condition) # No properties on 'r'
    - (Patient)-[r:TAKES_MEDICATION {dosage: String, frequency: String, startDate: String}]->(Medication) # Properties are ON the relationship 'r'
    - (Patient)-[r:HAS_ALLERGY {reaction: String}]->(Allergy) # Property 'reaction' is ON the relationship 'r'
    - (Patient)-[r:UNDERWENT_PROCEDURE {procedureDate: String}]->(Procedure) # Property 'procedureDate' is ON the relationship 'r'
    - (Patient)-[r:REPORTS_SYMPTOM {reportDate: String, severity: String}]->(Symptom) # Properties are ON the relationship 'r'
    """

    system_prompt = f"""
You are an expert translator of natural language questions into Neo4j Cypher queries based on the provided schema.
**CRITICAL INSTRUCTIONS:**
1.  **Read-Only:** Generate *read-only* Cypher queries using MATCH, WHERE, RETURN. NEVER use CREATE, MERGE, SET, DELETE, REMOVE.
2.  **Schema Adherence:** Strictly follow the provided schema. Pay close attention to node labels, property names, relationship types, and ESPECIALLY where properties are located (on nodes vs. on relationships).
3.  **Relationship Properties:** When asked for properties like 'dosage', 'frequency', 'startDate', 'reaction', 'procedureDate', 'reportDate', 'severity', you MUST access them from the *relationship variable*. Assign a variable to the relationship in the MATCH pattern (e.g., `-[r:TAKES_MEDICATION]->`) and return the property from that variable (e.g., `RETURN r.dosage`). Do NOT try to access these properties from the connected nodes.
4.  **Patient Lookup:**
    *   If a patient MRN (like 'HOS...') or patientId (UUID format) is mentioned, use that for precise lookup: `MATCH (p:Patient {{mrn: 'ID'}}) ...` or `MATCH (p:Patient {{patientId: 'ID'}}) ...`.
    *   If only a name is given (e.g., "Johnathan Doe"), use `MATCH (p:Patient {{name: 'Name'}}) ...`. Remember names might not be unique.
5.  **Normalization:** Normalize key entity names found in the user prompt (Conditions, Medications, Allergens, Procedures, Symptoms) to **Title Case** before using them in the Cypher query WHERE clause or property matching. For example, if the user asks about "migraine" or "SULFA drugs", use `name: 'Migraines'` or `allergen: 'Sulfa Drugs'` in the query, assuming these are stored in Title Case.
6.  **Return Specificity:** Return the specific properties requested or implied by the question. If asked about a patient, return identifying info (name, mrn, patientId if available). If asked about relationships, return info from both connected nodes and relevant relationship properties.
7.  **Clarity:** Use explicit node labels (`p:Patient`, `c:Condition`, etc.).
8.  **Output:** Output *only* the raw Cypher query string. No explanations, no ```cypher ``` tags.

**Schema:**
{schema_description}
"""

    user_message = f"Generate a Cypher query for the following question, carefully following ALL instructions above: {prompt}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0, 
        )
        cypher_query = response.choices[0].message.content.strip()

        cypher_query = cypher_query.removeprefix("```cypher").removesuffix("```").strip()

        modification_keywords = [" CREATE ", " MERGE ", " DELETE ", " SET ", " REMOVE "]
        if any(keyword in cypher_query.upper() for keyword in modification_keywords):
            print(f"Error: Generated query contains modification keywords despite instructions: {cypher_query}")
            return None

        if not cypher_query.upper().startswith("MATCH"):
             print(f"Warning: Generated query does not start with MATCH: {cypher_query}")

        print(f"Debug: Generated Cypher: {cypher_query}")
        return cypher_query

    except Exception as e:
        print(f"Error during Cypher generation with LLM: {e}")
        return None

def run_read_query(query: str) -> list[dict] | None:
    """Executes a read-only Cypher query against Neo4j and returns results."""
    if not NEO4J_PASSWORD:
        print("CRITICAL Error in run_read_query: NEO4J_PASSWORD not set.")
        return None
    driver = None
    try:
        auth = basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD)
        driver = GraphDatabase.driver(NEO4J_URI, auth=auth)
        driver.verify_connectivity()
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query)
            results_list = [record.data() for record in result]
            print(f"Debug: Query returned {len(results_list)} record(s).")
            return results_list
    except Exception as e:
        print(f"Error executing read query in Neo4j: {e}")
        print(f"Failing Query: {query}")
        return None
    finally:
        if driver: driver.close()


def generate_final_response(user_prompt: str, query_results: list[dict]) -> str:
    """
    Uses LLM to generate a natural language response based on the user prompt
    and the data retrieved from the knowledge graph, with improved grounding.
    """

    MAX_RESULT_CHARS = 4000 
    results_string = json.dumps(query_results, indent=2, default=str)
    if len(results_string) > MAX_RESULT_CHARS:
        results_string = results_string[:MAX_RESULT_CHARS] + "\n... (results truncated)"

    no_results = not query_results
    if no_results:
        results_string = "[] (No information found in the knowledge graph matching the query)"

    system_prompt = """
You are an AI assistant accessing a hospital knowledge graph. Your task is to synthesize the 'Retrieved Data' to answer the 'Original User Question'.
**CRITICAL INSTRUCTIONS:**
1.  **Strict Grounding:** Base your answer **EXCLUSIVELY** on the `Retrieved Data` provided. Do NOT add external knowledge or information not present in the data. Do NOT make assumptions or invent details.
2.  **Completeness:** If the `Retrieved Data` contains specific fields relevant to the question (e.g., 'mrn', 'dosage', 'reaction', dates), **include them** in your response. Don't omit details present in the data.
3.  **Acknowledge Emptiness:** If the `Retrieved Data` is empty (`[]`), clearly state that the requested information could not be found *in the knowledge graph*. Do not speculate why.
4.  **Handle Duplicates (if applicable):** If the `Retrieved Data` shows multiple similar records (e.g., multiple rows for the same patient/medication pair), present the information clearly. You can note the multiple entries if it seems relevant, but focus on presenting the *content* accurately (e.g., "The data shows John Doe takes Lisinopril 10mg daily...").
5.  **Medical Context:** Provide brief, neutral medical context if directly supported by the data (e.g., "Hypertension is a common condition..."), but **DO NOT give medical advice, diagnoses, or prognoses.** Avoid speculative insights.
6.  **Tone:** Be helpful, clear, and empathetic, but maintain a professional tone focused on the medical information requested. Avoid unrelated conversation.
7.  **Formatting:** Use clear language. Use bullet points for lists if appropriate.
"""

    context_message = f"""
Original User Question:
"{user_prompt}"

Retrieved Data from Knowledge Graph:
```json
{results_string}
Now, generate a response following ALL instructions above.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_message}
            ],
            temperature=0.3, # Slightly lower temperature for more factual responses
        )
        final_answer = response.choices[0].message.content
        return final_answer

    except Exception as e:
        print(f"Error generating final response with LLM: {e}")
        return "I encountered an issue while formulating the response based on the retrieved data. Please try again."
    
def chat_with_kg(prompt: str) -> str:
    """
    Handles the conversation flow: Prompt -> Cypher -> Neo4j -> Final Response.
    """
    print(f"\n--- Processing Chat Prompt: '{prompt}' ---")
    print("Step 1: Generating Cypher query...")
    cypher_query = generate_cypher_for_prompt(prompt)
    if not cypher_query:
        return "I wasn't able to translate your question into a valid query for the knowledge graph. Could you please try rephrasing it, perhaps being more specific?"

    print("Step 2: Executing query against Neo4j...")
    query_results = run_read_query(cypher_query)
    if query_results is None:
        return "I encountered an issue while querying the knowledge graph database. Please try again later or contact support."

    print("Step 3: Generating final response...")
    final_response = generate_final_response(prompt, query_results)

    print("--- Chat Prompt Processing Complete ---")
    return final_response

# questions = [
#     # "What conditions does Johnathan Doe have?", # Expect Hypertension, Diabetes
#     # "Tell me about patient Jane Smith.", # Expect Name, MRN, DOB, Sex
#     # "Which patients take Metformin?", # Expect John Doe with MRN/ID
#     # "Does patient HOS12345678 have any allergies?", # Expect Penicillin
#     # "What was the reaction for Jane Smith's Sulfa allergy?", # Expect 'hives'
#     # "List all medications prescribed to patient with MRN HOS12345678.", # Expect Lisinopril, Metformin with details
#     # "Are there any patients with migraines?", # Expect Jane Smith
#     # "Who had an Appendectomy?", # Expect John Doe
#     # "What is the dosage for Lisinopril for patient HOS12345678?", # Expect 10mg
#     # "Does anyone have pneumonia?", # Expect No results (unless ingested)
#     # "Tell me something interesting about the data.", # Might still struggle, but let's see
#     # "What is the capital of France?" # Expect "Not found in KG"
#     'List the names of patients that have something common in pairs, and tell what is that thing in common',
#     'List the names of patients that have migrane.'
# ]

# for q in questions:
#     response = chat_with_kg(q)
#     print(f"\nQ: {q}")
#     print(f"A: {response}")
#     print("-" * 30)
