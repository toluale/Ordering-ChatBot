# To launch the Ordering Chatbot — Quickstart

## Run locally
1) Start the API server
```bash
python server_test.py
```
2) Start the Streamlit UI (in a new terminal)
```bash
streamlit run streaming_ordering_chatbot/streamlit/app.py
```
or
```bash
streamlit run streaming_ordering_chatbot/streamlit/app_evaluation.py
```
3) Open the Local URL printed by Streamlit in your browser

Tip: The UI calls the API at `http://localhost:8000` by default. To override, set `STREAMLIT_API_BASE_URL` in your environment.

## Environment setup
Create a `.env` file in the project root with at least:
```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment

# Branding & style
BRAND_NAME=Brand name      # Used by the API flows
RESTAURANT_BRAND=Restaurant name  # Required by server_test.py
CONVERSATION_STYLE=default           # one of: default|casual|genz

# Streamlit (optional)
STREAMLIT_API_BASE_URL=http://localhost:8000

# Telemetry (optional)
APPLICATIONINSIGHTS_CONNECTION_STRING=...  
```

Install dependencies (choose one):
- Using requirements.txt:
```bash
pip install -r requirements.txt
```
- Using Poetry:
```bash
poetry install
```

# Restaurant Ordering Chatbot - Testing and Evaluation Guide

## Intent Classification Testing
NOTE: Copy test files from `test_scripts` directory to your root

To test and evaluate the intent classification: 
```bash
python intent_classification.py
```
This evaluates the chatbot's ability to classify user intents (order or conversation) and provides accuracy metrics using SKLearn.
**File Path**: `tests/data/intent_test_cases.json`

## Interactive Conversation Testing
To test the interactive conversation flow with brand personality feature:
```bash
python conversation_brand_feat.py
```
This runs an interactive conversation session where you can test the chatbot's responses with brand-specific personality traits.

## Brand Configuration
Brand configuration file is located at: `streaming_ordering_chatbot\resources\brand_configs.json`

Configure brand personality in your `.env` file by setting the `BRAND_NAME` variable.

Conversation styles supported: `default`, `casual`, `genz`. The Streamlit UI “Tone” dropdown maps to these; you can also set `CONVERSATION_STYLE` in `.env` to change the API’s default.

## Conversation Flow Evaluation with Brand Personality

### Conversation Generation
To simulate response from the Assistant (Chatbot)
```bash
python conversation_generator.py
```

### Evaluation with Custom Metric
To evaluate the generated conversation with custom metric
```bash
python conversation_evaluator.py
```

### Conversation Generation + Evaluation
To generate and evaluate conversation sync
```bash
python run_full_evaluation.py
```
This generate assistant response to user prompt and runs automated custom evaluation across all brands and scenarios, generating:
- Individual conversation JSON files per brand
- Comprehensive evaluation metrics (BrandVoiceMetric, RelevanceMetric, TaskCompletionMetric)
- Results and generated response are saved in `evaluation_results/` directory
- User prompt JSON file is located at: `streaming_ordering_chatbot/resources/evaluation_scenarios.json`

### Evaluation with Azure AI
For evaluation using Azure AI evaluators, use the Jupyter notebooks:

1. Conversation Data Processing and Evaluation
**File**: `evaluation_results/conversation_eval.ipynb`
- Processes evaluation results JSON files
- Extracts conversation pairs (user prompts + assistant responses)
- Generates CSV files for further analysis
- Extract custom evaluation metrics and saved it in CSV
- Evaluates conversations using Azure AI evaluators:
  - **RelevanceEvaluator**: How well responses address user queries
  - **CoherenceEvaluator**: Logical consistency and structure
  - **FluencyEvaluator**: Natural language quality and grammar
- Exports and save detailed results and summary statistics in `evaluation_results/` directory

### Adding Custom Scenarios
Add scenarios to the evaluation by editing the scenario JSON file in the resources directory with the following format:

```json
{
    "id": "unique_id",         // Optional, will be generated if not provided
    "text": "User message",    // Required - the user input to test
    "type": "scenario_type",   // Required - category (e.g., "order", "question", "modification")
    "expected_outcomes": [     // Optional - expected behaviors or outcomes
        "should_acknowledge_request",
        "should_provide_relevant_menu_info"
    ]
}
```
## Order Flow Test
To test the order flow run
```bash
python order_flow_test.py
```
**Test Files**: `tests/data/order_test_cases.json`
**Result Path**: `evaluation_results/order_evaluation/`

## API Endpoints (served by FastAPI)
- `POST /screen` — content safety screening + intent prediction
- `POST /order` — structured order flow with tool-calls (streamed)
- `POST /assistant` — general assistant replies (streamed)
- `POST /preamble` — brand/style preamble generation
- `POST /summary` — conversation summary generation
- `GET /conversation-styles` — list valid styles

Open API docs: http://localhost:8000/docs

## Major File Paths
Menu Config Path: `streaming_ordering_chatbot/api/flows/data/`
Flow Prompts Path: `streaming_ordering_chatbot/api/flows/prompts/`
Flows and Features Path: `streaming_ordering_chatbot/api/flows/`

Key Utilities:
- `streaming_ordering_chatbot/api/utils/azure_client.py` — Azure OpenAI client + unified chat parameter defaults
- `streaming_ordering_chatbot/api/utils/stream_utils.py` — streaming helpers for chat responses
- `streaming_ordering_chatbot/api/utils/order_stream_utils.py` — streaming tool-call parsing for order flow
- `streaming_ordering_chatbot/api/utils/prompt_utils.py` — brand/style overlay composition and caching

## Output Files
Evaluation results are saved to:
- `evaluation_results/{brand_name}_conversation_{timestamp}.json` - Individual brand conversations
- `evaluation_results/conversations.csv` - All conversation pairs
- `evaluation_results/evaluation_scores.csv` - Brand evaluation metrics
- `evaluation_results/conversation_metric_scores.csv` - Azure AI evaluation results

## Notes
- Context management: The latest 8 messages are kept verbatim for precision to keep prompts small and predictable. Optional summarization of older history can be added later if needed.
- Conversation style and deployment are configured via `.env` (no code changes needed). The Streamlit UI can override style per-session.
- Optional second UI: `streaming_ordering_chatbot/streamlit/app_evaluation.py` provides a lightweight dashboard for evaluation metrics.