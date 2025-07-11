# Restaurant Ordering Chatbot - Testing and Evaluation Guide

## Intent Classification Testing
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
- User prompt JSON file is located at: `streaming_ordering_chatbot\resources\evaluation_scenarios.json`

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

## Environment Configuration
Set up your `.env` file with the following variables:
```env
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
BRAND_NAME=your_preferred_brand_name
RESTAURANT_BRAND=your_preferred_restaurant_brand
MENU_CONFIG_PATH=brand_menu_config_path
```

## Major File Paths
Menu Config Path: `streaming_ordering_chatbot/api/flows/data/`
Flow Prompts Path: `streaming_ordering_chatbot/api/flows/prompts/`
Flows Path: `streaming_ordering_chatbot/api/flows/`

## Output Files
Evaluation results are saved to:
- `evaluation_results/{brand_name}_conversation_{timestamp}.json` - Individual brand conversations
- `evaluation_results/conversations.csv` - All conversation pairs
- `evaluation_results/evaluation_scores.csv` - Brand evaluation metrics
- `evaluation_results/conversation_metric_scores.csv` - Azure AI evaluation results

## Notes: 
### To Do
1. Evaluate order flow
2. Integrate conversation style option (casual, gen z or default brand style)
3. Implement `main.py` file, integrating all the flows