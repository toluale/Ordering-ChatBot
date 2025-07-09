# Restaurant Ordering Chatbot - Testing and Evaluation Guide

## Intent Classification Testing
To test and evaluate the intent classification: 
```bash
python intent_classification.py
```
This evaluates the chatbot's ability to classify user intents (order or conversation) and provides accuracy metrics using SKLearn.

## Interactive Conversation Testing
To test the conversation flow with brand personality feature:
```bash
python conversation_brand_feat.py
```
This runs an interactive conversation session where you can test the chatbot's responses with brand-specific personality traits.

## Brand Configuration
Brand configuration file is located at: `streaming_ordering_chatbot\resources\brand_configs.json`

Configure brand personality in your `.env` file by setting the `BRAND_NAME` variable.

## Conversation Flow Evaluation
To test and evaluate the conversation flow with brand personality alignment using custom evaluation function:
```bash
python conversation_brand_evaluator_v2.py
```
This generate assistant response to user prompt and runs automated custom evaluation across all brands and scenarios, generating:
- Individual conversation JSON files per brand
- Comprehensive evaluation metrics (BrandVoiceMetric, RelevanceMetric, TaskCompletionMetric)
- Results and generated response are saved in `evaluation_results/` directory
- User prompt JSON file is located at: `streaming_ordering_chatbot\resources\evaluation_scenarios.json`

## Evaluation with Azure AI
For evaluation using Azure AI evaluators, use the Jupyter notebooks:

### 1. Conversation Data Processing and Evaluation
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

## Adding Custom Scenarios
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

## Environment Configuration
Set up your `.env` file with the following variables:
```env
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
BRAND_NAME=your_preferred_brand_name
```

## Output Files
Evaluation results are saved to:
- `evaluation_results/{brand_name}_conversation_{timestamp}.json` - Individual brand conversations
- `evaluation_results/conversations.csv` - All conversation pairs
- `evaluation_results/evaluation_scores.csv` - Brand evaluation metrics
- `evaluation_results/conversation_metric_scores.csv` - Azure AI evaluation results

## Notes: 
### TO DO
1. Update the menu schema (each brand to evaluate will have its own menu ).
   - Convert the menu file into a JSON file with dynamic menu-code generation
   - Transfer the menu configuration to be defined in the environment variable (menu_path)
   - Run another evaluation of the conversation flow with update menu schema
2. Complete the implementation of order_flow in Semantic Kernel
   - Include context file in the implemntation (promptflow implementation lacks context)
   - Evaluate the order groundness and accuracy against the menu schema
3. Design the main implementation to integrate the flows.
   - Final evaluation of the end-to-end implementation 
