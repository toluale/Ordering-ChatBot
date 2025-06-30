To test and evaluate the intent classification: run intent_classification.py

To test the conversation flow with brand personality feature: run conversation_brand_feat.py

Brand configuration file is in Ordering-ChatBot\streaming_ordering_chatbot\resources\brand_configs.json

To test and evaluate the conversation flow with brand personality alignment: run conversation_brand_evaluator_v2.py

Scenario json file for the evaluation is in the resources directory with the following format to add additional scenario

{
    "id": "unique_id",         // Optional, will be generated if not provided
    "text": "User message",    // Required
    "type": "scenario_type",   // Required
    "expected_outcomes": [     // Optional
    ]
}

You define the brand configuration in .env file

NB: 
