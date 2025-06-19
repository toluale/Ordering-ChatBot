import os

from azure.ai.contentsafety import BlocklistClient
from azure.ai.contentsafety.models import (
    AddOrUpdateTextBlocklistItemsOptions, TextBlocklist, TextBlocklistItem)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

key = os.environ["AZURE_CONTENT_SAFETY_KEY"]
endpoint = os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"]
BLOCKLIST_NAME = os.environ["BLOCKLIST_NAME"]
  
BLOCKLIST_DESC = "Blocklist for non-burger items"

BLOCKLIST_CONTENT = [
    "salad",
    "burritos",
    "tacos",
    "pizza",
    "pasta",
]

client = BlocklistClient(endpoint, AzureKeyCredential(key))

def create_blocklist():
    blocklist_name = BLOCKLIST_NAME
    blocklist_description = BLOCKLIST_DESC 

    try:
        blocklist = client.create_or_update_text_blocklist(
            blocklist_name=blocklist_name,
            options=TextBlocklist(blocklist_name=blocklist_name, description=blocklist_description),
        )
        if blocklist:
            print("\nBlocklist created or updated: ")
            print(f"Name: {blocklist.blocklist_name}, Description: {blocklist.description}")
    except HttpResponseError as e:
        print("\nCreate or update text blocklist failed: ")
        if e.error:
            print(f"Error code: {e.error.code}")
            print(f"Error message: {e.error.message}")
            raise
        print(e)
        raise

def update_blocklist():
    blocklist_items = [TextBlocklistItem(text=item) for item in BLOCKLIST_CONTENT]
    try:
        result = client.add_or_update_blocklist_items(
            blocklist_name=BLOCKLIST_NAME, options=AddOrUpdateTextBlocklistItemsOptions(blocklist_items=blocklist_items)
        )
        for blocklist_item in result.blocklist_items:
            print(
                f"BlocklistItemId: {blocklist_item.blocklist_item_id}, Text: {blocklist_item.text}, Description: {blocklist_item.description}"
            )
    except HttpResponseError as e:
        print("\nAdd blocklistItems failed: ")
        if e.error:
            print(f"Error code: {e.error.code}")
            print(f"Error message: {e.error.message}")
            raise
        print(e)
        raise

if __name__ == "__main__":
    ## create and update blocklist contents
    create_blocklist()
    update_blocklist()

