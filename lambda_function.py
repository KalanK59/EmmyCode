import boto3
import json
import openai
import os
from openai import OpenAI
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    """
    AWS Lambda function to perform language processing on a given title.
    """
    try:
        # Parse input from event
        record_id = event.get('RecordID')
        title = event.get('Title')
        original_language = event.get('OriginalLanguage')

        # Check to ensure the event argument contains a Title and RecordID
        if not record_id or not title:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "RecordID and Title are required."})
            }

        # Initialize DynamoDB
        dynamodb_table_name = 'TitleLanguageData'
        dynamodb_client = boto3.resource('dynamodb').Table(dynamodb_table_name)

        # Check if a record already exists for the given title
        record = get_record(dynamodb_client, record_id, title)
        
        # If we found a record in DB, return it in JSON format
        if record:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Record found.", "data": record})
            }

        # If there isn't a record in the DB then make an OpenAI API call to process the title
        openai_data = process_with_openai(title)

        # Store the AI response as a new record in DynamoDB so future calls don't need to go to OpenAI
        store_record(dynamodb_client, record_id, title, original_language, openai_data)

        # Return the API response in JSON
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Record processed and stored.", "data": openai_data})
        }

    # General exception handling for any unaccounted errors
    except Exception as e:
        return {
            "statusCode": 501,
            "body": json.dumps({"error": str(e)})
        }

def get_record(dynamodb_client, record_id, title):
    """
    Retrieve a record from the DynamoDB table.
    """
    try:
        # Query the DynamoDB table for the record
        response = dynamodb_client.get_item(
            # Need both recordID and title because the table uses a composite key
            Key={
                "RecordID": record_id,
                "Title": title
            }
        )

        # return only the item (which contains the data) from the response
        return response.get("Item")
    except ClientError as e:
        raise Exception(f"DynamoDB error: {e.response['Error']['Message']}")

def process_with_openai(title):
    """
    Use OpenAI API to get DetectedLanguage, Transliteration, and Translation.
    """
    try:
        openai.api_key = os.environ.get('openai_key')

        # System and User Prompts as messages
        system_prompt = "You are an expert language model trained to detect languages, provide transliterations, and generate translations for movie titles. Provide a confidence score between 0 and 1 for the language detection."
        user_prompt = f"Given the title: '{title}', detect the language both by name and ISO 639 language code, provide a transliteration into the Latin-1 character set, and translate it to English. Return the output in JSON format with keys: DetectedLanguage, ISO639LanguageCode, Confidence, Transliteration, and Translation. Do not include code block formatting."
        
        # Making the API call using the updated method
        response = openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0  # Deterministic output
        )

        # Fetch the message given from the response (this should be in JSON format from the AI)
        response_content = response.choices[0].message.content.strip()

        # Convert from JSON string to object
        parsed_result = json.loads(response_content)

        # Return the structured result
        return parsed_result

    except json.JSONDecodeError:
        raise Exception("Failed to parse the response from OpenAI. Ensure the response is in JSON format.")
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def store_record(dynamodb_client, record_id, title, original_language, data):
    """
    Store the processed record into DynamoDB.
    """
    try:
        dynamodb_client.put_item(
            Item={
                "RecordID": record_id,
                "Title": title,
                "OriginalLanguage": original_language,
                "DetectedLanguage": data["DetectedLanguage"],
                "DetectedCode": data["ISO639LanguageCode"],
                "Confidence": str(data["Confidence"]), # must be converted to a string so that DynamoDB can store it as a Decimal instead of float
                "Transliteration": data["Transliteration"],
                "Translation": data["Translation"]
            }
        )
    except ClientError as e:
        raise Exception(f"DynamoDB error: {e.response['Error']['Message']}")
