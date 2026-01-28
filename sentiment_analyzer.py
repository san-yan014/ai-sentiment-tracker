import os
import json
from datetime import datetime
from newsapi import NewsApiClient
from transformers import pipeline
from azure.storage.blob import BlobServiceClient

def main():
    print("starting sentiment analysis...")
    
    # initialize newsapi
    newsapi = NewsApiClient(api_key=os.environ.get('NEWSAPI_KEY'))
    
    # initialize sentiment model
    print("loading sentiment model...")
    sentiment = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    
    # get ai-related news
    print("fetching news...")
    articles = newsapi.get_everything(
        q='artificial intelligence OR AI',
        language='en',
        page_size=50
    )
    
    # analyze sentiment
    results = []
    for article in articles['articles']:
        title = article['title']
        description = article.get('description', '')
        
        # combine title and description
        text_to_analyze = f"{title}. {description}"
        
        # run sentiment analysis
        result = sentiment(text_to_analyze)[0]
        
        results.append({
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'description': description,
            'sentiment': result['label'],
            'confidence': result['score'],
            'source': article['source']['name']
        })
    
    print(f"analyzed {len(results)} articles")
    
    # save to azure storage
    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if connection_string:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_name = "sentiment-results"
        
        # create container if doesn't exist
        try:
            blob_service_client.create_container(container_name)
        except:
            pass
        
        # save results
        blob_name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(json.dumps(results, indent=2))
        print(f"results saved to {blob_name}")
    
    print("done!")

if __name__ == "__main__":
    main()