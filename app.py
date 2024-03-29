from flask import Flask, render_template, request, jsonify
from langchain_community.llms import VLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from pymongo import MongoClient
import requests

app = Flask(__name__)

# Create a client connection
client = MongoClient("mongodb+srv://xy3d:XgB8JVGTuWGd50kp@cluster0.20iimjx.mongodb.net/?retryWrites=true&w=majority") # replace "mongodb_uri" with your actual MongoDB URI

# Reference the proper database and collection
db = client["blog"] # replace "mydatabase" with your MongoDB database name
collection = db["data"]

# Loading the model
def load_llm(max_tokens, prompt_template):
    llm = VLLM(model="mistralai/Mistral-7B-v0.1",
               trust_remote_code=True,  # mandatory for hf models
               max_new_tokens=1024,
               top_k=10,
               top_p=0.95,
               temperature=0.7,
               tensor_parallel_size=2,
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    print(llm_chain)
    return llm_chain

def get_src_original_url(query):
    url = 'https://api.pexels.com/v1/search'
    headers = {
        'Authorization': "1jG3qyQ6h6EIzPk0TA2jSrzz4J9BEdDbgS17y5hkyTkbK2c9fgFe6Xhb",
    }

    params = {
        'query': query,
        'per_page': 1,
    }

    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()
        photos = data.get('photos', [])
        if photos:
            src_original_url = photos[0]['src']['original']
            return src_original_url
        else:
            return None
    else:
        return None

def generate_article(user_input, image_input):
    prompt_template = """You are a JavaScript teacher and your task is to write a fundamental article on the given topic: "${user_input}". 
        The article must be under 900 words and should cover the topic comprehensively. 
        Ensure to include examples, explanations, and best practices to help learners understand the concept thoroughly.         
            """
    llm_call = load_llm(max_tokens=1024, prompt_template=prompt_template)
    result = llm_call.invoke(user_input)
    return result

@app.route('/gen', methods=['POST'])
def generate():
    data = request.json
    user_input = data.get('user_input')
    image_input = data.get('image_input')

    if not user_input or not image_input:
        return jsonify({"error": "Both 'user_input' and 'image_input' must be provided."}), 400

    article = generate_article(user_input, image_input)
    image_url = get_src_original_url(image_input)

    # if article and image_url and base64_image:
    if article and image_url:
        # Post data to MongoDB
        post = {
            "topic": user_input,
            "article": article["text"],
            "image_url": image_url,
        }
        collection.insert_one(post)
        
        return jsonify({
            "topic": user_input,
            "article": article["text"],
            "image_url": image_url,
        }), 200
    else:
        return jsonify({"error": "Failed to generate article or fetch image."}), 500
    
@app.route('/')
def index():
    # Fetch data from MongoDB
    data = collection.find()

    # Pass the data to the HTML template
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=False)
