from flask import Flask, request, render_template, jsonify
from google.cloud import aiplatform
import base64
import time

app = Flask(__name__)

# Vertex AI endpoint details
PROJECT_ID = "acs-course-new-449514" 
ENDPOINT_ID = "5659936215093215232"  
LOCATION = "europe-west4"  

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

def call_vertex_ai(image_content):
    # Load the endpoint
    endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}")

    # Prepare the request
    instances = [{"content": image_content.decode("ISO-8859-1")}]
    try:
        start_time = time.time()  # Track the time before the prediction request
        response = endpoint.predict(instances=instances)
        predictions = response.predictions
        api_response_time = time.time() - start_time  # Calculate response time

        # Extract results
        result = predictions[0] if predictions else {}

        return result, api_response_time
    except Exception as e:
        print(f"Error calling Vertex AI: {e}")
        return {"error": str(e)}, None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image file
        file = request.files["image"]
        if not file:
            return jsonify({"error": "No file uploaded."}), 400

        # Read and encode the file
        image_content = base64.b64encode(file.read())

        # Call Vertex AI for prediction
        result, api_response_time = call_vertex_ai(image_content)

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        image_url = f"data:image/jpeg;base64,{base64.b64encode(file.read()).decode()}"  # Display uploaded image
        return render_template("index.html", result=result, image_url=image_url, api_response_time=api_response_time)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)








