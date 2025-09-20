import os
import torch
import torch.nn as nn
from torchvision import models
from flask import Flask, request, jsonify, render_template
from PIL import Image
import pandas as pd
from torchvision import transforms

app = Flask(__name__, static_folder='static', template_folder='templates')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Directory for uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load skin_type_recommendation.csv data for recommendations 
try:
    skindata_all = pd.read_csv('skin_type_recommendations.csv', low_memory=False)
    skindata_all.columns = skindata_all.columns.str.strip()  
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the file 'skin_type_recommendations.csv' is in the 'skin_dataset' directory.")
    exit()

# Check required columns in skin_type_recommendation
required_columns_skin = ['Skin_Type', 'Product_Category', 'Product_Name', 'Brand', 'Website_Store']
for col in required_columns_skin:
    if col not in skindata_all.columns:
        print(f"Error: Required column '{col}' not found in skindata_all.")
        exit()

# VGG16 model for skin type detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

vgg16_model.classifier[6] = nn.Linear(vgg16_model.classifier[6].in_features, 3)
vgg16_model.classifier[2] = nn.Dropout(0.5)  

for param in vgg16_model.features.parameters():
    param.requires_grad = False

vgg16_model = vgg16_model.to(device)

# Load model weights VGG 16
try:
    checkpoint = torch.load('VGG16_model.pth', map_location=device)
    vgg16_model.load_state_dict(checkpoint, strict=False)  
    vgg16_model.eval()  
    print("Model loaded successfully")
except FileNotFoundError:
    print("Error: The model file 'VGG16_model.pth' was not found.")
except KeyError as e:
    print(f"Error: Key missing in the checkpoint: {e}")
except RuntimeError as e:
    print(f"RuntimeError during model loading: {e}")

# image transformations for VGG16
vgg16_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function detect skin type using VGG16
def detect_skin_type(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = vgg16_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = vgg16_model(image)
            _, predicted = torch.max(output, 1)
            class_labels = ['oily', 'acne', 'dry']
            detected_skin_type = class_labels[predicted.item()]
            print(f"Detected Skin Type: {detected_skin_type}")
            return detected_skin_type
    except Exception as e:
        print(f"Error detecting skin type: {e}")
        return None

# Function to recommend products based on skin_type and product_category
def recommend_skincare(Skin_Type, Product_Category, num_recommendations=5):
    print(f"Filter Criteria - Skin Type: {Skin_Type}, Product Category: {Product_Category}")
    try:
        # Filter recommendations based on skin type and product category
        filtered_skin_data = skindata_all[
            (skindata_all['Skin_Type'].str.lower() == Skin_Type.lower()) &
            ((skindata_all['Product_Category'].str.lower() == Product_Category.lower()) if Product_Category != 'No preference' else True)
        ]
        print(f"Filtered data count: {len(filtered_skin_data)}")  
        print(filtered_skin_data.head()) 

        if filtered_skin_data.empty:
            return pd.DataFrame({"message": ["No recommendations found"]})

        recommendations = filtered_skin_data[['Product_Name', 'Brand', 'Website_Store']].head(num_recommendations)
        return recommendations
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return pd.DataFrame({"error": ["Could not generate recommendations"]})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    skin_type = None
    if request.method == 'POST':
        skin_image = request.files.get('image')

        if skin_image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], skin_image.filename)
            try:
                skin_image.save(image_path)
                skin_type = detect_skin_type(image_path)  
            except Exception as e:
                print(f"Error processing image: {e}")

    return render_template('upload.html', skin_type=skin_type)

    
# recommendations route
@app.route('/recommendations', methods=['POST'])
def recommendations():
    print(skindata_all.head())

    skin_type = request.form.get('skin_type')
    product_category = request.form.get('product_category')

    print(f"Skin Type: {skin_type}, Product Category: {product_category}")

    skindata_all['Skin_Type'] = skindata_all['Skin_Type'].str.strip().str.lower()
    skindata_all['Product_Category'] = skindata_all['Product_Category'].str.strip().str.lower()

    # Check if product_category is None or empty
    if not product_category:
        filtered_recommendations = skindata_all[skindata_all['Skin_Type'] == skin_type.lower()]
    else:
        filtered_recommendations = skindata_all[
            (skindata_all['Skin_Type'] == skin_type.lower()) &
            (skindata_all['Product_Category'] == product_category.lower())
        ]

    has_recommendations = not filtered_recommendations.empty

    message = "No recommendations found." if not has_recommendations else None

    print(f"Filtered Recommendations: {filtered_recommendations}")

    return render_template('recommendation.html', 
                           recommendations=filtered_recommendations, 
                           has_recommendations=has_recommendations, 
                           message=message, 
                           skin_type=skin_type, 
                           product_category=product_category)



if __name__ == "__main__":
    app.run(debug=True)
