import tensorflow as tf
import keras
import pickle
import os
import numpy as np
from PIL import Image
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings

# --- ADVANCED MODEL LOADING ---
MODEL_PATH = os.path.join(settings.BASE_DIR, 'final_psl_model.h5')
CLASSES_PATH = os.path.join(settings.BASE_DIR, 'classes.pkl')

model = None
classes = None

# This "Fixer" removes the keys that are causing your error
def fixed_input_layer(config):
    config.pop('batch_shape', None)
    config.pop('optional', None)
    return keras.layers.InputLayer.from_config(config)

try:
    if os.path.exists(MODEL_PATH):
        # We use custom_objects to tell Keras how to handle the "InputLayer"
        model = keras.models.load_model(
            MODEL_PATH, 
            custom_objects={'InputLayer': fixed_input_layer},
            compile=False
        )
        print("Model loaded successfully!")
    
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'rb') as f:
            classes = pickle.load(f)
        print("Classes loaded successfully!")
except Exception as e:
    print(f"Error during initialization: {e}")

@api_view(['POST'])
def predict_sign(request):
    if model is None or classes is None:
        return Response({'error': 'Model not initialized'}, status=500)
    
    try:
        file = request.FILES['image']
        img = Image.open(file).convert('RGB')
        img = img.resize((180, 180))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        result_index = np.argmax(predictions)
        predicted_class = classes[result_index]
        confidence = float(np.max(tf.nn.softmax(predictions))) * 100

        return Response({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return Response({'error': str(e)}, status=400)