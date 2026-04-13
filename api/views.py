import numpy as np
import tensorflow as tf
import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from PIL import Image
import io

# # 1. Load the model and classes once when the server starts
# model = tf.keras.models.load_model('final_psl_model.h5')
# with open('classes.pkl', 'rb') as f:
#     classes = pickle.load(f)

# @api_view(['POST'])
# def predict_sign(request):
#     try:
#         # 2. Get the image from the Frontend request
#         file = request.FILES['image']
#         img = Image.open(file).convert('RGB')
        
#         # 3. Pre-process (Match your 180x180 Colab logic)
#         img = img.resize((180, 180))
#         img_array = tf.keras.utils.img_to_array(img)
#         img_array = img_array / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # 4. Predict
#         predictions = model.predict(img_array)
#         predicted_class = classes[np.argmax(predictions)]
#         confidence = float(np.max(predictions) * 100)

#         # 5. Send result back to React
#         return Response({
#             "prediction": predicted_class,
#             "confidence": f"{confidence:.2f}%"
#         })

#     except Exception as e:
#         return Response({"error": str(e)}, status=400)

import tensorflow as tf
import keras
import pickle
import os
from django.conf import settings

# Define paths
MODEL_PATH = os.path.join(settings.BASE_DIR, 'final_psl_model.h5')
CLASSES_PATH = os.path.join(settings.BASE_DIR, 'classes.pkl')

model = None
classes = None

try:
    # Load the model normally - Keras 3 handles the DTypePolicy automatically!
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Model file not found at {MODEL_PATH}")

    # Load your classes
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'rb') as f:
            classes = pickle.load(f)
        print("Classes loaded successfully!")
    else:
        print(f"Classes file not found at {CLASSES_PATH}")

except Exception as e:
    print(f"Error during initialization: {e}")
    model = None
    classes = None