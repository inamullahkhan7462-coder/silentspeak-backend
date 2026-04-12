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
from keras.layers import InputLayer
import pickle

# 1. This fixes the 'batch_shape' error in your specific .h5 file
class FixedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

# 2. Load the model with the fix
model = tf.keras.models.load_model(
    'final_psl_model.h5', 
    custom_objects={'InputLayer': FixedInputLayer}
)

# 3. Load your classes
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)