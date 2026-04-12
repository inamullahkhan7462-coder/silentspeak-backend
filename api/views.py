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
from keras.layers import InputLayer, Conv2D, Dense, Flatten, MaxPooling2D

# 1. This "Universal Fixer" handles the new DTypePolicy and batch_shape errors
class FixedLayer:
    @classmethod
    def from_config(cls, config):
        # Remove the modern keys that break older TensorFlow versions
        config.pop('dtype', None)
        config.pop('batch_shape', None)
        config.pop('optional', None)
        return super(cls, cls).from_config(config)

# Apply the fix to all layer types used in your model
class FixedInput(FixedLayer, InputLayer): pass
class FixedConv2D(FixedLayer, Conv2D): pass
class FixedDense(FixedLayer, Dense): pass

# 2. Load the model using these fixed versions
model = tf.keras.models.load_model(
    'final_psl_model.h5', 
    custom_objects={
        'InputLayer': FixedInput,
        'Conv2D': FixedConv2D,
        'Dense': FixedDense,
        'DTypePolicy': lambda **x: None # This ignores the DTypePolicy error directly
    },
    compile=False # Adding this avoids errors with the optimizer settings
)