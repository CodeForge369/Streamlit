import tensorflow as tf

model = tf.keras.models.load_model("Fruit_Class2.keras", compile=False)
print("Loaded successfully")