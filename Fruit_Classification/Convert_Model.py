import tensorflow as tf

model = tf.keras.models.load_model("Fruit_Class2.keras", compile=False)

model.save("Fruit_Class2_fixed.keras")   # or Fruit_Class2_fixed.keras

print("Saved successfully")