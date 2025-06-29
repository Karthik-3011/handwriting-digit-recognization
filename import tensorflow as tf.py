import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Save the trained model
model.save('digit_model.h5')
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("digit_model.h5")

# GUI Class
class DigitRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.geometry("600x400")

        self.canvas = tk.Canvas(self, width=300, height=300, bg='white')
        self.canvas.grid(row=0, column=0, pady=10, padx=10, columnspan=3)

        self.label = tk.Label(self, text="Draw a digit...", font=("Helvetica", 24))
        self.label.grid(row=0, column=3, padx=20)

        self.button_recognize = tk.Button(self, text="Recognise", command=self.predict_digit, width=15)
        self.button_recognize.grid(row=1, column=0, pady=10)

        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas, width=15)
        self.button_clear.grid(row=1, column=1, pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image1 = Image.new("L", (300, 300), color="white")
        self.draw_image = ImageDraw.Draw(self.image1)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image.rectangle([0, 0, 300, 300], fill="white")
        self.label.configure(text="Draw a digit...")

    def predict_digit(self):
        resized = self.image1.resize((28, 28))
        inverted = ImageOps.invert(resized)
        img_array = np.array(inverted).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = int(np.max(prediction) * 100)

        self.label.configure(text=f"{digit}, {confidence}%")

# Run the application
if __name__ == "__main__":
    app = DigitRecognizer()
    app.mainloop()
