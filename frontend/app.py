from flask import Flask, request, jsonify, send_file, request,session, redirect, url_for
import mysql.connector
from flask_bcrypt import Bcrypt
from flask_session import Session
from PIL import Image
import numpy as np
import sys
import tenseal as ts
import os
from train import CryptoNet, create_ckks_context
import matplotlib.pyplot as plt

app = Flask(__name__) 
bcrypt = Bcrypt(app)

# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = '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'
Session(app)

connection = mysql.connector.connect(
    host='localhost',       
    user='root',    
    password='2004', 
    database='homomorphic'  
)

@app.route("/")
@app.route("/index")
def index():
    return send_file("index.html")

@app.route("/signup_page")
def signup_page():
    return send_file("signup.html")

@app.route("/login_page")
def login_page():
    return send_file("login.html")

@app.route("/aboutus_page")
def aboutus_page():
    return send_file("aboutus.html")

@app.route("/model_page")
def model_page():
    if 'user' in session:
        print(session['user'])
        return send_file("model.html")
    else:
        return redirect(url_for('login_page'))

@app.route("/aboutproj_page")
def aboutproj_page():
    return send_file("aboutproj.html")

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username")
    full_name = data.get("fullname")
    email = data.get("email")
    phone = data.get("contact")
    password = data.get("password")
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    print(username, full_name, email, phone, password, hashed_password)
    cursor = connection.cursor()
    cursor.execute("""INSERT INTO users (username,fullname, email, contact, password) 
                      VALUES (%s, %s, %s, %s, %s)""", 
                   (username, full_name, email, phone, hashed_password))
    connection.commit()
    return jsonify({"success": True})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    cursor = connection.cursor()
    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    
    print(username, password)
    
    if user and bcrypt.check_password_hash(user[0], password):
        session['user'] = username  
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Invalid credentials or user does not exist."}), 401

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized. Please log in first.'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Preprocess the image (grayscale, resize, normalize)
    image = Image.open(file.stream).convert('L')
    image = image.resize((28, 28))
    image_np = np.array(image).astype(np.float32)
    image_np = (image_np - np.mean(image_np)) / (np.std(image_np) + 1e-8)

    # Save the original input image for frontend display
    original_image_path = os.path.join('static', 'original_image.png')
    image.save(original_image_path)

    # Save the preprocessed input image for frontend display
    input_image_path = os.path.join('static', 'input_image.png')
    plt.figure(figsize=(3,3))
    plt.imshow(image_np, cmap='gray')
    plt.title('Input Image (preprocessed)')
    plt.axis('off')
    plt.savefig(input_image_path)
    plt.close()

    # Encrypt the image
    context = create_ckks_context()
    flat_img = image_np.flatten()
    enc_img = ts.ckks_vector(context, flat_img)

    # Plot the encrypted image (decrypted for visualization)
    enc_img_np = np.array(enc_img.decrypt())  # For visualization, decrypt for plotting
    plt.figure(figsize=(3,3))
    plt.imshow(enc_img_np.reshape(28,28), cmap='gray')
    plt.title('Encrypted Image (decrypted for visualization)')
    plt.axis('off')
    plot_path = os.path.join('static', 'encrypted_image_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Plot the encrypted image without decrypting (visualize serialized bytes as image)
    enc_img_bytes = np.frombuffer(enc_img.serialize(), dtype=np.uint8)
    # Pad or trim to 784 for 28x28 image
    if enc_img_bytes.size < 784:
        enc_img_bytes = np.pad(enc_img_bytes, (0, 784 - enc_img_bytes.size), 'constant')
    elif enc_img_bytes.size > 784:
        enc_img_bytes = enc_img_bytes[:784]
    plt.figure(figsize=(3,3))
    plt.imshow(enc_img_bytes.reshape(28,28), cmap='gray')
    plt.title('Encrypted Image (raw ciphertext bytes)')
    plt.axis('off')
    raw_plot_path = os.path.join('static', 'encrypted_image_raw_plot.png')
    plt.savefig(raw_plot_path)
    plt.close()

    # Load the model
    model = CryptoNet()
    model.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/Cryptonet_weights_final.npz')))

    # Predict
    output = model.forward(enc_img, context)
    decrypted_output = np.array(output.decrypt())
    prediction = int(np.argmax(decrypted_output))

    # Create the correct label vector (one-hot) and encrypt it
    correct_label = prediction  # If you have ground truth, use it; here, using prediction as placeholder
    one_hot = np.zeros(10)
    one_hot[correct_label] = 1.0
    enc_label = ts.ckks_vector(context, one_hot)

    # Print file name, prediction, and encrypted label to console
    print(session['user'])
    print(f"File uploaded: {file.filename}")
    print(f"Predicted label: {prediction}")
    print(f"Encrypted label vector: {enc_label}")

    # For returning encrypted vectors, convert to list for JSON serialization
    encrypted_image_list = list(enc_img.serialize())
    encrypted_label_list = list(enc_label.serialize())

    return jsonify({
        'message': 'Image uploaded and classified successfully',
        'filename': file.filename,
        'predicted_label': prediction,
        'original_image': original_image_path,
        'input_image': input_image_path,
        'encrypted_image_plot': plot_path,
        'encrypted_image_raw_plot': raw_plot_path
    })

if __name__ == "__main__":
    app.run(debug=True)
