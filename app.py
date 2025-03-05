from flask import Flask, request, jsonify, send_file, redirect, url_for, session
import requests
import mysql.connector
from flask_bcrypt import Bcrypt
from flask_session import Session

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
    return send_file("model.html")

@app.route("/aboutproj_page")
def aboutproj_page():
    return send_file("aboutproj.html")

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username=data.get("username")
    full_name = data.get("fullname")
    email = data.get("email")
    phone = data.get("contact")
    password =data.get("password")
    print(username, full_name, email, phone, password)
    cursor = connection.cursor()
    cursor.execute("""INSERT INTO users (username,fullname, email, contact, password) 
                      VALUES (%s, %s, %s, %s, %s)""", 
                   (username, full_name, email, phone, password))
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
    
    if user and user[0] == password:
        session['user'] = username  
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Invalid credentials or user does not exist."}), 401

if __name__ == "__main__":
    app.run(debug=True)
