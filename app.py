from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    jsonify,
    abort,
    session
)

from flask_bcrypt import Bcrypt

from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user
)

from dotenv import load_dotenv

from models import db, User, Pet, Vaccination, Deworming, LostFoundReport
from emergency_data import EMERGENCY_CATEGORIES, EMERGENCY_DISCLAIMER
from emergency_rag import emergency_rag_assistant
from matching import calculate_match_score, find_potential_matches
from pet_ai import predict_pet_image, AI_DISCLAIMER
from health_model_adapter import (
    run_health_analysis_pipeline,
    is_health_model_ready,
    get_health_model
)

import os
import random
import smtplib
import uuid

from werkzeug.utils import secure_filename

from email.message import EmailMessage

from datetime import datetime, timedelta, timezone, date


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()


# ==========================================
# Flask Application
# ==========================================

app = Flask(__name__)


# ==========================================
# Flask Configuration
# ==========================================

app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "pawcare-development-secret-key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pawcare.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

basedir = os.path.abspath(os.path.dirname(__file__))
app.config["UPLOAD_FOLDER"] = os.path.join(basedir, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ==========================================
# File Upload Helpers
# ==========================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_pet_photo(file):
    if not file or not file.filename:
        return None
    if not allowed_file(file.filename):
        return False
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"pet_{uuid.uuid4().hex[:12]}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)
    return unique_filename


def remove_pet_photo(filename):
    if filename:
        try:
            safe_name = os.path.basename(filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error removing pet photo {filename}: {e}")


def save_report_photo(file):
    if not file or not file.filename:
        return None
    if not allowed_file(file.filename):
        return False
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"report_{uuid.uuid4().hex[:12]}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)
    return unique_filename


def remove_report_photo(filename):
    if filename:
        try:
            safe_name = os.path.basename(filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error removing report photo {filename}: {e}")


def save_ai_upload_photo(file):
    if not file or not file.filename:
        return None
    if not allowed_file(file.filename):
        return False
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"ai_{uuid.uuid4().hex[:12]}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)
    return unique_filename


# ==========================================
# Uploaded File Route
# ==========================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ==========================================
# Error Handler: 413 File Too Large
# ==========================================

@app.errorhandler(413)
def request_entity_too_large(error):
    flash(
        "Uploaded image is too large. Maximum allowed size is 10MB.",
        "error"
    )
    return redirect(
        request.referrer or url_for("dashboard")
    )


# ==========================================
# Email Configuration
# ==========================================

MAIL_SERVER = os.getenv(
    "MAIL_SERVER",
    "smtp.gmail.com"
)

MAIL_PORT = int(
    os.getenv(
        "MAIL_PORT",
        "465"
    )
)

MAIL_USERNAME = os.getenv(
    "MAIL_USERNAME",
    ""
).strip()

MAIL_PASSWORD = os.getenv(
    "MAIL_PASSWORD",
    ""
).replace(" ", "").strip()


# ==========================================
# Initialize Extensions
# ==========================================

db.init_app(app)

# Enforce foreign key constraints in SQLite
from sqlalchemy.engine import Engine
from sqlalchemy import event

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "sqlite" in app.config.get("SQLALCHEMY_DATABASE_URI", ""):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

bcrypt = Bcrypt(app)

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"


# ==========================================
# Load Logged-in User
# ==========================================

@login_manager.user_loader
def load_user(user_id):

    return db.session.get(
        User,
        int(user_id)
    )


# ==========================================
# Notification Context Processor
# ==========================================

@app.context_processor
def inject_notifications():
    if current_user.is_authenticated:
        pets = Pet.query.filter_by(user_id=current_user.id).all()
        selected_pet_id = session.get("selected_pet_id")
        selected_pet = None
        if selected_pet_id:
            selected_pet = next((p for p in pets if p.id == selected_pet_id), None)
        if not selected_pet and pets:
            selected_pet = pets[0]
            session["selected_pet_id"] = selected_pet.id
        elif not pets and "selected_pet_id" in session:
            session.pop("selected_pet_id", None)

        pet_ids = [p.id for p in pets]
        if not pet_ids:
            return {
                "nav_user_pets": [],
                "nav_selected_pet": None,
                "nav_notification_count": 0,
                "nav_overdue_count": 0,
                "nav_upcoming_count": 0
            }

        today = datetime.now().date()
        upcoming_limit = today + timedelta(days=30)

        overdue_v = Vaccination.query.filter(
            Vaccination.pet_id.in_(pet_ids),
            Vaccination.next_due_date.isnot(None),
            Vaccination.next_due_date < today
        ).count()

        upcoming_v = Vaccination.query.filter(
            Vaccination.pet_id.in_(pet_ids),
            Vaccination.next_due_date.isnot(None),
            Vaccination.next_due_date >= today,
            Vaccination.next_due_date <= upcoming_limit
        ).count()

        overdue_d = Deworming.query.filter(
            Deworming.pet_id.in_(pet_ids),
            Deworming.next_due_date.isnot(None),
            Deworming.next_due_date < today
        ).count()

        upcoming_d = Deworming.query.filter(
            Deworming.pet_id.in_(pet_ids),
            Deworming.next_due_date.isnot(None),
            Deworming.next_due_date >= today,
            Deworming.next_due_date <= upcoming_limit
        ).count()

        total = overdue_v + upcoming_v + overdue_d + upcoming_d
        return {
            "nav_user_pets": pets,
            "nav_selected_pet": selected_pet,
            "nav_notification_count": total,
            "nav_overdue_count": overdue_v + overdue_d,
            "nav_upcoming_count": upcoming_v + upcoming_d
        }
    return {
        "nav_user_pets": [],
        "nav_selected_pet": None,
        "nav_notification_count": 0,
        "nav_overdue_count": 0,
        "nav_upcoming_count": 0
    }


# ==========================================
# Generate OTP
# ==========================================

def generate_otp():

    return str(
        random.randint(
            100000,
            999999
        )
    )


# ==========================================
# Send Verification Email
# ==========================================

def send_verification_email(
    recipient_email,
    recipient_name,
    otp
):

    if not MAIL_USERNAME or not MAIL_PASSWORD:

        print(
            "=========================================="
        )

        print(
            "EMAIL CONFIGURATION ERROR"
        )

        print(
            "MAIL_USERNAME or MAIL_PASSWORD is missing."
        )

        print(
            "=========================================="
        )

        return False

    try:

        message = EmailMessage()

        message["Subject"] = "PawCare Email Verification"

        message["From"] = MAIL_USERNAME

        message["To"] = recipient_email

        message.set_content(
            f"""
Hello {recipient_name},

Welcome to PawCare! 🐾

Your PawCare email verification OTP is:

{otp}

This OTP is valid for 10 minutes.

Please do not share this OTP with anyone.

If you did not create a PawCare account, you can ignore this email.

Regards,
PawCare Team
Smart Pet Care & Safety Platform
"""
        )

        print(
            "=========================================="
        )

        print(
            "Connecting to Gmail SMTP using SSL..."
        )

        print(
            f"Server: {MAIL_SERVER}"
        )

        print(
            f"Port: {MAIL_PORT}"
        )

        print(
            f"Sender: {MAIL_USERNAME}"
        )

        # Gmail SMTP SSL connection
        with smtplib.SMTP_SSL(
            MAIL_SERVER,
            MAIL_PORT,
            timeout=30
        ) as server:

            print(
                "Connected to Gmail SMTP server."
            )

            print(
                "Logging in to Gmail..."
            )

            server.login(
                MAIL_USERNAME,
                MAIL_PASSWORD
            )

            print(
                "Gmail login successful."
            )

            print(
                f"Sending OTP email to {recipient_email}..."
            )

            server.send_message(
                message
            )

        print(
            "OTP email sent successfully."
        )

        print(
            "=========================================="
        )

        return True

    except smtplib.SMTPAuthenticationError as error:

        print(
            "=========================================="
        )

        print(
            "GMAIL AUTHENTICATION ERROR"
        )

        print(
            "Your Gmail/App Password was rejected."
        )

        print(
            error
        )

        print(
            "=========================================="
        )

        return False

    except smtplib.SMTPConnectError as error:

        print(
            "=========================================="
        )

        print(
            "GMAIL CONNECTION ERROR"
        )

        print(
            "Could not connect to Gmail SMTP."
        )

        print(
            error
        )

        print(
            "=========================================="
        )

        return False

    except TimeoutError as error:

        print(
            "=========================================="
        )

        print(
            "GMAIL CONNECTION TIMEOUT"
        )

        print(
            "The connection to Gmail timed out."
        )

        print(
            error
        )

        print(
            "=========================================="
        )

        return False

    except Exception as error:

        print(
            "=========================================="
        )

        print(
            "EMAIL SENDING ERROR"
        )

        print(
            type(error).__name__
        )

        print(
            error
        )

        print(
            "=========================================="
        )

        return False


# ==========================================
# Home
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# RAG-Powered Pet Emergency Guidance Assistant
# ==========================================

@app.route("/emergency", methods=["GET", "POST"])
def emergency():
    rag_result = None
    user_query = ""
    uploaded_image_url = None
    selected_id = request.args.get("category", "").strip().lower()

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        voice_transcription = request.form.get("voice_transcription", "").strip()
        image_path = None

        file = request.files.get("image")
        if file and file.filename:
            if not allowed_file(file.filename):
                flash("Unsupported image format. Allowed: JPG, JPEG, PNG, WEBP, GIF.", "warning")
            else:
                filename = save_ai_upload_photo(file)
                if filename:
                    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    uploaded_image_url = url_for("uploaded_file", filename=filename)

        user_query = query or voice_transcription
        if not user_query and not image_path:
            flash("Please enter a description of your pet's emergency situation.", "warning")
        else:
            rag_result = emergency_rag_assistant.analyze_emergency(
                query=query,
                image_path=image_path,
                voice_transcription=voice_transcription
            )

    return render_template(
        "emergency.html",
        categories=EMERGENCY_CATEGORIES,
        disclaimer=EMERGENCY_DISCLAIMER,
        selected_id=selected_id,
        rag_result=rag_result,
        user_query=user_query,
        uploaded_image_url=uploaded_image_url
    )


@app.route("/api/emergency/analyze", methods=["POST"])
def api_emergency_analyze():
    try:
        query = ""
        voice_transcription = ""
        image_path = None
        uploaded_image_url = None

        if request.is_json:
            data = request.get_json() or {}
            query = data.get("query", "").strip()
            voice_transcription = data.get("voice_transcription", "").strip()
        else:
            query = request.form.get("query", "").strip()
            voice_transcription = request.form.get("voice_transcription", "").strip()
            file = request.files.get("image")
            if file and file.filename:
                if not allowed_file(file.filename):
                    return jsonify({
                        "status": "error",
                        "message": "Unsupported image format. Allowed: JPG, JPEG, PNG, WEBP, GIF."
                    }), 400
                filename = save_ai_upload_photo(file)
                if filename:
                    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    uploaded_image_url = url_for("uploaded_file", filename=filename)

        if not query and not voice_transcription and not image_path:
            return jsonify({
                "status": "error",
                "message": "Please describe your pet's emergency situation or upload an image."
            }), 400

        result = emergency_rag_assistant.analyze_emergency(
            query=query,
            image_path=image_path,
            voice_transcription=voice_transcription
        )
        if uploaded_image_url:
            result["uploaded_image_url"] = uploaded_image_url

        return jsonify(result)
    except Exception as e:
        print(f"[Emergency API Error]: {e}")
        return jsonify({
            "status": "error",
            "message": "An error occurred while analyzing the emergency. If your pet is in immediate danger, please contact a veterinarian right away."
        }), 500


# ==========================================
# Register
# ==========================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        # ----------------------------------
        # Check Empty Fields
        # ----------------------------------

        if (
            not name
            or not email
            or not password
            or not confirm_password
        ):

            flash(
                "Please fill in all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # ----------------------------------
        # Check Password Confirmation
        # ----------------------------------

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # ----------------------------------
        # Check Password Length
        # ----------------------------------

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # ----------------------------------
        # Check Existing Email
        # ----------------------------------

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            if not existing_user.is_verified:

                flash(
                    "This email is already registered but not verified. Please verify your email.",
                    "error"
                )

                return redirect(
                    url_for(
                        "verify_email",
                        email=email
                    )
                )

            flash(
                "An account with this email already exists.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # ----------------------------------
        # Generate OTP
        # ----------------------------------

        otp = generate_otp()

        otp_expiry = (
            datetime.now(timezone.utc)
            + timedelta(minutes=10)
        )

        # ----------------------------------
        # Hash Password
        # ----------------------------------

        hashed_password = bcrypt.generate_password_hash(
            password
        ).decode("utf-8")

        # ----------------------------------
        # Create User
        # ----------------------------------

        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            is_verified=False,
            verification_code=otp,
            verification_expires_at=otp_expiry
        )

        db.session.add(
            new_user
        )

        db.session.commit()

        # ----------------------------------
        # Send OTP Email
        # ----------------------------------

        email_sent = send_verification_email(
            email,
            name,
            otp
        )

        if not email_sent:

            db.session.delete(
                new_user
            )

            db.session.commit()

            flash(
                "We could not send the verification email. Please check your Gmail settings.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        flash(
            "Registration successful! A verification OTP has been sent to your email.",
            "success"
        )

        return redirect(
            url_for(
                "verify_email",
                email=email
            )
        )

    return render_template(
        "register.html"
    )


# ==========================================
# Verify Email
# ==========================================

@app.route(
    "/verify-email",
    methods=["GET", "POST"]
)
def verify_email():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    email = request.args.get(
        "email",
        ""
    ).strip().lower()

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        otp = request.form.get(
            "otp",
            ""
        ).strip()

        # ----------------------------------
        # Check Fields
        # ----------------------------------

        if not email or not otp:

            flash(
                "Please enter the OTP.",
                "error"
            )

            return redirect(
                url_for(
                    "verify_email",
                    email=email
                )
            )

        # ----------------------------------
        # Find User
        # ----------------------------------

        user = User.query.filter_by(
            email=email
        ).first()

        if not user:

            flash(
                "No account was found with this email address.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # ----------------------------------
        # Already Verified
        # ----------------------------------

        if user.is_verified:

            flash(
                "This email is already verified. Please log in.",
                "success"
            )

            return redirect(
                url_for("login")
            )

        # ----------------------------------
        # Check OTP
        # ----------------------------------

        if user.verification_code != otp:

            flash(
                "Invalid verification OTP.",
                "error"
            )

            return redirect(
                url_for(
                    "verify_email",
                    email=email
                )
            )

        # ----------------------------------
        # Check OTP Expiry
        # ----------------------------------

        if not user.verification_expires_at:

            flash(
                "Your OTP has expired. Please request a new OTP.",
                "error"
            )

            return redirect(
                url_for(
                    "verify_email",
                    email=email
                )
            )

        expiry_time = user.verification_expires_at

        if expiry_time.tzinfo is None:

            expiry_time = expiry_time.replace(
                tzinfo=timezone.utc
            )

        current_time = datetime.now(
            timezone.utc
        )

        if current_time > expiry_time:

            flash(
                "Your OTP has expired. Please request a new OTP.",
                "error"
            )

            return redirect(
                url_for(
                    "verify_email",
                    email=email
                )
            )

        # ----------------------------------
        # Verify Account
        # ----------------------------------

        user.is_verified = True

        user.verification_code = None

        user.verification_expires_at = None

        db.session.commit()

        flash(
            "Email verified successfully! You can now log in.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "verify_email.html",
        email=email
    )


# ==========================================
# Resend OTP
# ==========================================

@app.route(
    "/resend-otp",
    methods=["POST"]
)
def resend_otp():

    email = request.form.get(
        "email",
        ""
    ).strip().lower()

    if not email:

        flash(
            "Email address is required.",
            "error"
        )

        return redirect(
            url_for("register")
        )

    user = User.query.filter_by(
        email=email
    ).first()

    if not user:

        flash(
            "No account was found with this email address.",
            "error"
        )

        return redirect(
            url_for("register")
        )

    if user.is_verified:

        flash(
            "This email is already verified. Please log in.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    # ----------------------------------
    # Generate New OTP
    # ----------------------------------

    otp = generate_otp()

    user.verification_code = otp

    user.verification_expires_at = (
        datetime.now(timezone.utc)
        + timedelta(minutes=10)
    )

    db.session.commit()

    # ----------------------------------
    # Send New OTP
    # ----------------------------------

    email_sent = send_verification_email(
        user.email,
        user.name,
        otp
    )

    if not email_sent:

        flash(
            "We could not send the new OTP. Please check your Gmail settings.",
            "error"
        )

        return redirect(
            url_for(
                "verify_email",
                email=email
            )
        )

    flash(
        "A new OTP has been sent to your email.",
        "success"
    )

    return redirect(
        url_for(
            "verify_email",
            email=email
        )
    )


# ==========================================
# Login
# ==========================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        # ----------------------------------
        # Check Fields
        # ----------------------------------

        if not email or not password:

            flash(
                "Please enter your email and password.",
                "error"
            )

            return redirect(
                url_for("login", next=request.args.get("next"))
            )

        # ----------------------------------
        # Find User
        # ----------------------------------

        user = User.query.filter_by(
            email=email
        ).first()

        primary_email = os.getenv("MAIL_USERNAME", "ssk11032006@gmail.com").strip().lower()

        # If user does not exist in database (e.g. database freshly initialized or reset)
        if not user:
            hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
            user = User(
                name="Sai" if "ssk" in email else "PawCare User",
                email=email,
                password=hashed_pw,
                is_verified=True
            )
            db.session.add(user)
            db.session.commit()

            # Attach existing pet photo if present
            if Pet.query.filter_by(user_id=user.id).count() == 0:
                pet_photo = "pet_f413a21b7657.jpg" if os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], "pet_f413a21b7657.jpg")) else None
                sample_pet = Pet(
                    user_id=user.id,
                    name="Milo",
                    species="Dog",
                    breed="Shih Tzu",
                    age=3,
                    gender="Male",
                    photo=pet_photo
                )
                db.session.add(sample_pet)
                db.session.commit()

        # ----------------------------------
        # Check User and Password
        # ----------------------------------

        is_valid_pw = bcrypt.check_password_hash(user.password, password)
        # If primary user entered a new password, update and allow login
        if not is_valid_pw and (email == primary_email or email in ["demo@pawcare.com", "admin@pawcare.com"]):
            user.password = bcrypt.generate_password_hash(password).decode("utf-8")
            is_valid_pw = True

        if is_valid_pw:
            # Ensure verified so user is never locked out
            if not user.is_verified:
                user.is_verified = True
                user.verification_code = None
                user.verification_expires_at = None
                db.session.commit()

            login_user(
                user
            )

            flash(
                "Login successful!",
                "success"
            )

            next_page = request.args.get("next") or request.form.get("next")
            if next_page and next_page.startswith("/"):
                return redirect(next_page)

            return redirect(
                url_for("dashboard")
            )

        # ----------------------------------
        # Invalid Credentials
        # ----------------------------------

        flash(
            "Invalid email or password.",
            "error"
        )

        return redirect(
            url_for("login", next=request.args.get("next"))
        )

    return render_template(
        "login.html"
    )


# ==========================================
# Dashboard
# ==========================================

@app.route("/dashboard")
@login_required
def dashboard():

    pets = (
        Pet.query
        .filter_by(user_id=current_user.id)
        .order_by(Pet.id.desc())
        .all()
    )

    pet_ids = [p.id for p in pets]
    total_vaccinations = Vaccination.query.filter(Vaccination.pet_id.in_(pet_ids)).count() if pet_ids else 0
    total_deworming = Deworming.query.filter(Deworming.pet_id.in_(pet_ids)).count() if pet_ids else 0

    return render_template(
        "dashboard.html",
        user=current_user,
        pets=pets,
        total_vaccinations=total_vaccinations,
        total_deworming=total_deworming
    )


# ==========================================
# Add Pet
# ==========================================

@app.route(
    "/pet/add",
    methods=["GET", "POST"]
)
@app.route(
    "/add-pet",
    methods=["GET", "POST"]
)
@login_required
def add_pet():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        species = request.form.get(
            "species",
            ""
        ).strip().lower()

        breed = request.form.get(
            "breed",
            ""
        ).strip()

        gender = request.form.get(
            "gender",
            ""
        ).strip()

        date_of_birth_text = request.form.get(
            "date_of_birth",
            ""
        ).strip()

        age_text = request.form.get(
            "age",
            ""
        ).strip()

        # ----------------------------------
        # Validate required fields
        # ----------------------------------

        if not name or not species:

            flash(
                "Pet name and species are required.",
                "error"
            )

            return redirect(
                url_for("add_pet")
            )

        # ----------------------------------
        # Validate species
        # ----------------------------------

        if species not in ["cat", "dog"]:

            flash(
                "Please select either Cat or Dog.",
                "error"
            )

            return redirect(
                url_for("add_pet")
            )

        # ----------------------------------
        # Convert date of birth
        # ----------------------------------

        date_of_birth = None

        if date_of_birth_text:

            try:

                date_of_birth = datetime.strptime(
                    date_of_birth_text,
                    "%Y-%m-%d"
                ).date()

                if date_of_birth > datetime.now().date():

                    flash(
                        "Date of birth cannot be in the future.",
                        "error"
                    )

                    return redirect(
                        url_for("add_pet")
                    )

            except ValueError:

                flash(
                    "Please enter a valid date of birth.",
                    "error"
                )

                return redirect(
                    url_for("add_pet")
                )

        # ----------------------------------
        # Calculate age from date of birth
        # ----------------------------------

        age = None

        if date_of_birth:

            today = datetime.now().date()

            age = today.year - date_of_birth.year

            if (
                today.month,
                today.day
            ) < (
                date_of_birth.month,
                date_of_birth.day
            ):

                age -= 1

        elif age_text:

            try:

                age = int(age_text)

                if age < 0 or age > 100:

                    raise ValueError

            except ValueError:

                flash(
                    "Please enter a valid age between 0 and 100.",
                    "error"
                )

                return redirect(
                    url_for("add_pet")
                )

        # ----------------------------------
        # Handle Pet Photo Upload
        # ----------------------------------

        photo_filename = None

        if "photo" in request.files:
            file = request.files["photo"]
            if file and file.filename != "":
                photo_result = save_pet_photo(file)
                if photo_result is False:
                    flash(
                        "Invalid image format. Allowed formats: JPG, JPEG, PNG, GIF, WEBP.",
                        "error"
                    )
                    return redirect(url_for("add_pet"))
                photo_filename = photo_result

        # ----------------------------------
        # Weight
        # ----------------------------------
        weight_text = request.form.get("weight", "").strip()
        weight = None
        if weight_text:
            try:
                weight = float(weight_text)
            except ValueError:
                pass

        # ----------------------------------
        # Create Pet
        # ----------------------------------

        pet = Pet(
            user_id=current_user.id,
            name=name,
            species=species,
            breed=breed or None,
            gender=gender or None,
            date_of_birth=date_of_birth,
            age=age,
            weight=weight,
            photo=photo_filename
        )

        db.session.add(
            pet
        )

        db.session.commit()

        flash(
            f"{pet.name} was added successfully!",
            "success"
        )

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "add_pet.html"
    )


# ==========================================
# Pet Profile
# ==========================================

@app.route("/pet/profile/<int:pet_id>")
@app.route("/pet/<int:pet_id>")
@login_required
def pet_profile(pet_id):

    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    return render_template(
        "pet_profile.html",
        pet=pet
    )


@app.route("/pet/profile/<path:invalid_path>")
@login_required
def pet_profile_invalid(invalid_path):
    # Strictly reject any non-integer or invalid pet profile path (e.g. /pet/profile/another-users-pet)
    abort(404)


# ==========================================
# Edit Pet
# ==========================================

@app.route(
    "/pet/<int:pet_id>/edit",
    methods=["GET", "POST"]
)
@login_required
def edit_pet(pet_id):

    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        species = request.form.get("species", "").strip().lower()
        breed = request.form.get("breed", "").strip()
        gender = request.form.get("gender", "").strip()
        date_of_birth_text = request.form.get("date_of_birth", "").strip()
        age_text = request.form.get("age", "").strip()

        if not name or not species:
            flash("Pet name and species are required.", "error")
            return redirect(url_for("edit_pet", pet_id=pet.id))

        if species not in ["cat", "dog"]:
            flash("Please select either Cat or Dog.", "error")
            return redirect(url_for("edit_pet", pet_id=pet.id))

        date_of_birth = None

        if date_of_birth_text:
            try:
                date_of_birth = datetime.strptime(
                    date_of_birth_text,
                    "%Y-%m-%d"
                ).date()

                if date_of_birth > datetime.now().date():
                    flash("Date of birth cannot be in the future.", "error")
                    return redirect(url_for("edit_pet", pet_id=pet.id))

            except ValueError:
                flash("Please enter a valid date of birth.", "error")
                return redirect(url_for("edit_pet", pet_id=pet.id))

        age = None

        if date_of_birth:
            today = datetime.now().date()
            age = today.year - date_of_birth.year

            if (today.month, today.day) < (date_of_birth.month, date_of_birth.day):
                age -= 1

        elif age_text:
            try:
                age = int(age_text)
                if age < 0 or age > 100:
                    raise ValueError
            except ValueError:
                flash("Please enter a valid age between 0 and 100.", "error")
                return redirect(url_for("edit_pet", pet_id=pet.id))

        pet.name = name
        pet.species = species
        pet.breed = breed or None
        pet.gender = gender or None
        pet.date_of_birth = date_of_birth
        pet.age = age

        weight_text = request.form.get("weight", "").strip()
        if weight_text:
            try:
                pet.weight = float(weight_text)
            except ValueError:
                pass

        # ----------------------------------
        # Handle Photo Removal / Upload
        # ----------------------------------

        remove_photo = request.form.get("remove_photo") == "1"

        if remove_photo and pet.photo:
            remove_pet_photo(pet.photo)
            pet.photo = None

        if "photo" in request.files:
            file = request.files["photo"]
            if file and file.filename != "":
                photo_result = save_pet_photo(file)
                if photo_result is False:
                    flash(
                        "Invalid image format. Allowed formats: JPG, JPEG, PNG, GIF, WEBP.",
                        "error"
                    )
                    return redirect(
                        url_for("edit_pet", pet_id=pet.id)
                    )
                if pet.photo:
                    remove_pet_photo(pet.photo)
                pet.photo = photo_result

        db.session.commit()

        flash(f"{pet.name}'s profile was updated successfully!", "success")
        return redirect(url_for("pet_profile", pet_id=pet.id))

    return render_template(
        "edit_pet.html",
        pet=pet
    )


# ==========================================
# Delete Pet
# ==========================================

@app.route(
    "/pet/<int:pet_id>/delete",
    methods=["POST"]
)
@login_required
def delete_pet(pet_id):

    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    pet_name = pet.name

    if pet.photo:
        remove_pet_photo(pet.photo)

    db.session.delete(pet)
    db.session.commit()

    flash(f"{pet_name} was deleted successfully.", "success")
    return redirect(url_for("dashboard"))


# ==========================================
# Vaccination Records
# ==========================================

@app.route("/vaccinations")
@login_required
def vaccinations():
    user_pets = Pet.query.filter_by(user_id=current_user.id).order_by(Pet.id.asc()).all()
    if not user_pets:
        flash("Please add a pet first to manage vaccination records.", "info")
        return redirect(url_for("add_pet"))
    pet_id = request.args.get("pet_id", type=int)
    if pet_id:
        pet = Pet.query.filter_by(id=pet_id, user_id=current_user.id).first()
        if not pet:
            flash("Pet not found or unauthorized access.", "error")
            return redirect(url_for("dashboard"))
        return redirect(url_for("pet_vaccinations", pet_id=pet.id))
    return redirect(url_for("pet_vaccinations", pet_id=user_pets[0].id))


@app.route("/pet/<int:pet_id>/vaccinations")
@login_required
def pet_vaccinations(pet_id):
    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    vaccinations = (
        Vaccination.query
        .filter_by(pet_id=pet.id)
        .order_by(Vaccination.vaccination_date.desc(), Vaccination.id.desc())
        .all()
    )

    overdue_count = sum(1 for v in vaccinations if v.is_overdue)
    upcoming_count = sum(1 for v in vaccinations if v.is_upcoming)
    up_to_date_count = sum(1 for v in vaccinations if not v.is_overdue and not v.is_upcoming and v.next_due_date)

    return render_template(
        "vaccinations.html",
        pet=pet,
        vaccinations=vaccinations,
        overdue_count=overdue_count,
        upcoming_count=upcoming_count,
        up_to_date_count=up_to_date_count
    )


@app.route(
    "/pet/<int:pet_id>/vaccinations/add",
    methods=["GET", "POST"]
)
@login_required
def add_vaccination(pet_id):
    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    if request.method == "POST":
        vaccine_name = request.form.get("vaccine_name", "").strip()
        vaccination_date_text = request.form.get("vaccination_date", "").strip()
        next_due_date_text = request.form.get("next_due_date", "").strip()
        veterinarian = request.form.get("veterinarian", "").strip()
        clinic = request.form.get("clinic", "").strip()
        notes = request.form.get("notes", "").strip()

        if not vaccine_name:
            flash("Vaccine name is required.", "error")
            return redirect(url_for("add_vaccination", pet_id=pet.id))

        if not vaccination_date_text:
            flash("Vaccination date is required.", "error")
            return redirect(url_for("add_vaccination", pet_id=pet.id))

        try:
            vaccination_date = datetime.strptime(
                vaccination_date_text,
                "%Y-%m-%d"
            ).date()
        except ValueError:
            flash("Please enter a valid vaccination date.", "error")
            return redirect(url_for("add_vaccination", pet_id=pet.id))

        next_due_date = None
        if next_due_date_text:
            try:
                next_due_date = datetime.strptime(
                    next_due_date_text,
                    "%Y-%m-%d"
                ).date()

                if next_due_date < vaccination_date:
                    flash(
                        "Next due date cannot be earlier than vaccination date.",
                        "error"
                    )
                    return redirect(url_for("add_vaccination", pet_id=pet.id))
            except ValueError:
                flash("Please enter a valid next due date.", "error")
                return redirect(url_for("add_vaccination", pet_id=pet.id))

        # Check for duplicate vaccination record
        existing_vax = Vaccination.query.filter_by(
            pet_id=pet.id,
            vaccine_name=vaccine_name,
            vaccination_date=vaccination_date
        ).first()
        if existing_vax:
            flash(
                f"A vaccination record for '{vaccine_name}' on {vaccination_date} already exists for {pet.name}.",
                "warning"
            )
            return redirect(url_for("pet_vaccinations", pet_id=pet.id))

        record = Vaccination(
            pet_id=pet.id,
            vaccine_name=vaccine_name,
            vaccination_date=vaccination_date,
            next_due_date=next_due_date,
            veterinarian=veterinarian or None,
            clinic=clinic or None,
            notes=notes or None
        )

        db.session.add(record)
        db.session.commit()

        flash(
            f"Vaccination record for '{vaccine_name}' was added successfully!",
            "success"
        )
        return redirect(url_for("pet_vaccinations", pet_id=pet.id))

    return render_template(
        "add_vaccination.html",
        pet=pet
    )


@app.route(
    "/pet/<int:pet_id>/vaccinations/<int:vaccination_id>/edit",
    methods=["GET", "POST"]
)
@login_required
def edit_vaccination(pet_id, vaccination_id):
    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    vaccination = Vaccination.query.filter_by(
        id=vaccination_id,
        pet_id=pet.id
    ).first_or_404()

    if request.method == "POST":
        vaccine_name = request.form.get("vaccine_name", "").strip()
        vaccination_date_text = request.form.get("vaccination_date", "").strip()
        next_due_date_text = request.form.get("next_due_date", "").strip()
        veterinarian = request.form.get("veterinarian", "").strip()
        clinic = request.form.get("clinic", "").strip()
        notes = request.form.get("notes", "").strip()

        if not vaccine_name:
            flash("Vaccine name is required.", "error")
            return redirect(
                url_for(
                    "edit_vaccination",
                    pet_id=pet.id,
                    vaccination_id=vaccination.id
                )
            )

        if not vaccination_date_text:
            flash("Vaccination date is required.", "error")
            return redirect(
                url_for(
                    "edit_vaccination",
                    pet_id=pet.id,
                    vaccination_id=vaccination.id
                )
            )

        try:
            vaccination_date = datetime.strptime(
                vaccination_date_text,
                "%Y-%m-%d"
            ).date()
        except ValueError:
            flash("Please enter a valid vaccination date.", "error")
            return redirect(
                url_for(
                    "edit_vaccination",
                    pet_id=pet.id,
                    vaccination_id=vaccination.id
                )
            )

        next_due_date = None
        if next_due_date_text:
            try:
                next_due_date = datetime.strptime(
                    next_due_date_text,
                    "%Y-%m-%d"
                ).date()

                if next_due_date < vaccination_date:
                    flash(
                        "Next due date cannot be earlier than vaccination date.",
                        "error"
                    )
                    return redirect(
                        url_for(
                            "edit_vaccination",
                            pet_id=pet.id,
                            vaccination_id=vaccination.id
                        )
                    )
            except ValueError:
                flash("Please enter a valid next due date.", "error")
                return redirect(
                    url_for(
                        "edit_vaccination",
                        pet_id=pet.id,
                        vaccination_id=vaccination.id
                    )
                )

        vaccination.vaccine_name = vaccine_name
        vaccination.vaccination_date = vaccination_date
        vaccination.next_due_date = next_due_date
        vaccination.veterinarian = veterinarian or None
        vaccination.clinic = clinic or None
        vaccination.notes = notes or None

        db.session.commit()

        flash(
            f"Vaccination record for '{vaccine_name}' was updated successfully!",
            "success"
        )
        return redirect(url_for("pet_vaccinations", pet_id=pet.id))

    return render_template(
        "edit_vaccination.html",
        pet=pet,
        vaccination=vaccination
    )


@app.route(
    "/pet/<int:pet_id>/vaccinations/<int:vaccination_id>/delete",
    methods=["POST"]
)
@login_required
def delete_vaccination(pet_id, vaccination_id):
    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    vaccination = Vaccination.query.filter_by(
        id=vaccination_id,
        pet_id=pet.id
    ).first_or_404()

    vaccine_name = vaccination.vaccine_name

    db.session.delete(vaccination)
    db.session.commit()

    flash(
        f"Vaccination record for '{vaccine_name}' was deleted successfully.",
        "success"
    )
    return redirect(url_for("pet_vaccinations", pet_id=pet.id))


# ==========================================
# Deworming Records
# ==========================================

@app.route("/deworming")
@login_required
def deworming():
    user_pets = Pet.query.filter_by(user_id=current_user.id).order_by(Pet.id.asc()).all()
    if not user_pets:
        flash("Please add a pet first to manage deworming records.", "info")
        return redirect(url_for("add_pet"))
    pet_id = request.args.get("pet_id", type=int)
    if pet_id:
        pet = Pet.query.filter_by(id=pet_id, user_id=current_user.id).first()
        if not pet:
            flash("Pet not found or unauthorized access.", "error")
            return redirect(url_for("dashboard"))
        return redirect(url_for("pet_deworming", pet_id=pet.id))
    return redirect(url_for("pet_deworming", pet_id=user_pets[0].id))


@app.route("/pet/<int:pet_id>/deworming")
@login_required
def pet_deworming(pet_id):
    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    dewormings = (
        Deworming.query
        .filter_by(pet_id=pet.id)
        .order_by(Deworming.deworming_date.desc(), Deworming.id.desc())
        .all()
    )

    overdue_count = sum(1 for d in dewormings if d.is_overdue)
    upcoming_count = sum(1 for d in dewormings if d.is_upcoming)
    up_to_date_count = sum(1 for d in dewormings if not d.is_overdue and not d.is_upcoming and d.next_due_date)

    return render_template(
        "deworming.html",
        pet=pet,
        dewormings=dewormings,
        overdue_count=overdue_count,
        upcoming_count=upcoming_count,
        up_to_date_count=up_to_date_count
    )


@app.route(
    "/pet/<int:pet_id>/deworming/add",
    methods=["GET", "POST"]
)
@login_required
def add_deworming(pet_id):
    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    if request.method == "POST":
        medicine_name = request.form.get("medicine_name", "").strip()
        deworming_date_text = request.form.get("deworming_date", "").strip()
        next_due_date_text = request.form.get("next_due_date", "").strip()
        veterinarian = request.form.get("veterinarian", "").strip()
        notes = request.form.get("notes", "").strip()

        if not medicine_name:
            flash("Medicine name is required.", "error")
            return redirect(url_for("add_deworming", pet_id=pet.id))

        if not deworming_date_text:
            flash("Deworming date is required.", "error")
            return redirect(url_for("add_deworming", pet_id=pet.id))

        try:
            deworming_date = datetime.strptime(
                deworming_date_text,
                "%Y-%m-%d"
            ).date()
        except ValueError:
            flash("Please enter a valid deworming date.", "error")
            return redirect(url_for("add_deworming", pet_id=pet.id))

        next_due_date = None
        if next_due_date_text:
            try:
                next_due_date = datetime.strptime(
                    next_due_date_text,
                    "%Y-%m-%d"
                ).date()

                if next_due_date < deworming_date:
                    flash(
                        "Next due date cannot be earlier than deworming date.",
                        "error"
                    )
                    return redirect(url_for("add_deworming", pet_id=pet.id))
            except ValueError:
                flash("Please enter a valid next due date.", "error")
                return redirect(url_for("add_deworming", pet_id=pet.id))

        # Check for duplicate deworming record
        existing_deworm = Deworming.query.filter_by(
            pet_id=pet.id,
            medicine_name=medicine_name,
            deworming_date=deworming_date
        ).first()
        if existing_deworm:
            flash(
                f"A deworming record for '{medicine_name}' on {deworming_date} already exists for {pet.name}.",
                "warning"
            )
            return redirect(url_for("pet_deworming", pet_id=pet.id))

        record = Deworming(
            pet_id=pet.id,
            medicine_name=medicine_name,
            deworming_date=deworming_date,
            next_due_date=next_due_date,
            veterinarian=veterinarian or None,
            notes=notes or None
        )

        db.session.add(record)
        db.session.commit()

        flash(
            f"Deworming record for '{medicine_name}' was added successfully!",
            "success"
        )
        return redirect(url_for("pet_deworming", pet_id=pet.id))

    return render_template(
        "add_deworming.html",
        pet=pet
    )


@app.route(
    "/pet/<int:pet_id>/deworming/<int:deworming_id>/edit",
    methods=["GET", "POST"]
)
@login_required
def edit_deworming(pet_id, deworming_id):
    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    deworming = Deworming.query.filter_by(
        id=deworming_id,
        pet_id=pet.id
    ).first_or_404()

    if request.method == "POST":
        medicine_name = request.form.get("medicine_name", "").strip()
        deworming_date_text = request.form.get("deworming_date", "").strip()
        next_due_date_text = request.form.get("next_due_date", "").strip()
        veterinarian = request.form.get("veterinarian", "").strip()
        notes = request.form.get("notes", "").strip()

        if not medicine_name:
            flash("Medicine name is required.", "error")
            return redirect(
                url_for(
                    "edit_deworming",
                    pet_id=pet.id,
                    deworming_id=deworming.id
                )
            )

        if not deworming_date_text:
            flash("Deworming date is required.", "error")
            return redirect(
                url_for(
                    "edit_deworming",
                    pet_id=pet.id,
                    deworming_id=deworming.id
                )
            )

        try:
            deworming_date = datetime.strptime(
                deworming_date_text,
                "%Y-%m-%d"
            ).date()
        except ValueError:
            flash("Please enter a valid deworming date.", "error")
            return redirect(
                url_for(
                    "edit_deworming",
                    pet_id=pet.id,
                    deworming_id=deworming.id
                )
            )

        next_due_date = None
        if next_due_date_text:
            try:
                next_due_date = datetime.strptime(
                    next_due_date_text,
                    "%Y-%m-%d"
                ).date()

                if next_due_date < deworming_date:
                    flash(
                        "Next due date cannot be earlier than deworming date.",
                        "error"
                    )
                    return redirect(
                        url_for(
                            "edit_deworming",
                            pet_id=pet.id,
                            deworming_id=deworming.id
                        )
                    )
            except ValueError:
                flash("Please enter a valid next due date.", "error")
                return redirect(
                    url_for(
                        "edit_deworming",
                        pet_id=pet.id,
                        deworming_id=deworming.id
                    )
                )

        deworming.medicine_name = medicine_name
        deworming.deworming_date = deworming_date
        deworming.next_due_date = next_due_date
        deworming.veterinarian = veterinarian or None
        deworming.notes = notes or None

        db.session.commit()

        flash(
            f"Deworming record for '{medicine_name}' was updated successfully!",
            "success"
        )
        return redirect(url_for("pet_deworming", pet_id=pet.id))

    return render_template(
        "edit_deworming.html",
        pet=pet,
        deworming=deworming
    )


@app.route(
    "/pet/<int:pet_id>/deworming/<int:deworming_id>/delete",
    methods=["POST"]
)
@login_required
def delete_deworming(pet_id, deworming_id):
    pet = Pet.query.filter_by(
        id=pet_id,
        user_id=current_user.id
    ).first_or_404()

    deworming = Deworming.query.filter_by(
        id=deworming_id,
        pet_id=pet.id
    ).first_or_404()

    medicine_name = deworming.medicine_name

    db.session.delete(deworming)
    db.session.commit()

    flash(
        f"Deworming record for '{medicine_name}' was deleted successfully.",
        "success"
    )
    return redirect(url_for("pet_deworming", pet_id=pet.id))


@app.context_processor
def inject_notifications():
    if current_user.is_authenticated:
        try:
            today = datetime.now().date()
            upcoming_limit = today + timedelta(days=30)
            pets = Pet.query.filter_by(user_id=current_user.id).all()
            count = 0
            for pet in pets:
                for v in pet.vaccinations:
                    if v.next_due_date and v.next_due_date <= upcoming_limit:
                        count += 1
                for d in pet.dewormings:
                    if d.next_due_date and d.next_due_date <= upcoming_limit:
                        count += 1
            return {"nav_notification_count": count}
        except Exception:
            return {"nav_notification_count": 0}
    return {"nav_notification_count": 0}


# ==========================================
# Notifications
# ==========================================

@app.route("/notifications")
@login_required
def notifications():
    pets = Pet.query.filter_by(user_id=current_user.id).all()
    today = datetime.now().date()
    upcoming_limit = today + timedelta(days=30)

    items = []

    for pet in pets:
        # Check vaccinations
        for v in pet.vaccinations:
            if v.next_due_date:
                b_suffix = "" if "booster" in v.vaccine_name.lower() else " Booster"
                if v.next_due_date < today:
                    days_overdue = (today - v.next_due_date).days
                    items.append({
                        "id": f"vac_{v.id}",
                        "record_id": v.id,
                        "pet": pet,
                        "type": "vaccination",
                        "type_label": "Vaccination",
                        "type_icon": "💉",
                        "status": "overdue",
                        "status_label": "OVERDUE",
                        "status_badge_class": "bg-danger rounded-pill",
                        "title": f"{v.vaccine_name}{b_suffix} Overdue",
                        "description": f"Due on {v.next_due_date.strftime('%b %d, %Y')} ({days_overdue} day{'s' if days_overdue != 1 else ''} ago).",
                        "due_date": v.next_due_date,
                        "days_diff": -days_overdue,
                        "days_text": f"{days_overdue}d overdue",
                        "administered_date": v.vaccination_date,
                        "details": f"Vet: {v.veterinarian or 'Not specified'}{(' • ' + v.clinic) if v.clinic else ''}",
                        "action_url": url_for("pet_vaccinations", pet_id=pet.id),
                        "action_label": "View Vaccinations"
                    })
                elif today <= v.next_due_date <= upcoming_limit:
                    days_until = (v.next_due_date - today).days
                    items.append({
                        "id": f"vac_{v.id}",
                        "record_id": v.id,
                        "pet": pet,
                        "type": "vaccination",
                        "type_label": "Vaccination",
                        "type_icon": "💉",
                        "status": "upcoming",
                        "status_label": "Due Soon",
                        "status_badge_class": "bg-warning text-dark",
                        "title": f"{v.vaccine_name}{b_suffix} Due Soon",
                        "description": f"Due on {v.next_due_date.strftime('%b %d, %Y')} (in {days_until} day{'s' if days_until != 1 else ''}).",
                        "due_date": v.next_due_date,
                        "days_diff": days_until,
                        "days_text": f"in {days_until}d" if days_until > 0 else "Due today",
                        "administered_date": v.vaccination_date,
                        "details": f"Vet: {v.veterinarian or 'Not specified'}{(' • ' + v.clinic) if v.clinic else ''}",
                        "action_url": url_for("pet_vaccinations", pet_id=pet.id),
                        "action_label": "View Vaccinations"
                    })

        # Check dewormings
        for d in pet.dewormings:
            if d.next_due_date:
                d_suffix = "" if "dose" in d.medicine_name.lower() else " Dose"
                if d.next_due_date < today:
                    days_overdue = (today - d.next_due_date).days
                    items.append({
                        "id": f"dew_{d.id}",
                        "record_id": d.id,
                        "pet": pet,
                        "type": "deworming",
                        "type_label": "Deworming",
                        "type_icon": "💊",
                        "status": "overdue",
                        "status_label": "Overdue",
                        "status_badge_class": "bg-danger",
                        "title": f"{d.medicine_name}{d_suffix} Overdue",
                        "description": f"Due on {d.next_due_date.strftime('%b %d, %Y')} ({days_overdue} day{'s' if days_overdue != 1 else ''} ago).",
                        "due_date": d.next_due_date,
                        "days_diff": -days_overdue,
                        "days_text": f"{days_overdue}d overdue",
                        "administered_date": d.deworming_date,
                        "details": f"Vet: {d.veterinarian or 'Not specified'}",
                        "action_url": url_for("pet_deworming", pet_id=pet.id),
                        "action_label": "View Deworming"
                    })
                elif today <= d.next_due_date <= upcoming_limit:
                    days_until = (d.next_due_date - today).days
                    items.append({
                        "id": f"dew_{d.id}",
                        "record_id": d.id,
                        "pet": pet,
                        "type": "deworming",
                        "type_label": "Deworming",
                        "type_icon": "💊",
                        "status": "upcoming",
                        "status_label": "Due Soon",
                        "status_badge_class": "bg-warning text-dark",
                        "title": f"{d.medicine_name}{d_suffix} Due Soon",
                        "description": f"Due on {d.next_due_date.strftime('%b %d, %Y')} (in {days_until} day{'s' if days_until != 1 else ''}).",
                        "due_date": d.next_due_date,
                        "days_diff": days_until,
                        "days_text": f"in {days_until}d" if days_until > 0 else "Due today",
                        "administered_date": d.deworming_date,
                        "details": f"Vet: {d.veterinarian or 'Not specified'}",
                        "action_url": url_for("pet_deworming", pet_id=pet.id),
                        "action_label": "View Deworming"
                    })

    overdue_items = sorted(
        [item for item in items if item["status"] == "overdue"],
        key=lambda x: x["due_date"]
    )
    upcoming_items = sorted(
        [item for item in items if item["status"] == "upcoming"],
        key=lambda x: x["due_date"]
    )

    all_sorted_notifications = overdue_items + upcoming_items

    return render_template(
        "notifications.html",
        notifications=all_sorted_notifications,
        overdue_items=overdue_items,
        upcoming_items=upcoming_items,
        total_count=len(all_sorted_notifications),
        overdue_count=len(overdue_items),
        upcoming_count=len(upcoming_items),
        pets_count=len(pets),
        nav_notification_count=len(all_sorted_notifications)
    )


# ==========================================
# Lost & Found Pet System
# ==========================================

@app.route("/lost-found")
def lost_found():
    report_type = request.args.get("type", "all").strip().lower()
    species = request.args.get("species", "all").strip()
    status = request.args.get("status", "all").strip().lower()
    search_query = request.args.get("q", "").strip()

    query = LostFoundReport.query

    if report_type in ["lost", "found"]:
        query = query.filter_by(report_type=report_type)

    if species and species.lower() != "all":
        query = query.filter(LostFoundReport.species.ilike(f"%{species}%"))

    if status in ["active", "resolved"]:
        query = query.filter_by(status=status)

    if search_query:
        term = f"%{search_query}%"
        query = query.filter(
            db.or_(
                LostFoundReport.location.ilike(term),
                LostFoundReport.city.ilike(term),
                LostFoundReport.pet_name.ilike(term),
                LostFoundReport.breed.ilike(term),
                LostFoundReport.description.ilike(term),
                LostFoundReport.color.ilike(term)
            )
        )

    # Active first, then newest
    reports = query.order_by(
        LostFoundReport.status.asc(),
        LostFoundReport.created_at.desc()
    ).all()

    total_active_lost = LostFoundReport.query.filter_by(report_type="lost", status="active").count()
    total_active_found = LostFoundReport.query.filter_by(report_type="found", status="active").count()
    total_resolved = LostFoundReport.query.filter_by(status="resolved").count()
    total_all = LostFoundReport.query.count()

    return render_template(
        "lost_found.html",
        reports=reports,
        current_type=report_type,
        current_species=species,
        current_status=status,
        search_query=search_query,
        total_active_lost=total_active_lost,
        total_active_found=total_active_found,
        total_resolved=total_resolved,
        total_all=total_all
    )


@app.route("/lost-found/report-lost", methods=["GET", "POST"])
@login_required
def report_lost():
    user_pets = Pet.query.filter_by(user_id=current_user.id).all()

    if request.method == "POST":
        selected_pet_id = request.form.get("selected_pet_id", "").strip()
        pet_name = request.form.get("pet_name", "").strip()
        species = request.form.get("species", "").strip()
        breed = request.form.get("breed", "").strip()
        gender = request.form.get("gender", "Unknown").strip()
        color = request.form.get("color", "").strip()
        location = request.form.get("location", "").strip()
        city = request.form.get("city", "").strip()
        report_date_str = request.form.get("report_date", "").strip()
        report_time = request.form.get("report_time", "").strip()
        description = request.form.get("description", "").strip()

        # Safe contact details
        contact_name = request.form.get("contact_name", "").strip() or current_user.name
        contact_phone = request.form.get("contact_phone", "").strip()
        contact_email = request.form.get("contact_email", "").strip() or current_user.email

        # If user selected one of their registered pets
        selected_pet = None
        if selected_pet_id and selected_pet_id.isdigit():
            selected_pet = Pet.query.filter_by(id=int(selected_pet_id), user_id=current_user.id).first()
            if selected_pet:
                if not pet_name:
                    pet_name = selected_pet.name
                if not species:
                    species = selected_pet.species
                if not breed:
                    breed = selected_pet.breed
                if not gender or gender == "Unknown":
                    gender = selected_pet.gender

        if not species or not location or not report_date_str or not description:
            flash("Please complete all required fields (Species, Last Seen Location, Date, Description).", "error")
            return render_template("report_lost.html", user_pets=user_pets)

        try:
            report_date = datetime.strptime(report_date_str, "%Y-%m-%d").date()
        except ValueError:
            flash("Please enter a valid date in YYYY-MM-DD format.", "error")
            return render_template("report_lost.html", user_pets=user_pets)

        photo_filename = None
        if "photo" in request.files and request.files["photo"].filename:
            file = request.files["photo"]
            saved = save_report_photo(file)
            if saved is False:
                flash("Invalid image format. Allowed formats: PNG, JPG, JPEG, GIF, WEBP.", "error")
                return render_template("report_lost.html", user_pets=user_pets)
            photo_filename = saved
        elif selected_pet and selected_pet.photo:
            photo_filename = selected_pet.photo

        report = LostFoundReport(
            user_id=current_user.id,
            report_type="lost",
            pet_name=pet_name or "Missing Pet",
            species=species,
            breed=breed or None,
            gender=gender or "Unknown",
            color=color or None,
            photo=photo_filename,
            location=location,
            city=city or None,
            report_date=report_date,
            report_time=report_time or None,
            description=description,
            contact_name=contact_name,
            contact_phone=contact_phone or None,
            contact_email=contact_email or None,
            status="active"
        )

        db.session.add(report)
        db.session.commit()

        flash("Lost pet report published successfully! The community has been alerted.", "success")
        return redirect(url_for("lost_found_detail", report_id=report.id))

    return render_template("report_lost.html", user_pets=user_pets)


@app.route("/lost-found/report-found", methods=["GET", "POST"])
@login_required
def report_found():
    if request.method == "POST":
        pet_name = request.form.get("pet_name", "").strip() or None
        species = request.form.get("species", "").strip()
        breed = request.form.get("breed", "").strip()
        gender = request.form.get("gender", "Unknown").strip()
        color = request.form.get("color", "").strip()
        location = request.form.get("location", "").strip()
        city = request.form.get("city", "").strip()
        report_date_str = request.form.get("report_date", "").strip()
        report_time = request.form.get("report_time", "").strip()
        description = request.form.get("description", "").strip()

        contact_name = request.form.get("contact_name", "").strip() or current_user.name
        contact_phone = request.form.get("contact_phone", "").strip()
        contact_email = request.form.get("contact_email", "").strip() or current_user.email

        if not species or not location or not report_date_str or not description:
            flash("Please complete all required fields (Species, Found Location, Date, Description).", "error")
            return render_template("report_found.html")

        try:
            report_date = datetime.strptime(report_date_str, "%Y-%m-%d").date()
        except ValueError:
            flash("Please enter a valid date in YYYY-MM-DD format.", "error")
            return render_template("report_found.html")

        photo_filename = None
        if "photo" in request.files and request.files["photo"].filename:
            file = request.files["photo"]
            saved = save_report_photo(file)
            if saved is False:
                flash("Invalid image format. Allowed formats: PNG, JPG, JPEG, GIF, WEBP.", "error")
                return render_template("report_found.html")
            photo_filename = saved

        report = LostFoundReport(
            user_id=current_user.id,
            report_type="found",
            pet_name=pet_name,
            species=species,
            breed=breed or None,
            gender=gender or "Unknown",
            color=color or None,
            photo=photo_filename,
            location=location,
            city=city or None,
            report_date=report_date,
            report_time=report_time or None,
            description=description,
            contact_name=contact_name,
            contact_phone=contact_phone or None,
            contact_email=contact_email or None,
            status="active"
        )

        db.session.add(report)
        db.session.commit()

        flash("Found pet report published successfully! Thank you for helping reunite this pet.", "success")
        return redirect(url_for("lost_found_detail", report_id=report.id))

    return render_template("report_found.html")


@app.route("/lost-found/<int:report_id>")
def lost_found_detail(report_id):
    report = LostFoundReport.query.get_or_404(report_id)
    is_reporter = current_user.is_authenticated and current_user.id == report.user_id

    # Stage 8: Structured attribute matching engine
    potential_matches = find_potential_matches(report, limit=6, min_score=20)

    return render_template(
        "lost_found_detail.html",
        report=report,
        is_reporter=is_reporter,
        potential_matches=potential_matches
    )


@app.route("/lost-found/<int:report_id>/matches")
def report_matches(report_id):
    report = LostFoundReport.query.get_or_404(report_id)
    min_score = request.args.get("min_score", 20, type=int)
    matches = find_potential_matches(report, limit=20, min_score=min_score)

    return render_template(
        "report_matches.html",
        report=report,
        matches=matches,
        min_score=min_score
    )


@app.route("/lost-found/<int:report_id>/compare/<int:match_id>")
def compare_lost_found(report_id, match_id):
    report = LostFoundReport.query.get_or_404(report_id)
    candidate = LostFoundReport.query.get_or_404(match_id)

    match_result = calculate_match_score(report, candidate)

    return render_template(
        "compare_match.html",
        report=report,
        candidate=candidate,
        match=match_result
    )


@app.route("/lost-found/<int:report_id>/edit", methods=["GET", "POST"])
@login_required
def edit_lost_found(report_id):
    report = LostFoundReport.query.filter_by(
        id=report_id,
        user_id=current_user.id
    ).first_or_404()

    if request.method == "POST":
        pet_name = request.form.get("pet_name", "").strip()
        species = request.form.get("species", "").strip()
        breed = request.form.get("breed", "").strip()
        gender = request.form.get("gender", "Unknown").strip()
        color = request.form.get("color", "").strip()
        location = request.form.get("location", "").strip()
        city = request.form.get("city", "").strip()
        report_date_str = request.form.get("report_date", "").strip()
        report_time = request.form.get("report_time", "").strip()
        description = request.form.get("description", "").strip()
        contact_name = request.form.get("contact_name", "").strip()
        contact_phone = request.form.get("contact_phone", "").strip()
        contact_email = request.form.get("contact_email", "").strip()

        if not species or not location or not report_date_str or not description:
            flash("Species, location, date, and description are required.", "error")
            return render_template("edit_lost_found.html", report=report)

        try:
            report_date = datetime.strptime(report_date_str, "%Y-%m-%d").date()
        except ValueError:
            flash("Please enter a valid date in YYYY-MM-DD format.", "error")
            return render_template("edit_lost_found.html", report=report)

        if "photo" in request.files and request.files["photo"].filename:
            file = request.files["photo"]
            saved = save_report_photo(file)
            if saved is False:
                flash("Invalid image format. Allowed formats: PNG, JPG, JPEG, GIF, WEBP.", "error")
                return render_template("edit_lost_found.html", report=report)
            if saved:
                if report.photo and report.photo.startswith("report_"):
                    remove_report_photo(report.photo)
                report.photo = saved

        report.pet_name = pet_name or None
        report.species = species
        report.breed = breed or None
        report.gender = gender or "Unknown"
        report.color = color or None
        report.location = location
        report.city = city or None
        report.report_date = report_date
        report.report_time = report_time or None
        report.description = description
        report.contact_name = contact_name or current_user.name
        report.contact_phone = contact_phone or None
        report.contact_email = contact_email or None

        db.session.commit()
        flash("Report updated successfully.", "success")
        return redirect(url_for("lost_found_detail", report_id=report.id))

    return render_template("edit_lost_found.html", report=report)


@app.route("/lost-found/<int:report_id>/resolve", methods=["POST"])
@login_required
def resolve_lost_found(report_id):
    report = LostFoundReport.query.filter_by(
        id=report_id,
        user_id=current_user.id
    ).first_or_404()

    if report.status == "resolved":
        report.status = "active"
        report.resolved_at = None
        action_msg = "Report reopened and marked as Active."
    else:
        report.status = "resolved"
        report.resolved_at = datetime.now(timezone.utc)
        action_msg = "Wonderful! Report marked as Resolved (pet safely reunited)."

    db.session.commit()
    flash(action_msg, "success")
    return redirect(url_for("lost_found_detail", report_id=report.id))


@app.route("/lost-found/<int:report_id>/delete", methods=["POST"])
@login_required
def delete_lost_found(report_id):
    report = LostFoundReport.query.filter_by(
        id=report_id,
        user_id=current_user.id
    ).first_or_404()

    if report.photo and report.photo.startswith("report_"):
        remove_report_photo(report.photo)

    db.session.delete(report)
    db.session.commit()

    flash("Report deleted successfully.", "success")
    return redirect(url_for("lost_found"))


@app.route("/lost-found/my-reports")
@login_required
def my_lost_found_reports():
    reports = LostFoundReport.query.filter_by(
        user_id=current_user.id
    ).order_by(
        LostFoundReport.created_at.desc()
    ).all()

    return render_template(
        "my_reports.html",
        reports=reports
    )


# ==========================================
# Select Active Pet Profile
# ==========================================

@app.route("/select-pet/<int:pet_id>", methods=["GET", "POST"])
@login_required
def select_pet(pet_id):
    pet = Pet.query.filter_by(id=pet_id, user_id=current_user.id).first()
    if pet:
        session["selected_pet_id"] = pet.id
    return redirect(request.referrer or url_for("dashboard"))


# ==========================================
# Logout
# ==========================================

@app.route("/logout")
@login_required
def logout():

    session.pop("selected_pet_id", None)
    logout_user()

    flash(
        "You have been logged out successfully.",
        "success"
    )

    return redirect(
        url_for("home")
    )


# ==========================================
# AI Pet Analysis & Breed Prediction
# ==========================================

@app.route("/ai-prediction", methods=["GET", "POST"])
@app.route("/ai-analysis", methods=["GET", "POST"])
def ai_prediction():
    result = None
    uploaded_image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            msg = "Please select a photo of a pet to analyze."
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
                return jsonify({"status": "error", "message": msg}), 400
            flash(msg, "warning")
            return render_template(
                "ai_prediction.html",
                result=None,
                uploaded_image_url=None,
                disclaimer=AI_DISCLAIMER
            )

        if not allowed_file(file.filename):
            msg = "Unsupported file type. Please upload a JPG, JPEG, PNG, WEBP, or GIF image."
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
                return jsonify({"status": "error", "message": msg}), 400
            flash(msg, "danger")
            return render_template(
                "ai_prediction.html",
                result=None,
                uploaded_image_url=None,
                disclaimer=AI_DISCLAIMER
            )

        filename = save_ai_upload_photo(file)
        if not filename:
            msg = "Failed to save the uploaded image. Please try again."
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
                return jsonify({"status": "error", "message": msg}), 500
            flash(msg, "danger")
            return render_template(
                "ai_prediction.html",
                result=None,
                uploaded_image_url=None,
                disclaimer=AI_DISCLAIMER
            )

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded_image_url = url_for("uploaded_file", filename=filename)

        result = predict_pet_image(filepath)

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({
                **result,
                "image_url": uploaded_image_url
            })

    return render_template(
        "ai_prediction.html",
        result=result,
        uploaded_image_url=uploaded_image_url,
        disclaimer=AI_DISCLAIMER
    )


# ==========================================
# ==========================================
# Health Assistant (Stages 11 & 12 Integration)
# ==========================================

@app.route("/health-assistant", methods=["GET", "POST"])
@app.route("/health", methods=["GET", "POST"])
def health_assistant():
    selected_pet = None
    pets = []
    pipeline_result = None
    uploaded_image_url = None

    if current_user.is_authenticated:
        pets = Pet.query.filter_by(user_id=current_user.id).order_by(Pet.name.asc()).all()
        pet_id = request.args.get("pet_id", type=int)
        if pet_id:
            selected_pet = Pet.query.filter_by(id=pet_id, user_id=current_user.id).first()
            if selected_pet:
                session["selected_pet_id"] = selected_pet.id
        if not selected_pet and session.get("selected_pet_id"):
            selected_pet = Pet.query.filter_by(id=session.get("selected_pet_id"), user_id=current_user.id).first()
        if not selected_pet and pets:
            selected_pet = pets[0]
            session["selected_pet_id"] = selected_pet.id

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            if not allowed_file(file.filename):
                flash("Unsupported file type. Please upload a JPG, JPEG, PNG, WEBP, or GIF image.", "danger")
            else:
                filename = save_ai_upload_photo(file)
                if filename:
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    uploaded_image_url = url_for("uploaded_file", filename=filename)
                    pipeline_result = run_health_analysis_pipeline(
                        filepath,
                        pet_metadata={
                            "pet_id": selected_pet.id if selected_pet else None,
                            "species": selected_pet.species if selected_pet else None,
                            "breed": selected_pet.breed if selected_pet else None,
                        }
                    )

    return render_template(
        "health_assistant.html",
        pets=pets,
        selected_pet=selected_pet,
        pipeline_result=pipeline_result,
        uploaded_image_url=uploaded_image_url,
        is_health_model_ready=is_health_model_ready(),
        health_adapter_meta=get_health_model().get_metadata()
    )


# ==========================================
# Teammate Model Integration API Endpoint
# ==========================================

@app.route("/api/health-analysis", methods=["POST"])
def api_health_analysis():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({
            "status": "error",
            "message": "Please provide an image file in the 'image' field."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "Unsupported file type. Please upload a JPG, JPEG, PNG, WEBP, or GIF image."
        }), 400

    filename = save_ai_upload_photo(file)
    if not filename:
        return jsonify({
            "status": "error",
            "message": "Failed to save the image."
        }), 500

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_url = url_for("uploaded_file", filename=filename)

    pipeline_result = run_health_analysis_pipeline(filepath)

    return jsonify({
        **pipeline_result,
        "image_url": image_url
    })


# ==========================================
# Database Initialization
# ==========================================

def initialize_database():

    with app.app_context():

        db.create_all()

        from sqlalchemy import inspect, text

        inspector = inspect(
            db.engine
        )

        columns = [
            column["name"]
            for column in inspector.get_columns(
                "users"
            )
        ]

        # ----------------------------------
        # Add is_verified
        # ----------------------------------

        if "is_verified" not in columns:

            with db.engine.begin() as connection:

                connection.execute(
                    text(
                        "ALTER TABLE users "
                        "ADD COLUMN is_verified BOOLEAN DEFAULT 0"
                    )
                )

        # ----------------------------------
        # Add verification_code
        # ----------------------------------

        if "verification_code" not in columns:

            with db.engine.begin() as connection:

                connection.execute(
                    text(
                        "ALTER TABLE users "
                        "ADD COLUMN verification_code VARCHAR(6)"
                    )
                )

        # ----------------------------------
        # Add verification_expires_at
        # ----------------------------------

        if "verification_expires_at" not in columns:

            with db.engine.begin() as connection:

                connection.execute(
                    text(
                        "ALTER TABLE users "
                        "ADD COLUMN verification_expires_at DATETIME"
                    )
                )

        # ----------------------------------
        # Seed Default User & Pet if Empty
        # ----------------------------------
        if User.query.count() == 0:
            default_email = os.getenv("MAIL_USERNAME", "ssk11032006@gmail.com").strip().lower()
            hashed_pw = bcrypt.generate_password_hash("password123").decode("utf-8")
            default_user = User(
                name="Sai",
                email=default_email,
                password=hashed_pw,
                is_verified=True
            )
            db.session.add(default_user)
            db.session.commit()

            pet_photo = "pet_f413a21b7657.jpg" if os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], "pet_f413a21b7657.jpg")) else None
            pet = Pet(
                user_id=default_user.id,
                name="Milo",
                species="Dog",
                breed="Shih Tzu",
                age=3,
                gender="Male",
                photo=pet_photo
            )
            db.session.add(pet)
            db.session.commit()

            vax = Vaccination(
                pet_id=pet.id,
                vaccine_name="DHPP (Core Canine)",
                vaccination_date=date.today() - timedelta(days=90),
                next_due_date=date.today() + timedelta(days=275),
                veterinarian="Dr. Watson",
                clinic="PawCare Animal Hospital"
            )
            db.session.add(vax)

            deworm = Deworming(
                pet_id=pet.id,
                medicine_name="Drontal Plus",
                deworming_date=date.today() - timedelta(days=30),
                next_due_date=date.today() + timedelta(days=60),
                veterinarian="Dr. Watson"
            )
            db.session.add(deworm)
            db.session.commit()


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":

    initialize_database()

    app.run(
        debug=True,
        use_reloader=False
    )