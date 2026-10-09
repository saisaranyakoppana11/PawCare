from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, date, timedelta, timezone


db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    # Email verification
    is_verified = db.Column(
        db.Boolean,
        default=False,
        nullable=False
    )

    verification_code = db.Column(
        db.String(6),
        nullable=True
    )

    verification_expires_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # One user can have multiple pets
    pets = db.relationship(
        "Pet",
        backref="owner",
        lazy=True,
        cascade="all, delete-orphan"
    )

    # One user can have multiple lost and found reports
    lost_found_reports = db.relationship(
        "LostFoundReport",
        backref="reporter",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="desc(LostFoundReport.created_at)"
    )

    def __repr__(self):
        return f"<User {self.email}>"


class Pet(db.Model):
    __tablename__ = "pets"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # Owner of this pet
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    # Pet information
    name = db.Column(
        db.String(100),
        nullable=False
    )

    species = db.Column(
        db.String(20),
        nullable=False
    )

    breed = db.Column(
        db.String(100),
        nullable=True
    )

    gender = db.Column(
        db.String(20),
        nullable=True
    )

    date_of_birth = db.Column(
        db.Date,
        nullable=True
    )

    age = db.Column(
        db.Integer,
        nullable=True
    )

    weight = db.Column(
        db.Float,
        nullable=True
    )

    photo = db.Column(
        db.String(255),
        nullable=True
    )

    # Relationship: Pet has many vaccination records
    vaccinations = db.relationship(
        "Vaccination",
        backref="pet",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="desc(Vaccination.vaccination_date)"
    )

    # Relationship: Pet has many deworming records
    dewormings = db.relationship(
        "Deworming",
        backref="pet",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="desc(Deworming.deworming_date)"
    )

    @property
    def overdue_vaccinations(self):
        return [v for v in self.vaccinations if v.is_overdue]

    @property
    def upcoming_vaccinations(self):
        return [v for v in self.vaccinations if v.is_upcoming]

    @property
    def overdue_dewormings(self):
        return [d for d in self.dewormings if d.is_overdue]

    @property
    def upcoming_dewormings(self):
        return [d for d in self.dewormings if d.is_upcoming]

    def __repr__(self):
        return f"<Pet {self.name} - {self.species}>"


class Vaccination(db.Model):
    __tablename__ = "vaccinations"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    pet_id = db.Column(
        db.Integer,
        db.ForeignKey("pets.id", ondelete="CASCADE"),
        nullable=False
    )

    vaccine_name = db.Column(
        db.String(150),
        nullable=False
    )

    vaccination_date = db.Column(
        db.Date,
        nullable=False
    )

    next_due_date = db.Column(
        db.Date,
        nullable=True
    )

    veterinarian = db.Column(
        db.String(120),
        nullable=True
    )

    clinic = db.Column(
        db.String(150),
        nullable=True
    )

    notes = db.Column(
        db.Text,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_overdue(self):
        if not self.next_due_date:
            return False
        return self.next_due_date < date.today()

    @property
    def is_upcoming(self):
        if not self.next_due_date:
            return False
        today = date.today()
        return today <= self.next_due_date <= (today + timedelta(days=30))

    @property
    def status(self):
        if not self.next_due_date:
            return "Completed"
        today = date.today()
        if self.next_due_date < today:
            return "Overdue"
        elif today <= self.next_due_date <= (today + timedelta(days=30)):
            return "Due Soon"
        else:
            return "Up to Date"

    @property
    def status_badge_class(self):
        if not self.next_due_date:
            return "bg-secondary"
        today = date.today()
        if self.next_due_date < today:
            return "bg-danger"
        elif today <= self.next_due_date <= (today + timedelta(days=30)):
            return "bg-warning text-dark"
        else:
            return "bg-success"

    def __repr__(self):
        return f"<Vaccination {self.vaccine_name} for Pet {self.pet_id}>"


class Deworming(db.Model):
    __tablename__ = "dewormings"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    pet_id = db.Column(
        db.Integer,
        db.ForeignKey("pets.id", ondelete="CASCADE"),
        nullable=False
    )

    medicine_name = db.Column(
        db.String(150),
        nullable=False
    )

    deworming_date = db.Column(
        db.Date,
        nullable=False
    )

    next_due_date = db.Column(
        db.Date,
        nullable=True
    )

    veterinarian = db.Column(
        db.String(120),
        nullable=True
    )

    notes = db.Column(
        db.Text,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_overdue(self):
        if not self.next_due_date:
            return False
        return self.next_due_date < date.today()

    @property
    def is_upcoming(self):
        if not self.next_due_date:
            return False
        today = date.today()
        return today <= self.next_due_date <= (today + timedelta(days=30))

    @property
    def status(self):
        if not self.next_due_date:
            return "Completed"
        today = date.today()
        if self.next_due_date < today:
            return "Overdue"
        elif today <= self.next_due_date <= (today + timedelta(days=30)):
            return "Due Soon"
        else:
            return "Up to Date"

    @property
    def status_badge_class(self):
        if not self.next_due_date:
            return "bg-secondary"
        today = date.today()
        if self.next_due_date < today:
            return "bg-danger"
        elif today <= self.next_due_date <= (today + timedelta(days=30)):
            return "bg-warning text-dark"
        else:
            return "bg-success"

    def __repr__(self):
        return f"<Deworming {self.medicine_name} for Pet {self.pet_id}>"


class LostFoundReport(db.Model):
    __tablename__ = "lost_found_reports"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    # 'lost' or 'found'
    report_type = db.Column(
        db.String(20),
        nullable=False
    )

    # Name of the pet (optional for found pets)
    pet_name = db.Column(
        db.String(100),
        nullable=True
    )

    # 'Dog', 'Cat', or 'Other'
    species = db.Column(
        db.String(50),
        nullable=False
    )

    breed = db.Column(
        db.String(100),
        nullable=True
    )

    # 'Male', 'Female', or 'Unknown'
    gender = db.Column(
        db.String(20),
        nullable=True,
        default="Unknown"
    )

    # Primary colors / patterns
    color = db.Column(
        db.String(100),
        nullable=True
    )

    # Image filename stored in uploads/
    photo = db.Column(
        db.String(255),
        nullable=True
    )

    # Last seen (for lost) or found location
    location = db.Column(
        db.String(255),
        nullable=False
    )

    city = db.Column(
        db.String(100),
        nullable=True
    )

    # Date pet went missing or was found
    report_date = db.Column(
        db.Date,
        nullable=False
    )

    # Time of incident / sighting (e.g. "14:30" or "Morning")
    report_time = db.Column(
        db.String(50),
        nullable=True
    )

    # Distinguishing markings, collar/tag, microchip info, behavioral notes
    description = db.Column(
        db.Text,
        nullable=False
    )

    # Safe contact info specifically shared for this report
    contact_name = db.Column(
        db.String(100),
        nullable=True
    )

    contact_phone = db.Column(
        db.String(50),
        nullable=True
    )

    contact_email = db.Column(
        db.String(120),
        nullable=True
    )

    # 'active' or 'resolved'
    status = db.Column(
        db.String(20),
        default="active",
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    resolved_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # ----------------------------------
    # Helper Properties
    # ----------------------------------

    @property
    def is_lost(self):
        return (self.report_type or "").lower() == "lost"

    @property
    def is_found(self):
        return (self.report_type or "").lower() == "found"

    @property
    def is_resolved(self):
        return (self.status or "").lower() == "resolved"

    @property
    def is_active(self):
        return (self.status or "").lower() == "active"

    @property
    def type_label(self):
        return "Lost Pet" if self.is_lost else "Found Pet"

    @property
    def type_badge_class(self):
        return "bg-danger" if self.is_lost else "bg-success"

    @property
    def status_badge_class(self):
        return "bg-secondary" if self.is_resolved else "bg-warning text-dark"

    @property
    def display_name(self):
        if self.pet_name:
            return self.pet_name
        return f"Unknown {self.species or 'Pet'}"

    def __repr__(self):
        return f"<LostFoundReport {self.report_type} - {self.display_name} ({self.status})>"