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

    pets = db.relationship(
        "Pet",
        backref="owner",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User {self.email}>"


class Pet(db.Model):
    __tablename__ = "pets"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

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

    photo = db.Column(
        db.String(255),
        nullable=True
    )

    vaccinations = db.relationship(
        "Vaccination",
        backref="pet",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="desc(Vaccination.vaccination_date)"
    )

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
        db.ForeignKey("pets.id"),
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
        db.ForeignKey("pets.id"),
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