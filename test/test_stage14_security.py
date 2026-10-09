import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
from bs4 import BeautifulSoup
from app import app, db, User, Pet, Vaccination, Deworming, LostFoundReport, bcrypt
from datetime import date, timedelta

BASE_URL = "http://127.0.0.1:5000"

def setup_test_users_and_data():
    with app.app_context():
        # Ensure user A exists
        user_a = User.query.filter_by(email="usera@test.com").first()
        if not user_a:
            user_a = User(
                name="Alice User",
                email="usera@test.com",
                password=bcrypt.generate_password_hash("password123").decode("utf-8"),
                is_verified=True
            )
            db.session.add(user_a)
            db.session.commit()

        # Ensure user B exists
        user_b = User.query.filter_by(email="userb@test.com").first()
        if not user_b:
            user_b = User(
                name="Bob User",
                email="userb@test.com",
                password=bcrypt.generate_password_hash("password123").decode("utf-8"),
                is_verified=True
            )
            db.session.add(user_b)
            db.session.commit()

        # Ensure pet A exists for user A
        pet_a = Pet.query.filter_by(user_id=user_a.id, name="AliceDog").first()
        if not pet_a:
            pet_a = Pet(
                user_id=user_a.id,
                name="AliceDog",
                species="dog",
                breed="Golden Retriever",
                age=2,
                gender="Female"
            )
            db.session.add(pet_a)
            db.session.commit()

        # Ensure pet B exists for user B
        pet_b = Pet.query.filter_by(user_id=user_b.id, name="BobCat").first()
        if not pet_b:
            pet_b = Pet(
                user_id=user_b.id,
                name="BobCat",
                species="cat",
                breed="Siamese",
                age=1,
                gender="Male"
            )
            db.session.add(pet_b)
            db.session.commit()

        # Add vaccination and deworming for Pet B
        vax_b = Vaccination.query.filter_by(pet_id=pet_b.id).first()
        if not vax_b:
            vax_b = Vaccination(
                pet_id=pet_b.id,
                vaccine_name="Rabies",
                vaccination_date=date.today() - timedelta(days=60),
                next_due_date=date.today() + timedelta(days=305)
            )
            db.session.add(vax_b)

        deworm_b = Deworming.query.filter_by(pet_id=pet_b.id).first()
        if not deworm_b:
            deworm_b = Deworming(
                pet_id=pet_b.id,
                medicine_name="Bravecto",
                deworming_date=date.today() - timedelta(days=20),
                next_due_date=date.today() + timedelta(days=70)
            )
            db.session.add(deworm_b)

        # Add Lost report for User B
        report_b = LostFoundReport.query.filter_by(user_id=user_b.id).first()
        if not report_b:
            report_b = LostFoundReport(
                user_id=user_b.id,
                report_type="lost",
                pet_name="BobCat",
                species="cat",
                breed="Siamese",
                location="Main Street Park",
                report_date=date.today() - timedelta(days=2),
                description="Lost Siamese cat with blue collar",
                contact_name="Bob User"
            )
            db.session.add(report_b)

        db.session.commit()

        return {
            "user_a_id": user_a.id,
            "user_b_id": user_b.id,
            "pet_a_id": pet_a.id,
            "pet_b_id": pet_b.id,
            "vax_b_id": vax_b.id,
            "deworm_b_id": deworm_b.id,
            "report_b_id": report_b.id
        }


def test_stage_14_security():
    ids = setup_test_users_and_data()
    print("Setup test data:", ids)

    pet_a_id = ids["pet_a_id"]
    pet_b_id = ids["pet_b_id"]
    vax_b_id = ids["vax_b_id"]
    deworm_b_id = ids["deworm_b_id"]
    report_b_id = ids["report_b_id"]

    # ----------------------------------------------------
    # Test 1: Unauthenticated access is blocked (redirect to login)
    # ----------------------------------------------------
    unauth_session = requests.Session()
    r = unauth_session.get(f"{BASE_URL}/pet/{pet_a_id}", allow_redirects=False)
    assert r.status_code == 302 and "/login" in r.headers.get("Location", ""), "Unauth pet profile should redirect to login"

    r = unauth_session.get(f"{BASE_URL}/pet/profile/{pet_a_id}", allow_redirects=False)
    assert r.status_code == 302 and "/login" in r.headers.get("Location", ""), "Unauth /pet/profile/<id> should redirect to login"

    r = unauth_session.get(f"{BASE_URL}/pet/{pet_a_id}/vaccinations", allow_redirects=False)
    assert r.status_code == 302 and "/login" in r.headers.get("Location", ""), "Unauth vaccinations should redirect to login"
    print("✓ Test 1: Unauthenticated access successfully blocked.")

    # ----------------------------------------------------
    # Test 2: User A Login & Authorized Access
    # ----------------------------------------------------
    session_a = requests.Session()
    r_login = session_a.post(f"{BASE_URL}/login", data={
        "email": "usera@test.com",
        "password": "password123"
    }, allow_redirects=True)
    assert r_login.status_code == 200, "User A login failed"

    # User A CAN access their own pet
    r = session_a.get(f"{BASE_URL}/pet/{pet_a_id}")
    assert r.status_code == 200, f"User A should be able to view own pet /pet/{pet_a_id}"

    r = session_a.get(f"{BASE_URL}/pet/profile/{pet_a_id}")
    assert r.status_code == 200, f"User A should be able to view own pet /pet/profile/{pet_a_id}"
    print("✓ Test 2: User A authorized access to own pet verified.")

    # ----------------------------------------------------
    # Test 3: STRICT PET ISOLATION — Cross-User Pet Access Forbidden!
    # "Never allow /pet/profile/another-users-pet to expose another user's private pet"
    # ----------------------------------------------------
    r = session_a.get(f"{BASE_URL}/pet/{pet_b_id}")
    assert r.status_code == 404, f"User A viewing User B's pet should return 404, got {r.status_code}"

    r = session_a.get(f"{BASE_URL}/pet/profile/{pet_b_id}")
    assert r.status_code == 404, f"User A viewing User B's pet via /pet/profile/<id> should return 404, got {r.status_code}"

    r = session_a.get(f"{BASE_URL}/pet/profile/another-users-pet")
    assert r.status_code == 404, f"Accessing /pet/profile/another-users-pet must return 404, got {r.status_code}"

    r = session_a.get(f"{BASE_URL}/pet/{pet_b_id}/edit")
    assert r.status_code == 404, f"User A editing User B's pet should return 404, got {r.status_code}"

    r = session_a.post(f"{BASE_URL}/pet/{pet_b_id}/delete")
    assert r.status_code == 404, f"User A deleting User B's pet should return 404, got {r.status_code}"
    print("✓ Test 3: Cross-user pet access, edit, delete, and profile slug strictly return 404.")

    # ----------------------------------------------------
    # Test 4: Cross-User Health Records Protection
    # ----------------------------------------------------
    r = session_a.get(f"{BASE_URL}/pet/{pet_b_id}/vaccinations")
    assert r.status_code == 404, f"User A viewing User B's vaccinations must return 404, got {r.status_code}"

    r = session_a.get(f"{BASE_URL}/pet/{pet_b_id}/vaccinations/{vax_b_id}/edit")
    assert r.status_code == 404, f"User A editing User B's vaccination must return 404, got {r.status_code}"

    r = session_a.post(f"{BASE_URL}/pet/{pet_b_id}/vaccinations/{vax_b_id}/delete")
    assert r.status_code == 404, f"User A deleting User B's vaccination must return 404, got {r.status_code}"

    r = session_a.get(f"{BASE_URL}/pet/{pet_b_id}/deworming")
    assert r.status_code == 404, f"User A viewing User B's deworming must return 404, got {r.status_code}"

    r = session_a.get(f"{BASE_URL}/pet/{pet_b_id}/deworming/{deworm_b_id}/edit")
    assert r.status_code == 404, f"User A editing User B's deworming must return 404, got {r.status_code}"

    r = session_a.post(f"{BASE_URL}/pet/{pet_b_id}/deworming/{deworm_b_id}/delete")
    assert r.status_code == 404, f"User A deleting User B's deworming must return 404, got {r.status_code}"
    print("✓ Test 4: Cross-user vaccination and deworming operations strictly blocked with 404.")

    # ----------------------------------------------------
    # Test 5: Lost & Found Ownership Protection
    # ----------------------------------------------------
    r = session_a.get(f"{BASE_URL}/lost-found/{report_b_id}/edit")
    assert r.status_code == 404, f"User A editing User B's lost/found report must return 404, got {r.status_code}"

    r = session_a.post(f"{BASE_URL}/lost-found/{report_b_id}/delete")
    assert r.status_code == 404, f"User A deleting User B's lost/found report must return 404, got {r.status_code}"

    r = session_a.post(f"{BASE_URL}/lost-found/{report_b_id}/resolve")
    assert r.status_code == 404, f"User A resolving User B's lost/found report must return 404, got {r.status_code}"
    print("✓ Test 5: Lost & Found edit, delete, resolve protected by ownership checks.")

    # ----------------------------------------------------
    # Test 6: Invalid IDs (Non-existent records)
    # ----------------------------------------------------
    r = session_a.get(f"{BASE_URL}/pet/999999")
    assert r.status_code == 404, "Non-existent pet ID should return 404"

    r = session_a.get(f"{BASE_URL}/pet/profile/999999")
    assert r.status_code == 404, "Non-existent pet profile ID should return 404"

    r = session_a.get(f"{BASE_URL}/pet/999999/vaccinations")
    assert r.status_code == 404, "Non-existent pet vaccinations should return 404"

    r = session_a.get(f"{BASE_URL}/lost-found/999999")
    assert r.status_code == 404, "Non-existent lost report should return 404"
    print("✓ Test 6: Invalid IDs gracefully return 404 without unhandled exceptions.")

    # ----------------------------------------------------
    # Test 7: Duplicate Medical Record Prevention
    # ----------------------------------------------------
    # Add a vaccination for Pet A
    vax_data = {
        "vaccine_name": "Bordetella",
        "vaccination_date": "2026-05-10",
        "next_due_date": "2027-05-10",
        "veterinarian": "Dr. Davis",
        "clinic": "Pet Care Clinic"
    }
    r1 = session_a.post(f"{BASE_URL}/pet/{pet_a_id}/vaccinations/add", data=vax_data, allow_redirects=True)
    assert r1.status_code == 200

    # Submit duplicate vaccination for Pet A
    r2 = session_a.post(f"{BASE_URL}/pet/{pet_a_id}/vaccinations/add", data=vax_data, allow_redirects=True)
    assert "already exists" in r2.text, "Duplicate vaccination should warn that it already exists"

    # Add a deworming for Pet A
    deworm_data = {
        "medicine_name": "Milbemax",
        "deworming_date": "2026-06-01",
        "next_due_date": "2026-09-01",
        "veterinarian": "Dr. Davis"
    }
    r3 = session_a.post(f"{BASE_URL}/pet/{pet_a_id}/deworming/add", data=deworm_data, allow_redirects=True)
    assert r3.status_code == 200

    # Submit duplicate deworming for Pet A
    r4 = session_a.post(f"{BASE_URL}/pet/{pet_a_id}/deworming/add", data=deworm_data, allow_redirects=True)
    assert "already exists" in r4.text, "Duplicate deworming should warn that it already exists"
    print("✓ Test 7: Duplicate vaccination and deworming records prevented with warning.")

    # ----------------------------------------------------
    # Test 8: Cascade Deletion Verification
    # ----------------------------------------------------
    with app.app_context():
        # Create a temp pet with vaccination and deworming to test cascade delete
        temp_pet = Pet(user_id=ids["user_a_id"], name="TempPet", species="dog")
        db.session.add(temp_pet)
        db.session.commit()

        temp_vax = Vaccination(pet_id=temp_pet.id, vaccine_name="TempVax", vaccination_date=date.today())
        temp_deworm = Deworming(pet_id=temp_pet.id, medicine_name="TempDeworm", deworming_date=date.today())
        db.session.add_all([temp_vax, temp_deworm])
        db.session.commit()

        temp_pet_id = temp_pet.id
        temp_vax_id = temp_vax.id
        temp_deworm_id = temp_deworm.id

        # Verify they exist
        assert Vaccination.query.get(temp_vax_id) is not None
        assert Deworming.query.get(temp_deworm_id) is not None

        # Delete the pet
        db.session.delete(temp_pet)
        db.session.commit()

        # Check cascading: child records must be deleted!
        assert Pet.query.get(temp_pet_id) is None, "Pet was not deleted"
        assert Vaccination.query.get(temp_vax_id) is None, "Vaccination should have been cascade-deleted"
        assert Deworming.query.get(temp_deworm_id) is None, "Deworming should have been cascade-deleted"

    print("✓ Test 8: Cascade deletion verified (Pet deletion cascades to Vaccination & Deworming).")

    print("\n========================================================")
    print("🎉 ALL STAGE 14 DATABASE & SECURITY TESTS PASSED! 🎉")
    print("========================================================")

if __name__ == "__main__":
    test_stage_14_security()
