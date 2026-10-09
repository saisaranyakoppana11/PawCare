import requests
from bs4 import BeautifulSoup
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_stage15_ui_ux_consistency():
    session = requests.Session()
    
    # 1. Login
    login_data = {
        "email": "ssk11032006@gmail.com",
        "password": "password123"
    }
    resp = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=True)
    assert resp.status_code == 200, f"Login failed: status {resp.status_code}"
    print("[PASS] Login successful")

    # 2. Check static/js/script.js
    resp_js = session.get(f"{BASE_URL}/static/js/script.js")
    assert resp_js.status_code == 200, f"script.js not served: {resp_js.status_code}"
    assert "togglePassword" in resp_js.text
    assert "needs-loading" in resp_js.text
    print("[PASS] static/js/script.js loaded and contains loading state & password helpers")

    # 3. Test core authenticated pages for consistent navbar
    pages_to_test = [
        ("Dashboard", "/dashboard"),
        ("Vaccinations", "/pet/1/vaccinations"),
        ("Deworming", "/pet/1/deworming"),
        ("Notifications", "/notifications"),
        ("Lost & Found", "/lost-found"),
        ("Emergency", "/emergency"),
        ("AI Analysis", "/ai-analysis"),
        ("Health Assistant", "/health-assistant"),
        ("Pet Profile", "/pet/profile/1"),
        ("My Reports", "/lost-found/my-reports"),
    ]

    expected_nav_items = [
        "Dashboard",
        "Vaccinations",
        "Deworming",
        "Notifications",
        "Lost & Found",
        "Emergency",
        "AI Analysis",
        "Health Assistant",
        "Logout",
    ]

    for page_name, route in pages_to_test:
        r = session.get(f"{BASE_URL}{route}")
        assert r.status_code == 200, f"Failed to load {page_name} at {route}: status {r.status_code}"
        
        soup = BeautifulSoup(r.text, "html.parser")
        navbar = soup.find("nav", class_="navbar")
        assert navbar is not None, f"Navbar missing on {page_name}"
        
        nav_text = navbar.get_text()
        for item in expected_nav_items:
            assert item in nav_text, f"Navbar item '{item}' missing on {page_name}"
        
        # Verify script.js is loaded
        scripts = [s.get("src") for s in soup.find_all("script") if s.get("src")]
        script_js_found = any("script.js" in s for s in scripts)
        assert script_js_found, f"script.js not referenced on {page_name}"

        # Verify flash container or alert-dismissible format
        print(f"[PASS] {page_name} ({route}) has consistent navbar and script.js")

    # 4. Verify empty state structures exist
    # Let's inspect empty states in templates
    with open("templates/vaccinations.html", "r", encoding="utf-8") as f:
        vac_html = f.read()
        assert "No Vaccination Records Yet" in vac_html
        assert "Add First Vaccination" in vac_html
    print("[PASS] Vaccinations empty state verified")

    with open("templates/deworming.html", "r", encoding="utf-8") as f:
        dew_html = f.read()
        assert "No Deworming Records Yet" in dew_html
        assert "Add First Deworming Record" in dew_html
    print("[PASS] Deworming empty state verified")

    with open("templates/notifications.html", "r", encoding="utf-8") as f:
        notif_html = f.read()
        assert "All Caught Up!" in notif_html
    print("[PASS] Notifications empty state verified")

    with open("templates/dashboard.html", "r", encoding="utf-8") as f:
        dash_html = f.read()
        assert "No Pets Added Yet" in dash_html
    print("[PASS] Dashboard empty state verified")

    with open("templates/lost_found.html", "r", encoding="utf-8") as f:
        lf_html = f.read()
        assert "No Reports Match Your Search" in lf_html
    print("[PASS] Lost & Found empty state verified")

    print("\n==========================================")
    print("ALL STAGE 15 UI/UX CONSISTENCY TESTS PASSED!")
    print("==========================================")

if __name__ == "__main__":
    test_stage15_ui_ux_consistency()
