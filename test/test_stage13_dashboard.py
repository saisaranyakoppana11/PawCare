import requests
from bs4 import BeautifulSoup

def run_tests():
    session = requests.Session()

    # 1. Login
    r_login = session.post('http://127.0.0.1:5000/login', data={
        'email': 'ssk11032006@gmail.com',
        'password': 'password123'
    }, allow_redirects=True)
    assert r_login.status_code == 200, f'Login failed with {r_login.status_code}'
    print('1. Logged in successfully.')

    # 2. GET /dashboard
    r_dash = session.get('http://127.0.0.1:5000/dashboard')
    assert r_dash.status_code == 200, f'Dashboard failed with {r_dash.status_code}'
    soup = BeautifulSoup(r_dash.text, 'html.parser')

    # Check that NO fake alert buttons exist
    alerts = [b for b in soup.find_all('button') if 'alert(' in (b.get('onclick') or '')]
    assert len(alerts) == 0, f'Found {len(alerts)} fake alert buttons!'
    print('2. Verified NO fake alert buttons exist on Dashboard.')

    # 3. Check presence of all 8 features in Feature Cards
    feature_section = soup.find('section', id='features')
    assert feature_section is not None, 'Features section #features not found!'

    feature_cards = feature_section.find_all('div', class_='card')
    print(f'3. Found {len(feature_cards)} feature cards in Features section.')
    assert len(feature_cards) >= 8, f'Expected at least 8 feature cards, got {len(feature_cards)}'

    card_titles = [c.find('h5').get_text(strip=True) for c in feature_cards if c.find('h5')]
    print('   Card Titles:', card_titles)

    required_features = [
        'My Pets',
        'AI Pet Analysis',
        'Health Assistant',
        'Emergency Guidance',
        'Vaccinations',
        'Deworming',
        'Notifications',
        'Lost & Found'
    ]
    for rf in required_features:
        matched = any(rf.lower() in ct.lower() for ct in card_titles)
        assert matched, f'Required feature card "{rf}" not found in dashboard cards: {card_titles}'
    print('4. All 8 required feature cards are present.')

    # 4. Check Navigation Bar links
    nav = soup.find('nav', class_='navbar')
    assert nav is not None, 'Navbar not found!'
    nav_links = [a.get('href') for a in nav.find_all('a')]
    print('   Nav link hrefs:', nav_links)

    assert any('#pets' in href for href in nav_links), 'Navbar missing My Pets link'
    assert any('/vaccinations' in href for href in nav_links), 'Navbar missing Vaccinations link'
    assert any('/deworming' in href for href in nav_links), 'Navbar missing Deworming link'
    assert any('/notifications' in href for href in nav_links), 'Navbar missing Notifications link'
    assert any('/lost-found' in href for href in nav_links), 'Navbar missing Lost & Found link'
    assert any('/emergency' in href for href in nav_links), 'Navbar missing Emergency link'
    assert any('/ai-analysis' in href or '/ai-prediction' in href for href in nav_links), 'Navbar missing AI link'
    assert any('/health' in href for href in nav_links), 'Navbar missing Health Assistant link'
    print('5. Navbar verified with all required feature links.')

    # 5. Check pet card quick action buttons on dashboard
    pets_section = soup.find('section', id='pets')
    assert pets_section is not None, 'Pets section not found!'
    pet_links = [a.get('href') for a in pets_section.find_all('a')]
    print('   Pet card action links:', pet_links)
    assert any('/pet/1' in href for href in pet_links), 'Pet profile link missing'
    assert any('/pet/1/vaccinations' in href for href in pet_links), 'Pet vaccinations link missing'
    assert any('/pet/1/deworming' in href for href in pet_links), 'Pet deworming link missing'
    assert any('/health' in href for href in pet_links), 'Pet health assistant link missing'
    print('6. Pet card quick action buttons verified.')

    # 6. Test each route returns 200 OK
    routes_to_test = [
        ('/vaccinations', 200),
        ('/deworming', 200),
        ('/pet/1/vaccinations', 200),
        ('/pet/1/deworming', 200),
        ('/notifications', 200),
        ('/emergency', 200),
        ('/ai-analysis', 200),
        ('/health-assistant', 200),
        ('/lost-found', 200),
        ('/pet/1', 200)
    ]

    for route, expected_status in routes_to_test:
        res = session.get(f'http://127.0.0.1:5000{route}', allow_redirects=True)
        assert res.status_code == expected_status, f'Route {route} returned {res.status_code}, expected {expected_status}'
        print(f'   Route {route:25} -> {res.status_code} OK (Final URL: {res.url})')

    print('\nALL STAGE 13 VERIFICATION TESTS PASSED SUCCESSFULLY!')

if __name__ == '__main__':
    run_tests()
