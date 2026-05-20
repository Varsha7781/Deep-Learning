"""Quick smoke test for all auth endpoints."""
import requests

BASE = "http://localhost:8000"

def sep(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

# 1. Health Check
sep("1. Health Check - GET /")
r = requests.get(f"{BASE}/")
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

# 2. Register
sep("2. Register - POST /register")
r = requests.post(f"{BASE}/register", json={
    "name": "Varsha Test",
    "email": "varsha@test.com",
    "password": "SecurePass1",
    "confirm_password": "SecurePass1"
})
print(f"Status: {r.status_code}")
data = r.json()
print(f"Response: {data}")
token = data.get("access_token", "")

# 2b. Duplicate registration
sep("2b. Duplicate Register (should fail 409)")
r = requests.post(f"{BASE}/register", json={
    "name": "Varsha Test",
    "email": "varsha@test.com",
    "password": "SecurePass1",
    "confirm_password": "SecurePass1"
})
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

# 3. Login
sep("3. Login - POST /login")
r = requests.post(f"{BASE}/login", json={
    "email": "varsha@test.com",
    "password": "SecurePass1"
})
print(f"Status: {r.status_code}")
data = r.json()
print(f"Response: {data}")
token = data.get("access_token", token)

# 3b. Bad login
sep("3b. Login with wrong password (should fail 401)")
r = requests.post(f"{BASE}/login", json={
    "email": "varsha@test.com",
    "password": "WrongPassword1"
})
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

# 4. Get Current User (protected)
sep("4. Get Current User - GET /me")
r = requests.get(f"{BASE}/me", headers={"Authorization": f"Bearer {token}"})
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

# 5. Session History (protected)
sep("5. Session History - GET /sessions")
r = requests.get(f"{BASE}/sessions", headers={"Authorization": f"Bearer {token}"})
print(f"Status: {r.status_code}")
sessions = r.json()
print(f"Sessions count: {len(sessions)}")
for s in sessions:
    print(f"  - {s['id'][:8]}... active={s['is_active']}  ip={s.get('ip_address')}")

# 6. Forgot Password
sep("6. Forgot Password - POST /forgot-password")
r = requests.post(f"{BASE}/forgot-password", json={"email": "varsha@test.com"})
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

# 7. Logout
sep("7. Logout - POST /logout")
r = requests.post(f"{BASE}/logout", headers={"Authorization": f"Bearer {token}"})
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

# 7b. Access after logout (should fail)
sep("7b. Access after logout (should fail 401)")
r = requests.get(f"{BASE}/me", headers={"Authorization": f"Bearer {token}"})
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

sep("ALL TESTS COMPLETE")
print()
