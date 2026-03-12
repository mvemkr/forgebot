"""
Schwab re-auth with auto-capture. Starts a local HTTPS server, prints the
auth URL, waits for Schwab to redirect back, exchanges the code immediately.

One-time setup: add https://127.0.0.1:8182 as a callback URL in the Schwab
Developer Portal (My Apps → Edit → Callback URLs — you can have multiple).

Usage:  python scripts/schwab_reauth.py
"""
import os, sys, ssl, json, base64, threading, time, secrets, urllib.parse, requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

APP_KEY      = os.getenv("SCHWAB_APP_KEY")
APP_SECRET   = os.getenv("SCHWAB_APP_SECRET")
CALLBACK_URL = "https://127.0.0.1:8182"
TOKEN_PATH   = Path(os.path.dirname(os.path.dirname(__file__))) / ".schwab_token.json"
TOKEN_URL    = "https://api.schwabapi.com/v1/oauth/token"
AUTH_URL     = "https://api.schwabapi.com/v1/oauth/authorize"
PORT         = 8182

captured = {}   # shared between server thread and main thread

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        code   = params.get("code", [None])[0]
        error  = params.get("error", [None])[0]

        if code:
            captured["code"] = code
            body = b"<h2>Authorized! You can close this tab.</h2>"
        elif error:
            captured["error"] = error
            body = f"<h2>Error: {error}</h2>".encode()
        else:
            body = b"<h2>Waiting...</h2>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass   # silence request logs


def make_self_signed_cert():
    """Generate a temporary self-signed cert for the local HTTPS server."""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime, tempfile

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"127.0.0.1")])
        cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(hours=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(__import__('ipaddress').ip_address('127.0.0.1'))]), critical=False)
            .sign(key, hashes.SHA256()))

        tmp = tempfile.mkdtemp()
        cert_path = os.path.join(tmp, "cert.pem")
        key_path  = os.path.join(tmp, "key.pem")
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(key_path, "wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption()))
        return cert_path, key_path
    except ImportError:
        return None, None


def exchange_code(code):
    creds = base64.b64encode(f"{APP_KEY}:{APP_SECRET}".encode()).decode()
    resp  = requests.post(TOKEN_URL,
        headers={"Authorization": f"Basic {creds}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "authorization_code", "code": code,
              "redirect_uri": CALLBACK_URL},
        timeout=15)
    return resp


if __name__ == "__main__":
    state  = secrets.token_urlsafe(22)
    params = {"response_type": "code", "client_id": APP_KEY,
              "redirect_uri": CALLBACK_URL, "state": state}
    url    = AUTH_URL + "?" + urllib.parse.urlencode(params)

    print("\n" + "="*60)
    print("SCHWAB RE-AUTH")
    print("="*60)
    print(f"\n1. Open this URL in your browser:\n\n   {url}\n")
    print("2. Log in and approve.")
    print("3. The browser will redirect to 127.0.0.1 — that's this server.")
    print("   Token will be exchanged automatically. No copy-paste needed.")
    print("\nWaiting for callback...\n")

    # Start HTTPS server
    server  = HTTPServer(("127.0.0.1", PORT), CallbackHandler)
    cert, key = make_self_signed_cert()
    if cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
    else:
        print("⚠️  cryptography not installed — running HTTP (may not work with https:// callback)")

    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    # Wait up to 3 minutes
    for _ in range(180):
        if "code" in captured or "error" in captured:
            break
        time.sleep(1)
    server.server_close()

    if "error" in captured:
        print(f"❌ Auth error: {captured['error']}")
        sys.exit(1)

    code = captured.get("code")
    if not code:
        print("❌ Timed out waiting for callback.")
        sys.exit(1)

    print(f"✅ Code captured! Exchanging...")
    resp = exchange_code(code)

    if resp.status_code != 200:
        print(f"❌ Exchange failed {resp.status_code}: {resp.text}")
        sys.exit(1)

    token = resp.json()
    TOKEN_PATH.write_text(json.dumps(token, indent=2))
    TOKEN_PATH.chmod(0o600)
    print(f"✅ Token saved! Expires in {token.get('expires_in')}s")

    # Verify
    verify = requests.get("https://api.schwabapi.com/trader/v1/accounts/accountNumbers",
        headers={"Authorization": f"Bearer {token['access_token']}"})
    if verify.status_code == 200:
        accounts = verify.json()
        print(f"✅ Connected — {len(accounts)} account(s):")
        for a in accounts:
            print(f"   {a.get('accountNumber')}  hash={a.get('hashValue')}")
        token['_accounts'] = accounts
        TOKEN_PATH.write_text(json.dumps(token, indent=2))
    else:
        print(f"⚠️  Verify: {verify.status_code}")

    print("\nDone. You can restart the orchestrators now.")
