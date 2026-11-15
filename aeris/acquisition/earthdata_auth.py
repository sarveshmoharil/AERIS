
import earthaccess
import logging

logging.basicConfig(level=logging.INFO)

print("=" * 45)
print("AERIS | NASA EARTHDATA AUTHENTICATION")
print("=" * 45)

try:
    auth = earthaccess.login(
        strategy="interactive",
        persist=True
    )

    if auth.authenticated:
        print("\nSUCCESS: NASA Earthdata authenticated!")
        print("AERIS can now access authenticated NASA datasets.")
    else:
        print("\nAuthentication unsuccessful.")

except Exception as error:
    print(f"\nAuthentication error: {error}")

