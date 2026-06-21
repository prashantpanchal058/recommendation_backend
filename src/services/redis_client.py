import redis
import os

# r = redis.Redis(
#     host=os.getenv("REDIS_HOST", "localhost"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=None,   # ← hardcode None, ignore env var
#     decode_responses=True
# )

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)

# Test connection on import
try:
    r.ping()
    print("✅ Redis connected successfully")
except redis.ConnectionError as e:
    print(f"❌ Redis connection failed: {e}")