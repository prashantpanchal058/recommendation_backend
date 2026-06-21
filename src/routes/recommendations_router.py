import json
from fastapi import APIRouter, Depends, HTTPException, Query
from src.services.supabase_client import supabase
from src.services.clerkAuth import get_current_user_clerk_id
from src.services.loader import get_artifacts
from src.services.recommender import recommend_by_asin, recommend_for_user, _format_products
from src.services.redis_client import r
from src.models.product import InteractionCreate

router = APIRouter()

ASIN_TTL    = 60 * 60   # 1 hour  — product recs stable
USER_TTL    = 60 * 3    # 5 mins  — user history changes

@router.get("/user")
async def recommend_for_user_route(
    top_n: int = Query(default=25, le=50),
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
    artifacts: dict = Depends(get_artifacts)
):
    try:
        cache_key = f"rec:user:{current_user_clerk_id}"
        cached = r.get(cache_key)
        if cached:
            return {"message": "User recommendations fetched", "source": "cache", "data": json.loads(cached)}

        events = (
            supabase.table("interactions")
            .select("product_id")
            .eq("user_id", current_user_clerk_id)
            .in_("event_type", ["cart", "click"])
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )

        viewed_asins = (
    [row["product_id"] for row in events.data]
    if events.data
    else ['B000GAWSDG', 'B077G7ZVCJ', 'B07BKLBQ47']
)
        recs = recommend_for_user(viewed_asins=viewed_asins, top_n=top_n, artifacts=artifacts)

        r.setex(cache_key, USER_TTL, json.dumps(recs))

        source = "history" if viewed_asins else "popular"

        return {
            "message":  "User recommendations fetched",
            "user_id":  current_user_clerk_id,
            "based_on": viewed_asins,
            "source":   source,
            "data":     recs
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/click")
async def recommend_for_click(
    top_n: int = Query(default=10, le=50),
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
    artifacts: dict = Depends(get_artifacts)
):
    try:
        cache_key = f"rec:click:{current_user_clerk_id}"
        cached = r.get(cache_key)
        if cached:
            return {"message": "User recommendations fetched", "source": "cache", "data": json.loads(cached)}

        events = (
            supabase.table("interactions")
            .select("product_id")
            .eq("user_id", current_user_clerk_id)
            .eq("event_type", "click")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        viewed_asins = [row["product_id"] for row in (events.data or [])]

        if not viewed_asins:
            return {"message": "No history found", "source": "popular", "data": []}

        recs = recommend_by_asin(asin=viewed_asins[0], top_n=top_n, artifacts=artifacts)

        r.setex(cache_key, USER_TTL, json.dumps(recs))

        return {
            "message":  "User recommendations fetched",
            "user_id":  current_user_clerk_id,
            "based_on": viewed_asins,
            "source":   "history",
            "data":     recs
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/added_card")
async def recommend_for_added_card(
    top_n: int = Query(default=10, le=50),
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
    artifacts: dict = Depends(get_artifacts)
):
    try:
        cache_key = f"rec:cart:{current_user_clerk_id}"
        cached = r.get(cache_key)
        if cached:
            return {"message": "User recommendations fetched", "source": "cache", "recommendations": json.loads(cached)}

        events = (
            supabase.table("interactions")
            .select("product_id")
            .eq("user_id", current_user_clerk_id)
            .eq("event_type", "cart")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        cart_asins = [row["product_id"] for row in (events.data or [])]

        if not cart_asins:
            return {"message": "No history found", "source": "popular", "data": []}

        recs = recommend_by_asin(asin=cart_asins[0], top_n=top_n, artifacts=artifacts)

        r.setex(cache_key, USER_TTL, json.dumps(recs))

        return {
            "message":        "User recommendations fetched",
            "user_id":        current_user_clerk_id,
            "based_on":       cart_asins,
            "source":         "history",
            "recommendations": recs
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/popular")
async def popular_by_category(
    categories: str = Query(..., description="Comma-separated category names"),
    top_n: int = Query(default=10, le=50),
    artifacts: dict = Depends(get_artifacts)
):
    try:
        cache_key = f"rec:popular:{categories}:{top_n}"
        cached = r.get(cache_key)
        if cached:
            return {"message": "Popular products fetched", "source": "cache", "data": json.loads(cached)}

        df = artifacts["df"]
        category_list = [c.strip() for c in categories.split(",") if c.strip()]

        filtered = df[df["category_name"].str.lower().isin([c.lower() for c in category_list])]

        if filtered.empty:
            raise HTTPException(404, "No products found for given categories")

        top  = filtered.sort_values("popularity_score", ascending=False).head(top_n)
        data = _format_products(top)

        r.setex(cache_key, ASIN_TTL, json.dumps(data))

        return {"message": "Popular products fetched", "categories": category_list, "data": data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── POST /event ───────────────────────────────────────────────────────
@router.post("/event")
async def log_event(
    body: InteractionCreate,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        result = (
            supabase.table("interactions")
            .insert({
                "user_id":    current_user_clerk_id,
                "product_id": body.product_id,
                "event_type": body.event_type,
            })
            .execute()
        )

        # Bust all user-specific caches on every new interaction
        r.delete(f"rec:user:{current_user_clerk_id}")
        r.delete(f"rec:click:{current_user_clerk_id}")
        r.delete(f"rec:cart:{current_user_clerk_id}")

        return {"message": "Event logged", "data": result.data}

    except Exception as e:
        raise HTTPException(500, str(e))


# ── GET /{asin} ───────────────────────────────────────────────────────
@router.get("/{asin}")
async def recommend_similar(
    asin: str,
    top_n: int = Query(default=10, le=50),
    artifacts: dict = Depends(get_artifacts)
):
    try:
        cache_key = f"rec:asin:{asin}"
        cached = r.get(cache_key)
        if cached:
            return {"message": "Recommendations fetched", "asin": asin, "source": "cache", "data": json.loads(cached)}

        recs = recommend_by_asin(asin=asin, top_n=top_n, artifacts=artifacts)

        if not recs:
            raise HTTPException(404, f"ASIN '{asin}' not found in model")

        r.setex(cache_key, ASIN_TTL, json.dumps(recs))

        return {"message": "Recommendations fetched", "asin": asin, "source": "model", "data": recs}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))