from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from src.services.supabase_client import supabase
from src.services.clerkAuth import get_current_user_clerk_id
from src.models.product import InteractionCreate

router = APIRouter()


# ----------------------------------------------------------------------
# GET all products
# ----------------------------------------------------------------------
@router.get("/")
async def get_products(
    categories: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    category_id: Optional[int] = None,
    search: Optional[str] = None,
):
    try:
        offset = (page - 1) * limit
        query = supabase.table("products").select("*", count="exact")

        if categories:
            query = query.in_("category_name", [c.strip() for c in categories.split(",")])
        if category_id:
            query = query.eq("category_id", category_id)
        if search:
            query = query.ilike("title", f"%{search}%")

        response = query.range(offset, offset + limit - 1).execute()

        return {
            "message": "Products retrieved successfully",
            "data": response.data or [],
            "total": response.count
        }

    except Exception as e:
        raise HTTPException(500, str(e))


# ----------------------------------------------------------------------
# GET recommendations  ← before /{asin}
# ----------------------------------------------------------------------
@router.get("/recommendations/me")
async def get_recommendations(
    limit: int = Query(10, ge=1, le=50),
):
    try:
        response = (
            supabase.table("products")
            .select("*")
            .order("boughtInLastMonth", desc=True)
            .limit(limit)
            .execute()
        )
        return {"message": "Recommendations retrieved", "data": response.data}

    except Exception as e:
        raise HTTPException(500, str(e))


# ----------------------------------------------------------------------
# GET products by category
# ----------------------------------------------------------------------
@router.get("/category/{category_id}")
async def get_products_by_category(
    category_id: int,
    page: int = 1,
    limit: int = 20
):
    try:
        offset = (page - 1) * limit

        response = (
            supabase.table("products")
            .select("*", count="exact")
            .eq("category_id", category_id)
            .range(offset, offset + limit - 1)
            .execute()
        )

        return {
            "message": "Products retrieved successfully",
            "data": response.data,
            "total": response.count
        }

    except Exception as e:
        raise HTTPException(500, str(e))
    raise HTTPException(404, "Product not found")


# ----------------------------------------------------------------------
# GET single product  ← ALWAYS LAST
# ----------------------------------------------------------------------
@router.get("/{asin}")
async def get_product(asin: str):
    try:
        response = (
            supabase.table("products")
            .select("*")
            .eq("asin", asin)
            .execute()
        )

        if not response.data:
            raise HTTPException(404, "Product not found")

        return {"message": "Product retrieved successfully", "data": response.data[0]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))