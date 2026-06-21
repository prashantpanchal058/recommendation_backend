from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from src.services.supabase_client import supabase
from src.services.clerkAuth import get_current_user_clerk_id
from src.models.product import InteractionCreate

router = APIRouter()

@router.get("/cart")
async def get_cart(
    current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        cart = (
            supabase.table("interactions")
            .select("product_id")
            .eq("user_id", current_user_clerk_id)
            .eq("event_type", "cart")
            .execute()
        )

        if not cart.data:
            return {"message": "Cart is empty", "data": []}

        # Deduplicate ASINs
        asins = list({row["product_id"] for row in cart.data})

        products = (
            supabase.table("products")
            .select("*")
            .in_("asin", asins)
            .execute()
        )

        return {"message": "Cart retrieved successfully", "data": products.data or []}

    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/cart")
async def add_to_cart(
    body: dict,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        product_id = body.get("product_id")

        if not product_id:
            raise HTTPException(
                status_code=400,
                detail="product_id is required"
            )

        # Verify product exists
        product = (
            supabase.table("products")
            .select("asin")
            .eq("asin", product_id)
            .execute()
        )

        if not product.data:
            raise HTTPException(
                status_code=404,
                detail="Product not found"
            )

        # Check if already in cart
        existing = (
            supabase.table("interactions")
            .select("*")
            .eq("user_id", current_user_clerk_id)
            .eq("product_id", product_id)
            .eq("event_type", "cart")
            .execute()
        )

        if existing.data:
            raise HTTPException(
                status_code=409,
                detail="Product already in cart"
            )

        response = (
            supabase.table("interactions")
            .insert({
                "user_id": current_user_clerk_id,
                "product_id": product_id,
                "event_type": "cart"
            })
            .execute()
        )

        return {
            "message": "Added to cart",
            "data": response.data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ----------------------------------------------------------------------
@router.delete("/cart/{asin}")
async def remove_from_cart(
    asin: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        exists = (
            supabase.table("interactions")
            .select("id")
            .eq("user_id", current_user_clerk_id)
            .eq("product_id", asin)
            .eq("event_type", "cart")
            .execute()
        )
        if not exists.data:
            raise HTTPException(404, "Product not in cart")

        response = (
            supabase.table("interactions")
            .delete()
            .eq("user_id", current_user_clerk_id)
            .eq("product_id", asin)
            .eq("event_type", "cart")
            .execute()
        )

        return {"message": "Removed from cart", "data": response.data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@router.delete("/cart")
async def clear_cart(
    current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        response = (
            supabase.table("interactions")
            .delete()
            .eq("user_id", current_user_clerk_id)
            .eq("event_type", "cart")
            .execute()
        )

        return {"message": "Cart cleared", "data": response.data}

    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/{asin}/interact")
async def record_interaction(
    asin: str,
    interaction: InteractionCreate,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        if interaction.event_type not in ("click", "cart", "purchase"):
            raise HTTPException(400, "Invalid event_type")

        # Check product exists
        # product = (
        #     supabase.table("products")
        #     .select("asin")
        #     .eq("asin", asin)
        #     .execute()
        # )

        # if not product.data:
        #     raise HTTPException(404, "Product not found")

        # Check existing interaction
        existing = (
            supabase.table("interactions")
            .select("*")
            .eq("user_id", current_user_clerk_id)
            .eq("product_id", asin)
            .execute()
        )

        if existing.data:
            current = existing.data[0]

            # Same interaction already exists
            if current["event_type"] == interaction.event_type:
                return {
                    "message": "Interaction already recorded",
                    "data": current
                }

            # Update interaction type
            updated = (
                supabase.table("interactions")
                .update({"event_type": interaction.event_type})
                .eq("id", current["id"])  # assuming id is PK
                .execute()
            )

            return {
                "message": "Interaction updated",
                "data": updated.data
            }

        # Insert new interaction
        inserted = (
            supabase.table("interactions")
            .insert({
                "user_id": current_user_clerk_id,
                "product_id": asin,
                "event_type": interaction.event_type,
            })
            .execute()
        )

        return {
            "message": "Interaction recorded",
            "data": inserted.data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))