from pydantic import BaseModel, Field
from typing import Optional, List


class ProductBase(BaseModel):
    asin: str = Field(..., description="Amazon product ASIN (unique ID)")
    title: str = Field(..., description="Product title")
    imgUrl: Optional[str] = Field(None, description="Product image URL")
    productURL: Optional[str] = Field(None, description="Amazon product URL")
    stars: Optional[float] = Field(0, description="Average rating")
    reviews: Optional[int] = Field(0, description="Number of reviews")
    price: Optional[float] = Field(0, description="Current price")
    listPrice: Optional[float] = Field(0, description="Original/list price")
    category_id: Optional[int] = Field(None, description="Category ID")
    category_name: Optional[str] = Field(None, description="Category name")
    isBestSeller: Optional[bool] = Field(False, description="Best seller flag")
    boughtInLastMonth: Optional[int] = Field(0, description="Units bought in last month")


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    title: Optional[str] = None
    imgUrl: Optional[str] = None
    productURL: Optional[str] = None
    stars: Optional[float] = None
    reviews: Optional[int] = None
    price: Optional[float] = None
    listPrice: Optional[float] = None
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    isBestSeller: Optional[bool] = None
    boughtInLastMonth: Optional[int] = None


class ProductOut(ProductBase):
    class Config:
        from_attributes = True


class ProductListResponse(BaseModel):
    message: str
    data: List[ProductOut] = []
    total: Optional[int] = None


class InteractionCreate(BaseModel):
    product_id: str = Field(..., description="ASIN of the product")
    event_type: str = Field(..., description="click | cart | purchase")