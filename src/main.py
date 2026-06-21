from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.routes.product_router import router as product_router
from src.routes.interaction_router import router as interaction_router
from src.routes.recommendations_router import router as recommend_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.vercel.app",  # ← add your deployed frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

app.include_router(recommend_router, prefix="/api/recommend", tags=["Recommendations"])
app.include_router(interaction_router, prefix="/api/products")
app.include_router(product_router, prefix="/api/products")

@app.get("/")
def root():
    return {"status": "ok"}