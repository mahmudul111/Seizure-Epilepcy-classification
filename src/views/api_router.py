from fastapi import APIRouter
from .view import router_v1

router = APIRouter()
router.include_router(router_v1, prefix="/api/v1")

