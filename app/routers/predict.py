from fastapi import APIRouter

from app.schemas.prestamo import PrediccionSalida, PrestamoEntrada
from app.services.modelo import predecir

router = APIRouter()


@router.post("/predict", response_model=PrediccionSalida)
def predict(entrada: PrestamoEntrada) -> PrediccionSalida:
    """Predicción directa con el modelo, sin LLM. Datos inválidos -> 422 (lo hace FastAPI)."""
    return predecir(entrada)
