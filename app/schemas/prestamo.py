from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class PrestamoEntrada(BaseModel):
    """Datos de una solicitud de préstamo (entrada del modelo y parámetros de la tool)."""

    # extra="forbid": si llega un campo que no existe (p. ej. inventado por el LLM).
    model_config = ConfigDict(extra="forbid")

    edad: int = Field(ge=18, le=75, description="Edad del solicitante en años")
    ingresos_mensuales: float = Field(gt=0.0, description="Ingresos netos mensuales en euros")
    importe_prestamo: float = Field(gt=0.0, description="Importe solicitado en euros")
    plazo_meses: int = Field(ge=6, le=84, description="Plazo de devolución en meses")
    antiguedad_laboral_anios: int = Field(
        ge=0, le=50, description="Años en el empleo actual (como máximo edad - 16)"
    )
    impagos_previos: int = Field(ge=0, le=20, description="Número de impagos anteriores")

    @field_validator("antiguedad_laboral_anios")
    @classmethod
    def antiguedad_coherente_con_edad(cls, valor: int, info: ValidationInfo) -> int:
        edad = info.data.get("edad")
        if edad is not None and valor > edad - 16:
            raise ValueError(
                f"la antigüedad laboral ({valor}) no puede ser mayor que edad - 16 ({edad - 16})"
            )
        return valor


class PrediccionSalida(BaseModel):
    """Resultado del modelo de impago."""

    probabilidad_impago: float = Field(ge=0.0, le=1.0)
    decision: Literal["aprobar", "revisar", "rechazar"]
