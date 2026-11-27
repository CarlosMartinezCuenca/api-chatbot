import json

from pydantic import ValidationError

from app.schemas.chat import ChatSalida, HerramientaUsada
from app.schemas.prestamo import PrestamoEntrada
from app.services.modelo import predecir

NOMBRE_TOOL = "calcular_riesgo_impago"
MAX_VUELTAS = 4

SISTEMA = (
    "Eres un asistente de riesgo de un banco. Cuando tengas todos los datos de una solicitud de "
    "prestamo, llama a la herramienta calcular_riesgo_impago. Si falta algun dato, pidelo. "
    "Nunca inventes valores. Responde en espanol y de forma breve."
)


def leer_argumentos(texto):
    """YA HECHO: convierte el texto JSON de la tool en dict ({} si no es JSON valido)."""
    try:
        datos = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(datos, dict):
        return datos
    return {}


MENSAJE_SIN_RESPUESTA = (
    "No he podido completar el cálculo con los datos recibidos. "
    "¿Puedes revisar los datos de la solicitud y volver a intentarlo?"
)


def definir_tool():
    """Definición de la tool en formato OpenAI. Los parámetros salen de PrestamoEntrada,
    así el contrato (campos, tipos, rangos y extra="forbid") está en un único sitio."""
    return {
        "type": "function",
        "function": {
            "name": NOMBRE_TOOL,
            "description": (
                "Calcula la probabilidad de impago de una solicitud de préstamo y devuelve "
                "la decisión (aprobar, revisar o rechazar). Requiere los 6 datos del solicitante."
            ),
            "parameters": PrestamoEntrada.model_json_schema(),
        },
    }


def formatear_errores(error: ValidationError) -> str:
    """Convierte los errores de Pydantic en un texto corto que el LLM pueda entender."""
    partes = []
    for e in error.errors():
        campo = ".".join(str(x) for x in e["loc"]) or "datos"
        partes.append(f"{campo}: {e['msg']}")
    return "; ".join(partes)


def ejecutar_tool(nombre, argumentos_json):
    """Valida los argumentos y llama al modelo. Nunca lanza: devuelve siempre un dict con "ok"."""
    if nombre != NOMBRE_TOOL:
        return {"ok": False, "error": f"herramienta desconocida: {nombre}"}

    try:
        argumentos = json.loads(argumentos_json)
    except (json.JSONDecodeError, TypeError):
        return {"ok": False, "error": "los argumentos no son un JSON válido"}
    if not isinstance(argumentos, dict):
        return {"ok": False, "error": "los argumentos deben ser un objeto JSON"}

    try:
        entrada = PrestamoEntrada(**argumentos)
    except ValidationError as e:
        return {"ok": False, "error": formatear_errores(e)}

    try:
        salida = predecir(entrada)
    except Exception as e:  # el modelo no debe tumbar el chat
        return {"ok": False, "error": f"error al calcular el riesgo: {e}"}

    return {"ok": True, **salida.model_dump()}


def responder(mensajes, llm):
    """Bucle LLM -> tool -> LLM hasta que el LLM conteste con texto (máximo MAX_VUELTAS)."""
    historial = [{"role": "system", "content": SISTEMA}]
    historial += [m.model_dump() for m in mensajes]
    herramientas = [definir_tool()]
    herramienta_usada = None

    for _ in range(MAX_VUELTAS):
        respuesta = llm.completar(historial, herramientas)
        llamadas = respuesta.get("tool_calls") or []

        if not llamadas:
            texto = respuesta.get("content") or MENSAJE_SIN_RESPUESTA
            return ChatSalida(respuesta=texto, herramienta=herramienta_usada)

        # El mensaje del asistente con las tool_calls tiene que ir antes que sus resultados
        historial.append(respuesta)
        for llamada in llamadas:
            nombre = llamada["function"]["name"]
            argumentos_json = llamada["function"]["arguments"]
            resultado = ejecutar_tool(nombre, argumentos_json)
            historial.append(
                {
                    "role": "tool",
                    "tool_call_id": llamada["id"],
                    "content": json.dumps(resultado, ensure_ascii=False),
                }
            )
            herramienta_usada = HerramientaUsada(
                nombre=nombre,
                argumentos=leer_argumentos(argumentos_json),
                resultado=resultado,
            )

    # Se agotaron las vueltas sin respuesta de texto: cortamos para no entrar en bucle
    return ChatSalida(respuesta=MENSAJE_SIN_RESPUESTA, herramienta=herramienta_usada)
