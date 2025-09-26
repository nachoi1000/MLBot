import time
import logging
from pydantic import BaseModel
from typing import Optional, List


class LLMClient:
    def __init__(self, client, model: str, base_prompt: str = None, max_retries=3):
        self.client = client
        self.base_prompt = base_prompt
        self.model = model
        self.max_retries = max_retries

    def _handle_response(self, response):
        if not response.choices:
            return False
        choice = response.choices[0]
        if choice.finish_reason not in ["stop", "length"]:
            return False
        return True

    def chat_completion_structured_response(
        self,
        messages: List[dict],  # AHORA RECIBE LA LISTA DE MENSAJES
        output_format: BaseModel,
    ) -> dict:
        # El bloque para construir 'content' y 'messages' se elimina.

        attempts = 0
        while attempts < self.max_retries:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=output_format,  # Se sigue usando para forzar el formato
            )
            if self._handle_response(response=response):
                tokens_input = response.usage.prompt_tokens
                tokens_output = response.usage.completion_tokens
                logging.info(
                    f"chat_completion_structured_response: tokens_input={tokens_input} - tokens_output={tokens_output}"
                )
                return {
                    # La respuesta ya viene parseada en el formato de tu clase Pydantic
                    "answer": response.choices[0].message.parsed,
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                }
            attempts += 1
            logging.warning(
                f"Intento {attempts} fallido. Reintentando en {2**attempts} segundos..."
            )
            time.sleep(2**attempts)
        raise Exception(
            f"Falló la llamada a la API después de {self.max_retries} reintentos."
        )

    def chat_completion_response(
        self,
        messages: List[dict],  # AHORA RECIBE LA LISTA DE MENSAJES DIRECTAMENTE
    ) -> dict:
        # Ya no necesitamos construir los mensajes aquí, vienen listos.
        # prompt: str,
        # question: str,
        # image_urls: Optional[List[str]] = None,

        # El bloque para construir 'content' y 'messages' se elimina.

        attempts = 0
        while attempts < self.max_retries:
            # La llamada a la API ahora usa directamente los mensajes recibidos
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            if self._handle_response(response=response):
                tokens_input = response.usage.prompt_tokens
                tokens_output = response.usage.completion_tokens
                logging.info(
                    f"chat_completion_response: tokens_input={tokens_input} - tokens_output={tokens_output}"
                )
                return {
                    "answer": response.choices[0].message.content,
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                }
            attempts += 1
            logging.warning(
                f"Intento {attempts} fallido. Reintentando en {2**attempts} segundos..."
            )
            time.sleep(2**attempts)
        raise Exception(
            f"Falló la llamada a la API después de {self.max_retries} reintentos."
        )