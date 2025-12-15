import json
import requests
import numpy as np
from typing import List, Dict, Optional, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClient:
    """Client để gọi API LLM và Embedding"""
    
    def __init__(self, api_config_path: str = 'api-keys.json'):
        with open(api_config_path, 'r') as f:
            configs_list = json.load(f)
        
        # Parse array format from api-keys.json
        self.llm_large = None
        self.llm_small = None
        self.embedding = None
        
        for config in configs_list:
            api_name = config.get('llmApiName', '').lower()
            if 'large' in api_name:
                self.llm_large = config
            elif 'small' in api_name:
                self.llm_small = config
            elif 'embed' in api_name:  # Match both "embedding" and "embedings"
                self.embedding = config
        
        # Verify all configs loaded
        if not all([self.llm_large, self.llm_small, self.embedding]):
            logger.error(f"llm_large: {self.llm_large is not None}")
            logger.error(f"llm_small: {self.llm_small is not None}")
            logger.error(f"embedding: {self.embedding is not None}")
            raise ValueError("Missing API configurations in api-keys.json")
        
        logger.info("API Client initialized")

    def call_chat(
        self,
        messages: List[Dict[str, str]],
        use_large: bool = True,
        max_tokens: int = 500,
        temperature: float = 0.1,
        top_p: float = 1.0,
        top_k: int = 20,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[Dict] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        timeout_s: int = 60,
    ) -> Dict[str, Any]:
        """
        Low-level chat API, trả về raw JSON response (để tận dụng n/logprobs/tools).
        """
        config = self.llm_large if use_large else self.llm_small

        if use_large:
            url = "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large"
            model = "vnptai_hackathon_large"
        else:
            url = "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small"
            model = "vnptai_hackathon_small"

        json_data: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n": int(max(1, n)),
            "max_completion_tokens": int(max_tokens),
        }
        if stop is not None:
            json_data["stop"] = stop
        if presence_penalty is not None:
            json_data["presence_penalty"] = float(presence_penalty)
        if frequency_penalty is not None:
            json_data["frequency_penalty"] = float(frequency_penalty)
        if response_format is not None:
            json_data["response_format"] = response_format
        if seed is not None:
            json_data["seed"] = int(seed)
        if logprobs is not None:
            json_data["logprobs"] = bool(logprobs)
        if top_logprobs is not None:
            json_data["top_logprobs"] = int(top_logprobs)
        if tools is not None:
            json_data["tools"] = tools
        if tool_choice is not None:
            json_data["tool_choice"] = tool_choice

        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": config["authorization"],
                    "Token-id": config["tokenId"],
                    "Token-key": config["tokenKey"],
                    "Content-Type": "application/json",
                },
                json=json_data,
                timeout=timeout_s,
            )

            # VNPT đôi khi trả HTTP 200 nhưng payload có error
            result: Any = {}
            try:
                result = response.json()
            except Exception:
                logger.error(f"LLM API non-JSON response: {response.status_code} - {response.text[:300]}")
                return {"__error__": {"http_status": response.status_code, "type": "NonJSON"}}

            if not isinstance(result, dict):
                logger.error(f"LLM API unexpected JSON type: {type(result).__name__}")
                return {
                    "__error__": {
                        "http_status": response.status_code,
                        "type": "NonDictJSON",
                        "payload_type": type(result).__name__,
                    }
                }

            if response.status_code == 200:
                if isinstance(result, dict) and isinstance(result.get("error"), dict):
                    err = result.get("error") or {}
                    logger.error(
                        f"LLM API error (200 payload): {err.get('type')} - {err.get('code')} - {err.get('message')}"
                    )
                return result

            # Non-200
            # Chuẩn hoá error để caller dễ xử lý (split/backoff) kể cả khi API trả error dạng string.
            if isinstance(result.get("error"), dict):
                err = result.get("error") or {}
                logger.error(
                    f"LLM API error: {response.status_code} - {err.get('type')} - {err.get('code')} - {err.get('message')}"
                )
            else:
                msg = result.get("message") if isinstance(result.get("message"), str) else None
                err_str = result.get("error") if isinstance(result.get("error"), str) else None
                logger.error(f"LLM API error: {response.status_code} - {msg or err_str or response.text[:200]}")
                result["error"] = {
                    "code": response.status_code,
                    "type": "HTTPError",
                    "message": msg or err_str or "HTTP error",
                }
            return result
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return {"__error__": {"type": "Exception", "message": str(e)}}
    
    def call_llm(
        self,
        prompt: str,
        use_large: bool = True,
        max_tokens: int = 500,
        temperature: float = 0.1,
        response_format: Optional[Dict] = None,
        timeout_s: int = 60,
    ) -> str:
        """
        Gọi LLM API
        
        Args:
            prompt: Prompt text
            use_large: True = LLM Large, False = LLM Small
            max_tokens: Max tokens to generate
            temperature: Temperature (0-1)
        
        Returns:
            Generated text
        """
        result = self.call_chat(
            messages=[{"role": "user", "content": prompt}],
            use_large=use_large,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            timeout_s=timeout_s,
        )
        # Map error payloads to sentinel strings for batch splitting/backoff logic.
        if isinstance(result, dict) and isinstance(result.get("error"), dict):
            err = result.get("error") or {}
            return f"__VNPT_ERROR__{err.get('code', '200')}__{err.get('type', '')}"
        if isinstance(result, dict) and isinstance(result.get("__error__"), dict):
            return "__VNPT_ERROR__EXCEPTION"
        choices = result.get("choices") if isinstance(result, dict) else None
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if isinstance(msg, dict):
                return (msg.get("content") or "").strip()
        logger.error(f"Unexpected response format: {result}")
        return ""

    def call_llm_n(
        self,
        prompt: str,
        use_large: bool,
        n: int,
        max_tokens: int = 500,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 20,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[Dict] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        timeout_s: int = 60,
    ) -> List[str]:
        """
        Gọi LLM và lấy n completions trong 1 request (dùng cho self-consistency vote).
        """
        result = self.call_chat(
            messages=[{"role": "user", "content": prompt}],
            use_large=use_large,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=max(1, int(n)),
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format,
            seed=seed,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            tools=tools,
            tool_choice=tool_choice,
            timeout_s=timeout_s,
        )
        if isinstance(result, dict) and isinstance(result.get("error"), dict):
            err = result.get("error") or {}
            return [f"__VNPT_ERROR__{err.get('code', '200')}__{err.get('type', '')}"]
        choices = result.get("choices") if isinstance(result, dict) else None
        if not isinstance(choices, list) or not choices:
            return []
        out = []
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message", {})
            if isinstance(msg, dict):
                out.append((msg.get("content") or "").strip())
        return out
    
    def get_embedding(self, text: str, encoding_format: str = "float") -> Optional[np.ndarray]:
        """
        Get embedding từ API
        
        Args:
            text: Text to embed
            encoding_format: "float" hoặc "base64" (theo tài liệu API)
        
        Returns:
            Numpy array of embedding (1536 dim) or None
        """
        # VNPT Embedding API URL - Đúng theo tài liệu chính thức
        url = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"
        
        try:
            response = requests.post(
                url,
                headers={
                    'Authorization': self.embedding['authorization'],
                    'Token-id': self.embedding['tokenId'],
                    'Token-key': self.embedding['tokenKey'],
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'vnptai_hackathon_embedding',
                    'input': text,
                    'encoding_format': encoding_format,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Tùy response format
                if 'data' in result and len(result['data']) > 0:
                    embedding = result['data'][0].get('embedding', [])
                elif 'embedding' in result:
                    embedding = result['embedding']
                elif 'embeddings' in result:
                    embedding = result['embeddings'][0] if result['embeddings'] else []
                else:
                    embedding = []
                
                if embedding:
                    return np.array(embedding, dtype='float32')
                else:
                    logger.error(f"No embedding in response: {result}")
                    return None
            else:
                logger.error(f"Embedding API error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def get_embeddings(
        self,
        texts: List[str],
        encoding_format: str = "float",
        sleep_s: float = 0.15,
    ) -> List[Optional[np.ndarray]]:
        """
        Embed nhiều đoạn text (loop từng request) và throttle nhẹ để tránh rate-limit.
        """
        import time

        out: List[Optional[np.ndarray]] = []
        for t in texts:
            out.append(self.get_embedding(t, encoding_format=encoding_format))
            if sleep_s and sleep_s > 0:
                time.sleep(sleep_s)
        return out


if __name__ == "__main__":
    # Test API
    print("Initializing API Client...")
    client = APIClient()
    print("✓ API Client initialized successfully")
    print(f"✓ LLM Large config loaded")
    print(f"✓ LLM Small config loaded")
    print(f"✓ Embedding config loaded")
    
    # Test with actual API
    print("\n" + "="*60)
    print("Testing LLM Large...")
    response = client.call_llm("Xin chào, bạn là ai?", use_large=True, max_tokens=50)
    print(f"LLM Response: {response}")
    
    print("\n" + "="*60)
    print("Testing LLM Small...")
    response = client.call_llm("2 + 2 = ?", use_large=False, max_tokens=10)
    print(f"LLM Response: {response}")
    
    print("\n" + "="*60)
    print("Testing Embedding...")
    emb = client.get_embedding("Test embedding")
    if emb is not None:
        print(f"✓ Embedding shape: {emb.shape}")
        print(f"✓ First 5 values: {emb[:5]}")
    else:
        print("✗ Embedding failed")
