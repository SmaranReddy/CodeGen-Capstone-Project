"""Model providers for the 68-example benchmark.

Each provider exposes `available()` and `generate(messages)` returning a dict:
    {text, input_tokens, output_tokens, latency_s, model_id, error}

Providers never fabricate access: `available()` must succeed before a run starts.
The 'reference' provider is not a model; it replays the project's own validated MQL and
is used only to verify the execution/scoring harness.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    pass


class Provider:
    name = "base"
    model_id = "unknown"

    def available(self):
        return False, "not implemented"

    def generate(self, messages, system, generation):
        raise NotImplementedError


class ReferenceProvider(Provider):
    """Replays the validated reference MQL for each example (harness oracle, NOT a model)."""
    name = "reference"
    model_id = "reference-oracle"

    def __init__(self, pipeline_by_iid):
        self.pipeline_by_iid = pipeline_by_iid

    def available(self):
        return True, "replays project reference MQL (harness validation only)"

    def generate(self, messages, system, generation):
        iid = generation.get("_iid")
        payload = self.pipeline_by_iid.get(iid)
        text = json.dumps(payload) if payload else "{}"
        return {"text": text, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0,
                "model_id": self.model_id, "error": None}


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, model, base_url="http://localhost:11434"):
        self.model_id = model
        self.base_url = base_url.rstrip("/")

    def available(self):
        try:
            with urllib.request.urlopen(self.base_url + "/api/tags", timeout=3) as r:
                tags = json.loads(r.read().decode("utf-8"))
            names = [m.get("name", "") for m in tags.get("models", [])]
            if any(self.model_id.split(":")[0] in n for n in names):
                return True, f"ollama up; model {self.model_id} present ({names})"
            return False, f"ollama up but model {self.model_id} not pulled (have {names})"
        except Exception as e:
            return False, f"ollama unreachable: {e}"

    def generate(self, messages, system, generation):
        body = {
            "model": self.model_id,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "options": {"temperature": generation.get("temperature", 0),
                        "top_p": generation.get("top_p", 1.0),
                        "seed": generation.get("seed", 42),
                        "num_predict": generation.get("max_new_tokens", 1024)},
        }
        req = urllib.request.Request(self.base_url + "/api/chat",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read().decode("utf-8"))
            return {"text": data.get("message", {}).get("content", ""),
                    "input_tokens": data.get("prompt_eval_count"),
                    "output_tokens": data.get("eval_count"),
                    "latency_s": time.time() - t0, "model_id": self.model_id, "error": None}
        except Exception as e:
            return {"text": "", "input_tokens": None, "output_tokens": None,
                    "latency_s": time.time() - t0, "model_id": self.model_id, "error": str(e)}


def parse_retry_in(msg):
    """Seconds from 'try again in 1h2m3.5s' / '4m35.6s' / '432ms' / '7.2s'; None if absent."""
    m = re.search(r"try again in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s|[\d.]+ms)?)", msg)
    if not m or not m.group(1):
        return None
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(h|ms|m|s)", m.group(1)):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


class OpenAICompatibleProvider(Provider):
    """OpenAI-compatible chat completions (OpenAI, Groq, OpenRouter, ...).
    Sends temperature/top_p/seed; retries 429/5xx honouring Retry-After (transport only,
    never on a wrong answer); stops immediately on a per-day quota error."""
    name = "openai"
    UA = "codegen-capstone-eval/1.0"

    def __init__(self, model, base_url, api_key_env="FRONTIER_API_KEY", max_tokens=None,
                 reasoning_effort=None, retries=8, quota_wait_max_s=0):
        self.model_id = model
        self.base_url = (base_url or "").rstrip("/")
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.retries = retries
        self.quota_wait_max_s = quota_wait_max_s  # >0: sleep through refilling daily-token buckets

    @property
    def api_key(self):
        return os.environ.get(self.api_key_env, "")

    def _headers(self):
        return {"Content-Type": "application/json", "User-Agent": self.UA,
                "Authorization": f"Bearer {self.api_key}"}

    def available(self):
        if not self.model_id:
            return False, "no frontier model id configured"
        if not self.base_url:
            return False, "no base_url configured"
        if not self.api_key:
            return False, f"missing API key in env var {self.api_key_env}"
        try:
            req = urllib.request.Request(self.base_url + "/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=20) as r:
                ids = [m.get("id") for m in json.loads(r.read().decode("utf-8")).get("data", [])]
            if self.model_id not in ids:
                return False, f"model {self.model_id} not served at {self.base_url}"
            return True, f"model {self.model_id} served at {self.base_url}"
        except Exception as e:
            return False, f"endpoint probe failed: {e}"

    def generate(self, messages, system, generation):
        body = {"model": self.model_id,
                "messages": [{"role": "system", "content": system}] + messages,
                "temperature": generation.get("temperature", 0),
                "top_p": generation.get("top_p", 1.0),
                "seed": generation.get("seed", 42),
                "max_tokens": self.max_tokens or generation.get("max_new_tokens", 1024)}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        t0 = time.time()
        last = None
        quota_waited = 0.0
        attempt = 0
        while attempt <= self.retries:
            attempt += 1
            t_req = time.time()
            req = urllib.request.Request(self.base_url + "/chat/completions",
                                         data=json.dumps(body).encode("utf-8"), headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.loads(r.read().decode("utf-8"))
                usage = data.get("usage", {})
                choice = data["choices"][0]
                details = usage.get("completion_tokens_details") or {}
                return {"text": choice["message"].get("content") or "",
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "thought_tokens": details.get("reasoning_tokens"),
                        "stop_reason": choice.get("finish_reason"),
                        "latency_s": time.time() - t_req, "wall_s": time.time() - t0,
                        "rate_limit_retries": attempt - 1, "quota_wait_s": round(quota_waited, 1),
                        "model_id": self.model_id,
                        "server_time_s": usage.get("total_time"), "error": None}
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                last = f"HTTP {e.code}: {raw[:300]}"
                if e.code == 429 and re.search(r"per day|\(RPD\)|\(TPD\)|daily", raw, re.I):
                    wait = parse_retry_in(raw)
                    if wait is not None and 0 < self.quota_wait_max_s and wait <= self.quota_wait_max_s:
                        quota_waited += wait + 2
                        time.sleep(wait + 2)
                        attempt -= 1
                        continue  # bucket refilled; not counted as a retry attempt
                    last = f"DAILY_QUOTA_EXHAUSTED: {raw[:300]}"
                    break
                if e.code not in (429, 500, 502, 503, 504):
                    break
                wait = e.headers.get("retry-after")
                try:
                    wait = float(wait)
                except (TypeError, ValueError):
                    wait = min(60, 5 * 2 ** attempt)
                time.sleep(min(wait + 1, 120))
                continue
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(min(60, 5 * 2 ** attempt))
        return {"text": "", "input_tokens": None, "output_tokens": None,
                "latency_s": time.time() - t0, "model_id": self.model_id, "error": last}


class TransformersProvider(Provider):
    """Local Hugging Face inference (Qwen baseline). Loads lazily on first generate()."""
    name = "transformers"

    def __init__(self, model_id, revision="main", quantization="none", device="auto"):
        self.model_id = model_id
        self.revision = revision
        self.quantization = quantization
        self.device = device
        self._pipe = None

    def available(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as e:
            return False, f"transformers/torch not importable in this interpreter: {e}"
        return True, "transformers importable (model loads on first call)"

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline
        kwargs = {}
        if self.quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            import torch
            kwargs["model_kwargs"] = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=self.quantization == "4bit",
                load_in_8bit=self.quantization == "8bit")}
            kwargs["device_map"] = "auto"
        elif self.device == "cpu":
            kwargs["device"] = -1
        self._pipe = pipeline("text-generation", model=self.model_id,
                              revision=self.revision, **kwargs)

    def generate(self, messages, system, generation):
        t0 = time.time()
        try:
            self._load()
            full = [{"role": "system", "content": system}] + messages
            out = self._pipe(full, max_new_tokens=generation.get("max_new_tokens", 1024),
                             do_sample=generation.get("temperature", 0) > 0,
                             temperature=max(generation.get("temperature", 0), 1e-5),
                             top_p=generation.get("top_p", 1.0),
                             return_full_text=False)
            text = out[0]["generated_text"]
            if isinstance(text, list):
                text = text[-1]["content"]
            return {"text": text, "input_tokens": None, "output_tokens": None,
                    "latency_s": time.time() - t0, "model_id": self.model_id, "error": None}
        except Exception as e:
            return {"text": "", "input_tokens": None, "output_tokens": None,
                    "latency_s": time.time() - t0, "model_id": self.model_id, "error": str(e)}


def build_provider(cfg, name, pipeline_by_iid=None):
    """cfg is one entry from configs/evaluation_68.yaml `models`."""
    backend = cfg.get("backend")
    if name == "reference":
        return ReferenceProvider(pipeline_by_iid or {})
    if backend == "ollama":
        return OllamaProvider(cfg.get("ollama_model"), cfg.get("ollama_base_url"))
    if backend == "openai":
        return OpenAICompatibleProvider(cfg.get("id"), cfg.get("base_url"),
                                        cfg.get("api_key_env", "FRONTIER_API_KEY"),
                                        cfg.get("max_tokens"), cfg.get("reasoning_effort"),
                                        quota_wait_max_s=cfg.get("quota_wait_max_s", 0))
    if backend == "transformers":
        return TransformersProvider(cfg.get("id"), cfg.get("revision", "main"),
                                    cfg.get("quantization", "none"), cfg.get("device", "auto"))
    if backend == "llamacpp":
        return LlamaCppProvider(cfg.get("id"), cfg.get("gguf_path"), cfg.get("n_ctx", 8192),
                                cfg.get("n_threads"))
    if backend == "gemini":
        return GeminiProvider(cfg.get("id"), cfg.get("api_key_env", "GEMINI_API_KEY"),
                              cfg.get("max_tokens", 16000))
    if backend == "anthropic":
        return AnthropicProvider(cfg.get("id"), cfg.get("effort", "medium"),
                                 cfg.get("max_tokens", 16000))
    raise ProviderError(f"unknown backend '{backend}' for provider '{name}'")


class LlamaCppProvider(Provider):
    """Local GGUF inference via llama-cpp-python (CPU). Used for the Qwen baseline when no
    GPU/torch is available. Greedy decoding (temperature 0) with a fixed seed."""
    name = "llamacpp"

    def __init__(self, model_id, gguf_path, n_ctx=8192, n_threads=None):
        self.model_id = model_id
        self.gguf_path = gguf_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None

    def available(self):
        try:
            import llama_cpp  # noqa: F401
        except Exception as e:
            return False, f"llama_cpp not importable: {e}"
        if not self.gguf_path or not os.path.exists(self.gguf_path):
            return False, f"GGUF not found: {self.gguf_path}"
        return True, f"llama-cpp-python {llama_cpp.__version__}; {os.path.basename(self.gguf_path)}"

    def _load(self, seed):
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(model_path=self.gguf_path, n_ctx=self.n_ctx, seed=seed,
                              n_threads=self.n_threads, verbose=False)

    def generate(self, messages, system, generation):
        t0 = time.time()
        try:
            self._load(generation.get("seed", 42))
            out = self._llm.create_chat_completion(
                messages=[{"role": "system", "content": system}] + messages,
                temperature=generation.get("temperature", 0), top_p=generation.get("top_p", 1.0),
                max_tokens=generation.get("max_new_tokens", 1024), seed=generation.get("seed", 42))
            usage = out.get("usage", {})
            return {"text": out["choices"][0]["message"]["content"],
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "latency_s": time.time() - t0, "model_id": self.model_id, "error": None}
        except Exception as e:
            return {"text": "", "input_tokens": None, "output_tokens": None,
                    "latency_s": time.time() - t0, "model_id": self.model_id, "error": str(e)}


class AnthropicProvider(Provider):
    """Claude via the official Anthropic SDK. Current Opus models reject sampling params
    (temperature/top_p) and always think adaptively; depth is set with `effort`.
    No server-side fallback is enabled, so every answer comes from `model_id`."""
    name = "anthropic"

    def __init__(self, model_id, effort="medium", max_tokens=16000):
        self.model_id = model_id
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = None

    def available(self):
        try:
            import anthropic
        except Exception as e:
            return False, f"anthropic SDK not importable: {e}"
        try:
            self._client = anthropic.Anthropic()
            m = self._client.models.retrieve(self.model_id)
            return True, f"model {m.id} reachable"
        except Exception as e:
            return False, f"Anthropic API probe failed: {type(e).__name__}: {str(e)[:200]}"

    def generate(self, messages, system, generation):
        import anthropic
        t0 = time.time()
        try:
            if self._client is None:
                self._client = anthropic.Anthropic()
            r = self._client.messages.create(
                model=self.model_id, max_tokens=self.max_tokens, system=system,
                output_config={"effort": self.effort}, messages=messages)
            text = "".join(b.text for b in r.content if b.type == "text")
            err = None if r.stop_reason != "refusal" else "refusal"
            return {"text": text, "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens, "stop_reason": r.stop_reason,
                    "request_id": r._request_id,
                    "latency_s": time.time() - t0, "model_id": self.model_id, "error": err}
        except Exception as e:
            return {"text": "", "input_tokens": None, "output_tokens": None,
                    "latency_s": time.time() - t0, "model_id": self.model_id,
                    "error": f"{type(e).__name__}: {str(e)[:300]}"}


class GeminiProvider(Provider):
    """Google Gemini via the REST generateContent API (key from env, never logged).
    temperature/topP/seed are passed through; thinking stays at the model default and
    thought tokens are recorded. 429/5xx are retried with backoff (transport only)."""
    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model_id, api_key_env="GEMINI_API_KEY", max_output_tokens=16000, retries=6):
        self.model_id = model_id or os.environ.get("GEMINI_MODEL")
        self.api_key_env = api_key_env
        self.max_output_tokens = max_output_tokens
        self.retries = retries

    def _key(self):
        return os.environ.get(self.api_key_env, "")

    def available(self):
        if not self._key():
            return False, f"missing {self.api_key_env}"
        try:
            req = urllib.request.Request(f"{self.BASE}/models/{self.model_id}",
                                         headers={"x-goog-api-key": self._key()})
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status == 200, f"model {self.model_id} reachable"
        except Exception as e:
            return False, f"Gemini probe failed: {type(e).__name__}: {str(e)[:200]}"

    def generate(self, messages, system, generation):
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user" if m["role"] == "user" else "model",
                          "parts": [{"text": m["content"]}]} for m in messages],
            "generationConfig": {"temperature": generation.get("temperature", 0),
                                 "topP": generation.get("top_p", 1.0),
                                 "seed": generation.get("seed", 42),
                                 "maxOutputTokens": self.max_output_tokens},
        }
        url = f"{self.BASE}/models/{self.model_id}:generateContent"
        t0 = time.time()
        last = None
        for attempt in range(self.retries + 1):
            t_req = time.time()
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json",
                                                  "x-goog-api-key": self._key()})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.loads(r.read().decode("utf-8"))
                cand = (data.get("candidates") or [{}])[0]
                parts = (cand.get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                u = data.get("usageMetadata", {})
                out_tok = (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)
                return {"text": text, "input_tokens": u.get("promptTokenCount"),
                        "output_tokens": out_tok, "thought_tokens": u.get("thoughtsTokenCount"),
                        "stop_reason": cand.get("finishReason"),
                        "latency_s": time.time() - t_req, "wall_s": time.time() - t0,
                        "rate_limit_retries": attempt, "model_id": self.model_id,
                        "error": None if text else f"empty response (finishReason={cand.get('finishReason')})"}
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                last = f"HTTP {e.code}: {raw[:300]}"
                if "PerDay" in raw:  # daily quota: retrying cannot help
                    quota = re.search(r'"quotaId":\s*"([^"]+)"', raw)
                    last = f"DAILY_QUOTA_EXHAUSTED ({quota.group(1) if quota else 'per-day'}): {raw[:200]}"
                    break
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(min(60, 5 * 2 ** attempt))
        return {"text": "", "input_tokens": None, "output_tokens": None,
                "latency_s": time.time() - t0, "model_id": self.model_id, "error": last}
