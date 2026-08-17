"""Local-only browser interface for testing a trained checkpoint."""

# ruff: noqa: E501, RUF001 - the embedded Chinese HTML/CSS/JS is kept self-contained

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from torch import nn

from .generation import GenerationConfig, generate_text

_MAX_REQUEST_BYTES = 64 * 1024
_MAX_PROMPT_CHARACTERS = 8_000


class RequestError(ValueError):
    """A safe, user-facing API validation error."""


class GenerationService:
    """Serialize GPU generation and expose small JSON-friendly results."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        checkpoint: str,
        model_name: str = "100M 中文 Base",
        use_bf16: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.use_bf16 = use_bf16
        self._lock = threading.Lock()
        self._device = next(model.parameters()).device
        self._parameter_count = sum(parameter.numel() for parameter in model.parameters())
        config = getattr(model, "config", None)
        self._context_length = int(getattr(config, "max_seq_len", 1024))

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "busy": self._lock.locked(),
            "model": self.model_name,
            "parameters": self._parameter_count,
            "context_length": self._context_length,
            "device": str(self._device),
            "checkpoint": self.checkpoint,
            "sft": False,
        }

    def generate(self, payload: Any) -> dict[str, Any]:
        settings = _parse_generation_payload(payload)
        if not self._lock.acquire(blocking=False):
            raise RequestError("模型正在生成，请等待当前请求结束。")
        started = time.perf_counter()
        try:
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self._device.type == "cuda" and self.use_bf16
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast:
                result = generate_text(
                    self.model,
                    self.tokenizer,
                    settings["prompt"],
                    GenerationConfig(
                        max_new_tokens=settings["max_new_tokens"],
                        temperature=settings["temperature"],
                        top_k=settings["top_k"],
                        seed=settings["seed"],
                        context_length=self._context_length,
                    ),
                )
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            elapsed = time.perf_counter() - started
            response = asdict(result)
            response.update(
                {
                    "elapsed_seconds": elapsed,
                    "generated_tokens": len(result.generated_token_ids),
                    "tokens_per_second": (
                        len(result.generated_token_ids) / elapsed if elapsed > 0 else 0.0
                    ),
                    "settings": settings,
                }
            )
            return response
        finally:
            self._lock.release()


def _finite_number(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{name} 必须是数字。")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise RequestError(f"{name} 必须在 {minimum:g}–{maximum:g} 之间。")
    return number


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f"{name} 必须是整数。")
    if not minimum <= value <= maximum:
        raise RequestError(f"{name} 必须在 {minimum}–{maximum} 之间。")
    return value


def _parse_generation_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestError("请求必须是 JSON 对象。")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError("请输入提示词。")
    prompt = prompt.strip()
    if len(prompt) > _MAX_PROMPT_CHARACTERS:
        raise RequestError(f"提示词不能超过 {_MAX_PROMPT_CHARACTERS} 个字符。")
    max_new_tokens = _integer(
        payload.get("max_new_tokens", 128), name="生成长度", minimum=1, maximum=256
    )
    temperature = _finite_number(
        payload.get("temperature", 0.8), name="温度", minimum=0.0, maximum=2.0
    )
    top_k_value = payload.get("top_k", 50)
    top_k = (
        None
        if top_k_value in {None, 0}
        else _integer(top_k_value, name="Top-K", minimum=1, maximum=500)
    )
    seed = _integer(payload.get("seed", 42), name="随机种子", minimum=0, maximum=2**31 - 1)
    return {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    }


def _handler_for(service: GenerationService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "llm-pretrain-local/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"web {self.address_string()} {format % args}", flush=True)

        def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            )
            self.end_headers()

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self._headers(status, content_type, len(body))
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, value: Any) -> None:
            self._send(
                status,
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, _INDEX_HTML, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._json(HTTPStatus.OK, service.status())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "路径不存在。"})

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self._json(HTTPStatus.NOT_FOUND, {"error": "路径不存在。"})
                return
            try:
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    raise RequestError("请求缺少 Content-Length。")
                length = int(raw_length)
                if not 0 < length <= _MAX_REQUEST_BYTES:
                    raise RequestError("请求正文过大或为空。")
                payload = json.loads(self.rfile.read(length))
                self._json(HTTPStatus.OK, service.generate(payload))
            except (RequestError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - hardware/runtime failure
                print(f"generation failed: {type(exc).__name__}: {exc}", flush=True)
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "生成失败，请查看服务日志。"})

    return Handler


def serve_generation_ui(
    service: GenerationService,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> None:
    """Serve until interrupted; non-loopback binding is deliberately rejected."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Web UI is local-only; host must be 127.0.0.1, localhost, or ::1")
    if not 1 <= port <= 65_535:
        raise ValueError("port must be in 1..65535")
    server = ThreadingHTTPServer((host, port), _handler_for(service))
    print(f"Local model lab: http://{host}:{port}", flush=True)
    print(f"Checkpoint: {service.checkpoint}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


_INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>壹亿字室 · 100M Base Model Lab</title>
  <style>
    :root{--paper:#eee9dc;--ink:#171715;--muted:#6e695f;--red:#b52a21;--line:#aaa396;--panel:#f8f4e9}
    *{box-sizing:border-box} html{background:var(--ink)} body{margin:0;color:var(--ink);background:var(--paper);font-family:"Noto Serif SC","Songti SC","STSong",serif;min-height:100vh}
    body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.16;background-image:repeating-linear-gradient(0deg,transparent 0 31px,#8d877b 32px),repeating-linear-gradient(90deg,transparent 0 31px,#8d877b 32px)}
    .shell{position:relative;width:min(1380px,calc(100% - 32px));margin:0 auto;padding:36px 0 48px}
    header{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:end;border-top:7px solid var(--ink);border-bottom:1px solid var(--ink);padding:18px 0 16px}
    h1{margin:0;font-size:clamp(42px,7vw,92px);line-height:.88;font-weight:900;letter-spacing:-.08em}.en{display:block;font:700 12px/1.2 "Cascadia Code","IBM Plex Mono",monospace;letter-spacing:.2em;text-transform:uppercase;margin:14px 0 0 6px}
    .stamp{border:3px solid var(--red);color:var(--red);padding:10px 14px;font:800 13px/1 "Cascadia Code",monospace;letter-spacing:.1em;transform:rotate(-3deg);text-align:center}.stamp small{display:block;margin-top:6px;font-size:9px}
    .notice{display:flex;gap:12px;align-items:center;padding:12px 0;color:var(--muted);font-size:13px}.notice b{color:var(--red)}.dot{width:9px;height:9px;background:var(--red);border-radius:50%;box-shadow:0 0 0 4px rgba(181,42,33,.13)}
    main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.65fr);gap:18px}.card{background:rgba(248,244,233,.94);border:1px solid var(--ink);box-shadow:6px 6px 0 var(--ink)}
    .card-head{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--ink);padding:11px 14px;font:700 11px/1 "Cascadia Code",monospace;letter-spacing:.12em;text-transform:uppercase}.index{color:var(--red)}
    .work{padding:18px}.label{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;font-weight:800}.label small{color:var(--muted);font:10px "Cascadia Code",monospace}
    textarea{width:100%;min-height:220px;resize:vertical;border:1px solid var(--ink);background:transparent;padding:16px;color:var(--ink);font:500 19px/1.75 "Noto Serif SC","Songti SC",serif;outline:none}textarea:focus{box-shadow:inset 0 -3px var(--red)}
    .presets{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0 18px}.presets button{background:transparent;border:1px solid var(--line);padding:7px 10px;color:var(--muted);font:11px "Cascadia Code",monospace;cursor:pointer}.presets button:hover{border-color:var(--red);color:var(--red)}
    .controls{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.control{border-top:1px solid var(--line);padding-top:9px}.control label{display:block;color:var(--muted);font:10px "Cascadia Code",monospace;text-transform:uppercase}.control input{width:100%;margin-top:6px;background:transparent;border:0;border-bottom:1px solid var(--ink);font:700 15px "Cascadia Code",monospace;padding:5px 2px;outline:none}
    .generate{width:100%;margin-top:18px;border:0;background:var(--red);color:#fff8ea;padding:16px 20px;font:800 14px "Cascadia Code",monospace;letter-spacing:.15em;cursor:pointer;transition:transform .12s,box-shadow .12s}.generate:hover{transform:translate(-2px,-2px);box-shadow:4px 4px 0 var(--ink)}.generate:disabled{background:var(--line);cursor:wait;transform:none;box-shadow:none}
    .output{min-height:310px;padding:20px;font-size:18px;line-height:1.8;white-space:pre-wrap;overflow-wrap:anywhere}.placeholder{color:var(--line)}.output.loading{animation:pulse 1s steps(2,end) infinite}@keyframes pulse{50%{opacity:.45}}
    .meta{display:grid;grid-template-columns:repeat(3,1fr);border-top:1px solid var(--ink)}.metric{padding:12px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric b{display:block;font:800 14px "Cascadia Code",monospace}.metric span{font:9px "Cascadia Code",monospace;color:var(--muted);text-transform:uppercase}
    .facts{margin-top:18px;padding:0 14px 14px}.facts dl{margin:0}.facts div{display:grid;grid-template-columns:110px 1fr;gap:10px;padding:10px 0;border-bottom:1px dotted var(--line)}dt{font:10px "Cascadia Code",monospace;color:var(--muted);text-transform:uppercase}dd{margin:0;font-size:13px;overflow-wrap:anywhere}
    .error{color:var(--red);font-weight:800}.statusline{display:flex;align-items:center;gap:8px}.live{width:7px;height:7px;border-radius:50%;background:#507a4b}.live.busy{background:#c18324}
    footer{display:flex;justify-content:space-between;gap:20px;margin-top:22px;padding-top:12px;border-top:1px solid var(--ink);font:10px/1.5 "Cascadia Code",monospace;color:var(--muted)}
    @media(max-width:820px){main{grid-template-columns:1fr}.controls{grid-template-columns:1fr 1fr}header{grid-template-columns:1fr}.stamp{justify-self:start}.shell{width:min(100% - 20px,680px);padding-top:18px}.meta{grid-template-columns:1fr}.metric{border-right:0;border-bottom:1px solid var(--line)}}
  </style>
</head>
<body><div class="shell">
  <header><div><h1>壹亿字室</h1><span class="en">100M Chinese Base Model Laboratory</span></div><div class="stamp">BASE MODEL<small>NO SFT · LOCAL ONLY</small></div></header>
  <div class="notice"><span class="dot"></span><span><b>实验性输出。</b> 模型只有约 1 亿参数，可能重复、捏造事实，也可能无法正确结束句子。</span></div>
  <main>
    <section class="card"><div class="card-head"><span><i class="index">01</i> / INPUT SPECIMEN</span><span id="statusText">连接中</span></div><div class="work">
      <div class="label">给模型一个开头 <small id="charCount">0 / 8000</small></div>
      <textarea id="prompt" autofocus placeholder="例如：杭州是一座"></textarea>
      <div class="presets"><button data-p="杭州是一座">杭州是一座</button><button data-p="请解释什么是机器学习：">机器学习</button><button data-p="从前有一只猫，">续写故事</button><button data-p="中国的首都是">事实探针</button></div>
      <div class="controls">
        <div class="control"><label>New tokens</label><input id="maxTokens" type="number" min="1" max="256" value="128"></div>
        <div class="control"><label>Temperature</label><input id="temperature" type="number" min="0" max="2" step="0.05" value="0.8"></div>
        <div class="control"><label>Top-K</label><input id="topK" type="number" min="0" max="500" value="50"></div>
        <div class="control"><label>Seed</label><input id="seed" type="number" min="0" value="42"></div>
      </div>
      <button class="generate" id="generate">送入模型 / GENERATE</button>
    </div></section>
    <aside>
      <section class="card"><div class="card-head"><span><i class="index">02</i> / OUTPUT RECORD</span><span class="statusline"><i id="live" class="live"></i><span id="runState">READY</span></span></div>
        <div class="output"><span id="output" class="placeholder">输出将记录在这里。模型第一次生成可能稍慢。</span></div>
        <div class="meta"><div class="metric"><b id="elapsed">—</b><span>seconds</span></div><div class="metric"><b id="tokens">—</b><span>tokens</span></div><div class="metric"><b id="speed">—</b><span>token / s</span></div></div>
      </section>
      <section class="card facts"><div class="card-head"><span><i class="index">03</i> / MODEL CARD</span></div><dl>
        <div><dt>MODEL</dt><dd id="modelName">读取中…</dd></div><div><dt>PARAMETERS</dt><dd id="parameters">—</dd></div><div><dt>CONTEXT</dt><dd id="context">—</dd></div><div><dt>DEVICE</dt><dd id="device">—</dd></div><div><dt>CHECKPOINT</dt><dd id="checkpoint">—</dd></div>
      </dl></section>
    </aside>
  </main>
  <footer><span>LLM-PRETRAIN / LOCAL INFERENCE BENCH</span><span>数据不会离开这台电脑 · 127.0.0.1</span></footer>
</div>
<script>
const $=id=>document.getElementById(id),prompt=$('prompt'),button=$('generate'),output=$('output');
const fmt=n=>new Intl.NumberFormat('zh-CN').format(n);
function state(text,busy=false){$('runState').textContent=text;$('live').classList.toggle('busy',busy);button.disabled=busy;output.classList.toggle('loading',busy)}
prompt.addEventListener('input',()=>{$('charCount').textContent=`${prompt.value.length} / 8000`});
document.querySelectorAll('[data-p]').forEach(x=>x.onclick=()=>{prompt.value=x.dataset.p;prompt.dispatchEvent(new Event('input'));prompt.focus()});
async function status(){try{const r=await fetch('/api/status'),s=await r.json();$('statusText').textContent=s.busy?'生成中':'模型就绪';$('modelName').textContent=s.model;$('parameters').textContent=fmt(s.parameters);$('context').textContent=fmt(s.context_length)+' tokens';$('device').textContent=s.device;$('checkpoint').textContent=s.checkpoint.split('/').slice(-2).join('/')}catch{$('statusText').textContent='连接失败'}}
button.onclick=async()=>{if(!prompt.value.trim())return prompt.focus();state('RUNNING',true);output.className='placeholder';output.textContent='模型正在逐 token 推演…';try{const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:prompt.value,max_new_tokens:Number($('maxTokens').value),temperature:Number($('temperature').value),top_k:Number($('topK').value),seed:Number($('seed').value)})});const d=await r.json();if(!r.ok)throw new Error(d.error||'请求失败');output.className='';output.textContent=d.completion||'（模型立即输出了 EOS）';$('elapsed').textContent=d.elapsed_seconds.toFixed(2);$('tokens').textContent=fmt(d.generated_tokens);$('speed').textContent=d.tokens_per_second.toFixed(1);state(d.stopped_on_eos?'EOS':'LIMIT',false)}catch(e){output.className='error';output.textContent=e.message;state('ERROR',false)}};
status();setInterval(status,5000);
</script></body></html>'''.encode()
