# revan-agent

Local LLM inference over Windows Named Pipes. Single-file C++, links llama.cpp directly, no HTTP anywhere.

- ~0.1ms IPC overhead instead of HTTP's 1-5ms per call
- Multi-turn conversations without re-processing history (KV cache snapshots)
- Model lazy-loads on first request, auto-unloads after 5 min idle to free VRAM
- No ports, no firewall prompts, no JSON parsing -- just a pipe and a binary message
- One C++ file, ~870 lines, nothing to configure beyond a model path
- Any GGUF model, any NVIDIA GPU with CUDA

If you've seen [revan-oss](https://github.com/bhopy/revan-oss), that project treats the model as a decision engine -- short structured answers only, 1-15 tokens max, with a Rust hub managing agents. This is the opposite. revan-agent is a general-purpose inference server for full text generation with multi-turn session persistence. No hub, no agent framework, just a pipe you send prompts to and get text back. Different tool for a different job.

Every local inference setup I could find uses HTTP -- llama-server, Ollama, LM Studio, all of them. That's fine for most things, but if you're building a native Windows app and just want to talk to a local model, you end up running a whole web server for what should be a function call. Named Pipes are built into the OS, give you atomic message delivery at around 0.1ms, and your app just connects to `\\.\pipe\revan-agent` with a binary message. No ports, no localhost networking, no firewall pop-ups.

The whole engine is one C++ file (~870 lines). It loads any GGUF model, listens on a Named Pipe, and does inference. Model weights go on the GPU, KV cache stays in system RAM to save VRAM. It lazy-loads on the first request and unloads after 5 minutes idle so it's not sitting there eating VRAM when nothing's happening.

The other thing it does is session persistence. If you pass a session ID with your request, the engine snapshots the KV cache state after each response and restores it next time. So turn 5 of a conversation doesn't re-process turns 1 through 4. Up to 8 sessions live in RAM at once, LRU eviction when it's full. Pass session ID 0 for stateless one-shot inference.

Works with any GGUF model. Tested on an RTX 3060 with Hermes-3 8B Q4_K_M.

## Building

You need Windows 10/11, CMake 3.18+, a C++ compiler (MSVC or MinGW), Ninja, CUDA Toolkit 12.x, and an NVIDIA GPU.

First, build llama.cpp. Any recent release works (tested with b8325):

```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --config Release
```

Then set up the `bin_lib` directory that CMakeLists.txt expects. You need the headers and import libraries from your llama.cpp build:

```
mkdir bin_lib\include
mkdir bin_lib\lib

copy llama.cpp\include\llama.h              bin_lib\include\
copy llama.cpp\ggml\include\ggml.h          bin_lib\include\
copy llama.cpp\ggml\include\ggml-backend.h  bin_lib\include\
copy llama.cpp\ggml\include\ggml-alloc.h    bin_lib\include\
copy llama.cpp\ggml\include\ggml-cpu.h      bin_lib\include\
copy llama.cpp\ggml\include\ggml-opt.h      bin_lib\include\
copy llama.cpp\ggml\include\ggml-cpp.h      bin_lib\include\
copy llama.cpp\ggml\include\gguf.h          bin_lib\include\

copy llama.cpp\build\src\llama.lib          bin_lib\lib\
copy llama.cpp\build\ggml\src\ggml.lib      bin_lib\lib\
copy llama.cpp\build\ggml\src\ggml-base.lib bin_lib\lib\
copy llama.cpp\build\ggml\src\ggml-cuda.lib bin_lib\lib\
```

Exact paths depend on your llama.cpp version but you need at minimum `llama.lib`, `ggml.lib`, `ggml-base.lib`, and `ggml-cuda.lib`.

Now build revan-agent itself:

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Binary goes to `build/revan-agent.exe`. You'll also need to copy the runtime DLLs into the same folder -- `llama.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cuda.dll`, and the `ggml-cpu-*.dll` files from your llama.cpp build, plus `cudart64_12.dll`, `cublas64_12.dll`, and `cublasLt64_12.dll` from your CUDA Toolkit install.

## Running

Copy `revan-agent.toml.example` to `revan-agent.toml`, point it at your model:

```toml
[model]
path = "D:/models/your-model-Q4_K_M.gguf"

[pipe]
name = "\\.\pipe\revan-agent"
```

Then just run it:

```
revan-agent.exe
revan-agent.exe --config path/to/revan-agent.toml
```

It starts idle. The model loads when the first request comes in, and unloads after 5 minutes of silence. The pipe name is configurable if you want multiple instances.

## Wire protocol

Everything goes over a single Named Pipe as binary messages, little-endian. No JSON on the wire.

Request header is 28 bytes:

| Offset | Size | Type | Field                                    |
| ------ | ---- | ---- | ---------------------------------------- |
| 0      | 4    | u32  | Total message length                     |
| 4      | 2    | u16  | Protocol version (2)                     |
| 6      | 2    | u16  | Request type                             |
| 8      | 2    | u16  | Max tokens (0 = until end-of-generation) |
| 10     | 2    | u16  | Temperature \* 100 (70 = 0.70)           |
| 12     | 4    | u32  | Session ID (0 = stateless)               |
| 16     | 4    | u32  | System prompt length in bytes            |
| 20     | 4    | u32  | User prompt length in bytes              |
| 24     | 4    | u32  | Reserved                                 |

After the header, append the system prompt and user prompt as raw UTF-8 bytes, no null terminators.

Request type `0x0001` is inference, `0x0002` is health check, `0x0003` is shutdown.

Response header is 24 bytes:

| Offset | Size | Type | Field                       |
| ------ | ---- | ---- | --------------------------- |
| 0      | 4    | u32  | Total message length        |
| 4      | 2    | u16  | Protocol version (2)        |
| 6      | 2    | u16  | Status (0 = OK)             |
| 8      | 2    | u16  | Tokens generated            |
| 10     | 2    | u16  | Tokens in prompt            |
| 12     | 4    | u32  | Prompt processing time (ms) |
| 16     | 4    | u32  | Generation time (ms)        |
| 20     | 4    | u32  | Reserved                    |

Generated text follows as UTF-8 after byte 24. Status codes: `0x0000` OK, `0x0001` model not loaded, `0x0002` prompt too long, `0x0003` inference failed, `0x0004` timeout.

## Python client example

Needs `pywin32` (`pip install pywin32`):

```python
import struct
import win32file
import win32pipe

PIPE = r'\\.\pipe\revan-agent'

def ask(system_prompt, user_prompt, max_tokens=256, temperature=0.7, session_id=0):
    sys_b = system_prompt.encode('utf-8')
    usr_b = user_prompt.encode('utf-8')

    header = struct.pack('<IHHHHIIII',
        28 + len(sys_b) + len(usr_b),  # msg_len
        2,                              # version
        0x0001,                         # inference
        max_tokens,
        int(temperature * 100),
        session_id,
        len(sys_b),
        len(usr_b),
        0                               # reserved
    )

    handle = win32file.CreateFile(
        PIPE,
        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
        0, None, win32file.OPEN_EXISTING, 0, None
    )
    win32pipe.SetNamedPipeHandleState(handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
    win32file.WriteFile(handle, header + sys_b + usr_b)
    _, data = win32file.ReadFile(handle, 1024 * 1024)
    win32file.CloseHandle(handle)

    msg_len, ver, status, tok_gen, tok_pp = struct.unpack_from('<IHHHH', data, 0)
    pp_ms, gen_ms, _ = struct.unpack_from('<III', data, 12)
    text = data[24:].decode('utf-8', errors='replace')

    return {
        'text': text,
        'status': status,
        'tokens_generated': tok_gen,
        'tokens_prompt': tok_pp,
        'prompt_ms': pp_ms,
        'generation_ms': gen_ms,
    }

# one-shot
r = ask("You are a helpful assistant.", "What is the capital of France?")
print(r['text'])

# multi-turn (session ID keeps context between calls)
ask("You are a helpful assistant.", "My name is Alex.", session_id=42)
r = ask("", "What's my name?", session_id=42)
print(r['text'])
```

Same idea works from C#, C++, Rust, or anything else that can open a Named Pipe. The `test_client.py` and `test_session.py` files have more complete examples.

## Tuning

The engine parameters are compile-time constants at the top of `revan_agent.cpp`. The main ones you might want to change:

`N_GPU_LAYERS` (default 33) controls how many transformer layers get offloaded to the GPU. A 6 GB card fits all layers of most 8B Q4 models. For bigger models, lower this to split between GPU and CPU.

`N_CTX` (default 32768) is the context window. `MAX_SESSIONS` (default 8) is how many KV snapshots live in RAM simultaneously. `IDLE_TIMEOUT_SECS` (default 300) is how long before the model unloads. `N_THREADS` (default 6) is CPU threads for prompt processing -- set it to your physical core count.

Change what you need, rebuild, done.

## What this doesn't do

Windows only -- Named Pipes and Win32 are baked in, no Linux or macOS. NVIDIA only -- needs CUDA. One request at a time -- concurrent connections queue up. No streaming -- you get the full response after generation finishes. No HTTP -- if you want HTTP, use [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server).

## Files

```
src/revan_agent.cpp       the engine (~870 lines)
CMakeLists.txt            build config
revan-agent.toml.example  sample config
test_client.py            basic inference test
test_session.py           multi-turn session test
```

## License

MIT
