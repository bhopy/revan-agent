"""Quick test client for revan-agent — sends one inference request via Named Pipe."""

import struct
import time
import win32file
import win32pipe
import pywintypes

PIPE_NAME = r'\\.\pipe\revan-agent'
PROTO_VERSION = 2
REQ_INFERENCE = 0x0001

def build_request(system_prompt: str, user_prompt: str,
                  max_tokens: int = 0, temperature: float = 0.7,
                  session_id: int = 0) -> bytes:
    sys_bytes  = system_prompt.encode('utf-8')
    usr_bytes  = user_prompt.encode('utf-8')
    header_size = 28
    msg_len = header_size + len(sys_bytes) + len(usr_bytes)
    temp_x100 = int(temperature * 100)

    header = struct.pack('<IHHHHIIII',
        msg_len,
        PROTO_VERSION,
        REQ_INFERENCE,
        max_tokens,
        temp_x100,
        session_id,
        len(sys_bytes),
        len(usr_bytes),
        0  # reserved
    )
    return header + sys_bytes + usr_bytes

def parse_response(data: bytes) -> dict:
    if len(data) < 24:
        return {'error': 'response too short'}
    msg_len, version, status, tokens_gen, tokens_pp = struct.unpack_from('<IHHHH', data, 0)
    pp_ms, gen_ms, reserved = struct.unpack_from('<III', data, 12)
    text = data[24:].decode('utf-8', errors='replace') if len(data) > 24 else ''
    return {
        'status': status,
        'tokens_gen': tokens_gen,
        'tokens_pp': tokens_pp,
        'pp_ms': pp_ms,
        'gen_ms': gen_ms,
        'pp_rate': tokens_pp * 1000 / pp_ms if pp_ms > 0 else 0,
        'tg_rate': tokens_gen * 1000 / gen_ms if gen_ms > 0 else 0,
        'text': text,
    }

def send_request(system_prompt: str, user_prompt: str,
                 max_tokens: int = 256, temperature: float = 0.7,
                 session_id: int = 0) -> dict:
    req = build_request(system_prompt, user_prompt, max_tokens, temperature, session_id)

    t0 = time.time()
    try:
        handle = win32file.CreateFile(
            PIPE_NAME,
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0, None,
            win32file.OPEN_EXISTING,
            win32file.FILE_FLAG_NO_BUFFERING,
            None
        )
        win32pipe.SetNamedPipeHandleState(
            handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)

        _, _ = win32file.WriteFile(handle, req)
        _, data = win32file.ReadFile(handle, 1024 * 1024)  # 1MB max
        win32file.CloseHandle(handle)

        result = parse_response(data)
        result['wall_ms'] = int((time.time() - t0) * 1000)
        return result

    except pywintypes.error as e:
        return {'error': str(e), 'wall_ms': int((time.time() - t0) * 1000)}

if __name__ == '__main__':
    print("Connecting to revan-agent...")
    print()

    tests = [
        ("short",  "You are a helpful assistant.", "Answer in one word: is the sky blue?", 32),
        ("medium", "You are a helpful assistant.", "In 50 words, explain what RAM is.", 128),
        ("long",   "You are a helpful assistant.", "Write a 200-word explanation of how neural networks learn, covering weights, gradients, and backpropagation.", 512),
    ]

    for name, sys_p, usr_p, max_tok in tests:
        print(f"=== {name} ===")
        r = send_request(sys_p, usr_p, max_tokens=max_tok)
        if 'error' in r:
            print(f"  ERROR: {r['error']}")
        else:
            status_str = 'OK' if r['status'] == 0 else f'FAIL 0x{r["status"]:04X}'
            print(f"  status:    {status_str}")
            print(f"  pp:        {r['tokens_pp']} tok @ {r['pp_rate']:.0f} t/s ({r['pp_ms']} ms)")
            print(f"  gen:       {r['tokens_gen']} tok @ {r['tg_rate']:.0f} t/s ({r['gen_ms']} ms)")
            print(f"  wall:      {r['wall_ms']} ms")
            print(f"  response:  {r['text'][:200]!r}")
        print()
