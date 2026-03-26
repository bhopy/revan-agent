"""Test multi-turn session persistence in revan-agent."""

import struct
import time
import win32file
import win32pipe
import pywintypes

PIPE_NAME     = r'\\.\pipe\revan-agent'
PROTO_VERSION = 2
REQ_INFERENCE = 0x0001

def build_request(system_prompt, user_prompt, max_tokens=256, temperature=0.7, session_id=0):
    sys_b = system_prompt.encode('utf-8')
    usr_b = user_prompt.encode('utf-8')
    msg_len = 28 + len(sys_b) + len(usr_b)
    hdr = struct.pack('<IHHHHIIII',
        msg_len, PROTO_VERSION, REQ_INFERENCE,
        max_tokens, int(temperature * 100), session_id,
        len(sys_b), len(usr_b), 0)
    return hdr + sys_b + usr_b

def send(sys_p, usr_p, max_tokens=256, temperature=0.7, session_id=0):
    req  = build_request(sys_p, usr_p, max_tokens, temperature, session_id)
    t0   = time.time()
    h    = win32file.CreateFile(PIPE_NAME,
               win32file.GENERIC_READ | win32file.GENERIC_WRITE,
               0, None, win32file.OPEN_EXISTING, 0, None)
    win32pipe.SetNamedPipeHandleState(h, win32pipe.PIPE_READMODE_MESSAGE, None, None)
    win32file.WriteFile(h, req)
    _, data = win32file.ReadFile(h, 2 * 1024 * 1024)
    win32file.CloseHandle(h)
    wall = time.time() - t0

    msg_len, ver, status, tgen, tpp = struct.unpack_from('<IHHHH', data, 0)
    pp_ms, gen_ms, _ = struct.unpack_from('<III', data, 12)
    text = data[24:].decode('utf-8', errors='replace')
    tg_rate = tgen * 1000 / gen_ms if gen_ms > 0 else 0

    print(f"  [{wall:.1f}s wall] pp={tpp}tok gen={tgen}tok@{tg_rate:.0f}t/s")
    print(f"  > {text[:200]}")
    return text

print("=== Multi-turn Session Test (session_id=42) ===")
SYS = "You are a helpful assistant. Remember everything the user tells you."
SESSION = 42

print("\nTurn 1: introduce a number")
t1 = send(SYS, "My favourite number is 7. Remember that.", session_id=SESSION)

print("\nTurn 2: recall the number (KV session should remember)")
t2 = send("", "What is my favourite number?", session_id=SESSION)

print("\nTurn 3: add another fact")
t3 = send("", "My favourite colour is blue.", session_id=SESSION)

print("\nTurn 4: recall both facts")
t4 = send("", "What are my favourite number and colour?", session_id=SESSION)

print("\n=== Session test complete ===")
