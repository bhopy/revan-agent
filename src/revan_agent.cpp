// revan_agent.cpp — Full Agent Engine
// KV cache in RAM (offload_kqv=false), 16k context, persistent sessions.
// IPC: Windows Named Pipes with 28-byte binary header.
// No HTTP, no external deps beyond llama.cpp + Win32.
//
// Wire protocol (little-endian):
//   [0:4]   u32 msg_len
//   [4:6]   u16 version
//   [6:8]   u16 request_type
//   [8:10]  u16 max_tokens  (0 = generate until EOG)
//   [10:12] u16 temperature * 100  (e.g., 70 = 0.70)
//   [12:16] u32 session_id  (0 = stateless, non-zero = persistent)
//   [16:20] u32 system_prompt_length
//   [20:24] u32 user_prompt_length
//   [24:28] u32 reserved
//   [28..]  system_prompt + user_prompt (UTF-8, no null terminator)

// ==================== INCLUDES ====================

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#define NOMINMAX
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <algorithm>

namespace fs = std::filesystem;

// ==================== PROTOCOL CONSTANTS ====================

static constexpr uint16_t PROTO_VERSION = 2;

static constexpr uint16_t REQ_INFERENCE    = 0x0001;
static constexpr uint16_t REQ_HEALTH_CHECK = 0x0002;
static constexpr uint16_t REQ_SHUTDOWN     = 0x0003;

static constexpr uint16_t STATUS_OK                     = 0x0000;
static constexpr uint16_t STATUS_ERROR_MODEL_NOT_LOADED = 0x0001;
static constexpr uint16_t STATUS_ERROR_PROMPT_TOO_LONG  = 0x0002;
static constexpr uint16_t STATUS_ERROR_INFERENCE_FAILED = 0x0003;
static constexpr uint16_t STATUS_ERROR_TIMEOUT          = 0x0004;

static constexpr int REQUEST_HEADER_SIZE  = 28;
static constexpr int RESPONSE_HEADER_SIZE = 24;

// ==================== ENGINE PARAMETERS ====================

static constexpr int   N_CTX            = 32768;   // 32k context window
static constexpr int   N_GPU_LAYERS     = 33;       // all layers on RTX 3060
static constexpr int   N_BATCH          = 8192;  // must fit full system prompt in one decode call
static constexpr int   N_UBATCH         = 512;
static constexpr int   N_THREADS        = 6;
static constexpr int   MAX_SESSIONS     = 8;        // max concurrent sessions in RAM
static constexpr int   PIPE_BUFFER_SIZE = 131072;   // 128 KB (large enough for long responses)
static constexpr int   WALL_TIMEOUT_MS  = 300000;   // 5 min max per request
static constexpr float DEFAULT_TEMP     = 0.7f;
static constexpr int   IDLE_TIMEOUT_SECS = 300;     // unload model after 5 min idle

// ==================== SESSION MAP ====================
// Each session snapshots the KV state after generation so subsequent
// turns can restore exactly where the conversation left off.

struct Session {
    uint32_t             id;
    std::vector<uint8_t> kv_snapshot;  // serialized KV for seq 0
    int                  n_tokens;     // total token count at snapshot time
    uint64_t             last_used;    // LRU counter for eviction
};

static std::unordered_map<uint32_t, Session> g_sessions;
static uint64_t g_lru_counter = 0;

static void evict_oldest_session() {
    if (g_sessions.empty()) return;
    auto oldest = g_sessions.begin();
    for (auto it = g_sessions.begin(); it != g_sessions.end(); ++it) {
        if (it->second.last_used < oldest->second.last_used) oldest = it;
    }
    fprintf(stderr, "[REVAN-AGENT] Evicting session %u (LRU)\n", oldest->first);
    g_sessions.erase(oldest);
}

// ==================== GLOBALS ====================

static std::atomic<bool>   g_shutdown{false};
static std::atomic<bool>   g_abort_flag{false};
static llama_model*        g_model = nullptr;
static llama_context*      g_ctx   = nullptr;
static std::string         g_pipe_name;
static std::string         g_model_path;           // stored for lazy reload
static std::mutex          g_load_mutex;           // serialises load/unload vs watchdog
static std::atomic<bool>   g_request_active{false};// true while a request is in-flight
static std::atomic<time_t> g_last_request_time{0}; // UNIX time of last completed request

// ==================== LOGGING ====================

static void log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[REVAN-AGENT] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

static void log_err(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[REVAN-AGENT] ERROR: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

static void llama_log_cb(enum ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}

static bool abort_callback(void* data) {
    return ((std::atomic<bool>*)data)->load(std::memory_order_relaxed);
}

// ==================== UTILITY ====================

static inline int64_t elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

static inline uint16_t read_u16(const uint8_t* b) {
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}
static inline uint32_t read_u32(const uint8_t* b) {
    return (uint32_t)b[0]       | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static inline void write_u16(uint8_t* b, uint16_t v) {
    b[0] = (uint8_t)(v & 0xFF); b[1] = (uint8_t)(v >> 8);
}
static inline void write_u32(uint8_t* b, uint32_t v) {
    b[0] = (uint8_t)(v & 0xFF);       b[1] = (uint8_t)((v >> 8) & 0xFF);
    b[2] = (uint8_t)((v >> 16) & 0xFF); b[3] = (uint8_t)((v >> 24) & 0xFF);
}

// ==================== CHATML FORMATTERS ====================

// Full first turn: system (optional) + user + open assistant tag
static std::string chatml_first_turn(const std::string& sys, const std::string& usr) {
    std::string s;
    s.reserve(sys.size() + usr.size() + 100);
    if (!sys.empty()) {
        s += "<|im_start|>system\n";
        s += sys;
        s += "<|im_end|>\n";
    }
    s += "<|im_start|>user\n";
    s += usr;
    s += "<|im_end|>\n";
    s += "<|im_start|>assistant\n";
    return s;
}

// Continuation turn: close previous assistant, open new user+assistant
// Called when resuming a session (previous assistant tokens are already in KV)
static std::string chatml_continuation(const std::string& usr) {
    std::string s;
    s.reserve(usr.size() + 60);
    s += "\n<|im_start|>user\n";
    s += usr;
    s += "<|im_end|>\n";
    s += "<|im_start|>assistant\n";
    return s;
}

// ==================== BATCH HELPERS ====================

// Create a batch at explicit token positions (used when restoring sessions)
static llama_batch make_batch_at_pos(const llama_token* tokens, int n, int start_pos) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = start_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n - 1) ? 1 : 0;  // logits for last token only
    }
    batch.n_tokens = n;
    return batch;
}

// ==================== RESPONSE BUILDER ====================

static std::vector<uint8_t> build_response(
    uint16_t    status,
    uint16_t    tokens_gen,
    uint16_t    tokens_pp,
    uint32_t    pp_ms,
    uint32_t    gen_ms,
    const std::string& text
) {
    uint32_t msg_len = RESPONSE_HEADER_SIZE + (uint32_t)text.size();
    std::vector<uint8_t> buf(msg_len);
    write_u32(&buf[0],  msg_len);
    write_u16(&buf[4],  PROTO_VERSION);
    write_u16(&buf[6],  status);
    write_u16(&buf[8],  tokens_gen);
    write_u16(&buf[10], tokens_pp);
    write_u32(&buf[12], pp_ms);
    write_u32(&buf[16], gen_ms);
    write_u32(&buf[20], 0);  // reserved
    if (!text.empty()) memcpy(&buf[24], text.data(), text.size());
    return buf;
}

// ==================== MODEL LOADING ====================

static bool load_model(const std::string& model_path) {
    if (!fs::exists(model_path)) {
        log_err("Model not found: %s", model_path.c_str());
        return false;
    }

    log_info("Loading model: %s", model_path.c_str());
    auto t0 = std::chrono::steady_clock::now();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = N_GPU_LAYERS;  // all layers on GPU

    g_model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!g_model) {
        log_err("Failed to load model (VRAM insufficient or file corrupt?)");
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx              = N_CTX;
    cp.n_threads          = N_THREADS;
    cp.n_threads_batch    = N_THREADS;
    cp.n_batch            = N_BATCH;
    cp.n_ubatch           = N_UBATCH;
    cp.flash_attn_type    = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv        = false;        // KV cache lives in system RAM, not VRAM
    cp.swa_full           = false;
    cp.type_k             = GGML_TYPE_Q4_0; // q4_0 K: halves attention read bandwidth (+55% tg, +82% mixed vs f16/f16)
    cp.type_v             = GGML_TYPE_F16;  // f16 V: quantizing V tanks tg speed, keep full precision
    cp.abort_callback      = abort_callback;
    cp.abort_callback_data = &g_abort_flag;

    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) {
        log_err("Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return false;
    }

    log_info("Model loaded in %.1fs | ctx=%d | kv_in_ram=true | ngl=%d",
             elapsed_ms(t0) / 1000.0f, N_CTX, N_GPU_LAYERS);
    return true;
}

static void unload_model() {
    // Caller must hold g_load_mutex.
    if (!g_model && !g_ctx) return;
    if (g_ctx)   { llama_free(g_ctx);        g_ctx   = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    log_info("Model unloaded (idle)");
}

static void warmup();  // forward declaration — defined after ensure_loaded

// Load model if not already loaded. Must be called with g_request_active=true
// set by the caller (prevents the watchdog from racing in and unloading).
static bool ensure_loaded() {
    std::lock_guard<std::mutex> lk(g_load_mutex);
    if (g_model && g_ctx) return true;   // already up
    if (!load_model(g_model_path)) return false;
    warmup();
    return true;
}

// ==================== WARM-UP ====================

static void warmup() {
    log_info("Warming up CUDA kernels...");
    auto t0 = std::chrono::steady_clock::now();

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    const char* text = "<|im_start|>system\nReady.<|im_end|>\n"
                       "<|im_start|>user\nPing.<|im_end|>\n"
                       "<|im_start|>assistant\n";

    std::vector<llama_token> toks(256);
    int n = llama_tokenize(vocab, text, (int)strlen(text),
                           toks.data(), (int)toks.size(), true, true);
    if (n > 0) {
        toks.resize(n);
        llama_batch batch = llama_batch_get_one(toks.data(), n);
        llama_decode(g_ctx, batch);
    }

    llama_memory_t mem = llama_get_memory(g_ctx);
    if (mem) llama_memory_clear(mem, false);

    log_info("Warm-up done (%.1fs)", elapsed_ms(t0) / 1000.0f);
}

// ==================== INFERENCE ====================

struct InferResult {
    uint16_t    status    = STATUS_OK;
    std::string output;
    uint16_t    tokens_gen = 0;
    uint16_t    tokens_pp  = 0;
    uint32_t    pp_ms      = 0;
    uint32_t    gen_ms     = 0;
};

static InferResult run_inference(
    uint32_t    session_id,
    const std::string& system_prompt,
    const std::string& user_prompt,
    int         max_tokens,
    float       temperature
) {
    InferResult res;

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    llama_memory_t     mem   = llama_get_memory(g_ctx);

    g_abort_flag.store(false);

    auto t_total = std::chrono::steady_clock::now();

    // ---- Determine context restore position ----
    int  start_pos       = 0;
    bool is_continuation = false;

    if (session_id != 0) {
        auto it = g_sessions.find(session_id);
        if (it != g_sessions.end()) {
            Session& sess = it->second;
            if (mem) llama_memory_clear(mem, false);

            size_t restored = llama_state_seq_set_data(
                g_ctx, sess.kv_snapshot.data(), sess.kv_snapshot.size(), 0);

            if (restored > 0) {
                start_pos       = sess.n_tokens;
                is_continuation = true;
                sess.last_used  = ++g_lru_counter;
                log_info("Session %u restored (pos=%d, %zuB)", session_id, start_pos, restored);
            } else {
                log_err("Session %u KV restore failed, starting fresh", session_id);
                if (mem) llama_memory_clear(mem, false);
            }
        } else {
            if (mem) llama_memory_clear(mem, false);
            log_info("Session %u: new session", session_id);
        }
    } else {
        if (mem) llama_memory_clear(mem, false);
    }

    // ---- Build and tokenize prompt ----
    std::string prompt = is_continuation
        ? chatml_continuation(user_prompt)
        : chatml_first_turn(system_prompt, user_prompt);

    std::vector<llama_token> prompt_tokens(N_CTX);
    int n_prompt = llama_tokenize(
        vocab, prompt.c_str(), (int)prompt.size(),
        prompt_tokens.data(), (int)prompt_tokens.size(),
        !is_continuation,  // add BOS only on the first turn
        true               // special tokens
    );

    if (n_prompt < 0) {
        log_err("Tokenize failed (code %d)", n_prompt);
        res.status = STATUS_ERROR_INFERENCE_FAILED;
        return res;
    }
    prompt_tokens.resize(n_prompt);

    int total_prompt = start_pos + n_prompt;
    int ctx_size     = (int)llama_n_ctx(g_ctx);

    if (total_prompt >= ctx_size - 1) {
        log_err("Context full: %d tokens (ctx=%d)", total_prompt, ctx_size);
        res.status = STATUS_ERROR_PROMPT_TOO_LONG;
        return res;
    }

    // ---- Decode prompt in N_BATCH-sized chunks ----
    auto t_pp = std::chrono::steady_clock::now();
    int ret   = 0;

    for (int off = 0; off < n_prompt && ret == 0; off += N_BATCH) {
        int chunk = std::min(N_BATCH, n_prompt - off);
        llama_batch pb = make_batch_at_pos(prompt_tokens.data() + off, chunk, start_pos + off);
        ret = llama_decode(g_ctx, pb);
        llama_batch_free(pb);
    }
    llama_synchronize(g_ctx);

    res.pp_ms     = (uint32_t)elapsed_ms(t_pp);
    res.tokens_pp = (uint16_t)total_prompt;

    if (ret != 0) {
        log_err("Prompt decode failed (code %d)", ret);
        res.status = STATUS_ERROR_INFERENCE_FAILED;
        if (mem) llama_memory_clear(mem, false);
        return res;
    }

    // ---- Sampler chain ----
    llama_sampler_chain_params chain_p = llama_sampler_chain_default_params();
    chain_p.no_perf = true;

    llama_sampler* smpl = llama_sampler_chain_init(chain_p);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ---- Generation loop ----
    auto t_gen = std::chrono::steady_clock::now();

    std::string output;
    int  generated   = 0;
    int  current_pos = total_prompt;
    int  limit = (max_tokens == 0) ? (ctx_size - total_prompt - 4) : max_tokens;

    for (int i = 0; i < limit; i++) {
        llama_token tok = llama_sampler_sample(smpl, g_ctx, -1);

        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }

        // Detokenize
        char piece[256];
        int  plen = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, false);
        if (plen > 0) output.append(piece, plen);
        generated++;

        // Wall-clock safety
        if (elapsed_ms(t_total) > WALL_TIMEOUT_MS) {
            log_err("Wall timeout (%ds)", WALL_TIMEOUT_MS / 1000);
            res.status = STATUS_ERROR_TIMEOUT;
            break;
        }

        // Decode generated token for next iteration
        llama_batch nb = llama_batch_get_one(&tok, 1);
        ret = llama_decode(g_ctx, nb);
        current_pos++;

        if (ret != 0) {
            res.status = (ret == 2) ? STATUS_ERROR_TIMEOUT : STATUS_ERROR_INFERENCE_FAILED;
            break;
        }
    }

    res.gen_ms     = (uint32_t)elapsed_ms(t_gen);
    res.output     = output;
    res.tokens_gen = (uint16_t)generated;

    llama_sampler_free(smpl);

    // ---- Snapshot session ----
    if (session_id != 0 && res.status == STATUS_OK) {
        // Evict LRU if at capacity
        if ((int)g_sessions.size() >= MAX_SESSIONS &&
            g_sessions.find(session_id) == g_sessions.end()) {
            evict_oldest_session();
        }

        int snapshot_pos = total_prompt + generated;
        size_t state_size = llama_state_seq_get_size(g_ctx, 0);
        if (state_size > 0) {
            std::vector<uint8_t> snap(state_size);
            size_t written = llama_state_seq_get_data(g_ctx, snap.data(), snap.size(), 0);
            if (written > 0) {
                snap.resize(written);
                Session& sess    = g_sessions[session_id];
                sess.id          = session_id;
                sess.kv_snapshot = std::move(snap);
                sess.n_tokens    = snapshot_pos;
                sess.last_used   = ++g_lru_counter;
                log_info("Session %u snapshot: pos=%d size=%.1fMB",
                         session_id, snapshot_pos, written / 1048576.0f);
            }
        }
    } else if (session_id == 0) {
        if (mem) llama_memory_clear(mem, false);
    }

    return res;
}

// ==================== CTRL+C HANDLER ====================

static BOOL WINAPI ctrl_handler(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_CLOSE_EVENT) {
        fprintf(stderr, "[REVAN-AGENT] Shutdown signal\n");
        g_shutdown.store(true);
        g_abort_flag.store(true);
        if (!g_pipe_name.empty()) {
            HANDLE h = CreateFileA(g_pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE,
                                   0, NULL, OPEN_EXISTING, 0, NULL);
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        }
        return TRUE;
    }
    return FALSE;
}

// ==================== REQUEST PARSING ====================

struct ParsedRequest {
    uint16_t    request_type = 0;
    uint16_t    max_tokens   = 0;
    float       temperature  = DEFAULT_TEMP;
    uint32_t    session_id   = 0;
    std::string system_prompt;
    std::string user_prompt;
    bool        valid = false;
};

static ParsedRequest parse_request(const uint8_t* data, DWORD size) {
    ParsedRequest req;

    if (size < (DWORD)REQUEST_HEADER_SIZE) {
        log_err("Request too short: %lu bytes", size);
        return req;
    }

    // [0:4] msg_len — validate
    uint32_t msg_len = read_u32(&data[0]);
    if (msg_len > size) {
        log_err("msg_len %u > buffer %lu", msg_len, size);
        return req;
    }

    uint16_t version = read_u16(&data[4]);
    if (version != PROTO_VERSION) {
        log_err("Unknown protocol version %u (expected %u)", version, PROTO_VERSION);
        return req;
    }

    req.request_type = read_u16(&data[6]);
    req.max_tokens   = read_u16(&data[8]);
    uint16_t temp_x100 = read_u16(&data[10]);
    req.session_id   = read_u32(&data[12]);
    uint32_t sys_len = read_u32(&data[16]);
    uint32_t usr_len = read_u32(&data[20]);
    // [24:28] reserved

    req.temperature = (temp_x100 > 0) ? (temp_x100 / 100.0f) : DEFAULT_TEMP;

    if (req.request_type == REQ_HEALTH_CHECK || req.request_type == REQ_SHUTDOWN) {
        req.valid = true;
        return req;
    }

    // Validate payload lengths
    uint64_t payload_needed = (uint64_t)REQUEST_HEADER_SIZE + sys_len + usr_len;
    if (payload_needed > size) {
        log_err("Payload length overflow: sys=%u usr=%u", sys_len, usr_len);
        return req;
    }

    if (sys_len > 0)
        req.system_prompt.assign((const char*)&data[REQUEST_HEADER_SIZE], sys_len);
    if (usr_len > 0)
        req.user_prompt.assign((const char*)&data[REQUEST_HEADER_SIZE + sys_len], usr_len);

    req.valid = true;
    return req;
}

// ==================== IDLE WATCHDOG ====================
// Checks every 30 s; unloads model if idle > IDLE_TIMEOUT_SECS.
// Session KV snapshots stay in RAM — they survive unload and are
// restored automatically on the next request.

static void watchdog_thread_fn() {
    while (!g_shutdown.load()) {
        // Sleep in 1-second ticks so we respond to shutdown quickly.
        for (int i = 0; i < 30 && !g_shutdown.load(); i++)
            std::this_thread::sleep_for(std::chrono::seconds(1));

        if (g_shutdown.load()) break;
        if (g_request_active.load()) continue;   // inference in-flight, skip

        time_t now  = time(nullptr);
        time_t last = g_last_request_time.load();
        if (last == 0) continue;                 // never had a request yet

        if (now - last < IDLE_TIMEOUT_SECS) continue;

        // Double-check inside the mutex to avoid racing with ensure_loaded().
        std::lock_guard<std::mutex> lk(g_load_mutex);
        if (g_request_active.load()) continue;
        unload_model();
    }
}

// ==================== PIPE SERVER ====================

static int run_pipe_server(const std::string& pipe_name) {
    g_pipe_name = pipe_name;

    HANDLE pipe = CreateNamedPipeA(
        pipe_name.c_str(),
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
        1,                   // max instances = 1 (single inference thread)
        PIPE_BUFFER_SIZE,
        PIPE_BUFFER_SIZE,
        0,
        NULL
    );

    if (pipe == INVALID_HANDLE_VALUE) {
        log_err("CreateNamedPipe failed: %lu", GetLastError());
        return 1;
    }

    log_info("Pipe ready: %s", pipe_name.c_str());

    // Buffer large enough for long responses (generation can be many KB)
    std::vector<uint8_t> read_buf(PIPE_BUFFER_SIZE);

    while (!g_shutdown.load()) {
        BOOL connected = ConnectNamedPipe(pipe, NULL);
        if (!connected && GetLastError() != ERROR_PIPE_CONNECTED) {
            if (g_shutdown.load()) break;
            log_err("ConnectNamedPipe: %lu", GetLastError());
            continue;
        }

        DWORD bytes_read = 0;
        BOOL  read_ok    = ReadFile(pipe, read_buf.data(), (DWORD)read_buf.size(),
                                    &bytes_read, NULL);

        if (!read_ok || bytes_read == 0) {
            FlushFileBuffers(pipe);
            DisconnectNamedPipe(pipe);
            continue;
        }

        ParsedRequest req = parse_request(read_buf.data(), bytes_read);

        auto send_and_disc = [&](const std::vector<uint8_t>& resp) {
            WriteFile(pipe, resp.data(), (DWORD)resp.size(), nullptr, NULL);
            FlushFileBuffers(pipe);
            DisconnectNamedPipe(pipe);
        };

        if (!req.valid) {
            send_and_disc(build_response(STATUS_ERROR_INFERENCE_FAILED, 0, 0, 0, 0, ""));
            continue;
        }

        if (req.request_type == REQ_SHUTDOWN) {
            send_and_disc(build_response(STATUS_OK, 0, 0, 0, 0, ""));
            g_shutdown.store(true);
            break;
        }

        if (req.request_type == REQ_HEALTH_CHECK) {
            uint16_t s = (g_model && g_ctx) ? STATUS_OK : STATUS_ERROR_MODEL_NOT_LOADED;
            send_and_disc(build_response(s, 0, 0, 0, 0, ""));
            continue;
        }

        if (req.request_type == REQ_INFERENCE) {
            g_request_active.store(true);

            if (!ensure_loaded()) {
                g_request_active.store(false);
                g_last_request_time.store(time(nullptr));
                send_and_disc(build_response(STATUS_ERROR_MODEL_NOT_LOADED, 0, 0, 0, 0, ""));
                continue;
            }

            log_info("REQ session=%u sys=%zuB usr=%zuB max_tok=%u temp=%.2f",
                     req.session_id, req.system_prompt.size(), req.user_prompt.size(),
                     req.max_tokens, req.temperature);

            InferResult r = run_inference(
                req.session_id,
                req.system_prompt,
                req.user_prompt,
                (int)req.max_tokens,
                req.temperature
            );

            g_last_request_time.store(time(nullptr));
            g_request_active.store(false);

            if (r.status == STATUS_OK) {
                float pp_rate = (r.pp_ms > 0)  ? (r.tokens_pp  * 1000.0f / r.pp_ms)  : 0.0f;
                float tg_rate = (r.gen_ms > 0) ? (r.tokens_gen * 1000.0f / r.gen_ms) : 0.0f;
                log_info("DONE gen=%dtok pp=%.0ft/s tg=%.0ft/s out_len=%zu",
                         r.tokens_gen, pp_rate, tg_rate, r.output.size());
            } else {
                log_err("FAIL status=0x%04X", r.status);
            }

            send_and_disc(build_response(r.status, r.tokens_gen, r.tokens_pp,
                                         r.pp_ms, r.gen_ms, r.output));
            continue;
        }

        // Unknown request type
        log_err("Unknown request type 0x%04X", req.request_type);
        send_and_disc(build_response(STATUS_ERROR_INFERENCE_FAILED, 0, 0, 0, 0, ""));
    }

    CloseHandle(pipe);
    return 0;
}

// ==================== CONFIG (minimal TOML parser) ====================

struct Config {
    std::string model_path;
    std::string pipe_name = "\\\\.\\pipe\\revan-agent";
};

static bool parse_config(const std::string& path, Config& cfg) {
    std::ifstream f(path);
    if (!f.is_open()) {
        log_err("Cannot open config: %s", path.c_str());
        return false;
    }

    std::string line, section;
    while (std::getline(f, line)) {
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos || line[a] == '#') continue;
        line = line.substr(a);

        if (line[0] == '[') {
            size_t e = line.find(']');
            if (e != std::string::npos) section = line.substr(1, e - 1);
            continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        auto trim = [](std::string& s) {
            size_t a = s.find_first_not_of(" \t\"");
            size_t b = s.find_last_not_of(" \t\"");
            if (a == std::string::npos) { s.clear(); return; }
            s = s.substr(a, b - a + 1);
        };
        trim(key); trim(val);

        if (section == "model" && key == "path") cfg.model_path = val;
        if (section == "pipe"  && key == "name") cfg.pipe_name  = val;
    }
    return true;
}

// ==================== MAIN ====================

int main(int argc, char* argv[]) {
    std::string config_path;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "-c") == 0) && i + 1 < argc)
            config_path = argv[++i];
    }

    // Auto-discover config file
    if (config_path.empty()) {
        if (fs::exists("revan-agent.toml"))
            config_path = "revan-agent.toml";
        else {
            fs::path exe_dir  = fs::path(argv[0]).parent_path();
            fs::path cand     = exe_dir / "revan-agent.toml";
            if (fs::exists(cand)) config_path = cand.string();
        }
    }

    if (config_path.empty()) {
        log_err("No config file found. Use --config <path> or place revan-agent.toml in the exe directory.");
        return 1;
    }

    Config cfg;
    if (!parse_config(fs::absolute(config_path).string(), cfg)) return 1;

    if (cfg.model_path.empty()) {
        log_err("model.path not set in config");
        return 1;
    }

    // Resolve relative model path against config's directory
    fs::path mp(cfg.model_path);
    if (!mp.is_absolute()) {
        cfg.model_path = (fs::absolute(config_path).parent_path() / mp).string();
    }

    log_info("Config: model=%s pipe=%s idle_timeout=%ds",
             cfg.model_path.c_str(), cfg.pipe_name.c_str(), IDLE_TIMEOUT_SECS);

    // ---- Initialize backends ----
    // Load ggml backend DLLs (ggml-cuda.dll, ggml-cpu-*.dll, ggml-rpc.dll)
    // from the same directory as the executable.
    {
        char exe_path[MAX_PATH] = {};
        GetModuleFileNameA(NULL, exe_path, MAX_PATH);
        fs::path exe_dir = fs::path(exe_path).parent_path();
        ggml_backend_load_all_from_path(exe_dir.string().c_str());
    }

    llama_log_set(llama_log_cb, nullptr);
    llama_backend_init();
    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    // Store model path for lazy loading — model loads on first request.
    g_model_path = cfg.model_path;
    log_info("Lazy mode: model will load on first request, unload after %ds idle",
             IDLE_TIMEOUT_SECS);

    // Start idle watchdog thread.
    std::thread watchdog(watchdog_thread_fn);

    int exit_code = run_pipe_server(cfg.pipe_name);

    watchdog.join();

    log_info("Shutting down...");
    if (g_ctx)   { llama_free(g_ctx);        g_ctx   = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    llama_backend_free();

    log_info("Shutdown complete (exit=%d)", exit_code);
    return exit_code;
}
