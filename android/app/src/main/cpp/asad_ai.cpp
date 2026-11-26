#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <algorithm>
#include "llama.h"

static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static const llama_vocab * g_vocab = nullptr;
static std::mutex g_mutex;

static std::string from_jstring(JNIEnv * env, jstring value) {
    if (!value) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

static std::string remove_thinking(const std::string & input) {
    std::string output;
    size_t pos = 0;
    while (pos < input.size()) {
        size_t start = input.find("<think>", pos);
        if (start == std::string::npos) {
            output.append(input.substr(pos));
            break;
        }
        output.append(input.substr(pos, start - pos));
        size_t end = input.find("</think>", start + 7);
        if (end == std::string::npos) break;
        pos = end + 8;
    }
    return output;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_asad_ai_NativeAI_nativeLoadModel(JNIEnv * env, jclass, jstring path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const std::string model_path = from_jstring(env, path);
    if (model_path.empty()) return JNI_FALSE;

    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_vocab = nullptr;

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // Stable CPU build; Vulkan is tested separately.
    g_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!g_model) return JNI_FALSE;

    unsigned int cores = std::thread::hardware_concurrency();
    int threads = cores == 0 ? 4 : (int)cores - 2;
    threads = std::max(2, std::min(4, threads));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 256;
    ctx_params.n_threads = threads;
    ctx_params.n_threads_batch = threads;
    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }
    g_vocab = llama_model_get_vocab(g_model);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_asad_ai_NativeAI_nativeGenerate(JNIEnv * env, jclass, jstring question, jint max_tokens) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx || !g_vocab)
        return env->NewStringUTF("Avval GGUF modelni yuklang.");

    const std::string q = from_jstring(env, question);
    if (q.empty()) return env->NewStringUTF("Savol bo‘sh.");

    const std::string prompt =
        "<|im_start|>system\nSen Asad AI nomli foydali yordamchisan. O‘zbek tilida aniq, tabiiy va to‘liq javob ber. Keraksiz takrorlardan qoch. /no_think\n<|im_end|>\n"
        "<|im_start|>user\n" + q + "\n<|im_end|>\n"
        "<|im_start|>assistant\n";

    int32_t needed = -llama_tokenize(g_vocab, prompt.c_str(),
        (int32_t) prompt.size(), nullptr, 0, true, true);
    if (needed <= 0) return env->NewStringUTF("Matnni tokenlashda xato.");

    std::vector<llama_token> tokens((size_t) needed);
    int32_t count = llama_tokenize(g_vocab, prompt.c_str(),
        (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), true, true);
    if (count < 0) return env->NewStringUTF("Tokenlashda xato.");
    tokens.resize((size_t) count);

    llama_memory_clear(llama_get_memory(g_ctx), true);
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(g_ctx, batch) != 0)
        return env->NewStringUTF("Model promptni qayta ishlay olmadi.");

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.12f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.90f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string answer;
    const int limit = max_tokens > 0 ? std::min(512, (int)max_tokens) : 512;
    for (int i = 0; i < limit; ++i) {
        llama_token token = llama_sampler_sample(sampler, g_ctx, -1);
        if (llama_vocab_is_eog(g_vocab, token)) break;
        char piece[512];
        int32_t n = llama_token_to_piece(g_vocab, token, piece, sizeof(piece), 0, true);
        if (n > 0) answer.append(piece, (size_t) n);
        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(g_ctx, batch) != 0) break;
    }
    llama_sampler_free(sampler);
    answer = remove_thinking(answer);
    return env->NewStringUTF(answer.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_asad_ai_NativeAI_nativeFree(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_vocab = nullptr;
}
