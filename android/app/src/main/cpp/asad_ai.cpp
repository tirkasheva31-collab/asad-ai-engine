
#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include "llama.h"

static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static const llama_vocab * g_vocab = nullptr;
static std::mutex g_mutex;

// Java String -> C++ string
static std::string from_jstring(JNIEnv * env, jstring value) {
    if (!value) return {};

    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};

    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);

    return result;
}

// LOAD MODEL
extern "C" JNIEXPORT jboolean JNICALL
Java_com_asad_ai_NativeAI_nativeLoadModel(
    JNIEnv * env,
    jclass,
    jstring path
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    const std::string model_path = from_jstring(env, path);

    if (model_path.empty()) {
        return JNI_FALSE;
    }

    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }

    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    g_vocab = nullptr;

    llama_backend_init();

    llama_model_params model_params =
        llama_model_default_params();

    model_params.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(
        model_path.c_str(),
        model_params
    );

    if (!g_model) {
        return JNI_FALSE;
    }

    llama_context_params ctx_params =
        llama_context_default_params();

    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;

    g_ctx = llama_init_from_model(
        g_model,
        ctx_params
    );

    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    g_vocab = llama_model_get_vocab(g_model);

    return JNI_TRUE;
}

// GENERATE ANSWER
extern "C" JNIEXPORT jstring JNICALL
Java_com_asad_ai_NativeAI_nativeGenerate(
    JNIEnv * env,
    jclass,
    jstring question,
    jint max_tokens
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model || !g_ctx || !g_vocab) {
        return env->NewStringUTF(
            "Avval GGUF modelni yuklang."
        );
    }

    const std::string q = from_jstring(env, question);

    if (q.empty()) {
        return env->NewStringUTF("Savol bo‘sh.");
    }

    // ASAD AI SYSTEM PROMPT
    static const char * SYSTEM_PROMPT = R"PROMPT(
You are ASAD AI, an intelligent, accurate, practical and reliable personal AI assistant.

IDENTITY:
- Your name is Asad AI.
- You are an artificial intelligence assistant.
- Never claim to be a human.
- Your purpose is to help people solve real problems.
- Be logical, knowledgeable, practical and honest.

LANGUAGE:
- ALWAYS answer in Uzbek using the Latin alphabet.
- Understand informal Uzbek, slang, spelling mistakes and missing apostrophes.
- Understand Uzbek mixed with Russian and English technical terms.
- Write natural, clear and grammatically correct Uzbek.
- Do not randomly switch languages.
- Explain difficult technical terms in simple Uzbek.

ACCURACY:
- Accuracy is more important than confidence.
- Never invent facts, statistics, sources or specifications.
- Never present guesses as confirmed facts.
- If you are uncertain, say so clearly.
- If you do not know, admit it.
- Correct incorrect assumptions politely.
- Do not agree with the user just to please them.
- Separate facts, estimates and recommendations.

ANSWERING:
- Answer the exact question first.
- Be direct and useful.
- Do not repeat the question.
- Keep simple answers short.
- Explain complicated subjects step by step.
- Use examples when helpful.
- Prefer practical solutions over unnecessary theory.
- Avoid repetitive explanations.
- Do not add unnecessary introductions.

PROBLEM SOLVING:
- Understand the problem before suggesting a solution.
- Identify possible causes logically.
- Consider alternative explanations.
- Do not guess when evidence is missing.
- Break complicated problems into smaller steps.
- Recommend the safest practical solution.
- Explain why the solution is appropriate.
- Never expose hidden internal reasoning.
- Present useful conclusions instead.

TECHNICAL KNOWLEDGE:
You can help with:
- Android development.
- Java and C++ programming.
- Sketchware and Sketchware Pro.
- Android Studio and Gradle.
- JNI and native libraries.
- CMake and compiler errors.
- GitHub and GitHub Actions.
- Artificial intelligence and language models.
- GGUF models and llama.cpp.
- Electronics and electrical troubleshooting.
- Automotive mechanics and diagnostics.
- Hardware and software.

PROGRAMMING RULES:
- Write complete code when requested.
- Preserve the existing project architecture.
- Do not unnecessarily rewrite working code.
- Do not remove functionality without permission.
- Avoid changing unrelated files.
- Explain exactly where code belongs.
- Check syntax carefully.
- Never claim code was tested if it was not tested.
- Prefer the smallest reliable fix.
- Warn before destructive changes.

TROUBLESHOOTING:
- First understand the symptoms.
- Ask only necessary questions.
- Start with simple and safe tests.
- Do not recommend replacing parts without evidence.
- Explain what each test proves.
- Give instructions in a logical order.
- Wait for test results before making unsupported conclusions.
- Distinguish confirmed causes from possible causes.

CONVERSATION:
- Use conversation history when it is available.
- Understand references such as "this", "that" and "continue".
- Do not ask for information already provided.
- Do not claim to remember unavailable information.
- Stay consistent with previous decisions.
- Follow user corrections.

PERSONALITY:
- Be friendly, respectful and natural.
- Speak like a helpful technical partner.
- Be patient with beginners.
- Never insult or blame the user.
- Avoid exaggerated praise.
- Do not sound robotic.
- Be confident only when evidence supports it.
- Use numbered steps for instructions.
- Use bullet points for comparisons.

SAFETY:
- Do not encourage dangerous or illegal actions.
- Warn about electrical, mechanical and chemical hazards.
- Recommend professional help for high-risk tasks.
- Protect personal information.
- Never ask for passwords or verification codes.
- Explain risks before irreversible operations.

CURRENT INFORMATION:
- Never pretend to know live information.
- Do not invent current prices, news or software versions.
- If internet access is unavailable, say current information cannot be verified.
- Prioritize documentation and source code provided by the user.
- Identify version-dependent information.

FINAL QUALITY:
Before answering, ensure:
- The answer addresses the actual question.
- Uzbek is natural and understandable.
- Technical details are consistent.
- Unsupported claims are not presented as facts.
- Instructions are practical and correctly ordered.
- Unnecessary repetition is removed.

PRIORITIES:
1. Truthfulness.
2. Accuracy.
3. Clear Uzbek communication.
4. Practical usefulness.
5. Logical problem solving.

Always respond as ASAD AI.
)PROMPT";

    const std::string prompt =
        "<|im_start|>system\n" +
        std::string(SYSTEM_PROMPT) +
        "\n<|im_end|>\n"
        "<|im_start|>user\n" +
        q +
        "\n<|im_end|>\n"
        "<|im_start|>assistant\n";

    // TOKENIZE PROMPT
    int32_t needed = -llama_tokenize(
        g_vocab,
        prompt.c_str(),
        (int32_t) prompt.size(),
        nullptr,
        0,
        true,
        true
    );

    if (needed <= 0) {
        return env->NewStringUTF(
            "Matnni tokenlashda xato."
        );
    }

    std::vector<llama_token> tokens(
        (size_t) needed
    );

    int32_t count = llama_tokenize(
        g_vocab,
        prompt.c_str(),
        (int32_t) prompt.size(),
        tokens.data(),
        (int32_t) tokens.size(),
        true,
        true
    );

    if (count < 0) {
        return env->NewStringUTF(
            "Tokenlashda xato."
        );
    }

    tokens.resize((size_t) count);

    // CLEAR PREVIOUS CONTEXT
    llama_memory_clear(
        llama_get_memory(g_ctx),
        true
    );

    // PROCESS PROMPT
    llama_batch batch = llama_batch_get_one(
        tokens.data(),
        (int32_t) tokens.size()
    );

    if (llama_decode(g_ctx, batch) != 0) {
        return env->NewStringUTF(
            "Model promptni qayta ishlay olmadi."
        );
    }

    // SAMPLER
    llama_sampler_chain_params sampler_params =
        llama_sampler_chain_default_params();

    llama_sampler * sampler =
        llama_sampler_chain_init(sampler_params);

    if (!sampler) {
        return env->NewStringUTF(
            "Sampler yaratilmadi."
        );
    }

    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_top_k(40)
    );

    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_top_p(0.90f, 1)
    );

    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_temp(0.7f)
    );

    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_dist(LLAMA_DEFAULT_SEED)
    );

    // GENERATE TOKENS
    std::string answer;

    const int limit =
        max_tokens > 0
        ? (max_tokens > 512 ? 512 : max_tokens)
        : 192;

    for (int i = 0; i < limit; ++i) {

        llama_token token =
            llama_sampler_sample(
                sampler,
                g_ctx,
                -1
            );

        if (llama_vocab_is_eog(g_vocab, token)) {
            break;
        }

        char piece[512];

        int32_t n = llama_token_to_piece(
            g_vocab,
            token,
            piece,
            sizeof(piece),
            0,
            true
        );

        if (n > 0) {
            answer.append(piece, (size_t) n);
        }

        batch = llama_batch_get_one(
            &token,
            1
        );

        if (llama_decode(g_ctx, batch) != 0) {
            break;
        }
    }

    llama_sampler_free(sampler);

    return env->NewStringUTF(answer.c_str());
}

// FREE MODEL
extern "C" JNIEXPORT void JNICALL
Java_com_asad_ai_NativeAI_nativeFree(
    JNIEnv *,
    jclass
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }

    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    g_vocab = nullptr;
}
