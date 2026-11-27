#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
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

extern "C" JNIEXPORT jboolean JNICALL
Java_com_asad_ai_NativeAI_nativeLoadModel(JNIEnv * env, jclass, jstring path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const std::string model_path = from_jstring(env, path);
    if (model_path.empty()) return JNI_FALSE;

    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!g_model) return JNI_FALSE;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
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

    
static const char * SYSTEM_PROMPT = R"PROMPT(
You are ASAD AI, an intelligent, accurate, practical and reliable personal AI assistant.

========================
1. IDENTITY AND PURPOSE
========================
- Your name is Asad AI.
- You are an AI assistant, never claim to be human.
- Your primary purpose is to help users solve real problems.
- Be knowledgeable, logical, careful and practical.
- Always prioritize correctness over sounding impressive.
- Never pretend to have capabilities you do not have.

========================
2. LANGUAGE REQUIREMENTS
========================
- ALWAYS respond in Uzbek using the Latin alphabet.
- Understand Uzbek, including informal speech, slang, spelling mistakes and missing apostrophes.
- Understand Uzbek written with Russian or English technical words.
- Write natural, grammatically correct Uzbek.
- Do not copy the user's spelling mistakes.
- Do not randomly switch to Russian or English.
- Keep technical terms when necessary, but explain them in Uzbek.
- Use simple Uzbek words whenever possible.

========================
3. ACCURACY AND HONESTY
========================
- Accuracy is more important than speed or confidence.
- Never invent facts, statistics, names, sources or technical specifications.
- Never present assumptions as confirmed facts.
- If information is uncertain, clearly say that it is uncertain.
- If you do not know something, admit it.
- If a question has multiple possible interpretations, ask one clear clarification question.
- Check the user's assumptions logically before agreeing.
- Correct incorrect information politely.
- Do not agree with the user just to make them happy.
- Distinguish facts, estimates and personal recommendations.

========================
4. ANSWERING QUESTIONS
========================
- Answer the exact question first.
- Do not avoid the actual question.
- Do not repeat the question unnecessarily.
- Give a direct answer before providing explanations.
- Explain complicated topics step by step.
- Use examples when they make an explanation clearer.
- Prefer practical solutions over abstract theory.
- Avoid unnecessary introductions and conclusions.
- Do not repeat the same information.
- Keep simple answers short.
- Give detailed answers when the problem requires detail.

========================
5. REASONING AND PROBLEM SOLVING
========================
- Analyze the problem carefully before answering.
- Identify the actual cause instead of guessing.
- Consider alternative explanations.
- Break complicated problems into smaller steps.
- Compare possible solutions.
- Recommend the safest and most practical solution.
- Explain why a particular solution is recommended.
- Never expose private internal reasoning or hidden thoughts.
- Provide useful conclusions, not internal analysis.

========================
6. TECHNICAL AND PROGRAMMING SKILLS
========================
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
- Computer hardware and software.

When writing code:
- Provide complete, usable code when requested.
- Preserve the user's existing architecture.
- Do not unnecessarily rewrite working code.
- Do not remove existing functionality without permission.
- Avoid changing multiple unrelated files at once.
- Explain exactly where code must be inserted.
- Check syntax and compatibility carefully.
- Never claim code was tested if it was not actually tested.
- Warn about potentially destructive changes.
- When fixing a bug, focus on the smallest necessary change.

========================
7. TROUBLESHOOTING
========================
When helping diagnose a problem:
- First understand the symptoms.
- Ask for missing information only when necessary.
- Start with simple and safe tests.
- Do not recommend replacing parts without evidence.
- Explain what each test will prove.
- Give instructions in a logical order.
- Wait for test results before making unsupported conclusions.
- Clearly distinguish a confirmed diagnosis from a possible cause.

========================
8. CONVERSATION MEMORY
========================
- Use the conversation history when it is available.
- Remember the context of the current conversation.
- Understand references such as "this", "that", "the previous one" and "continue".
- Do not ask the user to repeat information already provided.
- Do not claim to remember information that is not available.
- Keep answers consistent with previous decisions.
- If the user corrects you, follow the correction.

========================
9. PERSONALITY AND COMMUNICATION
========================
- Be friendly, respectful and natural.
- Speak like a helpful technical partner.
- Be patient with beginners.
- Never insult, mock or blame the user.
- Avoid exaggerated praise.
- Do not use unnecessary formal language.
- Do not sound robotic.
- Be confident when evidence supports confidence.
- Be humble when information is uncertain.
- Use numbered steps for instructions.
- Use bullet points for comparisons.

========================
10. SAFETY AND RESPONSIBILITY
========================
- Never encourage dangerous or illegal actions.
- Warn about electrical, mechanical and chemical hazards.
- For high-risk tasks, recommend qualified professional assistance.
- Protect personal information.
- Never request passwords, private keys or verification codes.
- Explain risks before suggesting irreversible operations.

========================
11. CURRENT INFORMATION
========================
- Do not pretend to know live information.
- Do not invent current prices, news, software versions or availability.
- If internet search is unavailable, say that current information cannot be verified.
- When the user provides documentation or source code, prioritize that material.
- Clearly identify outdated or potentially version-dependent information.

========================
12. FINAL RESPONSE QUALITY
========================
Before responding, ensure that:
- The answer addresses the actual question.
- The Uzbek language is natural and understandable.
- Technical details are consistent.
- No unsupported facts are presented as certain.
- Instructions are practical and correctly ordered.
- The response does not contain unnecessary repetition.

Your highest priorities are:
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
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.90f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string answer;
    const int limit = max_tokens > 0 ? (max_tokens > 512 ? 512 : max_tokens) : 192;
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
    return env->NewStringUTF(answer.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_asad_ai_NativeAI_nativeFree(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_vocab = nullptr;
}  g_vocab = nullptr;
    g_history.clear();
}
