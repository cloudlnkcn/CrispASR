#ifndef CRISPASR_PUNC_LOADER_H
#define CRISPASR_PUNC_LOADER_H

#include <string>

// Punctuation-restoration model selection, shared by the CLI one-shot path
// (crispasr_run.cpp) and the HTTP server (crispasr_server.cpp).
//
// The `--punc-model` value is a small set of aliases (auto / firered /
// fullstop / punctuate-all / pcs) plus the option of a direct GGUF path. The
// alias → (cache filename, download URL) table is the part that drifts: the
// server originally skipped punctuation entirely (PR #166), and once both
// front-ends restore punctuation they must resolve identical models. Keep the
// mapping here, pure and dependency-free (only <string>), so it can be unit
// tested without loading any model, and so the front-ends stay in lockstep.

enum class crispasr_punc_kind {
    none,        // disabled / no model
    fireredpunc, // FireRedPunc-family GGUF (auto|firered|fullstop|punctuate-all|path)
    pcs,         // XLM-R punctuation + capitalization + segmentation GGUF
};

struct crispasr_punc_spec {
    crispasr_punc_kind kind = crispasr_punc_kind::none;
    std::string cache_filename; // download target filename ("" for a direct path)
    std::string url;            // download URL              ("" for a direct path)
    std::string direct_path;    // explicit on-disk path     ("" for an alias)
};

// Map a `--punc-model` value to a model spec. Pure: no I/O, no model load.
// Mirrors the alias handling in crispasr_run.cpp exactly.
inline crispasr_punc_spec crispasr_resolve_punc_model(const std::string& punc_model) {
    crispasr_punc_spec s;
    const std::string& m = punc_model;

    if (m.empty() || m == "none" || m == "off")
        return s; // disabled

    if (m == "auto" || m == "firered") {
        s.kind = crispasr_punc_kind::fireredpunc;
        s.cache_filename = "fireredpunc-q4_k.gguf";
        s.url = "https://huggingface.co/cstr/fireredpunc-GGUF/resolve/main/fireredpunc-q4_k.gguf";
        return s;
    }
    if (m == "fullstop") {
        s.kind = crispasr_punc_kind::fireredpunc;
        s.cache_filename = "fullstop-punc-q4_k.gguf";
        s.url = "https://huggingface.co/cstr/fullstop-punc-multilang-GGUF/resolve/main/fullstop-punc-q4_k.gguf";
        return s;
    }
    if (m == "punctuate-all") {
        s.kind = crispasr_punc_kind::fireredpunc;
        s.cache_filename = "punctuate-all-q4_k.gguf";
        s.url = "https://huggingface.co/cstr/punctuate-all-GGUF/resolve/main/punctuate-all-q4_k.gguf";
        return s;
    }
    if (m == "pcs") {
        s.kind = crispasr_punc_kind::pcs;
        s.cache_filename = "pcs-xlmr-base-q4_k.gguf";
        s.url = "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/main/pcs-xlmr-base-q4_k.gguf";
        return s;
    }
    if (m.find("pcs") != std::string::npos) {
        // A path or keyword that mentions "pcs": only a concrete .gguf is
        // loadable as a PCS model. Anything else (e.g. a bare "pcs-de") matched
        // neither front-end's load path historically, so it stays disabled.
        if (m.find(".gguf") != std::string::npos) {
            s.kind = crispasr_punc_kind::pcs;
            s.direct_path = m;
        }
        return s;
    }

    // Otherwise: a direct path to a FireRedPunc-family GGUF.
    s.kind = crispasr_punc_kind::fireredpunc;
    s.direct_path = m;
    return s;
}

#endif // CRISPASR_PUNC_LOADER_H
