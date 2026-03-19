/* ═══════════════════════════════════════════════════════════════════════════
   VividnessMem — SillyTavern Extension
   ═══════════════════════════════════════════════════════════════════════════
   Hooks into SillyTavern's event pipeline to:
   • Inject VividnessMem context into every prompt (system prompt prepend)
   • Auto-store user + character messages as memories
   • Provide a settings panel for configuration
   • Show mood/stats in a sidebar widget
   ═══════════════════════════════════════════════════════════════════════════ */

import {
    eventSource,
    event_types,
    getRequestHeaders,
    saveSettingsDebounced,
} from "../../../../script.js";

import {
    extension_settings,
    getContext,
} from "../../../extensions.js";

// ── Constants ─────────────────────────────────────────────────────────────
const EXT_NAME     = "VividnessMem";
const EXT_KEY      = "vividnessmem";
const DEFAULT_URL  = "http://127.0.0.1:5050";

// Emotion keywords for auto-detection (maps keywords → emotion tag)
const EMOTION_KEYWORDS = {
    // positive
    "happy":       "happy",     "glad":        "happy",     "smile":   "happy",
    "laugh":       "amused",    "haha":        "amused",    "lol":     "amused",
    "love":        "affectionate", "adore":    "affectionate",
    "excited":     "excited",   "amazing":     "excited",   "awesome": "excited",
    "proud":       "proud",     "grateful":    "grateful",  "thanks":  "grateful",
    "curious":     "curious",   "wonder":      "curious",   "interesting": "curious",
    "hope":        "hopeful",   "wish":        "hopeful",
    "calm":        "peaceful",  "relax":       "peaceful",  "peace":   "peaceful",
    // negative
    "sad":         "sad",       "cry":         "sad",       "tears":   "sad",
    "angry":       "angry",     "furious":     "angry",     "mad":     "angry",
    "frustrated":  "frustrated","annoyed":     "frustrated",
    "scared":      "afraid",    "fear":        "afraid",    "terrified": "afraid",
    "anxious":     "anxious",   "nervous":     "anxious",   "worried": "anxious",
    "lonely":      "lonely",    "alone":       "lonely",
    "confused":    "confused",  "lost":        "confused",
    "hurt":        "hurt",      "pain":        "hurt",      "ache":    "hurt",
    "guilty":      "guilty",    "sorry":       "guilty",
    "embarrassed": "embarrassed", "blush":     "embarrassed",
    // reflective
    "remember":    "nostalgic", "memory":      "nostalgic", "past":    "nostalgic",
    "think":       "thoughtful","ponder":      "thoughtful",
};

// ── Default settings ──────────────────────────────────────────────────────
const DEFAULT_SETTINGS = {
    enabled:             true,
    server_url:          DEFAULT_URL,
    auto_store_user:     true,      // Store user messages as impressions
    auto_store_char:     true,      // Store character messages as reflections
    inject_context:      true,      // Inject memory context into prompts
    auto_detect_emotion: true,      // Auto-detect emotions from message text
    default_importance:  5,         // Default importance for auto-stored messages
    min_message_length:  10,        // Minimum message length to auto-store
    context_position:    "before",  // "before" or "after" system prompt
    max_context_tokens:  0,         // 0 = unlimited, else truncate context to this
    memory_scope:        "global",  // "global" = across all chats, "per_chat" = per chat
    injection_mode:      "auto",    // "auto" = extension prompt system, "macro" = {{vividmem}} only
    ooc_filter:          true,      // Filter out OOC messages (( )), /ooc, etc.
    show_mood_badge:     true,      // Show mood badge in chat
    bump_on_new_chat:    true,      // Auto-bump session on new chat
    debug_logging:       false,
};


// ═══════════════════════════════════════════════════════════════════════════
//  Settings panel HTML
// ═══════════════════════════════════════════════════════════════════════════

const SETTINGS_HTML = `
<div id="vividnessmem_settings">
    <div class="inline-drawer">
        <div class="inline-drawer-toggle inline-drawer-header">
            <b>VividnessMem</b>
            <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
        </div>
        <div class="inline-drawer-content">
            <!-- Connection -->
            <div class="vividmem-section">
                <h4>Connection</h4>
                <label>
                    Server URL
                    <input id="vividmem_server_url" type="text" class="text_pole"
                           placeholder="http://127.0.0.1:5050" />
                </label>
                <div id="vividmem_status" class="vividmem-status">
                    <span class="vividmem-dot"></span> <span class="vividmem-status-text">Not connected</span>
                </div>
                <button id="vividmem_test_btn" class="menu_button">Test Connection</button>
            </div>

            <!-- Toggle switches -->
            <div class="vividmem-section">
                <h4>Features</h4>
                <label class="checkbox_label">
                    <input id="vividmem_enabled" type="checkbox" />
                    <span>Enable VividnessMem</span>
                </label>
                <label class="checkbox_label">
                    <input id="vividmem_auto_store_user" type="checkbox" />
                    <span>Auto-store user messages</span>
                </label>
                <label class="checkbox_label">
                    <input id="vividmem_auto_store_char" type="checkbox" />
                    <span>Auto-store character messages</span>
                </label>
                <label class="checkbox_label">
                    <input id="vividmem_inject_context" type="checkbox" />
                    <span>Inject memory context into prompts</span>
                </label>
                <label class="checkbox_label">
                    <input id="vividmem_auto_detect_emotion" type="checkbox" />
                    <span>Auto-detect emotions</span>
                </label>
                <label class="checkbox_label">
                    <input id="vividmem_show_mood_badge" type="checkbox" />
                    <span>Show mood badge</span>
                </label>
                <label class="checkbox_label">
                    <input id="vividmem_ooc_filter" type="checkbox" />
                    <span>Filter OOC messages <small>(skip (( )), /ooc, //)</small></span>
                </label>
                <label class="checkbox_label">
                    <input id="vividmem_bump_on_new_chat" type="checkbox" />
                    <span>Bump session on new chat</span>
                </label>
                <label class="checkbox_label">
                    <input id="vividmem_debug_logging" type="checkbox" />
                    <span>Debug logging (console)</span>
                </label>
            </div>

            <!-- Tuning -->
            <div class="vividmem-section">
                <h4>Tuning</h4>
                <label>
                    Default importance (1-10)
                    <input id="vividmem_default_importance" type="number"
                           min="1" max="10" class="text_pole" />
                </label>
                <label>
                    Min message length to store
                    <input id="vividmem_min_message_length" type="number"
                           min="0" max="500" class="text_pole" />
                </label>
                <label>
                    Context position
                    <select id="vividmem_context_position" class="text_pole">
                        <option value="before">Before system prompt</option>
                        <option value="after">After system prompt</option>
                    </select>
                </label>
                <label>
                    Max context tokens (0 = unlimited)
                    <input id="vividmem_max_context_tokens" type="number"
                           min="0" max="16000" step="100" class="text_pole"
                           title="Limit how many tokens of memory context get injected. 0 means no limit." />
                </label>
                <label>
                    Memory scope
                    <select id="vividmem_memory_scope" class="text_pole"
                            title="Global: memories persist across all chats. Per-chat: fresh memory for each new chat.">
                        <option value="global">Global (across all chats)</option>
                        <option value="per_chat">Per chat (fresh each chat)</option>
                    </select>
                </label>
                <label>
                    Injection mode
                    <select id="vividmem_injection_mode" class="text_pole"
                            title="Auto: injected via extension system. Macro: only via {{vividmem}} in your prompt template.">
                        <option value="auto">Auto (extension prompt)</option>
                        <option value="macro">Macro only ({{vividmem}})</option>
                    </select>
                </label>
            </div>

            <!-- Manual actions -->
            <div class="vividmem-section">
                <h4>Actions</h4>
                <button id="vividmem_bump_session" class="menu_button">
                    Bump Session
                </button>
                <button id="vividmem_view_stats" class="menu_button">
                    View Stats
                </button>
                <button id="vividmem_view_memories" class="menu_button">
                    Browse Memories
                </button>
                <button id="vividmem_import_old" class="menu_button">
                    ⬆️ Import Old Memories (JSON)
                </button>
            </div>

            <!-- Advanced -->
            <div class="vividmem-section">
                <h4>Advanced</h4>
                <button id="vividmem_consolidate" class="menu_button"
                        title="Merge similar memories into coherent summaries">
                    🔄 Consolidate Memories
                </button>
                <button id="vividmem_dream" class="menu_button"
                        title="Run a dream cycle to discover hidden connections">
                    💤 Dream Cycle
                </button>
                <button id="vividmem_export" class="menu_button"
                        title="Export all memories as JSON for backup">
                    💾 Export Memories (JSON)
                </button>
                <button id="vividmem_preview_context" class="menu_button"
                        title="Preview the context block that will be injected">
                    👁️ Preview Context
                </button>
                <hr style="border-color:#444; margin:8px 0;">
                <label>
                    Add a manual memory note
                    <textarea id="vividmem_manual_note" class="text_pole" rows="2"
                              placeholder="Type a custom memory note..."></textarea>
                </label>
                <button id="vividmem_add_note" class="menu_button">
                    📝 Add Note as Memory
                </button>
                <hr style="border-color:#444; margin:8px 0;">
                <button id="vividmem_wipe" class="menu_button redWarningBG"
                        title="Delete ALL memories for this character. Cannot be undone!">
                    ⚠️ Wipe All Memories
                </button>
            </div>

            <!-- Macro info -->
            <div class="vividmem-section">
                <h4>Macros</h4>
                <p style="font-size:0.85em; opacity:0.8;">
                    Use <code>{{vividmem}}</code> in your prompt template to place memory context wherever you like.<br/>
                    Use <code>{{vividmood}}</code> to insert the current mood label.<br/>
                    Set injection mode to &ldquo;Macro only&rdquo; to avoid duplicate injection.
                </p>
                <div id="vividmem_token_count" style="font-size:0.85em; opacity:0.7; margin-top:4px;"></div>
            </div>

            <!-- Stats display -->
            <div id="vividmem_stats_display" class="vividmem-stats" style="display:none;">
            </div>

            <!-- Memory browser -->
            <div id="vividmem_memory_browser" class="vividmem-browser" style="display:none;">
                <div id="vividmem_memory_list"></div>
            </div>
        </div>
    </div>
</div>
`;


// ═══════════════════════════════════════════════════════════════════════════
//  Utility functions
// ═══════════════════════════════════════════════════════════════════════════

function log_debug(...args) {
    if (extension_settings[EXT_KEY]?.debug_logging) {
        console.log(`[${EXT_NAME}]`, ...args);
    }
}

function log_info(...args) {
    console.log(`[${EXT_NAME}]`, ...args);
}

function log_error(...args) {
    console.error(`[${EXT_NAME}]`, ...args);
}

/** Return the active character name, or empty string. */
function getCharacterName() {
    const ctx = getContext();
    if (ctx?.characterId !== undefined && ctx?.characters) {
        const char = ctx.characters[ctx.characterId];
        if (char?.name) return char.name;
    }
    return "";
}

/**
 * Return the effective character key for memory storage.
 * In "per_chat" mode, appends the chat file name so each chat gets
 * its own separate memory namespace.
 */
function getEffectiveCharacter() {
    const name = getCharacterName();
    if (!name) return "";
    const s = extension_settings[EXT_KEY];
    if (s?.memory_scope === "per_chat") {
        const ctx = getContext();
        const chatFile = ctx?.chatId || ctx?.chat_file_name || "";
        if (chatFile) return `${name}__chat_${chatFile}`;
    }
    return name;
}

/** Return the user's display name. */
function getUserName() {
    const ctx = getContext();
    return ctx?.name1 || "User";
}

/** Get the configured server URL (no trailing slash). */
function getServerUrl() {
    const url = extension_settings[EXT_KEY]?.server_url || DEFAULT_URL;
    return url.replace(/\/+$/, "");
}

/** Make an API call to the VividnessMem server. */
async function apiCall(method, path, body = null) {
    const url = `${getServerUrl()}${path}`;
    const opts = {
        method,
        headers: { "Content-Type": "application/json" },
    };
    if (body) {
        opts.body = JSON.stringify(body);
    }
    log_debug(`API ${method} ${path}`, body);
    const resp = await fetch(url, opts);
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`VividnessMem API error ${resp.status}: ${text}`);
    }
    return resp.json();
}

/**
 * Detect the dominant emotion from message text.
 * Returns the most-matched emotion tag, or "neutral".
 */
function detectEmotion(text) {
    if (!text) return "neutral";
    const lower = text.toLowerCase();
    const counts = {};
    for (const [keyword, emotion] of Object.entries(EMOTION_KEYWORDS)) {
        if (lower.includes(keyword)) {
            counts[emotion] = (counts[emotion] || 0) + 1;
        }
    }
    let best = "neutral", bestCount = 0;
    for (const [emotion, count] of Object.entries(counts)) {
        if (count > bestCount) {
            best = emotion;
            bestCount = count;
        }
    }
    return best;
}

/**
 * Estimate importance from message content (1-10).
 * Longer, more emotional, question-containing messages rate higher.
 */
function estimateImportance(text, baseImportance = 5) {
    if (!text) return baseImportance;
    let score = baseImportance;
    // Length bonus
    if (text.length > 200) score += 1;
    if (text.length > 500) score += 1;
    // Question = slightly more important
    if (text.includes("?")) score += 1;
    // Emotional intensity keywords
    const intense = ["love", "hate", "never", "always", "promise", "swear",
                     "forever", "please", "important", "secret", "trust"];
    for (const kw of intense) {
        if (text.toLowerCase().includes(kw)) { score += 1; break; }
    }
    return Math.min(10, Math.max(1, score));
}

/**
 * Detect OOC (out-of-character) messages that shouldn't be stored as memories.
 * Common RP patterns: (( )), /ooc, // comments, [OOC], etc.
 */
function isOOCMessage(text) {
    if (!text) return false;
    const trimmed = text.trimStart();
    if (trimmed.startsWith("((") || trimmed.startsWith("//") ||
        trimmed.startsWith("OOC:") || trimmed.startsWith("ooc:") ||
        trimmed.startsWith("[OOC") || trimmed.startsWith("/ooc")) {
        return true;
    }
    // Entire message wrapped in ((...))
    if (/^\(\([\s\S]*\)\)\s*$/.test(trimmed)) return true;
    return false;
}

/** Update the token count display in settings. */
function updateTokenCount() {
    const el = document.getElementById("vividmem_token_count");
    if (!el) return;
    const chars = _lastContextBlock?.length || 0;
    const tokens = Math.ceil(chars / 4);
    el.textContent = `\ud83d\udcca Current context: ~${tokens} tokens (${chars} chars)`;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Core extension logic
// ═══════════════════════════════════════════════════════════════════════════

let _lastContextBlock = "";
let _moodLabel = "neutral";
let _browsedMemories = [];

/**
 * Process a chat message — store it and retrieve updated context.
 */
async function processMessage(messageText, isUser) {
    const s = extension_settings[EXT_KEY];
    if (!s?.enabled) return;

    const character = getEffectiveCharacter();
    if (!character) return;

    // Skip OOC messages
    if (s.ooc_filter && isOOCMessage(messageText)) {
        log_debug("OOC message filtered, skipping storage:", messageText.slice(0, 50));
        return;
    }

    // Check minimum length
    if (messageText.length < (s.min_message_length || 10)) {
        log_debug("Message too short, skipping:", messageText.length);
        return;
    }

    // Should we store this message type?
    if (isUser && !s.auto_store_user) return;
    if (!isUser && !s.auto_store_char) return;

    const emotion = s.auto_detect_emotion ? detectEmotion(messageText) : "neutral";
    const importance = estimateImportance(messageText, s.default_importance || 5);

    try {
        const result = await apiCall("POST", "/api/memory/process", {
            character,
            user_name: getUserName(),
            message: messageText,
            is_user: isUser,
            emotion,
            importance,
            conversation_context: messageText,
            max_context_tokens: s.max_context_tokens || 0,
        });

        _lastContextBlock = result.context_block || "";
        _moodLabel = result.mood || "neutral";
        updateTokenCount();

        log_debug("Processed message:", {
            isUser,
            emotion,
            importance,
            mood: _moodLabel,
            memoryCount: result.stats?.total_self_reflections || 0,
        });

        // Update mood badge
        if (s.show_mood_badge) {
            updateMoodBadge(_moodLabel);
        }
    } catch (err) {
        log_error("Failed to process message:", err.message);
    }
}

/**
 * Fetch fresh context block (for prompt injection before generation).
 */
async function refreshContext() {
    const s = extension_settings[EXT_KEY];
    if (!s?.enabled || !s?.inject_context) return "";

    const character = getEffectiveCharacter();
    if (!character) return "";

    try {
        let url = `/api/memory/context/${encodeURIComponent(character)}` +
            `?entity=${encodeURIComponent(getUserName())}`;
        if (s.max_context_tokens > 0) {
            url += `&max_tokens=${s.max_context_tokens}`;
        }
        const result = await apiCall("GET", url);
        _lastContextBlock = result.context_block || "";
        updateTokenCount();
        return _lastContextBlock;
    } catch (err) {
        log_error("Failed to refresh context:", err.message);
        return _lastContextBlock;  // Use cached
    }
}

/** Update or create the mood badge element. */
function updateMoodBadge(mood) {
    let badge = document.getElementById("vividmem_mood_badge");
    if (!badge) {
        badge = document.createElement("div");
        badge.id = "vividmem_mood_badge";
        badge.className = "vividmem-mood-badge";
        const chatHeader = document.getElementById("chat_header") ||
                           document.querySelector(".chat_header") ||
                           document.querySelector("#form_sheld");
        if (chatHeader) {
            chatHeader.appendChild(badge);
        }
    }
    if (badge) {
        badge.textContent = `🧠 ${mood}`;
        badge.title = `VividnessMem mood: ${mood}`;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  SillyTavern event hooks
// ═══════════════════════════════════════════════════════════════════════════

/** Called when user sends a message. */
function onUserMessage(messageIndex) {
    const ctx = getContext();
    if (!ctx?.chat) return;
    const msg = ctx.chat[messageIndex];
    if (msg?.mes) {
        processMessage(msg.mes, true);
    }
}

/** Called when the character generates a message. */
function onCharacterMessage(messageIndex) {
    const ctx = getContext();
    if (!ctx?.chat) return;
    const msg = ctx.chat[messageIndex];
    if (msg?.mes) {
        processMessage(msg.mes, false);
    }
}

/** Called when a new chat is opened. */
async function onChatChanged() {
    const s = extension_settings[EXT_KEY];
    if (!s?.enabled) return;

    const character = getEffectiveCharacter();
    if (!character) return;

    log_info("Chat changed, character:", character);

    // Bump session if configured
    if (s.bump_on_new_chat) {
        try {
            const result = await apiCall("POST",
                `/api/memory/session/${encodeURIComponent(character)}/bump`
            );
            log_debug("Session bumped:", result);
        } catch (err) {
            log_error("Failed to bump session:", err.message);
        }
    }

    // Pre-fetch context
    await refreshContext();
}

/**
 * Prompt injection: called by ST before generating a response.
 * We modify the system prompt to include VividnessMem context.
 */
function onPromptReady(eventData) {
    const s = extension_settings[EXT_KEY];
    if (!s?.enabled || !s?.inject_context || !_lastContextBlock) return;

    // eventData is the prompt string or object depending on ST version
    // For the injection approach, we use the GENERATE_AFTER_COMBINE_PROMPTS event
    // to append our context block
    log_debug("Injecting context block, length:", _lastContextBlock.length);
}


// ═══════════════════════════════════════════════════════════════════════════
//  Settings management
// ═══════════════════════════════════════════════════════════════════════════

function loadSettings() {
    // Initialize with defaults
    if (!extension_settings[EXT_KEY]) {
        extension_settings[EXT_KEY] = {};
    }
    for (const [key, val] of Object.entries(DEFAULT_SETTINGS)) {
        if (extension_settings[EXT_KEY][key] === undefined) {
            extension_settings[EXT_KEY][key] = val;
        }
    }

    // Populate UI
    const s = extension_settings[EXT_KEY];
    $("#vividmem_server_url").val(s.server_url);
    $("#vividmem_enabled").prop("checked", s.enabled);
    $("#vividmem_auto_store_user").prop("checked", s.auto_store_user);
    $("#vividmem_auto_store_char").prop("checked", s.auto_store_char);
    $("#vividmem_inject_context").prop("checked", s.inject_context);
    $("#vividmem_auto_detect_emotion").prop("checked", s.auto_detect_emotion);
    $("#vividmem_show_mood_badge").prop("checked", s.show_mood_badge);
    $("#vividmem_bump_on_new_chat").prop("checked", s.bump_on_new_chat);
    $("#vividmem_debug_logging").prop("checked", s.debug_logging);
    $("#vividmem_default_importance").val(s.default_importance);
    $("#vividmem_min_message_length").val(s.min_message_length);
    $("#vividmem_context_position").val(s.context_position);
    $("#vividmem_max_context_tokens").val(s.max_context_tokens);
    $("#vividmem_memory_scope").val(s.memory_scope);
    $("#vividmem_injection_mode").val(s.injection_mode);
    $("#vividmem_ooc_filter").prop("checked", s.ooc_filter);
}

function bindSettingsUI() {
    const s = extension_settings[EXT_KEY];

    // Text / number inputs
    $("#vividmem_server_url").on("input", function () {
        s.server_url = $(this).val();
        saveSettingsDebounced();
    });
    $("#vividmem_default_importance").on("input", function () {
        s.default_importance = parseInt($(this).val()) || 5;
        saveSettingsDebounced();
    });
    $("#vividmem_min_message_length").on("input", function () {
        s.min_message_length = parseInt($(this).val()) || 10;
        saveSettingsDebounced();
    });
    $("#vividmem_context_position").on("change", function () {
        s.context_position = $(this).val();
        saveSettingsDebounced();
    });
    $("#vividmem_max_context_tokens").on("input", function () {
        s.max_context_tokens = parseInt($(this).val()) || 0;
        saveSettingsDebounced();
    });
    $("#vividmem_memory_scope").on("change", function () {
        s.memory_scope = $(this).val();
        saveSettingsDebounced();
    });
    $("#vividmem_injection_mode").on("change", function () {
        s.injection_mode = $(this).val();
        saveSettingsDebounced();
    });

    // Checkboxes
    const checkboxes = [
        "enabled", "auto_store_user", "auto_store_char", "inject_context",
        "auto_detect_emotion", "show_mood_badge", "bump_on_new_chat", "debug_logging",
        "ooc_filter",
    ];
    for (const key of checkboxes) {
        $(`#vividmem_${key}`).on("change", function () {
            s[key] = $(this).prop("checked");
            saveSettingsDebounced();
        });
    }

    // Buttons
    $("#vividmem_test_btn").on("click", testConnection);
    $("#vividmem_bump_session").on("click", manualBumpSession);
    $("#vividmem_view_stats").on("click", viewStats);
    $("#vividmem_view_memories").on("click", browseMemories);
    $("#vividmem_import_old").on("click", importOldMemories);
    $("#vividmem_consolidate").on("click", triggerConsolidation);
    $("#vividmem_dream").on("click", triggerDream);
    $("#vividmem_export").on("click", exportMemories);
    $("#vividmem_preview_context").on("click", showContextPreview);
    $("#vividmem_add_note").on("click", addManualNote);
    $("#vividmem_wipe").on("click", wipeMemories);
}

async function testConnection() {
    const statusEl = $("#vividmem_status");
    const dotEl = statusEl.find(".vividmem-dot");
    const textEl = statusEl.find(".vividmem-status-text");

    try {
        dotEl.removeClass("connected error").addClass("checking");
        textEl.text("Checking...");

        const result = await apiCall("GET", "/api/health");

        dotEl.removeClass("checking error").addClass("connected");
        textEl.text(`Connected — ${result.characters_loaded?.length || 0} characters loaded`);
    } catch (err) {
        dotEl.removeClass("checking connected").addClass("error");
        textEl.text(`Failed: ${err.message}`);
    }
}

async function manualBumpSession() {
    const character = getEffectiveCharacter();
    if (!character) {
        toastr.warning("No character selected.");
        return;
    }
    try {
        const result = await apiCall("POST",
            `/api/memory/session/${encodeURIComponent(character)}/bump`
        );
        toastr.success(`Session bumped to ${result.session_count}. ` +
            `Brief needed: ${result.needs_brief}, Dream needed: ${result.needs_dream}`);
    } catch (err) {
        toastr.error(`Failed: ${err.message}`);
    }
}

async function viewStats() {
    const character = getEffectiveCharacter();
    if (!character) {
        toastr.warning("No character selected.");
        return;
    }
    const display = $("#vividmem_stats_display");
    try {
        const stats = await apiCall("GET",
            `/api/memory/stats/${encodeURIComponent(character)}`
        );
        let html = `<h4>📊 ${character} Memory Stats</h4><table class="vividmem-stats-table">`;
        const labels = {
            total_self_reflections: "Self Reflections",
            active_self: "Active Self Memories",
            social_entities: "Social Entities",
            total_social: "Social Impressions",
            session_count: "Sessions",
            has_brief: "Has Brief",
            regret_count: "Regret Memories",
            dream_count: "Dream Insights",
            tracked_arcs: "Relationship Arcs",
            total_solutions: "Solution Patterns",
        };
        for (const [key, label] of Object.entries(labels)) {
            if (stats[key] !== undefined) {
                html += `<tr><td>${label}</td><td><b>${stats[key]}</b></td></tr>`;
            }
        }
        html += "</table>";
        display.html(html).show();
    } catch (err) {
        display.html(`<p class="error">Error: ${err.message}</p>`).show();
    }
}

async function browseMemories() {
    const character = getEffectiveCharacter();
    if (!character) {
        toastr.warning("No character selected.");
        return;
    }
    const browser = $("#vividmem_memory_browser");
    const list = $("#vividmem_memory_list");

    try {
        const result = await apiCall("POST", "/api/memory/query", {
            character,
            context: "",
            entity: "",
        });

        _browsedMemories = result.memories || [];

        if (!_browsedMemories.length) {
            list.html("<p><i>No memories yet.</i></p>");
            browser.show();
            return;
        }

        let html = `<h4>\ud83e\udde0 ${escapeHtml(character)} \u2014 ${result.count} memories</h4>`;
        for (let i = 0; i < _browsedMemories.length; i++) {
            const mem = _browsedMemories[i];
            const emotionClass = mem.emotion === "neutral" ? "" : `emotion-${mem.emotion}`;
            html += `
                <div class="vividmem-memory-card ${emotionClass}">
                    <div class="vividmem-memory-header">
                        <span class="vividmem-emotion">${escapeHtml(mem.emotion)}</span>
                        <span class="vividmem-importance">\u2605${mem.importance}</span>
                        <span class="vividmem-vividness">${(mem.vividness * 100).toFixed(0)}%</span>
                        <span class="vividmem-source">${escapeHtml(mem.source)}</span>
                        <button class="vividmem-delete-btn menu_button" data-idx="${i}"
                                title="Delete this memory" style="padding:2px 6px;font-size:0.8em;margin-left:auto;">\ud83d\uddd1\ufe0f</button>
                    </div>
                    <div class="vividmem-memory-content">${escapeHtml(mem.content)}</div>
                </div>
            `;
        }
        list.html(html);

        // Wire up delete buttons
        list.find(".vividmem-delete-btn").on("click", async function () {
            const idx = parseInt($(this).data("idx"));
            const mem = _browsedMemories[idx];
            if (!mem) return;
            if (!confirm(`Delete this memory?\n\n"${mem.content.slice(0, 100)}..."`)) return;
            try {
                await apiCall("POST", "/api/memory/delete", {
                    character,
                    content: mem.content,
                    entity: mem.entity || "",
                });
                toastr.success("Memory deleted.");
                browseMemories(); // Refresh the list
            } catch (err) {
                toastr.error(`Delete failed: ${err.message}`);
            }
        });

        browser.show();
    } catch (err) {
        list.html(`<p class="error">Error: ${err.message}</p>`);
        browser.show();
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

/** Trigger memory consolidation (merge similar memories). */
async function triggerConsolidation() {
    const character = getEffectiveCharacter();
    if (!character) { toastr.warning("No character selected."); return; }
    try {
        const result = await apiCall("POST",
            `/api/memory/consolidate/${encodeURIComponent(character)}`);
        if (result.clusters > 0) {
            toastr.success(`Found ${result.clusters} memory clusters for consolidation.`);
        } else {
            toastr.info("No memory clusters ready for consolidation yet.");
        }
    } catch (err) {
        toastr.error(`Consolidation failed: ${err.message}`);
    }
}

/** Trigger a dream cycle to find hidden memory connections. */
async function triggerDream() {
    const character = getEffectiveCharacter();
    if (!character) { toastr.warning("No character selected."); return; }
    try {
        const result = await apiCall("POST",
            `/api/memory/dream/${encodeURIComponent(character)}`);
        if (result.candidates > 0) {
            toastr.success(`Dream cycle found ${result.candidates} memory connections.`);
        } else {
            toastr.info("No dream candidates available yet. Needs more memories.");
        }
    } catch (err) {
        toastr.error(`Dream cycle failed: ${err.message}`);
    }
}

/** Export all memories as a downloadable JSON file. */
async function exportMemories() {
    const character = getEffectiveCharacter();
    if (!character) { toastr.warning("No character selected."); return; }
    try {
        const result = await apiCall("GET",
            `/api/memory/export/${encodeURIComponent(character)}`);
        const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `vividmem_${character.replace(/[^a-zA-Z0-9]/g, "_")}_export.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toastr.success("Memories exported!");
    } catch (err) {
        toastr.error(`Export failed: ${err.message}`);
    }
}

/** Show a preview popup of the current context block with token estimate. */
function showContextPreview() {
    const block = _lastContextBlock || "(No context available yet \u2014 send a message first)";
    const tokens = Math.ceil(block.length / 4);

    // Remove existing preview
    document.getElementById("vividmem_preview_popup")?.remove();

    const popup = document.createElement("div");
    popup.id = "vividmem_preview_popup";
    popup.style.cssText = `
        position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
        width:70vw; max-height:70vh; background:#1e1e2e; color:#cdd6f4;
        border:1px solid #585b70; border-radius:8px; z-index:99999;
        display:flex; flex-direction:column; box-shadow:0 8px 32px rgba(0,0,0,0.5);
    `;
    popup.innerHTML = `
        <div style="padding:12px 16px;border-bottom:1px solid #585b70;display:flex;justify-content:space-between;align-items:center;">
            <b>\ud83e\udde0 Context Preview (~${tokens} tokens, ${block.length} chars)</b>
            <button id="vividmem_close_preview" style="background:none;border:none;color:#cdd6f4;font-size:1.2em;cursor:pointer;">\u2715</button>
        </div>
        <pre style="padding:16px;margin:0;white-space:pre-wrap;overflow-y:auto;flex:1;font-size:0.85em;font-family:monospace;">${escapeHtml(block)}</pre>
    `;
    document.body.appendChild(popup);
    document.getElementById("vividmem_close_preview").addEventListener("click", () => popup.remove());
}

/** Add a user-written note as a self-reflection memory. */
async function addManualNote() {
    const character = getEffectiveCharacter();
    if (!character) { toastr.warning("No character selected."); return; }
    const text = document.getElementById("vividmem_manual_note")?.value?.trim();
    if (!text) { toastr.warning("Enter a memory note first."); return; }

    const emotion = detectEmotion(text);
    const importance = estimateImportance(text);

    try {
        await apiCall("POST", "/api/memory/reflection", {
            character,
            content: text,
            emotion,
            importance,
            source: "manual_note",
            why_saved: "Manually added by user",
        });
        document.getElementById("vividmem_manual_note").value = "";
        toastr.success("Memory note added!");
        await refreshContext();
    } catch (err) {
        toastr.error(`Failed: ${err.message}`);
    }
}

/** Wipe ALL memories for the current character (double-confirm). */
async function wipeMemories() {
    const character = getEffectiveCharacter();
    if (!character) { toastr.warning("No character selected."); return; }
    const charName = getCharacterName();

    if (!confirm(
        `\u26a0\ufe0f WARNING: This will permanently delete ALL memories for "${charName}".\n\n` +
        `This includes self-reflections, social impressions, and all memory data.\n` +
        `This cannot be undone.\n\nAre you sure?`
    )) return;

    const typed = prompt(`Type "${charName}" to confirm wiping all memories:`);
    if (typed?.trim() !== charName) {
        toastr.info("Wipe cancelled.");
        return;
    }

    try {
        await apiCall("DELETE",
            `/api/memory/${encodeURIComponent(character)}?confirm=yes`);
        _lastContextBlock = "";
        _moodLabel = "neutral";
        updateTokenCount();
        toastr.success(`All memories wiped for ${charName}.`);
    } catch (err) {
        toastr.error(`Wipe failed: ${err.message}`);
    }
}


/**
 * Import old-format memory JSON files into VividnessMem.
 * Supports common SillyTavern memory formats:
 * - Array of objects with "content"/"text"/"entry" fields
 * - Object with character name keys mapping to arrays
 * - VectorDB/ChromaDB export format
 * - Simple key-value { "key": "memory text" } format
 *
 * The server re-indexes each entry: detects emotion, estimates importance,
 * generates a meaningful why_saved, and stores without touching the original file.
 */
async function importOldMemories() {
    const character = getEffectiveCharacter();
    if (!character) {
        toastr.warning("No character selected. Open a chat first.");
        return;
    }

    // Create a hidden file input
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".json,.jsonl";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    fileInput.addEventListener("change", async () => {
        const file = fileInput.files[0];
        if (!file) return;
        document.body.removeChild(fileInput);

        try {
            const text = await file.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch (e) {
                // Try JSONL format
                const lines = text.split("\n").filter(l => l.trim());
                data = lines.map(l => JSON.parse(l));
            }

            // Normalize to array of strings/objects
            const entries = normalizeOldMemories(data, character);

            if (!entries.length) {
                toastr.warning("No memories found in that file. Check the format.");
                return;
            }

            const confirmed = confirm(
                `Found ${entries.length} memories to import for "${getCharacterName()}".\n\n` +
                `This will NOT modify your original file.\n` +
                `The memories will be re-indexed with emotion tags, importance scores, and reasons.\n\n` +
                `Proceed?`
            );
            if (!confirmed) return;

            toastr.info(`Importing ${entries.length} memories...`);

            const result = await apiCall("POST", "/api/memory/import/reindex", {
                character,
                user_name: getUserName(),
                entries,
            });

            toastr.success(
                `Imported ${result.imported} memories ` +
                `(${result.reflections} reflections, ${result.impressions} impressions).`
            );
            log_info("Import result:", result);

            // Refresh context
            await refreshContext();
        } catch (err) {
            toastr.error(`Import failed: ${err.message}`);
            log_error("Import error:", err);
        }
    });

    fileInput.click();
}

/**
 * Normalize various old memory JSON formats into a flat array of
 * { content, entity?, source? } objects for the server to re-index.
 */
function normalizeOldMemories(data, characterName) {
    const entries = [];

    function addEntry(text, entity, source) {
        if (typeof text !== "string") return;
        text = text.trim();
        if (text.length < 5) return;
        entries.push({
            content: text,
            entity: entity || "",
            source: source || "import",
        });
    }

    if (Array.isArray(data)) {
        // Array of objects or strings
        for (const item of data) {
            if (typeof item === "string") {
                addEntry(item, "", "import");
            } else if (typeof item === "object" && item !== null) {
                const text = item.content || item.text || item.entry ||
                             item.message || item.memory || item.value || "";
                const entity = item.entity || item.user || item.name || item.speaker || "";
                const source = item.source || item.type || "import";
                addEntry(text, entity, source);
            }
        }
    } else if (typeof data === "object" && data !== null) {
        // Check for common nested formats
        if (data.memories && Array.isArray(data.memories)) {
            return normalizeOldMemories(data.memories, characterName);
        }
        if (data.entries && Array.isArray(data.entries)) {
            return normalizeOldMemories(data.entries, characterName);
        }
        if (data.data && Array.isArray(data.data)) {
            return normalizeOldMemories(data.data, characterName);
        }

        // ChromaDB / VectorDB format: { ids: [...], documents: [...] }
        if (data.documents && Array.isArray(data.documents)) {
            for (const doc of data.documents) {
                addEntry(doc, "", "vectordb_import");
            }
        }
        // Key-value format: { "memory_1": "text", "memory_2": "text" }
        // Or character-keyed: { "CharName": [...] }
        else {
            for (const [key, val] of Object.entries(data)) {
                if (Array.isArray(val)) {
                    // Character-keyed format
                    for (const item of val) {
                        if (typeof item === "string") {
                            addEntry(item, "", "import");
                        } else if (typeof item === "object" && item !== null) {
                            const text = item.content || item.text || item.entry ||
                                         item.message || item.memory || "";
                            addEntry(text, key, "import");
                        }
                    }
                } else if (typeof val === "string") {
                    addEntry(val, "", "import");
                }
            }
        }
    }

    return entries;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Prompt injection via SillyTavern's extension API
// ═══════════════════════════════════════════════════════════════════════════

/**
 * SillyTavern calls this function to collect extension prompt injections.
 * We return the VividnessMem context block to be included in the system prompt.
 */
function getExtensionPrompt() {
    const s = extension_settings[EXT_KEY];
    if (!s?.enabled || !s?.inject_context || !_lastContextBlock) return "";
    // In macro-only mode, skip auto-injection (user places {{vividmem}} manually)
    if (s.injection_mode === "macro") return "";
    return _lastContextBlock;
}

/**
 * SillyTavern calls this to determine injection position.
 * 0 = before system prompt, 1 = after.
 */
function getExtensionPromptPosition() {
    const s = extension_settings[EXT_KEY];
    return (s?.context_position === "after") ? 1 : 0;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Initialization
// ═══════════════════════════════════════════════════════════════════════════

jQuery(async () => {
    // Inject settings HTML
    const settingsContainer = document.getElementById("extensions_settings2") ||
                              document.getElementById("extensions_settings");
    if (settingsContainer) {
        $(settingsContainer).append(SETTINGS_HTML);
    }

    loadSettings();
    bindSettingsUI();

    // Register event listeners
    if (typeof eventSource !== "undefined" && eventSource) {
        // Message events
        eventSource.on(event_types.MESSAGE_SENT, (messageIndex) => {
            onUserMessage(messageIndex);
        });

        eventSource.on(event_types.MESSAGE_RECEIVED, (messageIndex) => {
            onCharacterMessage(messageIndex);
        });

        // Chat switch
        eventSource.on(event_types.CHAT_CHANGED, () => {
            onChatChanged();
        });

        // Pre-generation: refresh context for injection
        eventSource.on(event_types.GENERATION_STARTED, async () => {
            await refreshContext();
        });

        log_info("Event listeners registered.");
    }

    // Auto-test connection on load
    setTimeout(testConnection, 2000);

    // Register custom macros for prompt templates
    try {
        const { MacrosParser } = await import("../../../../macros.js");
        MacrosParser.registerMacro('vividmem', () => _lastContextBlock || '');
        MacrosParser.registerMacro('vividmood', () => _moodLabel || 'neutral');
        log_info("Registered {{vividmem}} and {{vividmood}} macros.");
    } catch (e) {
        log_info("MacrosParser not available — custom macros disabled. Requires SillyTavern 1.12+");
    }

    log_info("Extension loaded.");
});
