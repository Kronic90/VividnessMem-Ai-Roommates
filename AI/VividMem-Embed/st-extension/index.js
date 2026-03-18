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


// ═══════════════════════════════════════════════════════════════════════════
//  Core extension logic
// ═══════════════════════════════════════════════════════════════════════════

let _lastContextBlock = "";
let _moodLabel = "neutral";

/**
 * Process a chat message — store it and retrieve updated context.
 */
async function processMessage(messageText, isUser) {
    const s = extension_settings[EXT_KEY];
    if (!s?.enabled) return;

    const character = getCharacterName();
    if (!character) return;

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
        });

        _lastContextBlock = result.context_block || "";
        _moodLabel = result.mood || "neutral";

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

    const character = getCharacterName();
    if (!character) return "";

    try {
        const result = await apiCall("GET",
            `/api/memory/context/${encodeURIComponent(character)}` +
            `?entity=${encodeURIComponent(getUserName())}`
        );
        _lastContextBlock = result.context_block || "";
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

    const character = getCharacterName();
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

    // Checkboxes
    const checkboxes = [
        "enabled", "auto_store_user", "auto_store_char", "inject_context",
        "auto_detect_emotion", "show_mood_badge", "bump_on_new_chat", "debug_logging",
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
    const character = getCharacterName();
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
    const character = getCharacterName();
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
    const character = getCharacterName();
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

        if (!result.memories?.length) {
            list.html("<p><i>No memories yet.</i></p>");
            browser.show();
            return;
        }

        let html = `<h4>🧠 ${character} — ${result.count} memories</h4>`;
        for (const mem of result.memories) {
            const emotionClass = mem.emotion === "neutral" ? "" : `emotion-${mem.emotion}`;
            html += `
                <div class="vividmem-memory-card ${emotionClass}">
                    <div class="vividmem-memory-header">
                        <span class="vividmem-emotion">${mem.emotion}</span>
                        <span class="vividmem-importance">★${mem.importance}</span>
                        <span class="vividmem-vividness">${(mem.vividness * 100).toFixed(0)}%</span>
                        <span class="vividmem-source">${mem.source}</span>
                    </div>
                    <div class="vividmem-memory-content">${escapeHtml(mem.content)}</div>
                </div>
            `;
        }
        list.html(html);
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

    log_info("Extension loaded.");
});
