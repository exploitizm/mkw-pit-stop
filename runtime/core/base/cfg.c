// mkw.ini: file-backed defaults for the MKW_* environment variables (docs/controls.md, mkw.ini.example).
// Format: "key = value" lines, '#' or ';' at the start of a line begins a comment, [sections] only group. A key is a
// variable name without "MKW_" (an optional MKW_ prefix is accepted), case-insensitive. A value is the rest of the
// line, trimmed, optionally in double quotes; an empty value counts as unset. The environment wins over the file.
//
// Live settings: cfg_set changes a value in memory (readers see the new pointer; the old string is kept, so a pointer
// cfg_get handed out stays valid) and cfg_save writes only the changed keys back, keeping the rest of the file.
// Readers take no lock: an entry is complete before n_entries counts it, and value pointers are swapped atomically.
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#ifdef _WIN32
#include "utf16.h"
#endif

#define PREFIX "MKW_"
#define PREFIX_LEN (sizeof PREFIX - 1)
#define CFG_MAX_KEYS 512

typedef struct { char *key, *value; int dirty, removed; } CfgEntry;   // key upper-case, without the prefix; value NULL after cfg_reset

static CfgEntry entries[CFG_MAX_KEYS];
static int n_entries;
static unsigned version;
static PlatMutex set_lock = PLAT_MUTEX_INIT;   // cfg_set, cfg_reset, cfg_save
static int save_ok = 1;                         // the last cfg_save_to wrote (or had nothing to write)

// One row per name the runtime reads (and the patch sets in generated/), without the prefix. A file key outside the
// table is logged as unknown, so a new variable gets its row here. S: a setting the menu shows (CfgMeta: the value in
// effect when the key is unset, what the key accepts, whether it applies at once) and the mkw.ini section cfg_set
// writes it to when the file lacks it (mkw.ini.example's sections; NULL: none). The only place a default lives:
// consumers read cfg_value / cfg_int / cfg_double / cfg_bool. A row with an empty default has no fixed value (it
// depends on the machine); its consumer reads cfg_get and keeps an "unset" branch. K: a key with no menu row (add it in
// alphabetical order, one per line, so two additions rarely meet); KS: the same with an ini section.
#ifdef __PROSPERO__
#define REPLAY_PAR_DEFAULT "1"   // the PS5 records a submission's chunks in parallel (vk_exec_par.c)
#else
#define REPLAY_PAR_DEFAULT "0"
#endif
#define LIVE 1
#define RESTART 0
#define S(k, sec, d, v, l) {{k, d, v, l}, sec}
#define K(k) {{k, NULL, NULL, 0}, NULL}
#define KS(k, sec) {{k, NULL, NULL, 0}, sec}
typedef struct { CfgMeta m; const char *section; } CfgRow;
static const CfgRow rows[] = {
    // Screen
    S("FULLSCREEN", "display", "0", "0|1", LIVE),
    S("WINDOW_SIZE", "display", "1920x1080", NULL, LIVE),
    S("VK_PRESENT", "display", "mailbox", "mailbox|fifo|immediate", LIVE),
    S("GPU_INDEX", "display", "", NULL, RESTART),
    S("PERF_OVERLAY", "display", "off", "off|fps|full", LIVE),
    S("FPS", "display", "auto", "auto|game|60|30|20|15", LIVE),
    // split screen (2-4 players on this machine) at 30 fps at most (game.c); same: MKW_FPS as for one player
    S("LOGIC_HZ", "display", "60", "60|120|144|240", RESTART),
    S("LOW_LATENCY", "display", "1", "0|1", LIVE),
    S("FRAMEGEN", "display", "off", "off|auto|2|3|4|120", LIVE),
    // the screen has variable refresh (FreeSync, G-Sync), told by the player: Vulkan does not say whether it is on.
    // Off, frame generation's Automatic uses only the factors that split the refreshes evenly (vk_framegen.c fits)
    S("VRR", "display", "0", "0|1", LIVE),
    // bilinear by default (Adrià 15/11/2026): FSR 1 only when the player picks it or Auto-set does
    S("UPSCALER", "display", "bilinear", "fsr|bilinear", LIVE),
    S("SHARPNESS", "display", "80", "0..100", LIVE),
    S("PICTURE_FIT", "display", "keep", "keep|stretch", LIVE),
    S("HDR", "display", "auto", "auto|off", LIVE),   // auto: HDR when the display is (hdr6c, 12/11/2026)
    // RenoDX tone mapping (gfx_shader.c, tools/shaders/renodx.py): the game's own curve by default, as the console
    S("HDR_TONEMAP", "display", "vanilla", "vanilla|psychov", RESTART),
    // the peak and paper white told to the game (gfx_hdr.c), only with HDR_TONEMAP psychov; auto: the system's
    S("HDR_PEAK", "display", "auto", NULL, LIVE),   // "auto" or 100..10000 nits
    S("HDR_WHITE", "display", "auto", NULL, LIVE),   // "auto" or 40..1000 nits
    // the system's peak/paper white nits of the last HDR10 swapchain: the RenoDX programs are loaded before the window
    // exists (pipeline prebuild), so their headroom comes from here (gfx_hdr_headroom); written only with HDR_TONEMAP
    // psychov; not a menu row
    S("HDR_SYSTEM", "display", "", NULL, LIVE),
    S("DEVICE_PROFILE", "display", "auto", "auto|desktop|deck|ps5", RESTART),
    S("RESOLUTION", "display", "original", "original|540|720|900|1080|1440|2160", RESTART),
    S("ASPECT", "display", "16:9", "16:9|window|21:9|32:9|48:9", RESTART),
    S("HUD_POSITION", "display", "16:9", "16:9|edges", RESTART),   // the race HUD in the middle 16:9 or at the screen edges
    // temporal upscaling in the place of the game's own (vk_temporal.c): the engine is live, the quality sets the
    // game's 3D resolution (romfs_res.c) and applies at the next start
    S("TEMPORAL", "display", "off", "off|dlss|fsr3", LIVE),
    S("TEMPORAL_QUALITY", "display", "quality", "native|quality|balanced|performance|ultra", RESTART),
    S("TEMPORAL_SHARPNESS", "display", "20", "0..100", LIVE),
    S("CAS", "display", "off", "off|on", LIVE),
    S("CAS_SHARPNESS", "display", "50", "0..100", LIVE),
    S("TEMPORAL_KART", "display", "1", "0|1", LIVE),
    S("DLSS_PRESET", "display", "default", "default|j|k|l|m", LIVE),
    // frame generation's engine: the port's own optical flow, DLSS Frame Generation or FSR 3.1 frame generation
    S("FG_ENGINE", "display", "flow", "flow|dlss|fsr3", LIVE),
    // the HUD in DLSS-G's and FSR 3.1's made frames: the game frame's own laid over them (own), or the engine's own
    // handling of the picture without the HUD (engine; vk_framegen.c, fg_hud.comp)
    S("FG_HUD", "display", "own", "own|engine", LIVE),
    S("REFLEX", "display", "off", "off|on|boost", LIVE),
    // Game
    S("LANG", "language", "", "ja|en-US|fr|de|it|es|zh-CN|ko|nl|pt|ru|zh-TW|en-GB|fr-CA|es-419|zh-Hans|zh-Hant|pt-BR|pl|th|ca|gl|eu", RESTART),
    S("OPERATION_MODE", "language", "docked", "docked|handheld", RESTART),
    S("LAN", "language", "0", "0|1", RESTART),   // LAN play (sdk/ov/ov_socket.c), off until played between two PCs (#144)
    S("LAN_IP", NULL, "", NULL, RESTART),   // the address LAN play uses; unset: the best interface (ov_lanaddr.c); the Network row (#707)
    // Graphics: the console's look by default
    S("QUALITY", "graphics", "console", "low|medium|console|high|ultra|custom", RESTART),   // the menu's presets; nothing reads it
    S("ANISO", "graphics", "game", "game|2|4|8|16", RESTART),
    S("TEX_QUALITY", "graphics", "full", "full|half|quarter", RESTART),
    S("DETAIL_DISTANCE", "graphics", "2", "1|1.5|2|3|4", RESTART),
    S("SHADOW_SIZE", "graphics", "console", "console|1024|2048|3072|4096", RESTART),
    // Sound, Controls, Menu
    S("VOLUME", "audio", "100", "0..100", LIVE),
    S("AUDIO_CHANNELS", "audio", "auto", "auto|2|6", RESTART),
    S("BACKGROUND_MUTE", "audio", "0", "0|1", LIVE),
    S("MOTION", "input", "1", "0|1", LIVE),
    S("RUMBLE_STRENGTH", "input", "130", "0..300", LIVE),
    S("RUMBLE_CURVE", "input", "0.6", "0.2..1.5", LIVE),
    S("RUMBLE_TRIGGERS", "input", "0", "0|1", LIVE),
    S("PAD_LAYOUT", "input", "auto", "auto|nintendo|xbox", LIVE),
    S("PAD_P2", "input", "0", "0|1", LIVE),
    S("BUTTON_ICONS", "input", "auto", "auto|nintendo|xbox|playstation|keyboard", LIVE),
    S("MENU_SHORTCUT", "input", "minus_plus", "minus_plus|guide", LIVE),
    S("KEYBOARD_MOUSE", "input", "1", "0|1", LIVE),   // the Controls row that turns the keyboard and mouse off for the game (#663)
    S("MOUSE_SENS", "input", "1", "0..10", LIVE),
    S("STICK_DEADZONE", "input", "8", "0..40", LIVE),
    S("SWITCH2", "input", "0", "0|1|2", RESTART),
    // Cemuhook / DSU motion (input/dsu.c, #435): a server's gyro for a gamepad without one; off unless asked for. The
    // server's address (26760 is the port every DSU server uses) and its slot for player 1 have no menu row
    S("DSU", "input", "0", "0|1", LIVE),
    S("DSU_SERVER", "input", "127.0.0.1:26760", NULL, LIVE),
    S("DSU_SLOT", "input", "0", "0..3", LIVE),
    S("MENU_LANG", "language", "", "en|ja|fr|de|it|es|nl|pt|pt-BR|ru|zh-Hans|zh-Hant|ko|es-419|ca|gl|eu|pl|tr|uk|th", LIVE),
    S("UPDATE_CHECK", "launcher", "1", "0|1", LIVE),
    S("START_SCREEN", "launcher", "auto", "auto|always|skip", LIVE),   // ui/launcher/launcher.cpp console_start
    S("DISCORD", "launcher", "0", "0|1", LIVE),   // Discord Rich Presence (core/presence): off by default, the player's friends see it
    S("MODS", "launcher", "", NULL, RESTART),   // the Mods page's enabled mods, comma-separated in load order; not a menu row
    S("MODS_ACK", "launcher", "0", "0|1", LIVE),   // the player ticked the Mods page's "I understand" once
    S("MODS_PLAYED", "launcher", "", NULL, LIVE),   // the mods of the last start: a new set backs the save up first (launcher)
    S("GB_MATURE", "launcher", "obfuscate", "obfuscate|show", LIVE),   // the GameBanana tab's rated mods: blurred, or shown
                                                     // (a blurred page still names the content ratings)
    S("GAME_DIR", "launcher", "", NULL, RESTART),   // unset: beside the executable; the launcher remembers the selection there
    S("KEYBOARD_PLAYER", "input", "shared", "shared|own", RESTART),   // own: the keyboard alone is player 1 (desktop)
    // Multiplayer (runtime/net): local wireless off until two consoles have raced together over it
    S("LOCAL_WIRELESS", "multiplayer", "lan", "off|lan|online", RESTART),
    S("LOBBY", "multiplayer", "", NULL, RESTART),
    S("NET_NAME", "multiplayer", "", NULL, LIVE),
    S("NET_PORT", "multiplayer", "11460", "1024..65535", RESTART),
    S("NET_HOST", "multiplayer", "private", "private|public", LIVE),
    S("NET_REGION", "multiplayer", "any", "any|eu|na|sa|asia|oc|af", LIVE),
    S("NET_IFACE", "multiplayer", "", NULL, RESTART),
    S("NET_JOIN", "multiplayer", "", NULL, LIVE),
    S("NET_RELAY_ONLY", "multiplayer", "0", "0|1", LIVE),
    // Advanced: scheduling
    S("ASYNC_PREEMPT", "performance", "1", "0|1|2", RESTART),
    S("REPLAY_ASYNC", "performance", "1", "0|1", RESTART),
    S("REPLAY_PAR", "performance", REPLAY_PAR_DEFAULT, "0|1|2", RESTART),
    S("PACK", NULL, "0", "0|1", RESTART),   // PS5 (on in its profile): cold guest memory compressed (vk_mem_upload.c pack_step)
    S("PACK_TEXFREE", NULL, "0", "0|1", RESTART),   // PS5 (on in its profile, with PACK): the texture heap's free blocks packed first (vk_mem_upload.c texfree_pack)
    S("PACK_BACKOFF", NULL, "1", "0|1|2", RESTART),   // MKW_PACK: a unit read back waits twice as long before the next pack; 2: is never packed again
    S("FREE_CORES", "performance", "0", "0|1", RESTART),
    S("TIMER_1MS", "performance", "1", "0|1", RESTART),
    // shaders and pipelines
    S("PIPE_WAIT", "performance", "default", "default|draw", RESTART),
    S("PIPE_SKIP", "performance", "fullscreen", "strict|any|fullscreen", RESTART),
    S("PIPE_PREBUILD", "performance", "1", "0|1", RESTART),
    S("GPL", "performance", "1", "0|1", RESTART),
    // textures and memory
    S("ASTC", "performance", "auto", "auto|gpu|native|cpu", RESTART),
    S("MEMORY", "performance", "auto", "auto|import|mirror|verify", RESTART),
    S("TEX_ALIAS", "performance", "1", "0|1", RESTART),
    S("ZSTD_NATIVE", NULL, "1", "0|1", RESTART),
    S("ZSTD_CHECK", NULL, "0", "0|1", RESTART),
    // render fixes
    S("FIX_C0", "performance", "1", "0|1", RESTART),
    S("FIX_ZERO", "performance", "1", "0|1", RESTART),
    S("BIND_SKIP", "performance", "1", "0|1", RESTART),
    S("SHADER_MEMO", "performance", "1", "0|1", RESTART),
    S("PIPE_MEMO", "performance", "1", "0|1", RESTART),
    S("TEX_MEMO", "performance", "1", "0|1", RESTART),
    S("DESC_TEMPLATE", "performance", "1", "0|1", RESTART),
    S("PUSH_SKIP", "performance", "1", "0|1", RESTART),
    S("SPLIT_DRAWS", "performance", "64", "0..1000", RESTART),
    S("SPLIT_US", "performance", "2000", "0..20000", RESTART),
    // Vulkan
    S("VK_VALIDATION", "performance", "0", "0|1", RESTART),
    S("VK_DYN_VERTEX", "performance", "", "0|1", RESTART),
    S("VK_DYN_A2C", "performance", "", "0|1", RESTART),
    S("VK_SUBGROUP_PIN", "performance", "1", "0|1", RESTART),
    // diagnostics
    S("TRACE", "diagnostics", "", NULL, RESTART),
    S("FLIGHT", "diagnostics", "0", "0|1", RESTART),
    S("FLIGHT_SPIKE_MS", "diagnostics", "100", "0..60000", RESTART),
    S("SEQ_WATCH", "diagnostics", "0", "0|1", RESTART),
    S("SEQ_UNTIL", "diagnostics", "", NULL, RESTART),
    S("INPUT_RECORD", "input", "", NULL, RESTART),
    // Hacks: none of them on the console, all off by default
    S("TURBO", "hacks", "off", "off|hold|toggle|on|until", LIVE),
    S("TURBO_SPEED", "hacks", "4", "2..8", LIVE),
    S("SKIP_MOVIES", "hacks", "0", "0|1", LIVE),
    S("SEED", "hacks", "", NULL, RESTART),
    S("UNLOCK_ALL", "hacks", "0", "0|1", RESTART),
    S("FREEZE_KEYS", "hacks", "0", "0|1", LIVE),
    S("BENCH", "hacks", "off", "off|tt|gp|ko", RESTART),
    S("BENCH_MAX", "hacks", "0", "0..14400", RESTART),
    S("BENCH_RALLIES", "hacks", "0", "0..99", RESTART),
    S("BENCH_NO_DROPOUT", "hacks", "0", "0|1", RESTART),
    S("QUICK_RACE", "hacks", "off", "off|tt|gp|free|ko", RESTART),
    S("CLASSIC_GP", "hacks", "0", "0|1", LIVE),   // core/hacks/classic.c: the cups' intermissions become laps on one course
    S("CLASSIC_VS", "hacks", "0", "0|1", LIVE),   // core/hacks/classic.c: VS races' routes become laps on one course
    S("DYN_FPS", "hacks", "0", "0|1", LIVE),   // core/boot/game.c: 1-3 game steps a frame; on in the PS5 profile
    // core/boot/game.c: split screen (2-4 players on this machine) at 30 fps at most; same: MKW_FPS as for one player.
    // Off by default, the PS5 included since 27/11/2026: the cap held 2 players at 30 where they run ~46 fps
    S("SPLIT_FPS", "hacks", "same", "same|30", LIVE),   // core/boot/game.c: 1-3 game steps a frame, full speed from 20 to 60 fps
    // no menu row
    KS("BIND_A", "input"),
    KS("BIND_B", "input"),
    KS("BIND_DOWN", "input"),
    KS("BIND_L", "input"),
    KS("BIND_LCLICK", "input"),
    KS("BIND_LEFT", "input"),
    KS("BIND_LSTICK_DOWN", "input"),
    KS("BIND_LSTICK_LEFT", "input"),
    KS("BIND_LSTICK_RIGHT", "input"),
    KS("BIND_LSTICK_UP", "input"),
    KS("BIND_MINUS", "input"),
    KS("BIND_PLUS", "input"),
    KS("BIND_R", "input"),
    KS("BIND_RCLICK", "input"),
    KS("BIND_RIGHT", "input"),
    KS("BIND_RSTICK_DOWN", "input"),
    KS("BIND_RSTICK_LEFT", "input"),
    KS("BIND_RSTICK_RIGHT", "input"),
    KS("BIND_RSTICK_UP", "input"),
    KS("BIND_UP", "input"),
    KS("BIND_X", "input"),
    KS("BIND_Y", "input"),
    KS("BIND_ZL", "input"),
    KS("BIND_ZR", "input"),
    KS("LAN_PEERS", "language"),
    KS("PLAYER", "launcher"),
    KS("SHORTCUT_ASKED", "launcher"),
    K("AB"),
    K("AB_INVERT"),
    K("ADDR_SPIN"),
    K("ADDR_SPIN_US"),
    K("APP_POOL"),
    K("ASSUME_FOCUS"),
    K("ASTC_BATCH"),
    K("ASTC_BC1"),
    K("ASTC_BC1_ALPHA"),
    K("ASTC_BC7"),
    K("ASTC_BC_ARRAYS"),
    K("ASTC_BC_CHECK"),
    K("ASTC_ERRORS"),
    K("ASTC_ERRORS_LAG"),
    K("ASTC_GPU_DESW"),
    K("ASTC_TIME"),
    K("AUDIO"),
    K("AUDIO_OFF"),
    K("AUDIO_PREFILL_MS"),
    K("AUDIO_SKEW"),
    K("BACKGROUND_WINDOW"),
    K("BIND_STATS"),
    K("BIND_XF"),
    K("BLACK_CHECK"),
    K("CAP_FROM_INIT"),
    K("COLD_CENSUS"),
    K("COMPUTE_WRITTEN"),
    K("CORE_DIAG"),
    K("CORE_HANDOFF"),
    K("CRIT"),
    K("CRIT_OUT"),
    K("CTL"),
    K("DISCORD_APP"),
    K("DLSSG_DISABLE_BUF"),
    K("DLSSG_EXP"),
    K("DUMP"),
    K("DUMP_BUF"),
    K("DUMP_DIR"),
    K("DUMP_DRAWS"),
    K("DUMP_DRAWS_RECT"),
    K("DUMP_DRAWS_SKIP"),
    K("DUMP_LOG"),
    K("DYN_DIFF"),
    K("EARLY_BATCH"),
    K("EARLY_SIBLING_CMP"),
    K("EARLY_SIBLING_EMU"),
    K("EARLY_SIBLING_MAX"),
    K("EARLY_SIBLING_MAX_B"),
    K("EARLY_SIBLING_SNAP"),
    K("EARLY_SIBLING_SNAP_CHECK"),
    K("EARLY_SIBLING_STEP"),
    K("EARLY_SIBLING_TEST"),
    K("EARLY_SIBLING_WAIT"),
    K("EARLY_WAIT_AWAY"),
    K("EVENT_WAIT"),
    K("EXEFS"),
    K("FEEDBACK"),
    K("FG_AB"),
    K("FG_BIAS"),
    K("FG_DUMP"),
    K("FG_DUMP_DIR"),
    K("FG_FLIP_THREAD"),
    K("FG_HUDLESS"),
    K("FG_LAMBDA"),
    K("FG_LOAD"),
    K("FG_LOG_EVERY"),
    K("FG_MAILBOX"),
    K("FG_OCCUPANCY"),
    K("FG_PROBE"),
    K("FG_QUEUE"),
    K("FG_QUEUE_PRIO"),
    K("FG_TEST"),
    K("FG_VBLANK"),
    K("FG_XWAIT"),
    K("FLIGHT_AT"),
    K("FLIGHT_DIR"),
    K("FLIGHT_FRAMES"),
    K("FLIGHT_THUMBS"),
    K("FREEZE_IDLE"),
    K("GLYPH_DUMP"),
    K("GOV_EVICT_IDLE"),
    K("GOV_RATE_LIMIT"),
    K("GOV_TOTAL"),
    K("GPL_DRAW_BUILD"),
    K("GPU"),
    K("GPU_CRUMBS"),
    K("GPU_CRUMBS_DRAW"),
    K("GPU_MEM_LIMIT"),
    K("GPU_NAME"),
    K("GPU_ONLY_RELEASE"),
    K("HDR_FAKE"),
    K("HEADLESS"),
    K("HEAP_CENSUS"),
    K("HEAP_PROFILE"),
    K("HEAP_SHRINK_RELEASE"),
    K("HL_TRACE"),
    K("HOST_MEMSET"),
    K("HOST_ROUND"),
    K("HUD_PROBE"),
    K("IND_READ"),
    K("IND_READ_FRAMES"),
    K("IND_SNAP"),
    K("IND_SNAP_COUNT"),
    K("INPUT"),
    K("INPUT_SCRIPT"),
    K("LATENCY"),
    K("LIST_WATCH"),
    K("LOCALE"),
    K("LOGIC_HZ_TT"),
    K("MEMCPY_HOLD"),
    K("MENU_TEST"),
    K("MENU_TEST_DIR"),
    K("MENU_TEST_SET"),
    K("MESH_COMPUTE"),
    K("MINIMAL_LAYOUT"),
    K("MIRROR_GAP"),
    K("NGX_LOG"),
    K("ODD_CHECK"),
    K("ODD_CHECK_WAR"),
    K("ONE_PER_CORE"),
    K("PACK_COLD_S"),
    K("PACK_FEED_S"),
    K("PACK_STORE_MIB"),
    K("PACK_VERIFY"),
    K("PASS_HAZARDS"),
    K("PERF_SLOW_MS"),
    K("PHASE_LOCK"),
    K("PIPELINE_CACHE"),
    K("PIPE_AUTOSAVE_S"),
    K("PIPE_STALL"),
    K("PIPE_SYNC"),
    K("PIPE_TRACE"),
    K("PIPE_WORKERS"),
    K("POOL_LOG"),
    K("POOL_RELEASE"),
    K("POOL_SPILL"),
    K("PRECHECK_PAR"),
    K("PRECHECK_YIELD"),
    K("PROBE"),
    K("PROFILE"),
    K("PROFILE_CHAINS"),
    K("PROFILE_DIR"),
    K("PROFILE_FN"),
    K("PROFILE_FN_MAINWAIT"),
    K("PROFILE_FRAMES"),
    K("PROFILE_FROM"),
    K("PROFILE_MAIN"),
    K("PROFILE_ONLY"),
    K("PROFILE_RENDER"),
    K("PROFILE_US"),
    K("PS5_FAST_LOCK"),
    K("PS5_CCX"),
    K("PS5_PRIO_INHERIT"),
    K("PRESENT_COPY"),
    K("DESC_BUFFER"),
    K("PS5_OVERLAP"),
    K("PS5_PW_BATCH"),
    K("PS5_TSC_CLOCK"),
    K("PUSH_STAGE"),
    K("PW_FAULT_BATCH"),
    K("PW_HOT"),
    K("QORDER_LOG"),
    K("QORDER_LOG_AT"),
    K("QORDER_PARTS"),
    K("QORDER_PRE"),
    K("QUEUE_HOLD_MS"),
    K("QUEUE_WAITS"),
    K("RDOC"),
    K("RDOC_DIR"),
    K("RDOC_DLL"),
    K("RDX_SPLIT"),
    K("REC_SET_WRITE"),
    K("REEXEC"),
    K("REG_LATE"),
    K("REG_QUEUED"),
    K("REPLAY_CHECK"),
    K("REPLAY_PAR_CHECK_BY_CHUNK"),
    K("REPLAY_PAR_FIRST"),
    K("REPLAY_PAR_HEAD"),
    K("REPLAY_PAR_JOIN"),
    K("REPLAY_PAR_PIPE"),
    K("REPLAY_PAR_PRECHECK"),
    K("REPLAY_PAR_THREADS"),
    K("RESOLVE_CHECK"),
    K("REUP_CENSUS"),
    K("REUP_SAME"),
    K("ROMFS"),
    K("ROMFS_MODS"),
    K("ROMFS_PATCHES"),
    K("RTSIG"),
    K("RTSTATS"),
    K("RUMBLE"),
    K("SCREEN_RAW"),
    K("SHADOW"),
    K("SHADOW_AUTO_RUNS"),
    K("SHADOW_DRAWS"),
    K("SHADOW_SUBMIT"),
    K("SHADOW_TRIM"),
    K("SHADOW_UNC"),
    K("SHARED_FONTS"),
    K("SHOTS"),
    K("SHOTS_DIR"),
    K("SHOTS_FULL"),
    K("SKIP_PROGRAMS"),
    K("SNAP_BG"),
    K("SNAP_SCOPED"),
    K("SPARSE_EVICT"),
    K("SPARSE_IMAGES"),
    K("SPARSE_MIB"),
    K("SPARSE_NO_BUFFER"),
    K("SPIN"),
    K("SPIN_HANDOFF_US"),
    K("SPIRV_RELEASE"),
    K("STALL_TRACE"),
    K("STEAM_X11"),
    K("SVC_STATS"),
    K("SWKBD_ANSWER"),
    K("SWKBD_TEST"),
    K("SYNC_POLL_WAIT"),
    K("TEMPORAL_DEBUG"),
    K("TEMPORAL_HDR"),
    K("TEMPORAL_JSIGN"),
    K("TEMPORAL_KART_CIRCLE"),
    K("TEMPORAL_KART_DEPTH"),
    K("TEMPORAL_NGX"),
    K("TEX_EVICT_IDLE"),
    K("TEX_GPU_DESW"),
    K("TEX_LIVE_DROP_IDLE"),
    K("TEX_LOCK_SPIN"),
    K("TEX_MAKE_FAST"),
    K("TEX_OVERWRITTEN"),
    K("TEX_PAR"),
    K("TEX_PRESSURE"),
    K("TEX_READBACK"),
    K("TEX_RELEASED_MIB"),
    K("TEX_REUP_BANDS"),
    K("TEX_REUP_PARTIAL"),
    K("TEX_UP_DUMP"),
    K("TEX_UP_DUMP_DIR"),
    K("TEX_UP_LOG"),
    K("TEX_UP_LOG_AT"),
    K("TEX_ZERO_LOG"),
    K("THAW_FAST"),
    K("TRACE_UBO"),
    K("TRACE_UBO_OUT"),
    K("UBO_SNAPSHOT"),
    K("UNMAP_DEFER"),
    K("UNPACK_BATCH"),
    K("UPDATE_AS"),
    K("UPDATE_REPO"),
    K("USNAP_ASYNC"),
    K("USNAP_AT_SUBMIT"),
    K("USNAP_CMP"),
    K("USNAP_TAKE_THREAD"),
    K("VIRTUAL_PADS"),
    K("VK_NO_FLOAT64"),
    K("VK_RECORDER"),
    K("VK_SSBO_ALIGN"),
    K("VK_SUBGROUP"),
    K("VK_TEX_POOL_MIN"),
    K("VRAM_HEADROOM"),
    K("VRAM_LOG"),
    K("VRAM_MARGIN"),
    K("VSYNC_LOCK"),
    K("VS_OUTS"),
    K("WB_PAR"),
    K("YIELD_NOLOCK"),
    K("YIELD_PAUSE"),
    K("ZERO_SCAN_CACHE"),
};
#define N_ROWS (sizeof rows / sizeof *rows)

static int is_known(const char *key) {
    for (size_t i = 0; i < N_ROWS; i++)
        if (!strcmp(rows[i].m.key, key)) return 1;
    return 0;
}

static void clear(void) {
    for (int i = 0; i < n_entries; i++) free(entries[i].key), free(entries[i].value);
    n_entries = 0;
}

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) *--e = 0;
    return s;
}

// The upper-case key without the prefix, in place.
static char *norm_key(char *key) {
    for (char *p = key; *p; p++) *p = (char)toupper((unsigned char)*p);
    return strncmp(key, PREFIX, PREFIX_LEN) ? key : key + PREFIX_LEN;
}

static char *unquote(char *value) {
    size_t vl = strlen(value);
    if (vl >= 2 && value[0] == '"' && value[vl - 1] == '"') value[vl - 1] = 0, value++;
    return value;
}

static void add(char *key, char *value, int line) {
    key = norm_key(key);
    value = unquote(value);
    if (!*key) return;
    if (!is_known(key)) rt_log("cfg: mkw.ini line %d: unknown key '%s'", line, key);
    if (!*value) return;   // "key =" leaves the setting at its default
    for (int i = 0; i < n_entries; i++)
        if (!strcmp(entries[i].key, key)) {   // a repeated key: the later line wins
            free(entries[i].value);
            entries[i].value = strdup(value);
            return;
        }
    if (n_entries == CFG_MAX_KEYS) return;
    entries[n_entries++] = (CfgEntry){strdup(key), strdup(value), 0, 0};
}

static void load_entry(char *key, char *value, int line, void *ctx) {
    (void)ctx;
    add(key, value, line);
}

static int read_ini(const char *path, void (*visit)(char *, char *, int, void *), void *ctx) {
    uint64_t size = 0;
    uint8_t *data = plat_file_read(path, &size);
    if (!data) return 0;
    char *p = (char *)data;
    if (size >= 3 && !memcmp(p, "\xEF\xBB\xBF", 3)) p += 3;   // editors add a BOM
    for (int line = 1; *p; line++) {
        char *eol = strchr(p, '\n');
        if (eol) *eol = 0;
        char *s = trim(p);
        char *eq = strchr(s, '=');
        if (*s && *s != '#' && *s != ';' && *s != '[') {
            if (!eq) rt_log("cfg: mkw.ini line %d: no '=' in '%s'", line, s);
            else {
                *eq = 0;
                visit(trim(s), trim(eq + 1), line, ctx);
            }
        }
        if (!eol) break;
        p = eol + 1;
    }
    free(data);
    return 1;
}

int cfg_load(const char *path) {
    clear();
    return read_ini(path, load_entry, NULL);
}

typedef struct { const char *key; char *value; } ReloadKey;

static void read_key(char *key, char *value, int line, void *ctx) {
    (void)line;
    ReloadKey *r = ctx;
    value = unquote(value);
    if (strcmp(norm_key(key), r->key) || !*value) return;
    free(r->value);
    r->value = strdup(value);   // as at startup, the last nonempty value wins
}

int cfg_reload_key(const char *name) {
    char key[64], path[PLAT_PATH_MAX * 2];
    if (snprintf(key, sizeof key, "%s", name) >= (int)sizeof key) return 0;
    ReloadKey r = {norm_key(key), NULL};
    if (!*r.key) return 0;
    snprintf(path, sizeof path, "%s/mkw.ini", cfg_game_dir());
    if (!read_ini(path, read_key, &r) || !r.value) {
        rt_log("cfg: could not reload %s from %s; keeping its previous value", name, path);
        return 0;
    }
    // Setup changed one key on disk. Keep other keys (including unsaved menu edits) and the strings already handed
    // to readers; cfg_load would free those strings and discard all live settings. The reloaded key is already saved.
    plat_mutex_lock(&set_lock);
    int i = 0, n = n_entries;
    while (i < n && strcmp(entries[i].key, r.key)) i++;
    if (i == n && n == CFG_MAX_KEYS) {
        plat_mutex_unlock(&set_lock);
        free(r.value);
        return 0;
    }
    if (i == n) {
        entries[n] = (CfgEntry){strdup(r.key), r.value, 0, 0};
        __atomic_store_n(&n_entries, n + 1, __ATOMIC_RELEASE);
        __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    } else {
        if (!entries[i].value || strcmp(entries[i].value, r.value)) {
            __atomic_store_n(&entries[i].value, r.value, __ATOMIC_RELEASE);
            __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
        } else free(r.value);
        entries[i].dirty = entries[i].removed = 0;
    }
    plat_mutex_unlock(&set_lock);
    return 1;
}

// ---- device profiles: defaults for a known machine, below the environment and mkw.ini and above the table, so any
// key the player sets wins and resetting a key in the menu returns to the profile's value. MKW_DEVICE_PROFILE = auto
// (default: the PS5 on the PS5 build, the Steam Deck when the firmware says so), desktop (none), deck or ps5.
//
// The Steam Deck (LCD Jupiter, OLED Galileo; 1280x800, 16:10, 4 Zen 2 cores, RDNA 2 with 8 CUs sharing 16 GB): a
// window at the screen's 1280x800, fullscreen and vsync, as gamescope composes Game Mode.
// RESOLUTION 720: with "original" the output already shrinks to the 720 lines a 16:9 picture gets on the 800-line
// screen (patches.c output_size), but the 3D scene still renders at the console's docked 2560x1440 (romfs_res.c),
// passes of that size taking 16 of the GPU's 42 ms a frame (MKW_PASS_TIMING with MKW_VK_RECORDER=0). With 720 they
// go: race median 36.3 -> 23.4 ms (time trial, low quality preset, UMA 1 GB, 13/11/2026), 35.7 -> 24.6 ms without the
// timing; a docked Deck on a large screen can pick a higher one in the menu.
static const struct { const char *key, *value; } DECK[] = {
    {"FULLSCREEN", "1"}, {"WINDOW_SIZE", "1280x800"}, {"VK_PRESENT", "fifo"}, {"RESOLUTION", "720"},
};
// The PS5: the game's 60 fps, as the console runs it, and no frame generation (the table's default, off). Until
// 20/11/2026 the profile set 30 fps with frame generation at 2x: a race frame's work was ~26-27 ms. With the shadow copies
// off and the replay's work cut it is ~17 ms: a Grand Prix race at 60 averages 17.0-17.4 ms, window medians at the cap
// (16.7), 1% lows 35-41 fps, game speed 0.95-0.98x (frames over 16.7 ms slow the race down; at 30 it kept 1.00x). 60 and
// not auto: auto swapped the 60 and 30 Hz intervals ~1,400 times a minute when frames ran ~26 ms, not measured since.
// 30 and frame generation stay in Pit Stop > Display.
static const struct { const char *key, *value; } PS5[] = {
    {"FPS", "60"},
    // No split screen cap since 27/11/2026: MKW_SPLIT_FPS=30 is a hack (Hacks tab), off. Measured 22/11/2026 (Grand Prix
    // benchmark at 60): 2 players 21.7 ms (46 fps, the race at 0.77x), 3 and 4 players 41.9 ms (24 fps, 0.40x); at 30
    // a split frame has 33.3 ms at full speed. MKW_DYN_FPS keeps the game at full speed without it.
    // Dynamic frame rate (core/boot/game.c), on since 27/11/2026: the frames that miss 16.7 ms take 2 or 3 game steps
    // instead of slowing the race (Grand Prix, 210 s, eboot 71a478db: 1 player 0.94-0.98x -> 1.00x at 57.6 fps, 2
    // players 0.75x -> 1.00x at 45.4 fps; dyn-1p-dyn, dyn-2p-dyn, dyn-2p-off). Off in Pit Stop > Hacks.
    {"DYN_FPS", "1"},
    // The parallel replay as before the 0.1.13 merge (18/11/2026, one player at MKW_FPS=60): the serial head records
    // ~600 draws a frame on the replay thread alone, 5.8-6.3 ms on these cores (reg-1p60: race median 25.9 against
    // 25.1 ms without); the slots checked by chunk and the replay thread recording after its walk (vk_exec_par.c).
    // Partition Zen 2's two CCXs (vk_backend.c replay_core): game threads on CCX-0 (CPUs 0-7, shared L3), Vulkan
    // replay, parallel chunk workers, submit and present threads on CCX-1 (CPUs 8-12, shared L3). In-race A/B on base
    // PS5 (ccx-ab-live, 28/11/2026, 150 A vs 150 B windows, 1.8.0 GP 1080p): frame 17.61 -> 17.06 ms (-0.55 ms,
    // median 16.95 -> 16.69 ms), busy 16.45 -> 13.63 ms (-2.82 ms), replay 12.29 -> 10.27 ms (-2.02 ms).
    {"PS5_CCX", "1"},
    {"REPLAY_PAR_HEAD", "0"},
    {"REPLAY_PAR_CHECK_BY_CHUNK", "1"},
    {"REPLAY_PAR_JOIN", "1"},
    // The walk in full before the chunks go out, as before 16/11/2026: handing them out during the walk gave single
    // game frames with the lit scene black, 5-55 a race against 0-1 without (black-flash runs pipeC-on / pipeC-off,
    // 18/11/2026; vk_tex_alias.c), for ~2 fps (27.8 against 29.9). Until that is fixed where it breaks; ab or 1 in
    // mkw.ini brings it back.
    {"REPLAY_PAR_PIPE", "0"},
    // CPU-visible pools imported, no shadow copies (vk_mem_shadow.c): shadows answer the cost of importing on Linux
    // (amdgpu userptrs), which the console's unified memory does not have, and their copies and fence write-backs sat in
    // the main thread's wait for the last frame's fence. Races 22.0-22.4 ms with them, 17.6-17.8 without (noshadow1/2,
    // 19/11/2026; GPU 7.6 -> 6.2 ms, free direct memory median 375 -> 642 MiB); races that happened to start after the
    // governor had shed shadows (vk_mem_shadow_shed) were the fast ones at 18.8-19.7. Menus checked by the user.
    {"SHADOW", "0"},
    // Keep shadow storage + submit-time capture for the 3 CPU_UNCACHED|GPU_UNCACHED dynamic ring-buffer pools
    // (~38 MiB total, vk_mem_back.c / vk_mem_capture.c): with SHADOW=0 and REPLAY_PAR=1, the game overwrote dynamic
    // SSBO/bone/transform slices in pool 0x2487069000 before replay and during GPU execution, stretching 3D meshes
    // across the screen every 6-30 frames (123-322 odd frames per Grand Prix race in MKW 1.8.0). Shadowing only those
    // 3 uncached pools with submit-time capture eliminates the corruption (0 odd frames across 6,180 frames;
    // verify-shunc-sub, 24/11/2026) at ~0.08-0.13 ms/sub on the game thread and 0.0 MiB governor trim.
    {"SHADOW_UNC", "1"},
    {"SHADOW_SUBMIT", "1"},
    // The uniform snapshot's bytes taken at the submit (vk_usnap_take.c), on a thread of its own (USNAP_TAKE_THREAD):
    // read later, during the replay's walk (USNAP_ASYNC), the game had already rewritten its double-buffered frame
    // blocks, and the countdown's scene lost half its scenery to the sky or went dark every few frames, 19-125 odd
    // frames a 90 s Grand Prix; 0 and 0 with the take at the submit, on the game's thread (cyan-atsub) and on its own
    // (take-fix1/2, against 73 without, take-ctrl), +0.04 ms a frame in-race A/B (take-ab2-inv0/1, 25/11/2026).
    {"USNAP_AT_SUBMIT", "1"},
    // Upstream's texture workers (MKW_TEX_PAR, tex_par.c) off, as before the 22/11/2026 merge: tried on the console
    // 19/11/2026 (texpar-*, six Grand Prix runs) the large deswizzles took 40 % less time but the race frame did not
    // move (A/B medians -0.06 ms) and the 0.1 % lows were ~1 fps worse. The ASTC alpha scan and decode keep their own
    // pool (astc_mt_for, MKW_TEX_MAKE_FAST).
    {"TEX_PAR", "0"},
    // Upstream's VRAM pressure eviction (vk_tex_pressure.c, off there): the console's direct memory is one 12,096 MiB
    // pool the game's pages and the GPU's share, and VMA's device-local blocks sit at ~90-93 % of their budget in a
    // race, so the idle recreatable textures (decoded ASTC, ~1.1 GiB) go before memory runs out (issue 11). Four 1.8.0
    // Grand Prix runs (fix180-*, 24/11/2026, with PR #15): lowest free direct memory 130 -> 166 MiB, 10th percentile
    // 242 -> 318 MiB, census minimum 168 -> 276 MiB; 1,039 textures (370 MiB) out in 4 passes; frame 17.53 -> 17.50 ms,
    // same 57.1 fps.
    {"TEX_PRESSURE", "1800"},
    // Texture checks peek their own pages instead of re-arming the 2 MiB write-watch snapshot (vk_mem_gen.c
    // vk_mem_changed_scoped). In-race A/B, two 1.0.0 races (24/11/2026): sceKernelMprotect arms 0.77 -> 0.24-0.28 ms a
    // frame, the game's write faults 13-14 -> 2.7-4.2, precheck spikes 11.0 -> 4.8 % and 6.9 -> 2.9 % of frames, busy
    // -0.55 / -0.84 ms. A 1.8.0 race with it on was watched by the user: no stale textures. PS5 only: Linux and Windows
    // have no peek (plat_mem_written_any answers "may be written"), so each check there would query its span.
    {"SNAP_SCOPED", "1"},
    // Cold guest memory packed (vk_mem_upload.c pack_step): the cold feed freezes the 2 MiB regions unwritten for 60 s
    // and the governor compresses their 64 KiB units into a direct-memory store. 1.8.0 Grand Prix (pack-gain-20261124,
    // with TEX_PRESSURE): 1,573 MiB packed into 453 MiB, hard governor passes 1,827 -> 87, census minimum free 168 ->
    // 304 MiB, frame 17.52 -> 17.50 ms; MKW_PACK_VERIFY found 0 of 3,447 unpacked units different (pack-verify).
    {"PACK", "1"},
    {"PACK_STORE_MIB", "1024"},
    // The free blocks of agl::TextureDataMgr's texture heap packed first (vk_mem_upload.c texfree_pack): 300-650 MiB of
    // backed memory sat in them while racing a Knockout Tour and 1,349 MiB at a rally's change (koTEX2, 27/11/2026).
    {"PACK_TEXFREE", "1"},
    // Re-uploads of a written texture whose bytes are those of its last upload left out (vk_tex_upload.c reup_skip):
    // 99.6 % of a 1.8.0 race's re-uploads (2,789 of 2,801, 678 MiB in 1,800 frames; reup-ab1), the write watch flagging
    // pages whose texture bytes did not change. In-race A/B: frame -0.11 / -0.49 ms, busy -0.10 / -0.91 (25/11/2026).
    {"REUP_SAME", "1"},
    // Large guest memcpy/memset destinations held writable in runs before the copy instead of a write-watch fault per
    // 64 KiB (nvn_api_core.c guest_write_hold): in-race A/B, two 1.8.0 Grand Prix (hold-ab / hold-inv, 28/11/2026),
    // 1 % low 37.9 -> 40.5 and 36.5 -> 42.6 fps, 0.1 % low 21.0 -> 20.4 and 15.9 -> 29.2 fps, frame -0.10 ms balanced;
    // the write-watch fault storms behind the 1 % low's page-protection frames left B's slowest frames.
    {"MEMCPY_HOLD", "1"},
    // The replay's precheck peeks its clean texture slots in parallel on the ASTC pool (vk_tex_slots.c
    // vk_texture_precheck_par): in-race A/B, two 1.8.0 Grand Prix (pp-ab0 / pp-ab1, 28/11/2026), ~2,100 slots a frame,
    // 1 % low 39.5 -> 41.3 and 38.3 -> 42.0 fps, frames over 20 ms 4.1 -> 2.7 % and 3.5 -> 3.2 %, frame unchanged; a
    // race watched by the user with it on showed no stale texture.
    {"PRECHECK_PAR", "1"},
    // A unit's backing (an unpack, a fresh commit) makes its page protections in one batch (ps5_guest_memory.c
    // back_unit_locked): in-race A/B, two 1.8.0 Grand Prix (ub-ab0 / ub-ab1, 28/11/2026), the slow frames' backing time
    // outside the commits' own steps 254 -> 71 and 435 -> 52 ms, 1 % low 42.0 -> 43.5 and 40.7 -> 44.2 fps, frames
    // over 33.3 ms 12 -> 14 and 20 -> 6, frame unchanged.
    {"UNPACK_BATCH", "1"},
    // The console's own controller: PlayStation icons rather than auto, which guesses from the pad's labels (Xbox when
    // it reports none), and the DualSense's vibration at 100 % (the table's 130 % suits PC pads; asked for by the user,
    // 24/11/2026). Both stay in Pit Stop > Controls.
    {"BUTTON_ICONS", "playstation"},
    {"RUMBLE_STRENGTH", "100"},
};
static int profile = -1;   // 0 none, 1 Steam Deck, 2 PS5

// MKW_BENCH=tt|gp|ko (core/diag/bench.c): the way to its races, below what the player set and above the device profile,
// so a benchmark run is the same time trial, Grand Prix or Knockout Tour on every machine. QUICK_RACE takes the mode's own name.
static const struct { const char *key, *value; } BENCH[] = {
    {"QUICK_RACE", NULL}, {"SEED", "1"}, {"SKIP_MOVIES", "1"},
};

static int read_line(const char *path, char *out, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int ok = fgets(out, (int)n, f) != NULL;
    fclose(f);
    out[ok ? strcspn(out, "\r\n") : 0] = 0;
    return ok;
}

// The Steam Deck: Valve's board, product Jupiter or Galileo (/sys/class/dmi/id); SteamDeck=1, which Steam sets on it.
static int on_steam_deck(void) {
    const char *env = getenv("SteamDeck");
    if (env && !strcmp(env, "1")) return 1;
    char vendor[64], product[64];
    return read_line("/sys/class/dmi/id/board_vendor", vendor, sizeof vendor) && !strcmp(vendor, "Valve") &&
           read_line("/sys/class/dmi/id/product_name", product, sizeof product) &&
           (!strcmp(product, "Jupiter") || !strcmp(product, "Galileo"));
}

static const char *raw_get(const char *name);

static int profile_of(void) {
    int p = __atomic_load_n(&profile, __ATOMIC_ACQUIRE);
    if (p >= 0) return p;
    const char *set = raw_get(PREFIX "DEVICE_PROFILE");
    if (set && !strcmp(set, "deck")) p = 1;
    else if (set && !strcmp(set, "ps5")) p = 2;
    else if (set && *set && strcmp(set, "auto")) p = 0;
#ifdef MKW_PS5_NATIVE
    else p = 2;
#else
    else p = on_steam_deck();
#endif
    __atomic_store_n(&profile, p, __ATOMIC_RELEASE);
    return p;
}

static const char *raw_get(const char *name) {
    const char *e = getenv(name);
    if (e || strncmp(name, PREFIX, PREFIX_LEN)) return e;
    int n = __atomic_load_n(&n_entries, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++)
        if (!strcasecmp(entries[i].key, name + PREFIX_LEN)) return __atomic_load_n(&entries[i].value, __ATOMIC_ACQUIRE);
    return NULL;
}

// Started from Steam's Big Picture or Gaming Mode (cfg.h): fullscreen, as a console shows its game; Big Picture stayed
// visible around a window before.
static const struct { const char *key, *value; } CONSOLE[] = {{"FULLSCREEN", "1"}};

int cfg_steam_started(void) {
    const char *id = getenv("SteamGameId"), *app = getenv("SteamAppId");
    return (id && *id) || (app && *app);
}

int cfg_steam_console(void) {
    const char *ui = getenv("SteamGamepadUI"), *tenfoot = getenv("SteamTenfoot"), *gs = getenv("GAMESCOPE_WAYLAND_DISPLAY");
    return (ui && !strcmp(ui, "1")) || (tenfoot && !strcmp(tenfoot, "1")) || (gs && *gs);
}

const char *cfg_steam_video_driver(void) {
#ifdef _WIN32
    // The Windows build under Wine/Proton sees the Linux session's variables too (DISPLAY, WAYLAND_DISPLAY, Steam's),
    // and SDL has no x11 driver there: "SDL video init failed (x11 not available)", no window and no Vulkan device, for
    // the game, the start screen and the setup's prewarm alike (SteamOS desktop, Proton 11.0, 20/11/2026). Wine's own
    // driver puts the window on X11 or Wayland.
    return NULL;
#else
    const char *own = getenv("SDL_VIDEODRIVER"), *wl = getenv("WAYLAND_DISPLAY"), *x = getenv("DISPLAY"),
               *gs = getenv("GAMESCOPE_WAYLAND_DISPLAY"), *off = cfg_get("MKW_STEAM_X11");
    if ((own && *own) || (gs && *gs) || (off && !strcmp(off, "0")) || !cfg_steam_started()) return NULL;
    return wl && *wl && x && *x ? "x11" : NULL;
#endif
}

const char *cfg_get(const char *name) {
    const char *v = raw_get(name);
    // A set key (environment, mkw.ini, the menu) wins over the profile.
    if (v || strncmp(name, PREFIX, PREFIX_LEN)) return v;
    for (size_t i = 0; i < sizeof BENCH / sizeof *BENCH; i++)   // MKW_BENCH read only for its own keys
        if (!strcasecmp(BENCH[i].key, name + PREFIX_LEN)) {
            const char *b = raw_get(PREFIX "BENCH");
            if (b && (!strcmp(b, "tt") || !strcmp(b, "gp") || !strcmp(b, "ko"))) return BENCH[i].value ? BENCH[i].value : b;
        }
    int p = profile_of();
    if (p == 1)
        for (size_t i = 0; i < sizeof DECK / sizeof *DECK; i++)
            if (!strcasecmp(DECK[i].key, name + PREFIX_LEN)) return DECK[i].value;
    if (p == 2)
        for (size_t i = 0; i < sizeof PS5 / sizeof *PS5; i++)
            if (!strcasecmp(PS5[i].key, name + PREFIX_LEN)) return PS5[i].value;
    if (p == 0 && cfg_steam_console())
        for (size_t i = 0; i < sizeof CONSOLE / sizeof *CONSOLE; i++)
            if (!strcasecmp(CONSOLE[i].key, name + PREFIX_LEN)) return CONSOLE[i].value;
    return NULL;
}

unsigned cfg_version(void) { return __atomic_load_n(&version, __ATOMIC_ACQUIRE); }

int cfg_from_env(const char *name) { return getenv(name) != NULL; }

void cfg_set(const char *name, const char *value) {
    if (strncmp(name, PREFIX, PREFIX_LEN) || !value) return;
    char key[64];
    snprintf(key, sizeof key, "%s", name + PREFIX_LEN);
    for (char *p = key; *p; p++) *p = (char)toupper((unsigned char)*p);
    plat_mutex_lock(&set_lock);
    int i = 0, n = n_entries;
    while (i < n && strcmp(entries[i].key, key)) i++;
    if (i < n) {
        if (!entries[i].value || strcmp(entries[i].value, value)) {
            __atomic_store_n(&entries[i].value, strdup(value), __ATOMIC_RELEASE);   // the old string stays for readers
            entries[i].dirty = 1;
        }
        entries[i].removed = 0;   // a value set after cfg_reset replaces the line instead of deleting it
    } else if (n < CFG_MAX_KEYS) {
        entries[n] = (CfgEntry){strdup(key), strdup(value), 1, 0};
        __atomic_store_n(&n_entries, n + 1, __ATOMIC_RELEASE);
    }
    __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&set_lock);
}

int cfg_reset(const char *const *keys, int n) {
    int done = 0;
    plat_mutex_lock(&set_lock);
    for (int k = 0; k < n; k++) {
        char key[64], var[80];
        snprintf(key, sizeof key, "%s", keys[k]);
        const char *name = norm_key(key);
        snprintf(var, sizeof var, PREFIX "%s", name);
        if (getenv(var)) continue;   // the environment wins; the file line stays
        int i = 0;
        while (i < n_entries && strcmp(entries[i].key, name)) i++;
        if (i == n_entries) {   // no value in memory ("key =" in the file): the line still has to go
            if (i == CFG_MAX_KEYS) continue;
            entries[i] = (CfgEntry){strdup(name), NULL, 0, 1};
            __atomic_store_n(&n_entries, i + 1, __ATOMIC_RELEASE);
        } else if (entries[i].value) {
            __atomic_store_n(&entries[i].value, NULL, __ATOMIC_RELEASE);   // the old string stays for readers
            entries[i].dirty = 0;
            entries[i].removed = 1;   // cfg_save_to deletes the key's lines
            done++;
        } else {
            entries[i].removed = 1;
        }
    }
    if (done) __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&set_lock);
    return done;
}

int cfg_last_save_ok(void) { return __atomic_load_n(&save_ok, __ATOMIC_ACQUIRE); }

void cfg_each_active(void (*cb)(const char *key, const char *value, int from_env, void *ctx), void *ctx) {
    int n = __atomic_load_n(&n_entries, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++) {
        const char *v = __atomic_load_n(&entries[i].value, __ATOMIC_ACQUIRE);
        if (v) cb(entries[i].key, v, 0, ctx);
    }
    for (size_t i = 0; i < N_ROWS; i++) {
        char var[80];
        snprintf(var, sizeof var, PREFIX "%s", rows[i].m.key);
        const char *e = getenv(var);
        if (e) cb(rows[i].m.key, e, 1, ctx);
    }
}

const CfgMeta *cfg_meta(const char *key) {
    char k[64];
    snprintf(k, sizeof k, "%s", key);
    const char *name = norm_key(k);
    for (size_t i = 0; i < N_ROWS; i++)
        if (rows[i].m.def && !strcmp(rows[i].m.key, name)) return &rows[i].m;   // def NULL: a K or KS row, no menu row
    return NULL;
}

const CfgMeta *cfg_meta_at(int i) {
    for (size_t r = 0; i >= 0 && r < N_ROWS; r++)
        if (rows[r].m.def && !i--) return &rows[r].m;
    return NULL;
}

const char *cfg_value(const char *name) {
    const char *e = cfg_get(name);
    if (e && *e) return e;
    const CfgMeta *m = cfg_meta(name);
    if (!m) rt_fatal("cfg: %s has no row in the settings table", name);
    return m->def;
}

// The row's default as a number, for a set value that does not parse.
static double default_number(const char *name) {
    return strtod(cfg_meta(name)->def, NULL);
}

double cfg_double(const char *name) {
    const char *v = cfg_value(name);
    char *end;
    double d = strtod(v, &end);
    return end == v || *end || d != d ? default_number(name) : d;
}

int cfg_int(const char *name) {
    const char *v = cfg_value(name);
    char *end;
    long l = strtol(v, &end, 10);
    return end == v || *end ? (int)default_number(name) : (int)l;
}

// "0", "off", "false", "no" are false and "1", "on", "true", "yes" true, in any case; another number is true as it was; a
// word that is none of these gives the default, as cfg_int does. Only "0" was false before: MKW_DISCORD=off switched
// Rich Presence on, MKW_LAN=off the LAN sockets and MKW_UPDATE_CHECK=off the update check, all rows of "0|1".
int cfg_bool(const char *name) {
    static const char *const no[] = {"0", "off", "false", "no"}, *const yes[] = {"1", "on", "true", "yes"};
    const char *v = cfg_value(name);
    for (size_t i = 0; i < sizeof no / sizeof *no; i++)
        if (!strcasecmp(v, no[i])) return 0;
    for (size_t i = 0; i < sizeof yes / sizeof *yes; i++)
        if (!strcasecmp(v, yes[i])) return 1;
    char *end;
    long l = strtol(v, &end, 10);
    if (end != v && !*end) return l != 0;
    const CfgMeta *m = cfg_meta(name);
    return m ? strcmp(m->def, "0") != 0 : 1;   // a row without a default (K) keeps the old reading: true
}

// ---- cfg_save: the file as lines, each with its line ending
typedef struct { const char *s; size_t n; } Line;   // n includes the '\n' (and '\r') when present

static int line_count(const char *p, size_t size) {
    int n = 0;
    for (size_t i = 0; i < size; i++) n += p[i] == '\n';
    return n + 1;
}

// The line's text without its ending, BOM and surrounding blanks, into out.
static void line_text(const Line *l, int first, char *out, size_t cap) {
    size_t n = l->n;
    const char *s = l->s;
    if (first && n >= 3 && !memcmp(s, "\xEF\xBB\xBF", 3)) s += 3, n -= 3;
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) n--;
    if (n >= cap) n = cap - 1;
    memcpy(out, s, n);
    out[n] = 0;
    char *t = trim(out);
    memmove(out, t, strlen(t) + 1);
}

// The key of a "key = value" line (comment = 0) or of a commented "# key = value" line (comment = 1), normalised,
// into key; 0 when the line is not one.
static int line_key(const char *text, int comment, char *key, size_t cap) {
    if (comment) {
        if (*text != '#' && *text != ';') return 0;
        text++;
        while (*text == ' ' || *text == '\t') text++;
    }
    if (!*text || *text == '#' || *text == ';' || *text == '[') return 0;
    const char *eq = strchr(text, '=');
    if (!eq) return 0;
    size_t n = (size_t)(eq - text);
    while (n && (text[n - 1] == ' ' || text[n - 1] == '\t')) n--;
    if (!n || n >= cap) return 0;
    for (size_t i = 0; i < n; i++)
        if (!isalnum((unsigned char)text[i]) && text[i] != '_') return 0;
    char tmp[128];
    memcpy(tmp, text, n);
    tmp[n] = 0;
    snprintf(key, cap, "%s", norm_key(tmp));
    return 1;
}

// The section name of a "[name]" line, lower-case, into out; 0 when the line is not one.
static int line_section(const char *text, char *out, size_t cap) {
    if (*text != '[') return 0;
    const char *e = strchr(text, ']');
    if (!e) return 0;
    char tmp[128];
    size_t n = (size_t)(e - text - 1);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    memcpy(tmp, text + 1, n);
    tmp[n] = 0;
    snprintf(out, cap, "%s", trim(tmp));
    for (char *p = out; *p; p++) *p = (char)tolower((unsigned char)*p);
    return 1;
}

typedef struct { char *buf; size_t n, cap; } Out;
static void put(Out *o, const char *s, size_t n) {
    if (o->n + n + 1 > o->cap) {
        o->cap = (o->n + n + 1) * 2;
        o->buf = realloc(o->buf, o->cap);
        if (!o->buf) rt_fatal("cfg: out of memory writing mkw.ini");
    }
    memcpy(o->buf + o->n, s, n);
    o->n += n;
}
static void puts_(Out *o, const char *s) { put(o, s, strlen(s)); }

static const char *section_of(const char *key) {
    for (size_t i = 0; i < N_ROWS; i++)
        if (!strcmp(rows[i].m.key, key)) return rows[i].section;
    return NULL;
}

static void put_value(Out *o, const CfgEntry *e) {
    // Keep the in-memory path literal. The parser trims whitespace outside quotes, then removes just the outer
    // quotes: paths ending in spaces/tabs, and literal quotes or backslashes inside them, must survive a restart.
    int quote = !strcmp(e->key, "GAME_DIR");
    if (quote) puts_(o, "\"");
    puts_(o, e->value);
    if (quote) puts_(o, "\"");
}

// "key = value" + eol for entry e (key lower-case)
static void put_entry(Out *o, const CfgEntry *e, const char *eol) {
    char k[64];
    snprintf(k, sizeof k, "%s", e->key);
    for (char *p = k; *p; p++) *p = (char)tolower((unsigned char)*p);
    puts_(o, k);
    puts_(o, " = ");
    put_value(o, e);
    puts_(o, eol);
}

int cfg_save_to(const char *path) {
    plat_mutex_lock(&set_lock);
    int n = n_entries, any = 0;
    for (int i = 0; i < n; i++) any |= entries[i].dirty | entries[i].removed;
    if (!any) {
        __atomic_store_n(&save_ok, 1, __ATOMIC_RELEASE);
        plat_mutex_unlock(&set_lock);
        return 1;
    }
    uint64_t size = 0;
    uint8_t *data = plat_file_read(path, &size);
    if (!data && plat_file_exists(path) && plat_file_size(path) != 0) {
        // the file is there but unreadable (locked, transient error): rebuilding it from nothing would lose its lines
        rt_log("cfg: cannot read %s, nothing written", path);
        __atomic_store_n(&save_ok, 0, __ATOMIC_RELEASE);
        plat_mutex_unlock(&set_lock);
        return 0;
    }
    const char *text = data ? (const char *)data : "";
    int nl = data ? line_count(text, size) : 0;
    Line *lines = calloc((size_t)nl + 1, sizeof *lines);
    // after[i]: entries to write after line i (-1 = none); replace[i]: the entry whose value line i now carries
    int *replace = malloc(sizeof(int) * ((size_t)nl + 1)), *after_key = malloc(sizeof(int) * (size_t)(n + 1));
    char *drop = calloc((size_t)nl + 1, 1);   // lines of a reset key
    if (!lines || !replace || !after_key || !drop) rt_fatal("cfg: out of memory writing mkw.ini");
    for (int i = 0, at = 0; i < nl; i++) {
        size_t start = (size_t)at;
        while ((uint64_t)at < size && text[at] != '\n') at++;
        if ((uint64_t)at < size) at++;
        lines[i] = (Line){text + start, (size_t)at - start};
        replace[i] = -1;
    }
    if (nl && !lines[nl - 1].n) nl--;   // the text ended with a newline: no last empty line
    const char *eol = data && strstr(text, "\r\n") ? "\r\n" : "\n";
    char t[1024], k[128], sec[128];
    for (int e = 0; e < n; e++) {
        after_key[e] = -2;   // -2 clean, -1 append in a new section at the end, >= 0 insert after that line
        if (entries[e].removed)
            for (int i = 0; i < nl; i++) {
                line_text(&lines[i], i == 0, t, sizeof t);
                if (line_key(t, 0, k, sizeof k) && !strcmp(k, entries[e].key)) drop[i] = 1;
            }
        if (!entries[e].dirty) continue;
        int last = -1;
        for (int i = 0; i < nl; i++) {
            line_text(&lines[i], i == 0, t, sizeof t);
            if (line_key(t, 0, k, sizeof k) && !strcmp(k, entries[e].key)) last = i;
        }
        if (last >= 0) {
            replace[last] = e;
            continue;
        }
        const char *want = section_of(entries[e].key);
        int in = 0, example = -1, tail = -1;
        for (int i = 0; i < nl && want; i++) {
            line_text(&lines[i], i == 0, t, sizeof t);
            if (line_section(t, sec, sizeof sec)) {
                in = !strcmp(sec, want);
                if (in) tail = i;
                continue;
            }
            if (!in) continue;
            if (*t) tail = i;
            if (line_key(t, 1, k, sizeof k) && !strcmp(k, entries[e].key)) example = i;
        }
        after_key[e] = example >= 0 ? example : tail;
    }
    Out o = {0};
    for (int i = 0; i < nl; i++) {
        if (drop[i]) {
            // the line leaves the file; keys inserted after it still need a line ending before them
        } else if (replace[i] >= 0) {
            // keep the line's indentation and key spelling, put the new value after " = "
            const char *s = lines[i].s;
            size_t len = lines[i].n;
            if (i == 0 && len >= 3 && !memcmp(s, "\xEF\xBB\xBF", 3)) put(&o, s, 3), s += 3, len -= 3;
            const char *eq = memchr(s, '=', len);
            size_t kn = (size_t)(eq - s);
            while (kn && (s[kn - 1] == ' ' || s[kn - 1] == '\t')) kn--;
            put(&o, s, kn);
            puts_(&o, " = ");
            put_value(&o, &entries[replace[i]]);
            int had_eol = len && s[len - 1] == '\n';
            if (had_eol) puts_(&o, eol);
        } else {
            put(&o, lines[i].s, lines[i].n);
        }
        for (int e = 0; e < n; e++)
            if (after_key[e] == i) {
                if (o.n && o.buf[o.n - 1] != '\n') puts_(&o, eol);
                put_entry(&o, &entries[e], eol);
            }
    }
    // keys whose section the file lacks: one new section each, at the end
    for (int e = 0; e < n; e++) {
        if (after_key[e] != -1) continue;
        const char *want = section_of(entries[e].key);
        if (o.n && o.buf[o.n - 1] != '\n') puts_(&o, eol);
        if (want) {
            if (o.n) puts_(&o, eol);
            puts_(&o, "[");
            puts_(&o, want);
            puts_(&o, "]");
            puts_(&o, eol);
        }
        put_entry(&o, &entries[e], eol);
        // the following keys of the same new section go right after this one
        for (int f = e + 1; f < n && want; f++)
            if (after_key[f] == -1 && section_of(entries[f].key) && !strcmp(section_of(entries[f].key), want)) {
                put_entry(&o, &entries[f], eol);
                after_key[f] = -3;
            }
    }
    int ok = plat_file_write_atomic(path, o.buf ? o.buf : "", o.n);
    if (ok)
        for (int e = 0; e < n; e++) entries[e].dirty = entries[e].removed = 0;
    else
        rt_log("cfg: cannot write %s", path);
    __atomic_store_n(&save_ok, ok ? 1 : 0, __ATOMIC_RELEASE);
    free(o.buf);
    free(drop);
    free(after_key);
    free(replace);
    free(lines);
    free(data);
    plat_mutex_unlock(&set_lock);
    return ok;
}

static char ini_path[PLAT_PATH_MAX * 3], launcher_ini[PLAT_PATH_MAX * 3], game_dir[PLAT_PATH_MAX * 2];
static int game_dir_invalid;

static int resolve_game_dir(const char *path, char *out, size_t cap) {
    if (!path || !*path || strchr(path, '\n')) return 0;
    const char *cut = plat_last_sep(launcher_ini);
    int absolute = path[0] == '/' || path[0] == '\\' || path[1] == ':';
    if (!absolute && !cut) return 0;
    int n = absolute ? snprintf(out, cap, "%s", path)
                     : snprintf(out, cap, "%.*s/%s", (int)(cut - launcher_ini), launcher_ini, path);
    if (n < 0 || n > CFG_GAME_DIR_MAX || (size_t)n >= cap) return 0;
#ifdef _WIN32
    // The manifest does not opt into long paths, and Win32 length counts UTF-16 units rather than UTF-8 bytes.
    // The byte limit above guarantees this conversion fits without truncating; include room for derived paths.
    // https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
    uint16_t wide[CFG_GAME_DIR_MAX + 1];
    if (utf8_to_utf16(out, wide, CFG_GAME_DIR_MAX + 1) > CFG_GAME_DIR_WIN32_MAX) return 0;
#endif
    return 1;
}

static int find_ini(void) {
    if (ini_path[0]) return 1;
    char path[PLAT_PATH_MAX * 2];
    if (!plat_exe_path(path, sizeof path)) return 0;
    char *cut = plat_last_sep(path);
    if (!cut) cut = strrchr(path, '/');
    if (!cut || cut + 10 >= path + sizeof path) return 0;
    strcpy(cut + 1, "mkw.ini");
    snprintf(ini_path, sizeof ini_path, "%s", path);
    snprintf(launcher_ini, sizeof launcher_ini, "%s", path);
    *cut = 0;
    snprintf(game_dir, sizeof game_dir, "%s", path);
    return 1;
}

int cfg_save(void) {
    if (find_ini()) return cfg_save_to(ini_path);
    __atomic_store_n(&save_ok, 0, __ATOMIC_RELEASE);
    return 0;
}

const char *cfg_game_dir(void) {
    // mem_init needs the game library's tables before it can reserve the guest address space.
    if (!game_dir[0]) cfg_init();
    return game_dir;
}

int cfg_game_dir_invalid(void) { return game_dir_invalid; }

// Save pending settings to the current installation first. Only the selector belongs beside the launcher; the
// caller restarts it after success so no DLL, shader cache or settings from the old installation survive the switch.
int cfg_select_game_dir(const char *path) {
    cfg_game_dir();
    if (cfg_from_env("MKW_GAME_DIR")) return 0;
    char resolved[CFG_GAME_DIR_MAX + 1];
    game_dir_invalid = !resolve_game_dir(path, resolved, sizeof resolved);
    if (game_dir_invalid) {
#ifdef _WIN32
        rt_log("cfg: game directory rejected: maximum %d UTF-16 units, no newlines", CFG_GAME_DIR_WIN32_MAX);
#else
        rt_log("cfg: game directory rejected: maximum %d UTF-8 bytes, no newlines", CFG_GAME_DIR_MAX);
#endif
        return 0;
    }
    if (!cfg_save()) return 0;
    int i = 0, old_n = n_entries;
    while (i < n_entries && strcmp(entries[i].key, "GAME_DIR")) i++;
    CfgEntry old = i < n_entries ? entries[i] : (CfgEntry){0};
    cfg_set("MKW_GAME_DIR", resolved);
    if (cfg_save_to(launcher_ini)) return 1;
    // A failed selector write must not leak into the current installation on the next settings save.
    if (i < old_n) entries[i] = old;
    else n_entries = old_n;
    return 0;
}

static void load_ini(void) {
    if (!find_ini()) return;
    if (!cfg_load(ini_path)) {
        rt_log("cfg: no settings file at %s", ini_path);
        return;
    }
    rt_log("cfg: read %s: %d keys", ini_path, n_entries);
    for (int i = 0; i < n_entries; i++) {
        char var[128];
        snprintf(var, sizeof var, PREFIX "%s", entries[i].key);
        if (getenv(var)) rt_log("cfg: %s is set in the environment and overrides mkw.ini", var);
    }
}

// The save lock only protects this queue. The worker runs file I/O without it, so a present or input poll can
// change the deadline while a write is in progress.
static PlatMutex save_lock = PLAT_MUTEX_INIT;
static PlatCond save_cv = PLAT_COND_INIT;
static uint64_t save_due;
static int save_worker_started, save_writing, save_draining;

static void save_worker(void *unused) {
    (void)unused;
    plat_mutex_lock(&save_lock);
    for (;;) {
        while (!save_due) plat_cond_wait(&save_cv, &save_lock);
        uint64_t now = plat_now_ns();
        if (!save_draining && now < save_due) {
            plat_cond_wait_ns(&save_cv, &save_lock, save_due - now);
            continue;
        }
        save_due = 0;
        save_writing = 1;
        plat_mutex_unlock(&save_lock);
        cfg_save();
        plat_mutex_lock(&save_lock);
        save_writing = 0;
        plat_cond_broadcast(&save_cv);
    }
}

void cfg_save_request(unsigned delay_ms) {
    plat_mutex_lock(&save_lock);
    if (!save_worker_started) {
        save_worker_started = plat_thread_spawn(save_worker, NULL, "settings save", 64 << 10);
        if (!save_worker_started) {
            __atomic_store_n(&save_ok, 0, __ATOMIC_RELEASE);
            plat_mutex_unlock(&save_lock);
            rt_log("cfg: could not start the settings save thread");
            return;
        }
    }
    save_due = plat_now_ns() + delay_ms * 1000000ull;
    plat_cond_broadcast(&save_cv);
    plat_mutex_unlock(&save_lock);
}

void cfg_save_drain(void) {
    plat_mutex_lock(&save_lock);
    save_draining = 1;
    plat_cond_broadcast(&save_cv);
    while (save_due || save_writing) plat_cond_wait(&save_cv, &save_lock);
    save_draining = 0;
    plat_mutex_unlock(&save_lock);
}

void cfg_init(void) {
    static int initialized;
    if (initialized) return;
    initialized = 1;
    load_ini();
    const char *selected = cfg_get("MKW_GAME_DIR");
    if (selected && *selected) {
        char dir[CFG_GAME_DIR_MAX + 1];
        game_dir_invalid = !resolve_game_dir(selected, dir, sizeof dir);
        if (game_dir_invalid) {
#ifdef _WIN32
            rt_log("cfg: invalid game directory: maximum %d UTF-16 units, no newlines; using %s",
                   CFG_GAME_DIR_WIN32_MAX, game_dir);
#else
            rt_log("cfg: invalid game directory: maximum %d UTF-8 bytes, no newlines; using %s", CFG_GAME_DIR_MAX,
                   game_dir);
#endif
        } else if (strcmp(game_dir, dir)) {
            snprintf(game_dir, sizeof game_dir, "%s", dir);
            snprintf(ini_path, sizeof ini_path, "%s/mkw.ini", game_dir);
            // Each installation keeps its own RomFS path and settings. The launcher ini holds the selection only.
            cfg_load(ini_path);
            char key[] = "GAME_DIR", value[sizeof game_dir];
            snprintf(value, sizeof value, "%s", game_dir);
            add(key, value, 0);
            rt_log("cfg: game directory %s (settings %s)", game_dir, ini_path);
        }
    }
    if (profile_of())
        rt_log("cfg: device profile %s: defaults for the keys mkw.ini does not set (MKW_DEVICE_PROFILE=desktop "
               "turns it off)", profile_of() == 2 ? "PS5" : "Steam Deck");
}
