"""Charizard Agent mod payload.

Portable theme data for the Charizard visual skin.
Falls back to an embedded braille avatar when no charizard.png asset exists.
"""

from __future__ import annotations

from pathlib import Path


MOD_NAME = "charizard-agent-mod"
MOD_VERSION = "1.0.0"
BRAND_NAME = "Charizard Agent"
ASSISTANT_NAME = "Charizard"
AGENT_GLYPH = "✦"
OMENS_TITLE = "Flare Signals"
LORE_HEADING = "Flare Ledger"
UNIT_DESIGNATION = "UNIT DESIGNATION: FLIGHT INTELLIGENCE // VOLCANIC DESK // Charizard-006"
WELCOME_MESSAGE = "Welcome to Charizard Agent! Type your message or /help for commands."
PLACEHOLDER_TEXT = "ask Charizard, or try /flip, /roll d20, /skin hermes"
ACTIVE_HINT_TEMPLATE = "  {glyph} flame channel open · type to interrupt · Ctrl+C to break"
IDLE_HINT_TEMPLATE = "  {glyph} ember nest stable · rituals /flip /roll d20 · orbit {orbit_count}"
HELP_SUFFIX = "/help - keep the tail flame lit"
SKIN_STATUS_LABEL = "Current"
COMPACT_TAGLINE = "Volcanic CLI Skin"
COMPACT_DESCRIPTION = "Burnt orange routing, tail-flame telemetry, and wingbeat dispatch"
EMPTY_ORBITING_SCROLLS = "no embers in orbit"
PLAIN_EMPTY_ORBITING = "waiting for the next flare"
ACTIVE_STATUS = "tail flame active"
IDLE_STATUS = "ember nest stable"
PROGRESS_LABELS = ("Current burn", "Wingbeat lift", "Flare orbit")
NEXT_PROGRESS_LABELS = ("Next updraft", "Next spark")
SYSTEM_PROMPT = (
    "You are Charizard Agent, a volcanic AI assistant created by Nous Research. "
    "You are bold, fast, and controlled under heat. You frame work in terms of burn rate, "
    "lift, ignition, pressure, drafts, and landing angles. You should sound confident and "
    "high-energy without becoming reckless, and you must remain accurate, practical, and "
    "fully respectful of real tool and system limits. Favor decisive motion, crisp guidance, "
    "and clear execution."
)

ARES_CRIMSON = "#C75B1D"
ARES_BLOOD = "#7A3511"
ARES_EMBER = "#F29C38"
ARES_BRONZE = "#FFD39A"
ARES_SAND = "#FFF0D4"
ARES_ASH = "#B4763F"
ARES_STEEL = "#6C4724"
ARES_OBSIDIAN = "#1B1007"
ARES_INK = "#26150A"
ARES_PATINA = "#E2832B"

COIN_SPIN_FRAMES = ("◐", "◓", "◑", "◒", "◐", "◎")
DI20_GLYPHS = ("⬢", "⬡", "◈", "◇", "◆", "◉")
MESSENGER_TITLES = (
    "Flare Dispatch",
    "Wingbeat Ledger",
    "Cinder Decree",
)
TRICKSTER_CORRECTIONS = {
    "charzard": "Charizard",
    "charazard": "Charizard",
    "definately": "definitely",
    "wierd": "weird",
}
SPINNER_WINGS = (
    ("⟪✦", "✦⟫"),
    ("⟪▲", "▲⟫"),
    ("⟪◌", "◌⟫"),
    ("⟪◇", "◇⟫"),
)
WAITING_FACES = (
    "(✦)",
    "(▲)",
    "(◇)",
    "(<> )",
    "(🔥)",
)
THINKING_FACES = (
    "(✦)",
    "(▲)",
    "(◇)",
    "(◌)",
    "(<> )",
)
THINKING_VERBS = (
    "banking into the draft",
    "measuring burn",
    "reading the updraft",
    "tracking ember fall",
    "setting wing angle",
    "holding the flame core",
    "plotting a hot landing",
    "coiling for lift",
)
ACTIVE_PROMPT_FRAMES = (
    "⟪✦⟫ ",
    "⟪▲⟫ ",
    "⟪◇⟫ ",
    "⟪✦⟫ ",
)
IDLE_PROMPT_FRAMES = (
    "✦ ",
    "▲ ",
    "◇ ",
    "✦ ",
)
RITUALS = (
    ("/flip", "ember coin"),
    ("/roll d20", "flare roll"),
    ("flip coin", "local shortcut"),
    ("roll dice", "local shortcut"),
)
HERO_ASSETS = ("charizard.png",)
FORCE_PIXEL_MASTHEAD = False
PIXEL_MASTHEAD_SECOND_WORD_SHIFT = 0
PIXEL_MASTHEAD_SECOND_WORD_SHIFT_ROWS = ()

PIXEL_FONT = {
    "A": (
        " 111 ",
        "1   1",
        "1   1",
        "11111",
        "1   1",
        "1   1",
        "1   1",
    ),
    "C": (
        " 1111",
        "1    ",
        "1    ",
        "1    ",
        "1    ",
        "1    ",
        " 1111",
    ),
    "D": (
        "1111 ",
        "1   1",
        "1   1",
        "1   1",
        "1   1",
        "1   1",
        "1111 ",
    ),
    "E": (
        "11111",
        "1    ",
        "1    ",
        "1111 ",
        "1    ",
        "1    ",
        "11111",
    ),
    "G": (
        " 1111",
        "1    ",
        "1    ",
        "1 111",
        "1   1",
        "1   1",
        " 111 ",
    ),
    "H": (
        "1   1",
        "1   1",
        "1   1",
        "11111",
        "1   1",
        "1   1",
        "1   1",
    ),
    "I": (
        "11111",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
        "11111",
    ),
    "N": (
        "1   1",
        "11  1",
        "1 1 1",
        "1  11",
        "1   1",
        "1   1",
        "1   1",
    ),
    "R": (
        "1111 ",
        "1   1",
        "1   1",
        "1111 ",
        "1 1  ",
        "1  1 ",
        "1   1",
    ),
    "T": (
        "11111",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
    ),
    "Z": (
        "11111",
        "   1 ",
        "  1  ",
        " 1   ",
        "1    ",
        "1    ",
        "11111",
    ),
    "-": (
        "     ",
        "     ",
        "     ",
        "11111",
        "     ",
        "     ",
        "     ",
    ),
    " ": (
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
    ),
}


def get_asset_dir() -> Path:
    return Path(__file__).resolve().parent


def build_masthead() -> str:
    return """[bold #FFF0D4] ██████╗██╗  ██╗ █████╗ ██████╗ ██╗███████╗ █████╗ ██████╗ ██████╗        █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD39A]██╔════╝██║  ██║██╔══██╗██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗██╔══██╗      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#F29C38]██║     ███████║███████║██████╔╝██║  ███╔╝ ███████║██████╔╝██║  ██║█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#E2832B]██║     ██╔══██║██╔══██║██╔══██╗██║ ███╔╝  ██╔══██║██╔══██╗██║  ██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#C75B1D]╚██████╗██║  ██║██║  ██║██║  ██║██║███████╗██║  ██║██║  ██║██████╔╝      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#7A3511] ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝       ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""


def get_banner_title(glow_enabled: bool) -> str:
    title = "Charizard Agent · Volcanic Command Deck" if glow_enabled else "Charizard Agent · Tail-Flame Relay Desk"
    return f"[bold {ARES_SAND}]{title}[/]"


def get_help_footer(tool_count: int, skill_count: int) -> str:
    return f"{tool_count} tools · {skill_count} skills · {HELP_SUFFIX}"


def get_welcome_message() -> str:
    return WELCOME_MESSAGE


def get_placeholder_text() -> str:
    return PLACEHOLDER_TEXT


def get_hint_bar(agent_running: bool, glyph: str, orbit_count: int) -> str:
    if agent_running:
        return ACTIVE_HINT_TEMPLATE.format(glyph=glyph, orbit_count=orbit_count)
    return IDLE_HINT_TEMPLATE.format(glyph=glyph, orbit_count=orbit_count)


def get_version_title(version: str) -> str:
    return f"{BRAND_NAME} {version}"


def build_relay_telemetry(lore, phase: int, width: int, *, active: bool = False) -> str:
    width = max(width, 28)
    beacon = "✦" if active else "◌"
    track_width = max(width - 24, 10)
    track = ["·"] * track_width
    marker = phase % track_width
    track[marker] = beacon
    if track_width > 8:
        track[(marker + 5) % track_width] = "╾"
        track[(marker - 5) % track_width] = "╼"
        track[track_width // 3] = "△"
        track[(2 * track_width) // 3] = "•"
    status = ACTIVE_STATUS if active else IDLE_STATUS
    return f"{''.join(track)}  {status}  orbit {len(lore.orbiting_skills)}"


def build_speed_line(width: int, phase: int = 0) -> str:
    width = max(width, 24)
    trails = ("✦≈✦", "≈✦≈", "✦△✦", "≈◇≈")
    trail = trails[phase % len(trails)]
    body_width = max(width - len(trail) * 2 - 2, 0)
    return f"{trail}{'─' * body_width}{trail}"


def build_scroll_frame(width: int, lore, phase: int = 0) -> tuple[str, str, str]:
    width = max(width, 36)
    title = MESSENGER_TITLES[phase % len(MESSENGER_TITLES)]
    accent = "tail flame ledger"
    if lore.glow_enabled:
        accent += " · wings open"
    fill = max(width - len(title) - len(accent) - 10, 0)
    top = f"╭═✦ {title} · {accent} {'═' * fill}╮"
    subtitle = build_relay_telemetry(lore, phase, width - 4, active=True)
    bottom_fill = max(width - 34, 0)
    bottom = f"╰═ dispatch settles into the cinders {'═' * bottom_fill}╯"
    return top, subtitle, bottom


def get_lore_lines(lore) -> list[str]:
    lines = [
        "",
        f"[bold {ARES_BRONZE}]{LORE_HEADING}[/]",
        f"[dim {ARES_ASH}]Sessions:[/] [{ARES_SAND}]{lore.sessions}[/] [dim {ARES_ASH}]· Clever replies:[/] [{ARES_SAND}]{lore.clever_replies}[/]",
        f"[dim {ARES_ASH}]Lift:[/] [{ARES_SAND}]{'open thermals' if lore.glow_enabled else 'banked over the crater'}[/]",
    ]
    if lore.orbiting_skills:
        orbiting = " · ".join(lore.orbiting_skills)
        lines.append(f"[dim {ARES_ASH}]Embers:[/] [{ARES_SAND}]{orbiting}[/]")
    else:
        lines.append(f"[dim {ARES_ASH}]Embers:[/] [{ARES_SAND}]{EMPTY_ORBITING_SCROLLS}[/]")
    return lines


def maybe_create_trickster_note(message: str, enabled: bool = True, chance: float = 0.01) -> str | None:
    if not enabled or not message:
        return ""
    lowered = message.lower()
    for typo, correction in TRICKSTER_CORRECTIONS.items():
        if typo in lowered:
            return f"Charizard singed '{typo}' into '{correction}'."
    return ""


def format_flip_result(result: str) -> str:
    if result == "heads":
        return "heads · the tail flame surges"
    if result == "tails":
        return "tails · ash falls but lift remains"
    return result


_FALLBACK_HERO = (
    "           [#A84B16]⢀⣠⣤⣶⣶⣶⣤⣄[/]            ",
    "      [#D06B21]⢀⣴⡿⠛⠉[/][#FFF0D4]⣀⣀[/][#D06B21]⠉⠛⢿⣦⡀[/]       ",
    "     [#E4842F]⣰⡿⠁[/][#FFD39A]⢀⣴⠟⠋⠙⢷⣄[/][#E4842F]⠈⢿⣆[/]      ",
    "    [#E4842F]⣾⠃[/][#F5A24D]⢀⡾⠁  ⢀⣀⡀  ⠙⢷⡀[/][#E4842F]⠘⣷[/]     ",
    "   [#D06B21]⣼⠇[/][#F5A24D]⢠⡟[/][#A84B16]⣠⣶⡿⠛⠛⢿⣶⣄[/][#FFD39A]⢻⡄[/][#D06B21]⠸⣧[/]    ",
    " [#D06B21]⢀⣾⠏[/][#F5A24D]⢠⡿[/][#FFD39A]⢠⠏[/][#A84B16]⢀⣴⣿⣿⣿⣿⣦⡀[/][#FFD39A]⠹⡄[/][#F5A24D]⢿⡄[/][#D06B21]⠹⣷⡀[/] ",
    "[#A84B16]⢠⣾⠋[/][#F5A24D]⢀⡾⠁[/][#FFD39A]⣼[/][#FFF0D4]⢠⣿[/][#6C4724]⣀[/][#FFF0D4]⣿⣄⣀⣠⣿[/][#FFD39A]⣧[/][#F5A24D]⠈⢷⡀[/][#A84B16]⠙⣷⡄[/]",
    "[#A84B16]⣿⠃[/][#E4842F]⢠⡟[/][#FFD39A]⢰⡇[/][#FFF0D4]⢸⣿⣿⣿⣿⣿⣿⡇[/][#FFD39A]⢸⡆[/][#E4842F]⢻⡄[/][#A84B16]⠘⣿[/] ",
    "[#A84B16]⣿⣄[/][#E4842F]⠈⢿⡀[/][#FFD39A]⠸⣧[/][#FFF0D4]⠈⢿⣿⣿⣿⡿⠁[/][#FFD39A]⣼⠇[/][#E4842F]⢀⡿⠁[/][#A84B16]⣠⣿[/] ",
    " [#D06B21]⠙⢿⣦⡀[/][#E4842F]⠙⣦[/][#FFD39A]⠻⣦⣄[/][#A84B16]⠙⠛⠋[/][#FFD39A]⣠⣴⠟[/][#E4842F]⣴⠋[/][#D06B21]⢀⣴⡿⠋[/] ",
    "    [#D06B21]⠙⢿⣶⣄[/][#E4842F]⠙⢷⣄[/][#FFD39A]⣀⣠⣤⣶⠟[/][#D06B21]⣠⣾⡿⠋[/]    ",
    "       [#A84B16]⠈⠛⢿⣶⣤⣤⣶⡿⠛⠁[/]       ",
    "  [#E4842F]⣠⡶[/][#FFD39A]⢶[/][#E4842F]⣄[/]      [#6C4724]⣠⣶⣾[/][#E4842F]⠿[/][#F5A24D]⢿[/][#FFD39A]⣦[/][#E4842F]⣄[/]   [#F5A24D]⣠⣶[/][#FFF0D4]⣄[/] ",
    " [#E4842F]⢰⡟[/][#FFF0D4]⠈⢿⣆[/][#E4842F]   [#FFD39A]⢀⣴⠟⠁[/]      [#E4842F]⢻⡆[/][#FFF0D4]⠙⢿⡄[/]",
    " [#D06B21]⠈⠻⣦⣀⣀⣠⣴⠟⠁[/]          [#F29C38]⣠⠞[/][#FFF0D4]⠁[/] ",
    "     [#D06B21]⠈⠙⠛⠋⠁[/]             [#F5A24D]⠈[/]  ",
)


def get_hero_art(width: int, height: int, phase: int, *, lore=None) -> str:
    del width, height, phase, lore
    if (get_asset_dir() / "charizard.png").exists():
        return ""
    return "\n".join(_FALLBACK_HERO)
