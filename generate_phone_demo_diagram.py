import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")

# Output directory
OUT_DIR = r"C:\Users\Mehedi\adavtive_model\adavtive_dataset\paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 9), facecolor="#0f172a")
plt.subplots_adjust(wspace=0.15, top=0.88, bottom=0.08, left=0.04, right=0.96)

fig.suptitle("Adaptive Smartphone UI Framework: Mechanism of Virtual Hitbox Expansion", 
             fontsize=20, fontweight="bold", color="white", y=0.96)

panels = [
    {
        "title": "Panel A: Static Default UI\n(Sitting / Stationary State)",
        "subtitle": "Standard Touch Target (40dp)",
        "hitbox_color": "#38bdf8",
        "hitbox_style": "-",
        "expanded": False,
        "tap_x": 0.50, "tap_y": 0.28, # On target
        "tap_result": "Success",
        "status_text": "Normal Touch: Accurate Hit\nMiss Rate = 2%",
        "status_color": "#22c55e",
        "badge": "Stationary Context"
    },
    {
        "title": "Panel B: Static UI during Walking\n(Situational Impairment)",
        "subtitle": "No Adaptation: Standard Target (40dp)",
        "hitbox_color": "#38bdf8",
        "hitbox_style": "-",
        "expanded": False,
        "tap_x": 0.65, "tap_y": 0.35, # Outside target
        "tap_result": "Miss",
        "status_text": "Vibration / Tremor: Miss Tap!\nTap lands in dead whitespace",
        "status_color": "#ef4444",
        "badge": "Walking Context"
    },
    {
        "title": "Panel C: Proposed Adaptive OS Layer\n(Virtual Hitbox Expansion)",
        "subtitle": "Adaptive Touch Target (56dp Expanded)",
        "hitbox_color": "#22c55e",
        "hitbox_style": "--",
        "expanded": True,
        "tap_x": 0.65, "tap_y": 0.35, # Same outside coordinate
        "tap_result": "Recovered",
        "status_text": "Framework Interception: SUCCESS!\nRedirected to ACTION_CLICK",
        "status_color": "#22c55e",
        "badge": "Active RL Adaptation"
    }
]

for idx, p in enumerate(panels):
    ax = axes[idx]
    ax.set_facecolor("#1e293b")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.7)
    ax.axis("off")

    # 1. Phone Bezel / Chassis
    phone_bezel = FancyBboxPatch((0.08, 0.08), 0.84, 1.48, boxstyle="round,pad=0.03,rounding_size=0.08",
                                 linewidth=3, edgecolor="#475569", facecolor="#020617")
    ax.add_patch(phone_bezel)

    # 2. Phone Screen
    phone_screen = FancyBboxPatch((0.11, 0.12), 0.78, 1.38, boxstyle="round,pad=0.01,rounding_size=0.05",
                                  linewidth=1, edgecolor="#334155", facecolor="#0f172a")
    ax.add_patch(phone_screen)

    # Top speaker notch & camera
    ax.add_patch(FancyBboxPatch((0.40, 1.44), 0.20, 0.02, boxstyle="round,pad=0.005", facecolor="#334155"))
    ax.plot(0.35, 1.45, "o", color="#38bdf8", markersize=4)

    # Context Badge
    ax.text(0.50, 1.37, f"● {p['badge']}", ha="center", va="center", color="#cbd5e1", 
            fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#475569"))

    # App Mockup Header (e.g. Social App / Third-Party App)
    header = FancyBboxPatch((0.14, 1.22), 0.72, 0.09, boxstyle="square,pad=0.0", facecolor="#1e293b")
    ax.add_patch(header)
    ax.text(0.20, 1.265, "SocialApp", color="white", fontsize=11, fontweight="bold", va="center")
    ax.plot([0.72, 0.78], [1.265, 1.265], "-", color="#94a3b8", lw=3)

    # App Post Card
    card = FancyBboxPatch((0.14, 0.42), 0.72, 0.76, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#1e293b")
    ax.add_patch(card)
    
    # User Profile line
    ax.plot(0.20, 1.10, "o", color="#f43f5e", markersize=10)
    ax.text(0.26, 1.11, "Sarah Jenkins", color="white", fontsize=10, fontweight="bold")
    ax.text(0.26, 1.07, "2 hours ago • Public", color="#94a3b8", fontsize=8)
    
    # Post image box
    post_img = FancyBboxPatch((0.18, 0.65), 0.64, 0.38, boxstyle="square,pad=0", facecolor="#334155")
    ax.add_patch(post_img)
    ax.text(0.50, 0.84, "[ App Content Image ]", color="#94a3b8", fontsize=9, ha="center", va="center")

    # Post Text lines
    ax.plot([0.18, 0.78], [0.58, 0.58], "-", color="#cbd5e1", lw=2)
    ax.plot([0.18, 0.60], [0.54, 0.54], "-", color="#cbd5e1", lw=2)
    
    # --- Action Bar Area (Like, Comment, Share) ---
    btn_bar = FancyBboxPatch((0.14, 0.22), 0.72, 0.16, boxstyle="square,pad=0", facecolor="#090d16", edgecolor="#334155")
    ax.add_patch(btn_bar)

    # Real Button Rectangles (Visual Size: 40dp)
    btn_y = 0.25
    btn_h = 0.10
    btn_w = 0.20
    
    # Button 1: Like
    ax.add_patch(FancyBboxPatch((0.18, btn_y), btn_w, btn_h, boxstyle="round,pad=0.005", facecolor="#1e293b", edgecolor="#475569"))
    ax.text(0.28, btn_y + 0.05, "Like", color="#cbd5e1", fontsize=9, ha="center", va="center", fontweight="bold")

    # Button 2: Target Button "Share" (Where adaptation happens)
    target_btn_x = 0.42
    ax.add_patch(FancyBboxPatch((target_btn_x, btn_y), btn_w, btn_h, boxstyle="round,pad=0.005", facecolor="#2563eb", edgecolor="#38bdf8", lw=1.5))
    ax.text(target_btn_x + btn_w/2, btn_y + 0.05, "Share", color="white", fontsize=9, ha="center", va="center", fontweight="bold")

    # Button 3: Menu
    ax.add_patch(FancyBboxPatch((0.66, btn_y), btn_w, btn_h, boxstyle="round,pad=0.005", facecolor="#1e293b", edgecolor="#475569"))
    ax.text(0.76, btn_y + 0.05, "Chat", color="#cbd5e1", fontsize=9, ha="center", va="center", fontweight="bold")

    # --- Virtual Hitbox Layer ---
    if p["expanded"]:
        # Expanded Virtual Hitbox (Invisible to user, managed by OS Accessibility Service)
        exp_pad = 0.05
        virtual_hitbox = FancyBboxPatch((target_btn_x - exp_pad, btn_y - exp_pad*0.8), btn_w + exp_pad*2, btn_h + exp_pad*1.6,
                                        boxstyle="round,pad=0.01", linewidth=2.2, linestyle="--", 
                                        edgecolor="#22c55e", facecolor="#22c55e", alpha=0.25)
        ax.add_patch(virtual_hitbox)
        ax.text(target_btn_x + btn_w/2, btn_y - 0.065, "[Virtual Hitbox +8dp]", color="#4ade80", 
                fontsize=9, fontweight="bold", ha="center")
    else:
        # Standard Touch Bounds
        std_hitbox = FancyBboxPatch((target_btn_x, btn_y), btn_w, btn_h,
                                    boxstyle="round,pad=0.005", linewidth=1.5, linestyle=":", 
                                    edgecolor="#38bdf8", facecolor="none")
        ax.add_patch(std_hitbox)
        ax.text(target_btn_x + btn_w/2, btn_y - 0.045, "Standard Bounds (40dp)", color="#94a3b8", fontsize=8, ha="center")

    # --- User Tap Coordinate Point ---
    tx, ty = p["tap_x"], p["tap_y"]
    if p["tap_result"] == "Success":
        ax.plot(tx, ty, "o", color="#22c55e", markersize=14, markeredgecolor="white", markeredgewidth=2)
        ax.text(tx, ty + 0.05, "✓ Clean Hit", color="#4ade80", fontsize=9, fontweight="bold", ha="center")
    elif p["tap_result"] == "Miss":
        ax.plot(tx, ty, "X", color="#ef4444", markersize=16, markeredgecolor="white", markeredgewidth=2)
        ax.plot([tx, target_btn_x + btn_w], [ty, btn_y + btn_h/2], ":", color="#ef4444", lw=1.5)
        ax.text(tx, ty + 0.05, "✗ Missed Button!\n(Dead Space)", color="#f87171", fontsize=9, fontweight="bold", ha="center")
    elif p["tap_result"] == "Recovered":
        ax.plot(tx, ty, "o", color="#22c55e", markersize=14, markeredgecolor="white", markeredgewidth=2)
        # Redirection arrow from tap to actual button center
        ax.annotate("", xy=(target_btn_x + btn_w/2, btn_y + btn_h/2), xytext=(tx, ty),
                    arrowprops=dict(arrowstyle="->", color="#4ade80", lw=2.5, linestyle="--"))
        ax.text(tx, ty + 0.06, "✓ Tap in Virtual Hitbox!\nOS Dispatches Click", color="#4ade80", 
                fontsize=9, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#022c22", edgecolor="#22c55e"))

    # Bottom Home indicator
    ax.add_patch(FancyBboxPatch((0.40, 0.14), 0.20, 0.01, boxstyle="round,pad=0.002", facecolor="#475569"))

    # Panel Bottom Description Card
    desc_box = FancyBboxPatch((0.05, -0.22), 0.90, 0.26, boxstyle="round,pad=0.02,rounding_size=0.03", 
                              facecolor="#1e293b", edgecolor=p["status_color"], linewidth=1.8)
    ax.add_patch(desc_box)
    ax.text(0.50, -0.02, p["title"], color="white", fontsize=11, fontweight="bold", ha="center", va="top")
    ax.text(0.50, -0.12, p["status_text"], color=p["status_color"], fontsize=10, fontweight="bold", ha="center", va="center")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "adaptive_ui_hitbox_mechanism_phone_demo.png")
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"✅ Generated Phone Demo Diagram: {out_path}")
