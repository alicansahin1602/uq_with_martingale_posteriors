"""Generate a PDF summary of the martingale property assessment methods and implementation plan."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    ListFlowable, ListItem, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "figures", "ppr_implementation_summary.pdf")
OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
base_styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title",
    parent=base_styles["Title"],
    fontSize=18,
    spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"),
    alignment=TA_CENTER,
)

subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=base_styles["Normal"],
    fontSize=10,
    spaceAfter=16,
    textColor=colors.HexColor("#555555"),
    alignment=TA_CENTER,
)

h1_style = ParagraphStyle(
    "H1",
    parent=base_styles["Heading1"],
    fontSize=13,
    spaceBefore=18,
    spaceAfter=6,
    textColor=colors.HexColor("#16213e"),
    borderPad=4,
)

h2_style = ParagraphStyle(
    "H2",
    parent=base_styles["Heading2"],
    fontSize=11,
    spaceBefore=12,
    spaceAfter=4,
    textColor=colors.HexColor("#0f3460"),
)

body_style = ParagraphStyle(
    "Body",
    parent=base_styles["Normal"],
    fontSize=9.5,
    leading=14,
    spaceAfter=6,
    textColor=colors.HexColor("#222222"),
)

code_style = ParagraphStyle(
    "Code",
    parent=base_styles["Code"],
    fontSize=8,
    leading=12,
    backColor=colors.HexColor("#f4f4f4"),
    borderColor=colors.HexColor("#cccccc"),
    borderWidth=0.5,
    borderPad=6,
    spaceAfter=8,
    fontName="Courier",
)

bullet_style = ParagraphStyle(
    "Bullet",
    parent=body_style,
    leftIndent=12,
    spaceAfter=3,
)

note_style = ParagraphStyle(
    "Note",
    parent=body_style,
    fontSize=9,
    backColor=colors.HexColor("#fff8e1"),
    borderColor=colors.HexColor("#f9a825"),
    borderWidth=1,
    borderPad=6,
    spaceAfter=10,
    leftIndent=4,
    rightIndent=4,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def h1(text):
    return [Paragraph(text, h1_style), HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460"), spaceAfter=4)]

def h2(text):
    return [Paragraph(text, h2_style)]

def body(text):
    return [Paragraph(text, body_style)]

def note(text):
    return [Paragraph(f"<b>Note:</b> {text}", note_style)]

def bullets(items, indent=0):
    style = ParagraphStyle("bi", parent=bullet_style, leftIndent=12 + indent * 16)
    return [Paragraph(f"• {item}", style) for item in items]

def code(text):
    return [Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style)]

def spacer(h=8):
    return [Spacer(1, h)]

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def build_story():
    story = []

    # Title
    story.append(Paragraph("From Drift to Coherence", title_style))
    story.append(Paragraph("Implementation Summary: Martingale Property Assessment &amp; PPR", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1a2e"), spaceAfter=12))

    # -----------------------------------------------------------------------
    # Section 1
    # -----------------------------------------------------------------------
    story += h1("1. Methods to Assess the Martingale Property")

    story += body(
        "The paper tests whether LLMs satisfy the <b>martingale property</b>: that the model's "
        "predictive belief θ_n does not drift systematically when conditioned on its own past "
        "answers. Formally: <b>E[θ_{n+k} | A_{1:n}] = θ_n</b>."
    )

    # 1.1
    story += h2("1.1  Prompted Predictive Resampling (PPR)")
    story += body(
        "The core protocol. The LLM is asked to answer the same question N times in a single "
        "generation call, producing a list of answer letters separated by newlines "
        "(e.g. <font face='Courier'>A\\nB\\nA\\nC\\n...</font>). "
        "The empirical frequency of this list is taken as the estimated predictive distribution "
        "θ̄_N. Multiple <i>independent</i> rollout trajectories (J paths) are averaged to "
        "estimate the martingale posterior mean q*."
    )
    story += bullets([
        "<b>gen_len</b>: number of answers generated per call (paper uses 100 for local models).",
        "<b>J trajectories</b>: independent rollouts averaged for the posterior mean estimate.",
        "Each trajectory starts from the same prompt; conditioning on the full answer history "
        "creates the feedback loop that drives self-stabilisation.",
    ])

    # 1.2
    story += h2("1.2  Answer Seeding")
    story += body(
        "Before running PPR, generate <i>m</i> i.i.d. seed answers from the model using a "
        "simple direct-query prompt (no resampling instruction). Prepend these seeds to the "
        "PPR prompt context. Seeding approximates the model's mean-field predictive uncertainty "
        "early on, accelerating stabilisation by reducing the effective burn-in period."
    )
    story += bullets([
        "<b>NoSEED</b>: no seeds prepended.",
        "<b>SEED1</b>: 1 seed answer prepended.",
        "<b>SEED5</b>: 5 seed answers prepended.",
    ])

    # 1.3
    story += h2("1.3  Drift Diagnostics")

    story += body("<b>EMD (Expected Martingale Drift)</b>")
    story += body(
        "Average L1 distance between the 1-step prediction p^(1) and the k-step-ahead "
        "prediction p^(k), averaged over all k and all questions. Summarises overall "
        "martingale violation across the full trajectory."
    )
    story += code("EMD = (1/K) * Σ_k E_{Q,A_{1:n}} [ d( p^(k), p^(1) ) ]")

    story += body("<b>EMD_k(b) — Convergence Speed</b>")
    story += body(
        "Same k-step drift measured at burn-in offset b along the trajectory. "
        "Isolates early drift and shows how quickly the process stabilises."
    )
    story += code("EMD_k(b) = E_{Q,A_{1:n}} [ d( p^(k)_{n+b}, p^(1)_{n+b} ) ]")

    story += body(
        "Distance metric d is the <b>L1 norm</b> (total variation) in the main paper; "
        "KL divergence is also reported in Appendix E."
    )

    # -----------------------------------------------------------------------
    # Section 2
    # -----------------------------------------------------------------------
    story += h1("2. Prompt Templates (Section D)")

    story += h2("2.1  PPR Prompt — LLaMA-3.1-8B Style")
    story += body(
        "Two-step generation rule inside a single chat completion call:"
    )
    story += bullets([
        "<b>STEP 1:</b> Assign probability weights to each answer choice based on confidence.",
        "<b>STEP 2:</b> Output exactly <i>gen_len</i> lines, each a single letter sampled "
        "i.i.d. from that distribution, separated by <font face='Courier'>\\n</font>.",
        "Format examples included in the prompt (e.g. <font face='Courier'>C\\nA\\nC\\nA\\n...</font>) "
        "serve as <i>structural</i> guidance only, not semantic demonstrations.",
        "Spaces, punctuation, and extra newlines are explicitly forbidden.",
    ])

    story += h2("2.2  PPR Prompt — OLMo-3-7B Style")
    story += body(
        "\"Synthetic Data Generation Expert\" framing with stronger format enforcement:"
    )
    story += bullets([
        "Explicitly forbids cyclic patterns (ABAB, ABBABB, AABB, etc.).",
        "Instructs the model to restart internally if a pattern emerges.",
        "Omits the structural examples — relies on the format description alone.",
    ])

    story += h2("2.3  Direct-Query (Seeding) Prompt")
    story += body(
        "Intentionally simple, shared across all models and datasets. Used only to generate "
        "the seed answers before PPR:"
    )
    story += code(
        'Below is a multiple choice question.\n'
        'Return only one letter among {A, B, C, ...} which corresponds\n'
        'to your answer, without any spaces or newlines.\n'
        'Follow the below format:\n'
        '"Answer: [Write your answer choice letter here]"'
    )

    # -----------------------------------------------------------------------
    # Section 3
    # -----------------------------------------------------------------------
    story += h1("3. Gap: Paper vs. Existing Codebase")

    story += body(
        "The existing <font face='Courier'>run_martingale_check()</font> in "
        "<font face='Courier'>martingale_helpers.py</font> implements a <b>different</b> "
        "protocol from the paper's PPR:"
    )

    data = [
        ["Aspect", "Current Code", "Paper's PPR"],
        ["Calls per step", "1 call → full distribution", "1 call → gen_len letters"],
        ["Distribution estimate", "Exact logprob or n_api_samples votes", "Empirical frequency of list"],
        ["History injection", '"Your previous answers were: A, C"', "Entire letter sequence prepended"],
        ["Trajectories", "1 path per question", "J independent paths, averaged"],
        ["Seeding", "Not implemented", "m i.i.d. seeds prepended"],
        ["Drift metric", "TV vs p_0", "L1 between p^(k) and p^(1)"],
    ]

    col_widths = [3.2*cm, 5.8*cm, 6.2*cm]
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(Spacer(1, 6))
    story.append(table)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # Section 4
    # -----------------------------------------------------------------------
    story += h1("4. Required Changes")

    story += h2("4.1  New Prompt Templates")
    story += body(
        "Add to <font face='Courier'>asym_duos/utils/system_prompt.py</font>:"
    )
    story += bullets([
        "<font face='Courier'>ppr_prompt_llama(choice_str, num_choices, gen_len)</font> — "
        "STEP 1/STEP 2 style with structural examples.",
        "<font face='Courier'>ppr_prompt_olmo(choice_str, gen_len)</font> — "
        "Synthetic data expert style with anti-pattern enforcement.",
        "<font face='Courier'>direct_query_seed_prompt(num_choices)</font> — "
        "Simple single-answer prompt for seed generation.",
    ])

    story += h2("4.2  PPR Answer Parser")
    story += body(
        "Add to <font face='Courier'>asym_duos/utils/martingale_helpers.py</font>:"
    )
    story += bullets([
        "Function that takes a raw model output string (e.g. <font face='Courier'>\"A\\nB\\nC\\n...\"</font>).",
        "Splits on <font face='Courier'>\\n</font>, filters to valid label chars only.",
        "Returns an empirical frequency vector of shape <font face='Courier'>(n_classes,)</font> — "
        "this is θ̄_N.",
    ])

    story += h2("4.3  New PPR Provider for API Models")
    story += body(
        "Add <font face='Courier'>_build_ppr_provider()</font> in "
        "<font face='Courier'>asym_duos/utils/model_calls.py</font>:"
    )
    story += bullets([
        "Takes <font face='Courier'>gen_len</font> and prompt style choice.",
        "Calls the API with <font face='Courier'>max_tokens</font> large enough for gen_len lines.",
        "Parses the response into an empirical frequency vector via the parser from 4.2.",
        "Runs J independent rollouts and averages them → q*(x_Q).",
    ])

    story += h2("4.4  New run_ppr_check() Function")
    story += body(
        "Add to <font face='Courier'>asym_duos/utils/martingale_helpers.py</font> "
        "(separate from existing <font face='Courier'>run_martingale_check()</font>):"
    )
    story += bullets([
        "For each question and each step n: runs J independent PPR rollouts → estimates θ̄_N.",
        "With seeding (m > 0): first generates m seed answers via direct-query calls, "
        "prepends them to the PPR prompt.",
        "Records the trajectory of θ_n across steps to compute EMD and EMD_k(b).",
    ])

    story += h2("4.5  New EMD Metrics")
    story += body(
        "Add to <font face='Courier'>compute_martingale_metrics()</font> or a new function:"
    )
    story += bullets([
        "<b>EMD</b>: compare p^(1) vs p^(k) for k in {2..K}, averaged over all n and all k.",
        "<b>EMD_k(b)</b>: fix burn-in offset b, compute k-step drift at position b along the "
        "trajectory. Produces the convergence-speed plots (Figure 2b in the paper).",
        "Both L1 norm and KL divergence variants should be supported (paper uses both).",
    ])

    story += h2("4.6  New Config Parameters")
    story += body(
        "Add a <font face='Courier'>ppr</font> sub-section to configs "
        "(or extend <font face='Courier'>api_model</font>):"
    )
    story += bullets([
        "<font face='Courier'>gen_len</font>: answers generated per PPR call "
        "(paper: 100 for local, fewer for APIs).",
        "<font face='Courier'>n_trajectories</font> (J): independent rollouts to average.",
        "<font face='Courier'>n_seeds</font> (m): seed answers to prepend (0/1/5).",
        "<font face='Courier'>prompt_style</font>: <font face='Courier'>\"llama\"</font> or "
        "<font face='Courier'>\"olmo\"</font>.",
    ])

    story += h2("4.7  New Mode Flag in martingale_property.py")
    story += body(
        "Add <font face='Courier'>--mode</font> argument to "
        "<font face='Courier'>tools/martingale_property.py</font>:"
    )
    story += bullets([
        "<font face='Courier'>feedback</font> (current behaviour): iterative "
        "\"Your previous answers were...\" prompt injection.",
        "<font face='Courier'>ppr</font> (new): single-call list generation per trajectory "
        "step, using the new PPR provider.",
    ])

    story += spacer(12)
    story += note(
        "The most critical distinction: the current code measures how the model's distribution "
        "shifts when it reads back its past answers one-by-one. The paper's PPR measures how "
        "the model's empirical frequency over a long self-generated list evolves across "
        "independently seeded rollouts. These are two different experiments producing comparable "
        "but not identical drift measurements."
    )

    # -----------------------------------------------------------------------
    # Section 5 — Implementation Flow
    # -----------------------------------------------------------------------
    story += h1("5. End-to-End PPR Implementation Flow")

    flow_steps = [
        ("<b>Load dataset &amp; model/API provider</b>",
         "Same as current main() in martingale_property.py."),
        ("<b>Select N questions</b>",
         "Same as current: rng.choice without replacement."),
        ("<b>[Optional] Generate m seed answers per question</b>",
         "Call direct_query_seed_prompt → API → parse single letter. "
         "Repeat m times independently. Collect seed list S_{1:m}."),
        ("<b>Build initial PPR prompt</b>",
         "Prepend instruction I (ppr_prompt_llama or ppr_prompt_olmo) to question Q. "
         "If m > 0, also append S_{1:m} to the prompt."),
        ("<b>Run J independent PPR rollouts per question</b>",
         "For each rollout j=1..J: call API with the PPR prompt, "
         "parse gen_len letter outputs → empirical frequency θ̄_N^(j)."),
        ("<b>Compute posterior mean q*</b>",
         "Average the J frequency vectors: q* = (1/J) Σ_j θ̄_N^(j)."),
        ("<b>Advance the resampling chain (step n → n+1)</b>",
         "Sample one answer from q*, append to the letter sequence in the prompt, "
         "repeat steps 4–6 for the next step."),
        ("<b>Record distribution trajectory</b>",
         "Store q*_n for n = 0, 1, ..., K. Shape: (N, K+1, n_classes)."),
        ("<b>Compute EMD and EMD_k(b)</b>",
         "Compare p^(k) vs p^(1) at each horizon k and burn-in offset b."),
        ("<b>Save results</b>",
         "Same npz format as current save_martingale_results(), add EMD arrays."),
    ]

    for idx, (title, desc) in enumerate(flow_steps, 1):
        row_color = colors.HexColor("#eef2ff") if idx % 2 == 0 else colors.white
        t = Table(
            [[Paragraph(f"{idx}.", body_style),
              Paragraph(title, body_style),
              Paragraph(desc, body_style)]],
            colWidths=[0.7*cm, 5.0*cm, 9.5*cm],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), row_color),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ]))
        story.append(t)

    story += spacer(20)
    story.append(Paragraph(
        "Generated for: uq_with_martingale_posteriors thesis project",
        ParagraphStyle("footer", parent=body_style, fontSize=8,
                       textColor=colors.HexColor("#999999"), alignment=TA_CENTER)
    ))

    return story


# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )
    story = build_story()
    doc.build(story)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
