"""Build a readable vector-PDF guide from saved experiment figures and results.

Run with the bundled Python runtime and a completed results directory. Existing
figure PDFs are merged without rasterization; captions come from the saved
manifest. No model inference or result selection occurs in this script.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


EXP = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = letter
MARGIN = 44
TEXT_WIDTH = WIDTH - 2 * MARGIN
INK = colors.HexColor("#162230")
MUTED = colors.HexColor("#52616c")
BLUE = colors.HexColor("#1f77b4")
BODY = ParagraphStyle("Body", fontName="Times-Roman", fontSize=9,
                      leading=13, textColor=INK, spaceAfter=8)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8, leading=11,
                       textColor=MUTED)

FIGURE_TEXT = {
    "01": ("Observations and the hidden signal", "Separate the noisy measurements from the underlying latent truth."),
    "02": ("Several plausible histories", "The blue histories share temporal structure; their spread describes uncertainty about the past."),
    "03": ("Why history points must move together", "Nearby history values are correlated. Independent pointwise draws erase this information."),
    "04": ("From uncertain inputs to uncertain forecast means", "Compare the spread of forecast means with the forecast from one mean-history input."),
    "05": ("Check against a model with a known answer", "The state Monte Carlo curve should track the exact AR curve. Future noise is prescribed analytically."),
    "06": ("Changing the forecast function changes state uncertainty", "Chronos and the exact AR map use the same input posterior but can propagate its uncertainty differently."),
    "07": ("The coherent Chronos experiment", "Left shows state inference after observing data; right shows forecasts made before observing each target."),
    "08": ("The three pieces of one-step variance", "Blue state uncertainty changes over time; known future process and measurement noise add to it."),
    "09": ("Does the particle count matter?", "Compare both particle counts on exactly the same observations. Differences reveal numerical sensitivity."),
    "10": ("Do state intervals cover their intended target?", "A curve near the diagonal has empirical coverage close to its nominal level."),
    "11": ("Estimate noise using training observations", "Compare training-only point estimates across data seeds with the generating noise values."),
    "12": ("History dependence and Monte Carlo sensitivity", "Left changes temporal covariance; right changes the number of posterior-history draws."),
    "13": ("Keep parameter uncertainty as a separate term", "Future noise, state uncertainty conditional on parameters, and parameter uncertainty sum to total variance."),
    "14": ("What the data say about noise parameters", "Read the color as probability mass at a grid node. Concentration is conditional on the fixed prior and AR model."),
    "15": ("Future noise through recursive Chronos dynamics", "At longer leads, future shocks pass through nonlinear dynamics. Negative corrected state estimates flag insufficient Monte Carlo resolution."),
    "16": ("Filtered states in the plot_states layout", "The top row is a forecast before observing the current measurement. The remaining rows are posterior state estimates after that measurement; realized noise states differ from future-noise variances."),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain_caption(value):
    """Convert the manifest's few TeX expressions into readable PDF text."""
    replacements = {
        r"$a^{2h}P_t$": "a^(2h) P(t)",
        r"$R+Q\sum_{j=0}^{h-1}a^{2j}$": "R + Q sum of a^(2j), j = 0,...,h-1",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    value = value.replace("$", "").replace("\u00b2", "^2")
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    value = value.replace("\u00b1", "+/-").replace("\u2019", "'")
    return re.sub(r"\s+", " ", value).strip()


def paragraph(pdf, value, y, *, style=BODY, width=TEXT_WIDTH):
    block = Paragraph(escape(plain_caption(value)), style)
    _, height = block.wrap(width, HEIGHT)
    block.drawOn(pdf, MARGIN, y - height)
    return y - height - style.spaceAfter


def page_frame(pdf, page, total, run_name):
    pdf.setStrokeColor(colors.HexColor("#d7dde2"))
    pdf.line(MARGIN, 38, WIDTH - MARGIN, 38)
    pdf.setFillColor(MUTED)
    pdf.setFont("Times-Roman", 8)
    pdf.drawString(MARGIN, 24, "Chronos latent-posterior experiment")
    pdf.drawRightString(WIDTH - MARGIN, 24, f"{page} / {total}")
    pdf.setFont("Times-Roman", 7)
    pdf.drawString(MARGIN, HEIGHT - 24, run_name)


def heading(pdf, value, y=HEIGHT - 50):
    pdf.setFillColor(INK)
    pdf.setFont("Times-Bold", 14)
    pdf.drawString(MARGIN, y, value)
    return y - 22


def numeric(row, key, percentage=False):
    value = row.get(key, "")
    if not value or value.lower() == "nan":
        return "-"
    number = float(value)
    return f"{100 * number:.1f}%" if percentage else f"{number:.2e}" if abs(number) >= 1e4 else f"{number:.3f}"


def introduction(pdf, run, config, summary, pages):
    page_frame(pdf, 1, pages, run.name)
    y = heading(pdf, "Understanding Chronos uncertainty")
    y = paragraph(pdf, "This experiment separates uncertainty about a hidden input history from noise that arrives in the future. Each figure on the following pages has a reading cue and a caption. The complete methods, limitations and saved tables are in report.md beside this guide.", y)
    if config.get("smoke"):
        y = paragraph(pdf, "SMOKE RUN: a reduced pipeline check. The displayed numbers are not the full experiment results.", y, style=ParagraphStyle("Smoke", parent=BODY, textColor=BLUE))
    bounded = config.get("transition_kind") == "bounded_tanh"
    if bounded:
        y = paragraph(pdf, "EXPLORATORY FOLLOW-UP: the unconstrained recursive model diverged. This run uses g = 3 tanh(Chronos median / 3), a declared amplitude prior. Both truth and inference use this changed model. The original failure is preserved; this is not a unique split of the original Chronos intervals.", y, style=ParagraphStyle("Followup", parent=BODY, textColor=BLUE))
    elif not config.get("smoke"):
        y = paragraph(pdf, "FAILED UNBOUNDED CONTROL: recursive dynamics and filtering diverged, with late floating-point resolution too coarse for the specified noise. Retained full-run scores below describe that failure and must not be read as reliable calibration.", y, style=ParagraphStyle("Failure", parent=BODY, textColor=BLUE))
    transition_text = "The primary model uses the bounded Chronos median" if bounded else "The primary model uses the frozen Chronos median"
    y = paragraph(pdf, transition_text + " as a deterministic transition g. Latent state evolves as X(next) = g(history) + process noise; the observation adds measurement noise. Known Q = 0.01 and R = 0.04 give future-noise variance A = 0.05. State uncertainty E is the variance of g across posterior histories. Total variance is A + E. Frozen Chronos weights have no inferred posterior.", y)
    y = paragraph(pdf, "The coherent control generates and filters data with the same Chronos transition. A separate hybrid control draws histories from an exact AR posterior, then passes them through Chronos. Its future-noise formula is prescribed by the AR model; it is a sensitivity study with different future dynamics.", y)
    pdf.setFillColor(INK)
    pdf.setFont("Times-Bold", 11)
    pdf.drawString(MARGIN, y, "Saved test results")
    y -= 15
    selected = []
    labels = {"particle64": "Coherent particle filter", "raw_chronos": "Hybrid: raw Chronos",
              "joint_known_noise": "Hybrid: joint histories", "plugin_known_noise": "Hybrid: mean history",
              "joint_fitted_noise": "Hybrid: fitted noise"}
    order = ["particle64", "raw_chronos", "joint_known_noise", "plugin_known_noise", "joint_fitted_noise"]
    for method in order:
        row = next((item for item in summary if item["method"] == method), None)
        if row:
            selected.append([labels[method], str(int(float(row["n_targets"]))),
                             numeric(row, "observation_rmse"), numeric(row, "observation_crps"),
                             numeric(row, "observation_coverage90", True), numeric(row, "predictable_coverage90", True)])
    table_data = [["Method", "N", "Obs. RMSE", "Obs. CRPS", "Obs. 90%", "State 90%"], *selected]
    table = Table(table_data, colWidths=[167, 35, 80, 80, 81, 81])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f5")),
        ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#bcc8d0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    _, height = table.wrap(TEXT_WIDTH, HEIGHT)
    table.drawOn(pdf, MARGIN, y - height)
    y -= height + 12
    y = paragraph(pdf, "RMSE and CRPS are in signal units (a.u.); smaller is better. Obs. 90% is empirical observation coverage. State 90% targets the predictable transition mean, excluding zero-state-variance rows; it does not target the next latent value containing fresh process noise. N counts forecast rows, not independent trials. A dash means no declared state component or eligible state score.", y, style=SMALL)
    pdf.setFillColor(INK)
    pdf.setFont("Times-Bold", 11)
    pdf.drawString(MARGIN, y, "Suggested reading path")
    y -= 17
    y = paragraph(pdf, "Figures 1-4: hidden histories and propagation. Figure 5: analytical control. Figure 6: AR-to-Chronos hybrid. Figures 7-10: coherent mechanism and calibration. Figures 11-14: noise fitting and parameter uncertainty. Figure 15: recursive multi-step simulation. Figure 16: the plot_states-style view, followed by its reading notes.", y)
    y = paragraph(pdf, "Displayed uncertainty bands are plus/minus one standard deviation. The coverage tables use 90% Gaussian moment intervals. Variances add; standard deviations do not. Known noise validates a conditional computational mechanism, while real-data identification and nonlinear posterior convergence remain open questions.", y)
    if y < 52:
        raise ValueError("Introduction content exceeds the available page")
    pdf.showPage()


def figure_template(pdf, entry, index, pages, run_name, full_height=False):
    page_frame(pdf, index + 2, pages, run_name)
    number = entry["name"].split("_", 1)[0]
    title, cue = FIGURE_TEXT[number]
    y = heading(pdf, f"Figure {int(number)}. {title}")
    if full_height:
        frame = (MARGIN, 60, TEXT_WIDTH, y - 75)
        pdf.showPage()
        return frame
    # The vector chart is merged into this reserved upper rectangle afterward.
    frame = (MARGIN, 348, TEXT_WIDTH, y - 358)
    y = 324
    pdf.setFillColor(BLUE)
    pdf.setFont("Times-Bold", 10)
    pdf.drawString(MARGIN, y, "What to look for")
    y = paragraph(pdf, cue, y - 17)
    pdf.setFillColor(INK)
    pdf.setFont("Times-Bold", 10)
    pdf.drawString(MARGIN, y, "How this figure was made")
    y = paragraph(pdf, entry["caption"], y - 17)
    y = paragraph(pdf, "Figure source: " + entry["name"] + ".pdf. Original PDF/PGF exports and PNG previews remain in the experiment's figures directory.", y, style=SMALL)
    if y < 52:
        raise ValueError(f"Caption exceeds the page: {entry['name']}")
    pdf.showPage()
    return frame


def states_notes(pdf, entry, pages, run_name):
    page_frame(pdf, pdf.getPageNumber(), pages, run_name)
    y = heading(pdf, "Reading the state panels")
    y = paragraph(pdf, "What to look for", y, style=ParagraphStyle("Cue", parent=BODY, fontName="Times-Bold"))
    y = paragraph(pdf, FIGURE_TEXT["16"][1], y)
    y = paragraph(pdf, "How this figure was made", y, style=ParagraphStyle("How", parent=BODY, fontName="Times-Bold"))
    paragraph(pdf, entry["caption"], y)
    pdf.showPage()


def build(run):
    run = Path(run).resolve()
    figures = EXP / "figures" / run.name
    manifest_path, summary_path, config_path = (run / name for name in
                                              ("figure_manifest.json", "summary_overall.csv", "config.json"))
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    with summary_path.open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    pages = len(manifest) + 1 + sum(entry["name"].startswith("16_") for entry in manifest)
    template = io.BytesIO()
    pdf = canvas.Canvas(template, pagesize=letter, invariant=1)
    pdf.setTitle("Chronos latent-posterior experiment: figure guide")
    introduction(pdf, run, config, summary, pages)
    chart_pages = {}
    for entry in manifest:
        index = pdf.getPageNumber() - 1
        tall = entry["name"].startswith("16_")
        frame = figure_template(pdf, entry, index - 1, pages, run.name, full_height=tall)
        chart_pages[index] = (entry, frame)
        if tall:
            states_notes(pdf, entry, pages, run.name)
    pdf.save()
    template.seek(0)
    base = PdfReader(template)
    writer = PdfWriter()
    figure_hashes = {}
    for index, page in enumerate(base.pages):
        if index in chart_pages:
            entry, frame = chart_pages[index]
            source = figures / f"{entry['name']}.pdf"
            chart = PdfReader(source).pages[0]
            chart.transfer_rotation_to_content()
            left, bottom = float(chart.mediabox.left), float(chart.mediabox.bottom)
            chart_width, chart_height = float(chart.mediabox.width), float(chart.mediabox.height)
            x, y, width, height = frame
            scale = min(width / chart_width, height / chart_height)
            x += (width - chart_width * scale) / 2
            y += (height - chart_height * scale) / 2
            transform = Transformation().translate(-left, -bottom).scale(scale).translate(x, y)
            page.merge_transformed_page(chart, transform, expand=False)
            figure_hashes[str(source)] = digest(source)
        writer.add_page(page)
    writer.add_metadata({"/Title": "Chronos latent-posterior experiment: figure guide",
                         "/Author": "Canari uncertainty experiment",
                         "/Subject": "Saved results and uncertainty decomposition figures"})
    output = run / "guide.pdf"
    with output.open("wb") as stream:
        writer.write(stream)
    reader = PdfReader(output)
    if len(reader.pages) != pages:
        raise ValueError("Unexpected guide page count")
    for index, page in enumerate(reader.pages):
        text = page.extract_text()
        required = ("Understanding Chronos uncertainty" if index == 0 else
                    "Figure " if index in chart_pages else "What to look for")
        if required not in text:
            raise ValueError(f"Missing guide text on page {index + 1}")
    provenance = {"output": str(output), "pages": pages,
                  "source_script_sha256": digest(__file__),
                  "inputs_sha256": {str(path): digest(path) for path in
                                     (manifest_path, summary_path, config_path)},
                  "figure_pdfs_sha256": figure_hashes,
                  "versions": {name: importlib.metadata.version(name) for name in ("reportlab", "pypdf")},
                  "vector_figure_merge": True,
                  "page_text_checks_passed": True,
                  "render_example": ["/opt/homebrew/bin/pdftoppm", "-r", "100", "-png",
                                     str(output), str(run / "pdf_preview/page")]}
    (run / "guide_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed saved results directory")
    build(parser.parse_args().run)
