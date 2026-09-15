// build_pv_ml_presentation.js — generates presentation/PV_ML_Final_Presentation.pptx
// Built from the approved presentation/SLIDE_OUTLINE.md (14 slides).
// Every number is taken directly from results/FINAL_EXPERIMENT_TABLE.csv and
// the underlying results/problemN/ files — verified before writing.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (p) => path.join(ROOT, "figures", p);

// ---------------------------------------------------------------------------
// Palette — solar/energy theme: deep navy (dominant) + solar gold (accent) +
// teal (positive-finding accent) + muted terracotta (negative-finding accent)
// (same palette as the earlier presentation pass, for visual consistency
// across this project's deliverables)
// ---------------------------------------------------------------------------
const NAVY = "132A46";
const NAVY_LIGHT = "24466E";
const GOLD = "F5A623";
const OFFWHITE = "F7F8FA";
const CARD = "FFFFFF";
const CHARCOAL = "1A1A1A";
const GRAY = "5B6B7C";
const TEAL = "2A9D8F";
const TERRACOTTA = "C0524A";
const LINE = "E1E6EC";

const FONT_HEAD = "Cambria";
const FONT_BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33" x 7.5"
pres.author = "ECE571 Student";
pres.company = "ECE571 Machine Learning Course Project";
pres.title = "Photovoltaic (Solar) Power Prediction on Large-Scale Spatiotemporal Data";

const SLIDE_W = 13.33, SLIDE_H = 7.5;
const MARGIN = 0.55;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function newSlide(bg) {
  const slide = pres.addSlide();
  slide.background = { color: bg || OFFWHITE };
  return slide;
}

function pageNum(slide, n, dark) {
  slide.addText(String(n), {
    x: SLIDE_W - 0.7, y: SLIDE_H - 0.45, w: 0.4, h: 0.3,
    fontFace: FONT_BODY, fontSize: 10, color: dark ? "8FA3B8" : GRAY,
    align: "right", isTextBox: true, margin: 0,
  });
}

function contentTitle(slide, kicker, title) {
  slide.addText(kicker.toUpperCase(), {
    x: MARGIN, y: 0.42, w: 10, h: 0.32,
    fontFace: FONT_BODY, fontSize: 13, bold: true, color: GOLD,
    charSpacing: 1.5, isTextBox: true, margin: 0,
  });
  slide.addText(title, {
    x: MARGIN, y: 0.74, w: 12.2, h: 0.7,
    fontFace: FONT_HEAD, fontSize: 30, bold: true, color: NAVY,
    isTextBox: true, margin: 0,
  });
}

function statCallout(slide, x, y, w, h, value, label, color) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: CARD }, line: { color: LINE, width: 1 },
    shadow: { type: "outer", color: "1A2B3C", opacity: 0.12, blur: 6, offset: 2, angle: 90 },
  });
  slide.addText(value, {
    x: x + 0.12, y: y + 0.12, w: w - 0.24, h: h * 0.55,
    fontFace: FONT_HEAD, fontSize: 26, bold: true, color: color || NAVY,
    align: "center", valign: "bottom", isTextBox: true, margin: 0,
  });
  slide.addText(label, {
    x: x + 0.12, y: y + h * 0.62, w: w - 0.24, h: h * 0.34,
    fontFace: FONT_BODY, fontSize: 11.5, color: GRAY,
    align: "center", valign: "top", isTextBox: true, margin: 0,
  });
}

function figureCard(slide, x, y, w, h, imgPath, caption) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: CARD }, line: { color: LINE, width: 1 },
    shadow: { type: "outer", color: "1A2B3C", opacity: 0.12, blur: 6, offset: 2, angle: 90 },
  });
  const pad = 0.12;
  const capH = caption ? 0.38 : 0;
  slide.addImage({
    path: imgPath, x: x + pad, y: y + pad, w: w - pad * 2, h: h - pad * 2 - capH,
    sizing: { type: "contain", w: w - pad * 2, h: h - pad * 2 - capH },
  });
  if (caption) {
    slide.addText(caption, {
      x: x + pad, y: y + h - capH - pad * 0.5, w: w - pad * 2, h: capH,
      fontFace: FONT_BODY, fontSize: 11, italic: true, color: GRAY,
      align: "center", isTextBox: true, margin: 0,
    });
  }
}

function bulletList(slide, x, y, w, h, items, opts = {}) {
  const fontSize = opts.fontSize || 15;
  const color = opts.color || CHARCOAL;
  const arr = items.map((t, idx) => ({
    text: t,
    options: {
      bullet: { code: "2022", indent: 18 },
      color, fontFace: FONT_BODY, fontSize,
      breakLine: idx < items.length - 1,
      paraSpaceAfter: opts.spaceAfter || 10,
    },
  }));
  slide.addText(arr, { x, y, w, h, valign: "top", isTextBox: true, margin: 0 });
}

// ===========================================================================
// SLIDE 1 — TITLE
// ===========================================================================
{
  const slide = newSlide(NAVY);

  slide.addShape(pres.ShapeType.ellipse, { x: 10.3, y: -1.3, w: 4.6, h: 4.6, fill: { color: NAVY_LIGHT }, line: { type: "none" } });
  slide.addShape(pres.ShapeType.ellipse, { x: 10.9, y: -0.7, w: 3.4, h: 3.4, fill: { color: "3A5A82" }, line: { type: "none" } });
  slide.addShape(pres.ShapeType.ellipse, { x: 11.5, y: -0.1, w: 2.2, h: 2.2, fill: { color: GOLD }, line: { type: "none" } });

  slide.addText("PHOTOVOLTAIC POWER PREDICTION", {
    x: MARGIN, y: 2.55, w: 11.5, h: 0.4,
    fontFace: FONT_BODY, fontSize: 15, bold: true, color: GOLD, charSpacing: 2,
    isTextBox: true, margin: 0,
  });
  slide.addText("Photovoltaic (Solar) Power Prediction on\nLarge-Scale Spatiotemporal Data", {
    x: MARGIN, y: 2.95, w: 11.5, h: 1.8,
    fontFace: FONT_HEAD, fontSize: 38, bold: true, color: "FFFFFF",
    isTextBox: true, margin: 0, lineSpacingMultiple: 1.08,
  });
  slide.addText("Five Machine Learning Paradigms, One Dataset, One Honest Investigation", {
    x: MARGIN, y: 4.7, w: 11.5, h: 0.5,
    fontFace: FONT_BODY, fontSize: 17, italic: true, color: "CADCFC",
    isTextBox: true, margin: 0,
  });

  slide.addShape(pres.ShapeType.line, { x: MARGIN, y: 5.5, w: 3.2, h: 0, line: { color: GOLD, width: 2 } });

  slide.addText([
    { text: "[YOUR NAME]", options: { breakLine: true, color: "D9E2EC", fontSize: 14 } },
    { text: "ECE571 Machine Learning", options: { breakLine: true, color: "D9E2EC", fontSize: 14 } },
    { text: "Instructor: [PROFESSOR NAME]", options: { breakLine: true, color: "D9E2EC", fontSize: 14 } },
    { text: "[DATE]", options: { color: "D9E2EC", fontSize: 14 } },
  ], { x: MARGIN, y: 5.8, w: 8, h: 1.4, fontFace: FONT_BODY, isTextBox: true, margin: 0, lineSpacing: 22 });

  slide.addNotes(
    "WHAT TO SAY: Hi, I'm [your name], and this is my ECE571 project — predicting solar power output using " +
    "five different machine learning approaches on the same dataset. Rather than five separate assignments, " +
    "I tried to tell one coherent story about what actually helps when forecasting photovoltaic power, and " +
    "just as importantly, what doesn't.\n" +
    "IMPORTANT POINT: Frame this immediately as ONE investigation, not five disconnected exercises — that's " +
    "the whole narrative thread for the next 10 minutes.\n" +
    "TRANSITION: Let's start with why this problem matters in the first place.\n" +
    "APPROX TIME: 0:30"
  );
}

// ===========================================================================
// SLIDE 2 — WHY SOLAR POWER FORECASTING?
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Motivation", "Why Solar Power Forecasting?");

  slide.addText(
    "Solar power output is not something you can turn up on demand — it is set entirely by the weather.",
    { x: MARGIN, y: 1.55, w: 6.15, h: 0.9, fontFace: FONT_BODY, fontSize: 16, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  const factors = ["Irradiance (GHI / DNI / DHI)", "Cloud cover & cloud type", "Temperature", "Time of day & season"];
  slide.addText("Output depends on:", { x: MARGIN, y: 2.75, w: 5.6, h: 0.35, fontFace: FONT_BODY, bold: true, fontSize: 14, color: NAVY, isTextBox: true, margin: 0 });
  bulletList(slide, MARGIN, 3.15, 5.6, 2.4, factors, { fontSize: 14.5 });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 6.9, y: 1.55, w: 5.9, h: 4.6, rectRadius: 0.08,
    fill: { color: NAVY }, line: { type: "none" },
  });
  slide.addText("Why it matters operationally", {
    x: 7.15, y: 1.8, w: 5.4, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 14, color: GOLD, isTextBox: true, margin: 0,
  });
  bulletList(slide, 7.15, 2.3, 5.4, 3.7, [
    "Grid operators need to schedule other generation and reserves around expected solar output",
    "Battery storage systems need forecasts to decide when to charge and discharge",
    "Utilities planning a new site with little or no history need some way to estimate output",
  ], { fontSize: 14.5, color: "E7ECF2", spaceAfter: 16 });

  slide.addText(
    "All three are, at their core, the same ML problem: relate weather and irradiance to electrical output — sometimes with plenty of historical data, sometimes with very little.",
    { x: MARGIN, y: 6.35, w: 12.2, h: 0.7, fontFace: FONT_BODY, italic: true, fontSize: 14.5, color: GRAY, isTextBox: true, margin: 0 }
  );

  pageNum(slide, 2);

  slide.addNotes(
    "WHAT TO SAY: Solar power is different from a fossil-fuel plant because you can't just turn it up when " +
    "you need more — the sun decides. That creates real problems: grid operators need to know how much " +
    "power is coming so they can plan other generation, storage systems need forecasts to know when to " +
    "charge or discharge, and utilities planning a brand-new site often have little or no historical data.\n" +
    "IMPORTANT POINT: This is called 'intermittency' — a generation source you can forecast but not control. " +
    "That's the reason PV forecasting is its own research area.\n" +
    "TRANSITION: Let's look at the actual dataset I used.\n" +
    "APPROX TIME: 0:45"
  );
}

// ===========================================================================
// SLIDE 3 — THE DATASET
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "The Data", "The Dataset");

  const cities = [
    ["Davis, CA", "2011–2016", "164.0 kW avg"],
    ["Amherst, MA", "2018–2020", "60.9 kW avg"],
    ["Huron, SD", "2011–2016", "50.1 kW avg"],
    ["Santa Barbara, CA", "2011–2016", "49.1 kW avg"],
    ["La Jolla, CA", "2011–2016", "47.3 kW avg"],
  ];
  const colW = 2.36, startX = MARGIN, y0 = 1.55;
  cities.forEach((c, idx) => {
    const x = startX + idx * (colW + 0.06);
    slide.addShape(pres.ShapeType.roundRect, {
      x, y: y0, w: colW, h: 1.5, rectRadius: 0.07,
      fill: { color: CARD }, line: { color: LINE, width: 1 },
    });
    slide.addText(c[0], { x: x + 0.1, y: y0 + 0.12, w: colW - 0.2, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 13, color: NAVY, isTextBox: true, margin: 0 });
    slide.addText(c[1], { x: x + 0.1, y: y0 + 0.55, w: colW - 0.2, h: 0.3, fontFace: FONT_BODY, fontSize: 11.5, color: GRAY, isTextBox: true, margin: 0 });
    slide.addText(c[2], { x: x + 0.1, y: y0 + 0.95, w: colW - 0.2, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 14, color: GOLD, isTextBox: true, margin: 0 });
  });

  statCallout(slide, MARGIN, 3.35, 2.9, 1.05, "22", "columns: irradiance, weather, Cloud Type, Output Power", TEAL);
  statCallout(slide, MARGIN + 3.05, 3.35, 2.9, 1.05, "30 min", "measurement interval, 10:00–15:00 daily window", TEAL);
  statCallout(slide, MARGIN + 6.1, 3.35, 2.9, 1.05, "~24,000", "rows for each 6-year city (Amherst: 12,056)", TEAL);
  statCallout(slide, MARGIN + 9.15, 3.35, 2.9, 1.05, "3.5×", "Davis's output scale vs. the smallest city", TERRACOTTA);

  slide.addShape(pres.ShapeType.roundRect, {
    x: MARGIN, y: 4.75, w: 12.23, h: 2.05, rectRadius: 0.08,
    fill: { color: NAVY }, line: { type: "none" },
  });
  slide.addText("\u201cThe dataset is spatiotemporal\u201d — why that matters:", {
    x: MARGIN + 0.25, y: 4.95, w: 11.7, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 14.5, color: GOLD, isTextBox: true, margin: 0,
  });
  slide.addText(
    "Data varies across both SPACE (5 different cities, different climates and plant sizes) and TIME " +
    "(30-minute readings across multiple years) — that's exactly what makes cross-city comparison (Problems 2 " +
    "& 5) and time-based forecasting (Problem 2's sequence model) possible in the first place.",
    { x: MARGIN + 0.25, y: 5.38, w: 11.7, h: 1.3, fontFace: FONT_BODY, fontSize: 14.5, color: "E7ECF2", isTextBox: true, margin: 0 }
  );

  pageNum(slide, 3);

  slide.addNotes(
    "WHAT TO SAY: This is the dataset — five cities, mostly six years of data at 30-minute intervals, 22 " +
    "columns covering irradiance, weather, cloud type, and Output Power. The big number on the right — " +
    "Davis's plant produces roughly three and a half times the average output of the smallest city. That's " +
    "not a data quality issue, it's just different plant sizes, but it matters enormously later.\n" +
    "IMPORTANT POINT: 'Spatiotemporal' just means the data varies across both space — five different cities " +
    "— and time — years of 30-minute readings. That combination is exactly what makes cross-city comparison " +
    "and time-based forecasting possible.\n" +
    "TRANSITION: Now let's look at how I actually structured the experiments around this data.\n" +
    "APPROX TIME: 1:00"
  );
}

// ===========================================================================
// SLIDE 4 — EXPERIMENTAL DESIGN
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Methodology", "Experimental Design — Five ML Paradigms");

  const paradigms = [
    ["1", "Classification", "Sky-condition & generation-regime"],
    ["2", "Regression", "Output Power — the core task"],
    ["3", "Dimension Reduction", "PCA & autoencoder compression"],
    ["4", "Semi-Supervised", "Learning from partly-labeled data"],
    ["5", "Transfer Learning", "Davis → Amherst knowledge transfer"],
  ];
  const pw = 2.36;
  paradigms.forEach((p, idx) => {
    const x = MARGIN + idx * (pw + 0.06);
    const y = 1.6;
    slide.addShape(pres.ShapeType.roundRect, { x, y, w: pw, h: 1.9, rectRadius: 0.07, fill: { color: CARD }, line: { color: LINE, width: 1 } });
    slide.addShape(pres.ShapeType.ellipse, { x: x + 0.15, y: y + 0.15, w: 0.5, h: 0.5, fill: { color: GOLD }, line: { type: "none" } });
    slide.addText(p[0], { x: x + 0.15, y: y + 0.15, w: 0.5, h: 0.5, fontFace: FONT_HEAD, bold: true, fontSize: 18, color: NAVY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    slide.addText(p[1], { x: x + 0.15, y: y + 0.78, w: pw - 0.3, h: 0.55, fontFace: FONT_BODY, bold: true, fontSize: 13.5, color: NAVY, isTextBox: true, margin: 0 });
    slide.addText(p[2], { x: x + 0.15, y: y + 1.28, w: pw - 0.3, h: 0.55, fontFace: FONT_BODY, fontSize: 11, color: GRAY, isTextBox: true, margin: 0 });
  });

  slide.addText("Same-city", {
    x: MARGIN, y: 3.85, w: 3.6, h: 0.5, fontFace: FONT_HEAD, bold: true, fontSize: 20, color: TEAL, align: "center", isTextBox: true, margin: 0,
  });
  slide.addText("vs.", {
    x: 4.15, y: 3.9, w: 1.0, h: 0.4, fontFace: FONT_BODY, fontSize: 16, color: GRAY, align: "center", isTextBox: true, margin: 0,
  });
  slide.addText("Cross-city", {
    x: 5.15, y: 3.85, w: 3.6, h: 0.5, fontFace: FONT_HEAD, bold: true, fontSize: 20, color: TERRACOTTA, align: "center", isTextBox: true, margin: 0,
  });
  slide.addText("(used throughout Problems 2 & 5 — train on one city, test on the same city vs. a different one)", {
    x: MARGIN, y: 4.35, w: 8.75, h: 0.4, fontFace: FONT_BODY, italic: true, fontSize: 12.5, color: GRAY, align: "center", isTextBox: true, margin: 0,
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: MARGIN, y: 4.95, w: 12.23, h: 1.85, rectRadius: 0.08,
    fill: { color: NAVY }, line: { type: "none" },
  });
  slide.addText("Chronological splitting — used everywhere:", {
    x: MARGIN + 0.25, y: 5.15, w: 11.7, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 14.5, color: GOLD, isTextBox: true, margin: 0,
  });
  slide.addText(
    "Earliest 80% of a city's timestamps train, latest 20% test — never shuffled. Random shuffling would " +
    "let a model trained on next Tuesday's readings predict last Monday's, which would make every metric " +
    "meaningless for real forecasting.",
    { x: MARGIN + 0.25, y: 5.58, w: 11.7, h: 1.1, fontFace: FONT_BODY, fontSize: 14.5, color: "E7ECF2", isTextBox: true, margin: 0 }
  );

  pageNum(slide, 4);

  slide.addNotes(
    "WHAT TO SAY: I tested five ML paradigms on this dataset, each with its own fair baseline: classification, " +
    "regression — the core task — dimension reduction, semi-supervised learning, and transfer learning. Two " +
    "and five specifically compare same-city versus cross-city performance. And every single experiment uses " +
    "chronological splitting: earliest eighty percent of a city's data trains, most recent twenty percent " +
    "tests, never shuffled.\n" +
    "IMPORTANT POINT: Why chronological and not random? Because this is time-series data — random shuffling " +
    "would let the model see the future relative to what it's being tested on, which would make every " +
    "reported number meaningless.\n" +
    "TRANSITION: Before the results, let me quickly show the preprocessing pipeline that feeds all five " +
    "problems.\n" +
    "APPROX TIME: 1:00"
  );
}

// ===========================================================================
// SLIDE 5 — PREPROCESSING PIPELINE
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Methodology", "Preprocessing Pipeline");

  const steps = ["Raw\nData", "Missing-Value\nHandling", "Feature\nEngineering", "Categorical\nEncoding", "Scaling", "Chronological\nSplit", "ML\nModels"];
  const stepW = 1.55, gap = 0.2, stepY = 1.65, stepH = 1.0;
  const totalW = steps.length * stepW + (steps.length - 1) * gap;
  let sx = (SLIDE_W - totalW) / 2;
  steps.forEach((s, idx) => {
    slide.addShape(pres.ShapeType.roundRect, {
      x: sx, y: stepY, w: stepW, h: stepH, rectRadius: 0.06,
      fill: { color: idx === steps.length - 1 ? TEAL : NAVY }, line: { type: "none" },
    });
    slide.addText(s, {
      x: sx, y: stepY, w: stepW, h: stepH, fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", isTextBox: true, margin: 0,
    });
    if (idx < steps.length - 1) {
      slide.addText("→", { x: sx + stepW, y: stepY, w: gap + 0.02, h: stepH, fontFace: FONT_BODY, fontSize: 16, color: GRAY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    }
    sx += stepW + gap;
  });

  const details = [
    ["Clear-Sky Index", "GHI ÷ Clearsky GHI — measures how close actual irradiance is to the clear-sky maximum; defines Problem 1's sky-condition label"],
    ["Cyclical time features", "Hour/month/day-of-year → sine/cosine pairs, so 23:00 and 00:00 are numerically close, not maximally far apart"],
    ["Categorical encoding", "Cloud Type is one-hot encoded everywhere — it's a category code, not an ordered number"],
    ["Lag / sequence windows", "Problem 2's forecasting sub-task uses the previous 12 readings (~6 hours) to predict the next one"],
  ];
  let dy = 3.05;
  details.forEach((d) => {
    slide.addShape(pres.ShapeType.ellipse, { x: MARGIN, y: dy + 0.05, w: 0.16, h: 0.16, fill: { color: GOLD }, line: { type: "none" } });
    slide.addText(d[0] + ":", { x: MARGIN + 0.3, y: dy, w: 3.0, h: 0.6, fontFace: FONT_BODY, bold: true, fontSize: 13.5, color: NAVY, isTextBox: true, margin: 0 });
    slide.addText(d[1], { x: MARGIN + 3.35, y: dy, w: 8.9, h: 0.6, fontFace: FONT_BODY, fontSize: 13, color: CHARCOAL, isTextBox: true, margin: 0 });
    dy += 0.82;
  });

  slide.addText(
    "Every scaler and encoder is fit on training data only, then applied unchanged to test data — never the reverse.",
    { x: MARGIN, y: 6.45, w: 12.2, h: 0.4, fontFace: FONT_BODY, italic: true, fontSize: 13, color: GRAY, isTextBox: true, margin: 0 }
  );

  pageNum(slide, 5);

  slide.addNotes(
    "WHAT TO SAY: All five problems share one preprocessing pipeline: clean the raw data, engineer features " +
    "like the Clear-Sky Index and cyclical time encodings, one-hot encode Cloud Type, scale everything, split " +
    "chronologically, then train. A few specific decisions worth mentioning: the Clear-Sky Index — GHI " +
    "divided by the theoretical clear-sky maximum — is what defines Problem 1's label. Time features use " +
    "sine and cosine pairs so that, say, 11pm and midnight end up close together numerically instead of far " +
    "apart. And Problem 2's forecasting sub-task uses the past 12 readings, about six hours, to predict the " +
    "next one.\n" +
    "IMPORTANT POINT: Every scaler is fit on TRAINING data only, then applied unchanged to test data — this " +
    "is the core leakage-prevention rule that holds everywhere in the project.\n" +
    "TRANSITION: Let's get into the actual results, starting with classification.\n" +
    "APPROX TIME: 1:00"
  );
}



// ===========================================================================
// SLIDE 6 — PROBLEM 1: CLASSIFICATION
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Problem 1", "Classification");

  slide.addText(
    "Predict sky-condition (Clear / Partly Cloudy / Overcast) from weather features — GHI excluded, since it defines the label itself.",
    { x: MARGIN, y: 1.55, w: 6.9, h: 0.85, fontFace: FONT_BODY, fontSize: 14.5, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  statCallout(slide, MARGIN, 2.55, 3.35, 1.15, "0.772", "Balanced Accuracy — Best Model (Davis)", TEAL);
  statCallout(slide, MARGIN + 3.5, 2.55, 3.35, 1.15, "Log. Reg.", "Best model (balanced weight) — not an ensemble!", TEAL);

  slide.addText("Hardest class: Partly Cloudy (F1 = 0.566)", {
    x: MARGIN, y: 3.9, w: 6.9, h: 0.35, fontFace: FONT_BODY, bold: true, fontSize: 13.5, color: NAVY, isTextBox: true, margin: 0,
  });
  bulletList(slide, MARGIN, 4.3, 6.9, 2.6, [
    "Sits between Clear and Overcast on the Clear-Sky Index scale — confused with both neighbors",
    "Clear conditions were easiest (F1 = 0.946) — also the majority class (72% of data)",
    "Class imbalance is why balanced accuracy, not raw accuracy, is the headline metric",
  ], { fontSize: 14 });

  figureCard(slide, 8.05, 1.55, 4.75, 5.0, FIG("problem1/problem1_sky_condition_confusion_matrix_Davis.png"),
    "Confusion matrix, best Davis sky-condition model");

  pageNum(slide, 6);

  slide.addNotes(
    "WHAT TO SAY: First result: classification. Predict whether the sky is Clear, Partly Cloudy, or " +
    "Overcast, from weather features. I excluded GHI and related irradiance columns, because the label " +
    "itself is defined by GHI divided by clear-sky GHI — including them would just leak the answer. The " +
    "best model, at 0.772 balanced accuracy, was plain logistic regression with class weighting — not a " +
    "random forest or neural network. Looking at the confusion matrix, Clear is easy, but Partly Cloudy is " +
    "the hardest class, because it sits on the boundary between the other two.\n" +
    "IMPORTANT POINT: Balanced accuracy averages each class's recall equally, so a model that always guessed " +
    "'Clear' — the majority class at 72% of the data — can't fake a good score.\n" +
    "TRANSITION: Now, the core task of the whole project — regression.\n" +
    "APPROX TIME: 1:00"
  );
}

// ===========================================================================
// SLIDE 7 — PROBLEM 2: REGRESSION (CORE TASK)
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Problem 2 — Core Task", "Regression");

  slide.addText(
    "Predict continuous Output Power (kW). Compared same-city (train & test on Davis) vs. cross-city (Davis model applied directly to 3 other cities, no retraining).",
    { x: MARGIN, y: 1.55, w: 6.9, h: 0.85, fontFace: FONT_BODY, fontSize: 14, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  statCallout(slide, MARGIN, 2.5, 2.2, 1.1, "15.17", "RMSE (kW), Davis", TEAL);
  statCallout(slide, MARGIN + 2.3, 2.5, 2.2, 1.1, "8.31", "MAE (kW), Davis", TEAL);
  statCallout(slide, MARGIN + 4.6, 2.5, 2.3, 1.1, "0.953", "R², Davis", TEAL);

  slide.addText("Best model: Gradient Boosting (Davis, same-city)", {
    x: MARGIN, y: 3.8, w: 6.9, h: 0.35, fontFace: FONT_BODY, bold: true, fontSize: 13.5, color: NAVY, isTextBox: true, margin: 0,
  });
  bulletList(slide, MARGIN, 4.2, 6.9, 2.7, [
    "Strongest result in the whole project — explains 95.3% of Output Power's variance from weather alone",
    "Cross-city (same model → Huron/Santa Barbara/La Jolla): RMSE ≈ 125 kW — ~8× worse",
    "Why: scale mismatch, not a modeling failure — Davis averages 164 kW, others 47–50 kW. Rescaling by the target city's mean recovers R²=0.55–0.84",
  ], { fontSize: 13.5 });

  figureCard(slide, 8.35, 1.55, 4.45, 5.0, FIG("problem2/problem2_predicted_vs_actual_davis.png"),
    "Predicted vs. actual Output Power, Davis");

  pageNum(slide, 7);

  slide.addNotes(
    "WHAT TO SAY: This is the project's core task — predict actual Output Power in kilowatts. Gradient " +
    "boosting on Davis's own data was the strongest result in the whole project: fifteen kilowatts of RMSE, " +
    "an R-squared of point nine five three — it explains ninety-five percent of the variance using just " +
    "weather. Applying that same Davis model directly to other cities without any adaptation failed badly — " +
    "RMSE around 125 kilowatts. But that's mostly a scale problem, not a failure to learn weather patterns: " +
    "rescaling the predictions by the target city's own average recovers R-squared of point five five to " +
    "point eight four.\n" +
    "IMPORTANT POINT: RMSE penalizes big misses more heavily than MAE does, since it squares errors before " +
    "averaging — that's why I report both. And nRMSE, normalized by the target's range, is what lets me " +
    "compare error across cities with very different power scales.\n" +
    "TRANSITION: Next — did compressing the feature space help or hurt?\n" +
    "APPROX TIME: 1:15"
  );
}

// ===========================================================================
// SLIDE 8 — PROBLEM 3: DIMENSION REDUCTION
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Problem 3", "Dimension Reduction");

  slide.addText(
    "Compress 23 features to d = 2, 5, or 10 with PCA (linear) and an autoencoder (nonlinear) — fit with zero label access — then test whether Problems 1 & 2 still work as well.",
    { x: MARGIN, y: 1.55, w: 12.2, h: 0.65, fontFace: FONT_BODY, fontSize: 14.5, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  figureCard(slide, MARGIN, 2.35, 5.85, 4.25, FIG("problem3/problem3_explained_variance.png"),
    "PCA explained variance by dimension");

  slide.addShape(pres.ShapeType.roundRect, {
    x: 6.65, y: 2.35, w: 6.13, h: 4.25, rectRadius: 0.08,
    fill: { color: NAVY }, line: { type: "none" },
  });
  slide.addText("Did reducing the feature space help prediction?", {
    x: 6.9, y: 2.6, w: 5.6, h: 0.5, fontFace: FONT_BODY, bold: true, fontSize: 15, color: GOLD, isTextBox: true, margin: 0,
  });
  slide.addText("No.", {
    x: 6.9, y: 3.15, w: 5.6, h: 0.5, fontFace: FONT_HEAD, bold: true, fontSize: 24, color: TERRACOTTA, isTextBox: true, margin: 0,
  });
  bulletList(slide, 6.9, 3.75, 5.6, 2.7, [
    "Raw features beat every PCA/autoencoder version, at every dimension, on both classification and regression",
    "Raw: 0.743 balanced accuracy vs. best reduced (Autoencoder, d=10): 0.661",
    "Traced to a cause: Cloud Type (highly predictive) compresses poorly into a few continuous dimensions",
  ], { fontSize: 13.5, color: "E7ECF2", spaceAfter: 14 });

  pageNum(slide, 8);

  slide.addNotes(
    "WHAT TO SAY: Problem three asks: can I compress this feature space without hurting the tasks that " +
    "depend on it? I tried PCA — the classical, linear method — and a small autoencoder — the deep, " +
    "nonlinear method — at three sizes, fit with zero access to any label. The honest answer is no: raw, " +
    "uncompressed features beat every reduced version, at every dimension, on both downstream tasks. That " +
    "surprised me going in. I traced part of this to Cloud Type — a feature that turns out to be highly " +
    "useful but doesn't compress well into a handful of continuous numbers.\n" +
    "IMPORTANT POINT: Explained variance tells you how much of the INPUT features' spread you kept — it " +
    "says nothing about whether that spread includes the specific information needed to predict a label. " +
    "That's exactly the gap this result exposes.\n" +
    "TRANSITION: Next — what if you barely have any labels at all?\n" +
    "APPROX TIME: 1:00"
  );
}

// ===========================================================================
// SLIDE 9 — PROBLEM 4: SEMI-SUPERVISED LEARNING
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Problem 4", "Semi-Supervised Learning");

  slide.addText(
    "Only 10%, 30%, or 50% of Davis's sky-condition labels are available. Can pseudo-labeling — using the model's own confident predictions on unlabeled data — beat training on the labeled subset alone?",
    { x: MARGIN, y: 1.55, w: 12.2, h: 0.7, fontFace: FONT_BODY, fontSize: 14.5, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  figureCard(slide, MARGIN, 2.4, 5.85, 4.2, FIG("problem4/problem4_label_efficiency_curve.png"),
    "Label-efficiency curve: supervised-only vs. SSL");

  slide.addText("Did unlabeled data help — even at 10% labels? NO.", {
    x: 6.65, y: 2.5, w: 6.13, h: 0.6, fontFace: FONT_BODY, bold: true, fontSize: 16, color: TERRACOTTA, isTextBox: true, margin: 0,
  });
  bulletList(slide, 6.65, 3.25, 6.13, 3.3, [
    "SSL gain was slightly negative at every fraction tested (10%, 30%, 50%) — an honest, reported finding",
    "Diagnosis: pseudo-labels were 100% accurate (checked after training only) — but 98% of the first round were the easy \u201cClear\u201d class",
    "Correct, but redundant — reinforced what the model already knew instead of helping with the hard Partly Cloudy / Overcast boundary",
  ], { fontSize: 13.5, spaceAfter: 14 });

  pageNum(slide, 9);

  slide.addNotes(
    "WHAT TO SAY: For semi-supervised learning, I hid most of Davis's sky-condition labels — down to as " +
    "few as ten percent — and tested whether pseudo-labeling, where the model labels the unlabeled data " +
    "itself when confident enough, could beat just training on the small labeled set alone. The honest " +
    "answer is no, especially interesting at the ten percent case — the gap is slightly negative at every " +
    "label fraction. I checked why, using the hidden true labels purely as a diagnostic after training was " +
    "already finished. The pseudo-labels were completely accurate — a hundred percent — but ninety-eight " +
    "percent of the first batch were all the same, easy 'Clear' class.\n" +
    "IMPORTANT POINT: This means the method wasn't adding WRONG information — it was adding REDUNDANT " +
    "information. That's a more precise, more useful finding than just 'SSL didn't work.'\n" +
    "TRANSITION: Now, the last paradigm, and the most interesting result in the project — what if a whole " +
    "different city could help?\n" +
    "APPROX TIME: 1:00"
  );
}

// ===========================================================================
// SLIDE 10 — PROBLEM 5: TRANSFER LEARNING (most depth — includes the
// negative-transfer story, per the approved outline's consolidation)
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Problem 5 — Most Interesting Result", "Transfer Learning");

  // Small Davis -> Amherst diagram
  const dy = 1.55, dh = 0.7;
  const diagSteps = [
    ["Davis\n(data-rich)", NAVY, "FFFFFF"],
    ["Pretrain", NAVY_LIGHT, "FFFFFF"],
    ["Fine-Tune\non k samples", GOLD, NAVY],
    ["Amherst\n(data-poor)", TEAL, "FFFFFF"],
  ];
  const dW = 2.5, dGap = 0.35;
  let dx = MARGIN;
  diagSteps.forEach((s, idx) => {
    slide.addShape(pres.ShapeType.roundRect, { x: dx, y: dy, w: dW, h: dh, rectRadius: 0.06, fill: { color: s[1] }, line: { type: "none" } });
    slide.addText(s[0], { x: dx, y: dy, w: dW, h: dh, fontFace: FONT_BODY, bold: true, fontSize: 11.5, color: s[2], align: "center", valign: "middle", isTextBox: true, margin: 0 });
    if (idx < diagSteps.length - 1) {
      slide.addText("→", { x: dx + dW, y: dy, w: dGap, h: dh, fontFace: FONT_BODY, fontSize: 16, color: GRAY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    }
    dx += dW + dGap;
  });

  statCallout(slide, MARGIN, 2.5, 2.85, 1.0, "+21.4%", "Transfer gain at k=10 samples", TEAL);
  statCallout(slide, MARGIN + 2.95, 2.5, 2.85, 1.0, "30.03", "Transfer RMSE (kW), k=10", TEAL);
  statCallout(slide, MARGIN + 5.9, 2.5, 2.85, 1.0, "184.17", "Zero-shot RMSE (kW)", TERRACOTTA);

  bulletList(slide, MARGIN, 3.75, 6.05, 1.0, [
    "Transfer (fine-tuned) beat few-shot (fresh model) at every k tested — most at k=10",
  ], { fontSize: 13, spaceAfter: 8 });

  slide.addShape(pres.ShapeType.roundRect, {
    x: MARGIN, y: 4.55, w: 6.05, h: 2.3, rectRadius: 0.08,
    fill: { color: TERRACOTTA }, line: { type: "none" },
  });
  slide.addText("Negative transfer DID occur — and was fixed", {
    x: MARGIN + 0.2, y: 4.72, w: 5.65, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 13.5, color: "FFFFFF", isTextBox: true, margin: 0,
  });
  slide.addText(
    "An early version, with too small a fine-tuning learning rate (the assignment's own suggested strategy), " +
    "scored 140.9 kW at k=10 — WORSE than few-shot's 38.2 kW. Diagnosed: output stuck near Davis's scale. " +
    "Fixed by matching the pretraining learning rate.",
    { x: MARGIN + 0.2, y: 5.12, w: 5.65, h: 1.6, fontFace: FONT_BODY, fontSize: 12.5, color: "FBEAE8", isTextBox: true, margin: 0 }
  );

  figureCard(slide, 6.75, 3.75, 6.05, 3.1, FIG("problem5/problem5_transfer_curve.png"), "RMSE vs. Amherst labeled sample count");

  pageNum(slide, 10);

  slide.addNotes(
    "WHAT TO SAY: The last paradigm, and the most interesting result in the whole project. Can Davis, with " +
    "six years of data, help Amherst, which only has three? I compared zero-shot — no Amherst training at " +
    "all — few-shot — a fresh model trained only on a handful of Amherst samples — and transfer — the " +
    "Davis-pretrained model fine-tuned on that same handful. Zero-shot failed badly, same scale problem as " +
    "before. Transfer clearly beat training from scratch at every sample count, most at just ten samples — " +
    "a twenty-one percent improvement.\n" +
    "IMPORTANT POINT — the real story: my very first version of this experiment made things WORSE, not " +
    "better, because I used too small a learning rate during fine-tuning — actually what the assignment " +
    "itself suggested. I dug into what the model was predicting, found its output was stuck near Davis's " +
    "scale, and fixed it by matching the pretraining learning rate instead. One hyperparameter was the " +
    "entire difference between failure and success.\n" +
    "TRANSITION: Let's zoom out and see all five problems side by side.\n" +
    "APPROX TIME: 1:15"
  );
}



// ===========================================================================
// SLIDE 11 — OVERALL RESULTS
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "The Whole Project, At A Glance", "Overall Results");

  const headerOpts = { fill: { color: NAVY }, color: "FFFFFF", bold: true, fontSize: 13, fontFace: FONT_BODY, align: "left", valign: "middle" };
  const cellOpts = (shade) => ({ fill: { color: shade ? "F0F3F7" : "FFFFFF" }, color: CHARCOAL, fontSize: 12.5, fontFace: FONT_BODY, valign: "middle" });

  const rows = [
    [
      { text: "Problem", options: headerOpts },
      { text: "Best Method", options: headerOpts },
      { text: "Main Result", options: headerOpts },
      { text: "Main Finding", options: headerOpts },
    ],
    ["P1 — Classification", "Logistic Regression", "0.772 Bal. Acc. (Davis)", "A simple linear model beat every ensemble and neural net"],
    ["P2 — Regression", "Gradient Boosting", "RMSE 15.17 kW (R²=0.953)", "Strongest result in the entire project"],
    ["P3 — Dimension Reduction", "Raw features (no reduction)", "0.743 Bal. Acc. / RMSE 23.86 kW", "Compression hurt at every dimension tested"],
    ["P4 — Semi-Supervised", "Pseudo-labeling", "Macro F1 ≈ 0.70 (all fractions)", "SSL gain ≈ 0 — labels correct, but redundant"],
    ["P5 — Transfer Learning", "Transfer (k=10)", "RMSE 30.03 kW (+21.4%)", "Transfer clearly helped, most when data was scarce"],
  ].map((r, i) => (i === 0 ? r : r.map((t) => ({ text: t, options: cellOpts(i % 2 === 0) }))));

  slide.addTable(rows, {
    x: MARGIN, y: 1.7, w: 12.23, h: 4.6,
    colW: [2.35, 2.55, 3.13, 4.2],
    border: { type: "solid", color: LINE, pt: 0.75 },
    autoPage: false,
    rowH: 0.75,
  });

  slide.addText(
    "Metrics are NOT combined into one fake overall score — balanced accuracy and RMSE aren't comparable quantities.",
    { x: MARGIN, y: 6.5, w: 12.23, h: 0.55, fontFace: FONT_BODY, italic: true, fontSize: 13.5, color: GRAY, isTextBox: true, margin: 0 }
  );

  pageNum(slide, 11);

  slide.addNotes(
    "WHAT TO SAY: This table is the whole project in one place. Two paradigms clearly helped — supervised " +
    "learning was strong throughout, and transfer learning gave a real boost when Amherst data was scarce. " +
    "Two did NOT help, and I want to be upfront about that rather than bury it — dimension reduction hurt " +
    "performance at every size I tried, and semi-supervised learning's gain was essentially zero.\n" +
    "IMPORTANT POINT: I deliberately did not combine these into one overall score — a balanced accuracy " +
    "number and an RMSE in kilowatts simply aren't the same kind of quantity, so a fake combined ranking " +
    "would be misleading, not helpful.\n" +
    "TRANSITION: Let's pull out the handful of findings that matter most.\n" +
    "APPROX TIME: 1:00"
  );
}

// ===========================================================================
// SLIDE 12 — MOST IMPORTANT FINDINGS
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Pulling It Together", "Most Important Findings");

  const findings = [
    ["1", "Same-city prediction was far easier than cross-city prediction", "15–20 kW vs. ~125–184 kW RMSE — driven mainly by output-scale mismatch, not a failure to learn weather patterns", TEAL],
    ["2", "A simple linear model beat every ensemble and neural net", "Logistic regression won Problem 1's classification task at 0.772 balanced accuracy — added complexity didn't help here", TEAL],
    ["3", "Semi-supervised learning did NOT help, even at 10% labels", "Pseudo-labels were 100% accurate but redundant — reinforced the easy class instead of the hard one", TERRACOTTA],
    ["4", "Transfer learning clearly helped — but only after fixing a real bug", "A learning-rate issue caused negative transfer during development; fixing it turned failure into a +21.4% win", TEAL],
  ];
  let fy = 1.7;
  findings.forEach((f) => {
    slide.addShape(pres.ShapeType.ellipse, { x: MARGIN, y: fy, w: 0.55, h: 0.55, fill: { color: f[3] }, line: { type: "none" } });
    slide.addText(f[0], { x: MARGIN, y: fy, w: 0.55, h: 0.55, fontFace: FONT_HEAD, bold: true, fontSize: 18, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
    slide.addText(f[1], { x: MARGIN + 0.75, y: fy - 0.03, w: 11.0, h: 0.45, fontFace: FONT_BODY, bold: true, fontSize: 15, color: NAVY, isTextBox: true, margin: 0 });
    slide.addText(f[2], { x: MARGIN + 0.75, y: fy + 0.42, w: 11.0, h: 0.5, fontFace: FONT_BODY, fontSize: 12.5, color: GRAY, isTextBox: true, margin: 0 });
    fy += 1.2;
  });

  pageNum(slide, 12);

  slide.addNotes(
    "WHAT TO SAY: Four findings I'd want you to remember. First, same-city prediction was far easier than " +
    "cross-city — mostly a scale problem, not a weather-learning problem. Second, a simple linear model beat " +
    "every more complex model on classification — a genuinely useful, somewhat counterintuitive result. " +
    "Third, semi-supervised learning didn't help even with just ten percent labels — the pseudo-labels were " +
    "accurate but redundant. And fourth, transfer learning clearly helped, but only after I found and fixed " +
    "a real negative-transfer bug during development.\n" +
    "IMPORTANT POINT: Notice these are genuinely mixed — two positive, two negative. That honesty is " +
    "intentional; I didn't force every technique to look good.\n" +
    "TRANSITION: Let's talk about what this project doesn't cover.\n" +
    "APPROX TIME: 0:45"
  );
}

// ===========================================================================
// SLIDE 13 — LIMITATIONS & FUTURE WORK
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Scope", "Limitations & Future Work");

  slide.addShape(pres.ShapeType.roundRect, { x: MARGIN, y: 1.6, w: 5.95, h: 5.0, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 } });
  slide.addText("Limitations", { x: MARGIN + 0.25, y: 1.85, w: 5.5, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 16, color: TERRACOTTA, isTextBox: true, margin: 0 });
  bulletList(slide, MARGIN + 0.25, 2.35, 5.45, 4.1, [
    "Mid-day-only data (10:00\u201315:00) — no nighttime behavior covered",
    "Only 5 cities, and only one transfer-learning source/target pair tested",
    "City-specific PV plant scale is a recurring complication, not a solved problem",
    "Amherst has only 3 years of data vs. 6 for the other cities",
    "Deliberately small hyperparameter searches (3\u20135 options) throughout",
    "No GPU during development — all models trained on CPU",
  ], { fontSize: 13.5, spaceAfter: 12 });

  slide.addShape(pres.ShapeType.roundRect, { x: 6.83, y: 1.6, w: 5.95, h: 5.0, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 } });
  slide.addText("Future Work  (not yet completed)", { x: 7.08, y: 1.85, w: 5.5, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 16, color: TEAL, isTextBox: true, margin: 0 });
  bulletList(slide, 7.08, 2.35, 5.45, 4.1, [
    "Additional cities for transfer learning (Huron, Santa Barbara, La Jolla — currently zero-shot only)",
    "A larger hyperparameter search now that working baselines exist",
    "Longer forecasting horizons and real weather-forecast inputs, not just historical readings",
    "More sophisticated temporal models for the sequence-forecasting sub-task",
    "Better domain adaptation for cross-city scale differences",
    "Uncertainty estimation alongside point predictions",
  ], { fontSize: 13.5, spaceAfter: 12 });

  pageNum(slide, 13);

  slide.addNotes(
    "WHAT TO SAY: Quick, honest limitations: every reading falls in a ten-to-three daytime window, so " +
    "nothing here describes nighttime behavior. Only five cities, and only one source-target pair tested for " +
    "transfer learning. Plant-scale differences kept showing up as something I had to work around, not fully " +
    "solve. And I kept hyperparameter searches small throughout, on purpose.\n" +
    "IMPORTANT POINT: The future-work items on the right are explicitly things I have NOT done — a larger " +
    "search, more cities, longer horizons, real forecast inputs, better domain adaptation, uncertainty " +
    "estimation. I want to be clear these are proposals, not completed work.\n" +
    "TRANSITION: Let's wrap up.\n" +
    "APPROX TIME: 0:45"
  );
}

// ===========================================================================
// SLIDE 14 — CONCLUSION
// ===========================================================================
{
  const slide = newSlide(NAVY);
  slide.addShape(pres.ShapeType.ellipse, { x: 10.3, y: -1.3, w: 4.6, h: 4.6, fill: { color: NAVY_LIGHT }, line: { type: "none" } });
  slide.addShape(pres.ShapeType.ellipse, { x: 10.9, y: -0.7, w: 3.4, h: 3.4, fill: { color: "3A5A82" }, line: { type: "none" } });
  slide.addShape(pres.ShapeType.ellipse, { x: 11.5, y: -0.1, w: 2.2, h: 2.2, fill: { color: GOLD }, line: { type: "none" } });

  slide.addText("CONCLUSION", { x: MARGIN, y: 0.55, w: 10, h: 0.35, fontFace: FONT_BODY, bold: true, fontSize: 13, color: GOLD, charSpacing: 1.5, isTextBox: true, margin: 0 });

  slide.addShape(pres.ShapeType.roundRect, {
    x: MARGIN, y: 1.15, w: 12.23, h: 1.35, rectRadius: 0.08,
    fill: { color: NAVY_LIGHT }, line: { type: "none" },
  });
  slide.addText(
    "\u201cMachine learning can predict PV output effectively within a city, but cross-city domain shift remains a major challenge — one that fine-tuning can address, but naive transfer cannot.\u201d",
    { x: MARGIN + 0.3, y: 1.32, w: 11.6, h: 1.0, fontFace: FONT_HEAD, italic: true, bold: true, fontSize: 16, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 }
  );

  const closers = [
    ["Strongest result", "Davis same-city regression — RMSE = 15.17 kW, R² = 0.953"],
    ["Most interesting finding", "The Problem 5 learning-rate story — one hyperparameter separated failure from a +21.4% win"],
    ["Biggest limitation", "Deliberately small hyperparameter searches throughout — never confirmed against a larger search"],
  ];
  let cy = 2.85;
  closers.forEach((c) => {
    slide.addText(c[0] + ":", { x: MARGIN, y: cy, w: 3.3, h: 0.5, fontFace: FONT_BODY, bold: true, fontSize: 14.5, color: GOLD, isTextBox: true, margin: 0 });
    slide.addText(c[1], { x: MARGIN + 3.4, y: cy, w: 8.8, h: 0.6, fontFace: FONT_BODY, fontSize: 14.5, color: "E7ECF2", isTextBox: true, margin: 0 });
    cy += 0.85;
  });

  slide.addShape(pres.ShapeType.line, { x: MARGIN, y: 5.7, w: 12.23, h: 0, line: { color: "3A5A82", width: 1 } });

  slide.addText(
    "Whether an advanced ML technique helps has to be tested directly, not assumed — and that's the real finding of this project.",
    { x: MARGIN, y: 5.95, w: 12.23, h: 0.9, fontFace: FONT_HEAD, italic: true, fontSize: 17, color: "FFFFFF", isTextBox: true, margin: 0 }
  );

  pageNum(slide, 14, true);

  slide.addNotes(
    "WHAT TO SAY: So — machine learning predicts PV output very effectively within a city, but cross-city " +
    "domain shift remains a major challenge — one that fine-tuning can address, but naive transfer cannot. " +
    "My strongest result was Davis same-city regression, RMSE of fifteen point one seven kilowatts, R-squared " +
    "of point nine five three. The most interesting finding was that single learning-rate story in Problem " +
    "five. And the biggest limitation is that I kept every hyperparameter search deliberately small, so I " +
    "can't say for certain there isn't more headroom out there.\n" +
    "IMPORTANT POINT: The real thread connecting all five problems — supervised learning and transfer " +
    "learning clearly helped, dimension reduction and semi-supervised learning didn't, and understanding WHY " +
    "in each case mattered as much as the numbers themselves.\n" +
    "TRANSITION: Thanks — happy to take questions.\n" +
    "APPROX TIME: 0:45"
  );
}

pres.writeFile({ fileName: "PV_ML_Final_Presentation.pptx" }).then(() => console.log("Wrote PV_ML_Final_Presentation.pptx"));
