// build_presentation.js — generates presentation/FINAL_PRESENTATION.pptx
// Every number below is taken directly from results/FINAL_EXPERIMENT_TABLE.csv
// and report/FINAL_REPORT.md — verified before writing, see PROFESSOR_QUESTIONS.md
// and the numerical-validation note printed at the end of this script.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (p) => path.join(ROOT, "figures", p);

// ---------------------------------------------------------------------------
// Palette — solar/energy theme: deep navy (dominant) + solar gold (accent) +
// teal (positive-finding accent) + muted terracotta (failure/negative accent)
// ---------------------------------------------------------------------------
const NAVY = "132A46";       // dominant — dark slide backgrounds, headers
const NAVY_LIGHT = "24466E"; // secondary navy for cards on dark bg
const GOLD = "F5A623";       // sun accent — used sparingly, titles/highlights
const OFFWHITE = "F7F8FA";   // light slide background
const CARD = "FFFFFF";       // card background on light slides
const CHARCOAL = "1A1A1A";   // body text on light bg
const GRAY = "5B6B7C";       // muted/secondary text
const TEAL = "2A9D8F";       // positive-finding accent
const TERRACOTTA = "C0524A"; // negative-finding accent
const LINE = "E1E6EC";       // subtle borders/dividers

const FONT_HEAD = "Cambria";
const FONT_BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33" x 7.5"
pres.author = "ECE571 Student";
pres.company = "ECE571 Machine Learning Course Project";
pres.title = "Photovoltaic Power Prediction on Large-Scale Spatiotemporal Data";

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

  // Simple sun motif: concentric circles, upper right, subtle
  slide.addShape(pres.ShapeType.ellipse, { x: 10.3, y: -1.3, w: 4.6, h: 4.6, fill: { color: NAVY_LIGHT }, line: { type: "none" } });
  slide.addShape(pres.ShapeType.ellipse, { x: 10.9, y: -0.7, w: 3.4, h: 3.4, fill: { color: "3A5A82" }, line: { type: "none" } });
  slide.addShape(pres.ShapeType.ellipse, { x: 11.5, y: -0.1, w: 2.2, h: 2.2, fill: { color: GOLD }, line: { type: "none" } });

  slide.addText("PHOTOVOLTAIC POWER PREDICTION", {
    x: MARGIN, y: 2.55, w: 11.5, h: 0.4,
    fontFace: FONT_BODY, fontSize: 15, bold: true, color: GOLD, charSpacing: 2,
    isTextBox: true, margin: 0,
  });
  slide.addText("Photovoltaic Power Prediction on\nLarge-Scale Spatiotemporal Data", {
    x: MARGIN, y: 2.95, w: 11.5, h: 1.8,
    fontFace: FONT_HEAD, fontSize: 40, bold: true, color: "FFFFFF",
    isTextBox: true, margin: 0, lineSpacingMultiple: 1.08,
  });
  slide.addText("Five Machine Learning Paradigms, One Dataset, One Honest Investigation", {
    x: MARGIN, y: 4.75, w: 11.5, h: 0.5,
    fontFace: FONT_BODY, fontSize: 17, italic: true, color: "CADCFC",
    isTextBox: true, margin: 0,
  });

  slide.addShape(pres.ShapeType.line, { x: MARGIN, y: 5.55, w: 3.2, h: 0, line: { color: GOLD, width: 2 } });

  slide.addText([
    { text: "ECE571 — Machine Learning Course Project", options: { breakLine: true, color: "D9E2EC", fontSize: 14 } },
    { text: "[Your Name]", options: { breakLine: true, color: "D9E2EC", fontSize: 14 } },
    { text: "Instructor: Prof. Puchovsky", options: { breakLine: true, color: "D9E2EC", fontSize: 14 } },
    { text: "[Presentation Date]", options: { color: "D9E2EC", fontSize: 14 } },
  ], { x: MARGIN, y: 5.85, w: 8, h: 1.4, fontFace: FONT_BODY, isTextBox: true, margin: 0, lineSpacing: 22 });

  slide.addNotes(
    "Hi everyone, I'm going to walk you through my ECE571 project: predicting solar power output " +
    "using five different machine learning approaches on the same dataset. Rather than treating this " +
    "as five separate assignments, I tried to tell one coherent story about what actually helps when " +
    "you're trying to forecast photovoltaic power — and, just as importantly, what doesn't. Let's start " +
    "with why this problem matters in the first place."
  );
}

// ===========================================================================
// SLIDE 2 — WHY PV FORECASTING MATTERS
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Motivation", "Why PV Forecasting Matters");

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
    "Solar power is different from a fossil-fuel plant because you can't just turn it up when you need " +
    "more — the sun decides. That intermittency creates real problems: grid operators need to know how " +
    "much power is coming so they can plan other generation, storage systems need forecasts to know when " +
    "to charge or discharge, and utilities planning a brand-new site often have little or no historical " +
    "data to work with. All three of those are fundamentally the same machine learning problem — relate " +
    "weather to power output — just with different amounts of data available. That's the thread connecting " +
    "all five parts of this project. Next, let's look at the actual dataset I used."
  );
}




// ===========================================================================
// SLIDE 3 — DATASET + EXPERIMENTAL CHALLENGE
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "The Data", "Dataset & Experimental Challenge");

  const cities = [
    ["Davis, CA", "2011–2016", "164.0 kW avg"],
    ["Amherst, MA", "2018–2020", "60.9 kW avg"],
    ["Huron, SD", "2011–2016", "50.1 kW avg"],
    ["Santa Barbara, CA", "2011–2016", "49.1 kW avg"],
    ["La Jolla, CA", "2011–2016", "47.3 kW avg"],
  ];
  const colW = 2.36, startX = MARGIN, y0 = 1.6;
  cities.forEach((c, idx) => {
    const x = startX + idx * (colW + 0.06);
    slide.addShape(pres.ShapeType.roundRect, {
      x, y: y0, w: colW, h: 1.55, rectRadius: 0.07,
      fill: { color: CARD }, line: { color: LINE, width: 1 },
    });
    slide.addText(c[0], { x: x + 0.1, y: y0 + 0.12, w: colW - 0.2, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 13, color: NAVY, isTextBox: true, margin: 0 });
    slide.addText(c[1], { x: x + 0.1, y: y0 + 0.56, w: colW - 0.2, h: 0.3, fontFace: FONT_BODY, fontSize: 11.5, color: GRAY, isTextBox: true, margin: 0 });
    slide.addText(c[2], { x: x + 0.1, y: y0 + 0.98, w: colW - 0.2, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 14, color: GOLD, isTextBox: true, margin: 0 });
  });

  statCallout(slide, MARGIN, 3.5, 2.9, 1.15, "22", "columns per row: irradiance, weather, Cloud Type, Output Power", TEAL);
  statCallout(slide, MARGIN + 3.05, 3.5, 2.9, 1.15, "30 min", "measurement interval, 10:00–15:00 daily window", TEAL);
  statCallout(slide, MARGIN + 6.1, 3.5, 2.9, 1.15, "~24,000", "rows for each 6-year city (Amherst: 12,056)", TEAL);
  statCallout(slide, MARGIN + 9.15, 3.5, 2.9, 1.15, "3.5×", "Davis's output scale vs. the smallest city", TERRACOTTA);

  slide.addShape(pres.ShapeType.roundRect, {
    x: MARGIN, y: 5.05, w: 12.23, h: 1.85, rectRadius: 0.08,
    fill: { color: NAVY }, line: { type: "none" },
  });
  slide.addText("The central challenge running through this project:", {
    x: MARGIN + 0.25, y: 5.25, w: 11.7, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 14.5, color: GOLD, isTextBox: true, margin: 0,
  });
  slide.addText(
    "Output Power has a different physical scale in every city, and the data is a time series — so " +
    "every train/test split in this project is chronological (earliest 80% train, latest 20% test), " +
    "never randomly shuffled. Shuffling first would let a model trained on next Tuesday predict last Monday.",
    { x: MARGIN + 0.25, y: 5.68, w: 11.7, h: 1.1, fontFace: FONT_BODY, fontSize: 14.5, color: "E7ECF2", isTextBox: true, margin: 0 }
  );

  pageNum(slide, 3);

  slide.addNotes(
    "This is the dataset: five cities, mostly six years of data at 30-minute intervals, 22 columns " +
    "covering irradiance, weather, cloud type, and Output Power. The big number on the right — " +
    "Davis's plant produces roughly three and a half times the average output of the " +
    "smallest city. That's not a data quality issue, just different plant sizes, but it " +
    "matters enormously later when I compare or transfer between cities. And because this is time-" +
    "series data, every experiment splits chronologically — train on the earliest eighty percent, test on " +
    "the most recent twenty — never a random shuffle, since that would let the model peek into " +
    "the future. Now, the overall pipeline I built to run all five ML problems consistently."
  );
}

// ===========================================================================
// SLIDE 4 — EXPERIMENTAL FRAMEWORK
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Methodology", "One Shared Experimental Framework");

  const steps = ["Raw Excel\nData", "Cleaning", "Feature\nEngineering", "Chronological\nSplit", "Scaling /\nEncoding", "Model\nTraining", "Evaluation"];
  const stepW = 1.55, gap = 0.22, stepY = 1.7, stepH = 1.0;
  const totalW = steps.length * stepW + (steps.length - 1) * gap;
  let sx = (SLIDE_W - totalW) / 2;
  steps.forEach((s, idx) => {
    slide.addShape(pres.ShapeType.roundRect, {
      x: sx, y: stepY, w: stepW, h: stepH, rectRadius: 0.06,
      fill: { color: idx === steps.length - 1 ? TEAL : NAVY }, line: { type: "none" },
    });
    slide.addText(s, {
      x: sx, y: stepY, w: stepW, h: stepH, fontFace: FONT_BODY, fontSize: 11.5, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", isTextBox: true, margin: 0,
    });
    if (idx < steps.length - 1) {
      slide.addText("→", { x: sx + stepW, y: stepY, w: gap + 0.02, h: stepH, fontFace: FONT_BODY, fontSize: 16, color: GRAY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    }
    sx += stepW + gap;
  });

  slide.addText(
    "Every scaler and encoder is fit on training data only, then applied unchanged to test data — never the reverse.",
    { x: MARGIN, y: 2.95, w: 12.2, h: 0.4, fontFace: FONT_BODY, italic: true, fontSize: 13.5, color: GRAY, align: "center", isTextBox: true, margin: 0 }
  );

  slide.addText("This one pipeline feeds five ML paradigms:", {
    x: MARGIN, y: 3.65, w: 12.2, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 15, color: NAVY, isTextBox: true, margin: 0,
  });

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
    const y = 4.15;
    slide.addShape(pres.ShapeType.roundRect, { x, y, w: pw, h: 1.9, rectRadius: 0.07, fill: { color: CARD }, line: { color: LINE, width: 1 } });
    slide.addShape(pres.ShapeType.ellipse, { x: x + 0.15, y: y + 0.15, w: 0.5, h: 0.5, fill: { color: GOLD }, line: { type: "none" } });
    slide.addText(p[0], { x: x + 0.15, y: y + 0.15, w: 0.5, h: 0.5, fontFace: FONT_HEAD, bold: true, fontSize: 18, color: NAVY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    slide.addText(p[1], { x: x + 0.15, y: y + 0.78, w: pw - 0.3, h: 0.55, fontFace: FONT_BODY, bold: true, fontSize: 13.5, color: NAVY, isTextBox: true, margin: 0 });
    slide.addText(p[2], { x: x + 0.15, y: y + 1.28, w: pw - 0.3, h: 0.55, fontFace: FONT_BODY, fontSize: 11, color: GRAY, isTextBox: true, margin: 0 });
  });

  pageNum(slide, 4);

  slide.addNotes(
    "Rather than build five separate one-off pipelines, I built one shared framework: load the raw Excel " +
    "data, clean it, engineer features like the clear-sky index and cyclical time encodings, split " +
    "chronologically, scale and encode — fitting only on training data — then train and evaluate. That " +
    "one pipeline feeds all five ML paradigms you see at the bottom: classification, regression — the " +
    "core task — dimension reduction, semi-supervised learning, and transfer learning. Building it this " +
    "way meant every paradigm was tested under the exact same leakage-prevention rules, so comparisons " +
    "between them are fair. Let's go through each one, starting with classification."
  );
}



// ===========================================================================
// SLIDE 5 — PROBLEM 1: CLASSIFICATION
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Problem 1", "Supervised Classification");

  slide.addText(
    "Predict sky-condition (Clear / Partly Cloudy / Overcast) from weather features — with GHI and " +
    "irradiance columns deliberately excluded, since they define the label itself.",
    { x: MARGIN, y: 1.55, w: 6.9, h: 0.85, fontFace: FONT_BODY, fontSize: 14.5, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  statCallout(slide, MARGIN, 2.55, 3.35, 1.15, "0.772", "Balanced Accuracy (Davis)", TEAL);
  statCallout(slide, MARGIN + 3.5, 2.55, 3.35, 1.15, "0.720", "Macro F1 (Davis)", TEAL);

  slide.addText("Best model: Logistic Regression (balanced weight)", {
    x: MARGIN, y: 3.9, w: 6.9, h: 0.35, fontFace: FONT_BODY, bold: true, fontSize: 13.5, color: NAVY, isTextBox: true, margin: 0,
  });
  bulletList(slide, MARGIN, 4.3, 6.9, 2.6, [
    "Notably not an ensemble or neural network — a well-regularized linear model, since the remaining features relate to sky-condition fairly directly",
    "Clear conditions: F1 = 0.946 (easiest, majority class)",
    "Partly Cloudy: F1 = 0.566 (hardest — sits between both other classes)",
  ], { fontSize: 14 });

  figureCard(slide, 8.05, 1.55, 4.75, 5.0, FIG("problem1/problem1_sky_condition_confusion_matrix_Davis.png"),
    "Confusion matrix, best Davis sky-condition model");

  pageNum(slide, 5);

  slide.addNotes(
    "The first problem is classification: predict whether the sky is Clear, Partly Cloudy, or Overcast, " +
    "using weather features. Important detail — I deliberately excluded GHI and related irradiance columns, " +
    "because the label itself is DEFINED by GHI divided by clear-sky GHI, so including them would just be " +
    "leaking the answer. The best model here, at 0.772 balanced accuracy, was actually plain logistic " +
    "regression with class weighting — not a random forest or neural network. Looking at the confusion " +
    "matrix on the right, Clear conditions are easy to identify, but Partly Cloudy is the hardest class, " +
    "because it sits on the boundary between the other two — it gets confused with both neighbors instead " +
    "of having one dominant failure mode. That's a pattern that shows up again later. Next, the core task " +
    "of the project: regression."
  );
}

// ===========================================================================
// SLIDE 6 — PROBLEM 2: REGRESSION
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Problem 2 — Core Task", "Supervised Regression");

  slide.addText(
    "Predict continuous Output Power (kW) from weather + irradiance features — same-city, cross-city, and as a short-term sequence forecast.",
    { x: MARGIN, y: 1.55, w: 6.9, h: 0.75, fontFace: FONT_BODY, fontSize: 14.5, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  statCallout(slide, MARGIN, 2.45, 2.2, 1.15, "15.17", "RMSE (kW)", TEAL);
  statCallout(slide, MARGIN + 2.3, 2.45, 2.2, 1.15, "8.31", "MAE (kW)", TEAL);
  statCallout(slide, MARGIN + 4.6, 2.45, 2.3, 1.15, "0.953", "R² (Davis)", TEAL);

  slide.addText("Best model: Gradient Boosting (Davis, same-city)", {
    x: MARGIN, y: 3.8, w: 6.9, h: 0.35, fontFace: FONT_BODY, bold: true, fontSize: 13.5, color: NAVY, isTextBox: true, margin: 0,
  });
  bulletList(slide, MARGIN, 4.2, 6.9, 2.7, [
    "Cross-city zero-shot (Davis → Huron/Santa Barbara/La Jolla) failed severely (RMSE ≈ 125 kW) — almost entirely a scale-mismatch problem, not a modeling failure",
    "K=12 sequence GRU (17.58 kW) clearly beat a persistence baseline (21.97 kW) — genuine temporal structure, though a harder task than same-timestep regression",
  ], { fontSize: 14 });

  figureCard(slide, 8.35, 1.55, 4.45, 5.0, FIG("problem2/problem2_predicted_vs_actual_davis.png"),
    "Predicted vs. actual Output Power, Davis");

  pageNum(slide, 6);

  slide.addNotes(
    "This is the project's core task: predict actual Output Power in kilowatts. Gradient boosting on " +
    "Davis's own data was the strongest result in the whole project — fifteen kilowatts of RMSE, an R-" +
    "squared of point nine five three, meaning it explains ninety-five percent of the variance using just " +
    "weather and irradiance. You can see that in the scatter plot on the right — points cluster tightly " +
    "around the perfect-prediction line. Two other things worth mentioning quickly: applying the Davis " +
    "model directly to other cities without any adaptation failed badly, RMSE around 125 kilowatts — but " +
    "that's mostly a scale problem I'll come back to in problem five. And a GRU sequence model, predicting " +
    "the next reading from the previous twelve, clearly beat a naive persistence baseline, so it's learning " +
    "real temporal structure. Now — did compressing the feature space help or hurt?"
  );
}

// ===========================================================================
// SLIDE 7 — PROBLEM 3: DIMENSION REDUCTION
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Problem 3", "Dimension Reduction");

  slide.addText(
    "Compress 23 features to d = 2, 5, or 10 dimensions with PCA (linear) and an autoencoder (nonlinear) — fit with zero label access — then test whether Problems 1 & 2 still work as well.",
    { x: MARGIN, y: 1.55, w: 12.2, h: 0.65, fontFace: FONT_BODY, fontSize: 14.5, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  figureCard(slide, MARGIN, 2.35, 5.85, 4.25, FIG("problem3/problem3_downstream_classification.png"),
    "Downstream classification accuracy: raw vs. every reduced representation");

  slide.addShape(pres.ShapeType.roundRect, {
    x: 6.65, y: 2.35, w: 6.13, h: 4.25, rectRadius: 0.08,
    fill: { color: NAVY }, line: { type: "none" },
  });
  slide.addText("The key question: did compression help or hurt?", {
    x: 6.9, y: 2.6, w: 5.6, h: 0.5, fontFace: FONT_BODY, bold: true, fontSize: 15, color: GOLD, isTextBox: true, margin: 0,
  });
  bulletList(slide, 6.9, 3.2, 5.6, 3.2, [
    "Raw features beat every PCA/autoencoder representation, at every tested dimension, on both classification and regression",
    "Raw: 0.743 balanced accuracy vs. best reduced (Autoencoder, d=10): 0.661",
    "Autoencoder consistently beat PCA — it can learn nonlinear compressions PCA cannot",
    "Traced to a cause: Cloud Type (highly predictive) compresses poorly into a few continuous dimensions",
  ], { fontSize: 13.5, color: "E7ECF2", spaceAfter: 14 });

  pageNum(slide, 7);

  slide.addNotes(
    "Problem three asks: can I compress this feature space without hurting the tasks that depend on it? " +
    "I tried PCA — the classical, linear method — and a small autoencoder — the deep, nonlinear method — " +
    "at three latent sizes, fit with zero access to any label. The chart on the left is the honest answer: " +
    "raw, uncompressed features beat every reduced version, at every dimension, on both tasks. That " +
    "surprised me going in. The autoencoder did consistently beat PCA, since it can learn curved " +
    "compressions instead of just straight lines — but neither closed the gap with using everything. I " +
    "traced part of this to Cloud Type, one of the most useful individual predictors, which doesn't " +
    "compress well into a handful of continuous numbers. Next: what if you barely have any labels at all?"
  );
}

// ===========================================================================
// SLIDE 8 — PROBLEM 4: SEMI-SUPERVISED LEARNING
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

  slide.addText("Did SSL beat supervised-only at 10% labels? NO.", {
    x: 6.65, y: 2.5, w: 6.13, h: 0.45, fontFace: FONT_BODY, bold: true, fontSize: 16, color: TERRACOTTA, isTextBox: true, margin: 0,
  });
  bulletList(slide, 6.65, 3.1, 6.13, 3.4, [
    "SSL gain was slightly negative at every fraction tested (10%, 30%, 50%) — an honest, reported finding",
    "Diagnosis: pseudo-labels were 100% accurate (checked after training only) — but 98% of the first round were the easy \u201cClear\u201d class",
    "Correct, but redundant — reinforced what the model already knew instead of helping with the hard Partly Cloudy / Overcast boundary",
  ], { fontSize: 13.5, spaceAfter: 14 });

  pageNum(slide, 8);

  slide.addNotes(
    "For semi-supervised learning, I hid most of Davis's sky-condition labels — down to as few as ten " +
    "percent — and tested whether pseudo-labeling, where the model labels the unlabeled data itself when " +
    "confident enough, could beat just training on the small labeled set alone. The honest answer, shown " +
    "in that curve, is no — the two lines are essentially on top of each other, slightly negative at every " +
    "label fraction, not just the smallest. I checked WHY, using the hidden true labels purely as a " +
    "diagnostic after training was already finished. The pseudo-labels were completely accurate, a hundred " +
    "percent — but ninety-eight percent of the first batch were the same, easy \"Clear\" class. So the " +
    "method wasn't adding wrong information, it was adding redundant information. Now, the last paradigm: " +
    "what if a whole different city could help?"
  );
}

// ===========================================================================
// SLIDE 9 — PROBLEM 5: TRANSFER LEARNING
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Problem 5", "Transfer Learning");

  // Small Davis -> Amherst diagram
  const dy = 1.65, dh = 0.85;
  slide.addShape(pres.ShapeType.roundRect, { x: MARGIN, y: dy, w: 1.9, h: dh, rectRadius: 0.06, fill: { color: NAVY }, line: { type: "none" } });
  slide.addText("Davis\n6 years", { x: MARGIN, y: dy, w: 1.9, h: dh, fontFace: FONT_BODY, bold: true, fontSize: 12.5, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addText("→", { x: MARGIN + 1.9, y: dy, w: 0.5, h: dh, fontFace: FONT_BODY, fontSize: 18, color: GRAY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addShape(pres.ShapeType.roundRect, { x: MARGIN + 2.4, y: dy, w: 1.9, h: dh, rectRadius: 0.06, fill: { color: NAVY_LIGHT }, line: { type: "none" } });
  slide.addText("Source\nModel", { x: MARGIN + 2.4, y: dy, w: 1.9, h: dh, fontFace: FONT_BODY, bold: true, fontSize: 12.5, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addText("→", { x: MARGIN + 4.3, y: dy, w: 0.5, h: dh, fontFace: FONT_BODY, fontSize: 18, color: GRAY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addShape(pres.ShapeType.roundRect, { x: MARGIN + 4.8, y: dy, w: 1.9, h: dh, rectRadius: 0.06, fill: { color: GOLD }, line: { type: "none" } });
  slide.addText("Fine-Tune", { x: MARGIN + 4.8, y: dy, w: 1.9, h: dh, fontFace: FONT_BODY, bold: true, fontSize: 12.5, color: NAVY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addText("→", { x: MARGIN + 6.7, y: dy, w: 0.5, h: dh, fontFace: FONT_BODY, fontSize: 18, color: GRAY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addShape(pres.ShapeType.roundRect, { x: MARGIN + 7.2, y: dy, w: 1.9, h: dh, rectRadius: 0.06, fill: { color: TEAL }, line: { type: "none" } });
  slide.addText("Amherst\n3 years", { x: MARGIN + 7.2, y: dy, w: 1.9, h: dh, fontFace: FONT_BODY, bold: true, fontSize: 12.5, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });

  statCallout(slide, MARGIN, 2.85, 2.95, 1.1, "+21.4%", "Transfer gain at k=10 samples", TEAL);
  statCallout(slide, MARGIN + 3.1, 2.85, 2.95, 1.1, "30.03", "Transfer RMSE (kW), k=10", TEAL);
  statCallout(slide, MARGIN + 6.2, 2.85, 2.95, 1.1, "184.17", "Zero-shot RMSE (kW) — no adaptation", TERRACOTTA);

  bulletList(slide, MARGIN, 4.15, 6.0, 2.9, [
    "Zero-shot (no Amherst training) fails badly — same scale mismatch as Problem 2",
    "Transfer (Davis-pretrained + fine-tuned) beats training from scratch on the same k samples, at every k — most when data is scarcest",
    "An early version, with too-small a fine-tuning learning rate, caused negative transfer (worse than from-scratch) — found and fixed",
  ], { fontSize: 13.5, spaceAfter: 10 });

  figureCard(slide, 6.65, 4.0, 6.15, 3.05, FIG("problem5/problem5_transfer_curve.png"), "RMSE vs. Amherst labeled sample count");

  pageNum(slide, 9);

  slide.addNotes(
    "The last paradigm: can Davis, with six years of data, help Amherst, which only has three? I tested " +
    "three approaches on Amherst's held-out test set: zero-shot, the Davis model applied directly with no " +
    "Amherst training at all; few-shot, a fresh model trained only on a handful of Amherst samples; and " +
    "transfer, the Davis-pretrained model fine-tuned on that same handful. Zero-shot failed badly, same " +
    "scale problem as before. But transfer clearly beat training from scratch at every sample count, and " +
    "the gain was biggest exactly where it matters most — just ten samples, a twenty-one percent " +
    "improvement in RMSE. I'll be upfront: my first version of this experiment actually made things worse, " +
    "because I used too small a learning rate during fine-tuning — I'll explain exactly what happened on " +
    "the next slide, since it's the most interesting finding in the whole project."
  );
}



// ===========================================================================
// SLIDE 10 — CROSS-PROBLEM RESULTS
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "The Whole Project, At A Glance", "Cross-Problem Results");

  const headerOpts = { fill: { color: NAVY }, color: "FFFFFF", bold: true, fontSize: 13, fontFace: FONT_BODY, align: "left", valign: "middle" };
  const cellOpts = (shade) => ({ fill: { color: shade ? "F0F3F7" : "FFFFFF" }, color: CHARCOAL, fontSize: 12.5, fontFace: FONT_BODY, valign: "middle" });

  const rows = [
    [
      { text: "Problem", options: headerOpts },
      { text: "Best Method", options: headerOpts },
      { text: "Best Result", options: headerOpts },
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
    "Two paradigms clearly helped (supervised learning, transfer learning); two produced genuinely negative results (dimension reduction, semi-supervised learning) — all reported honestly.",
    { x: MARGIN, y: 6.5, w: 12.23, h: 0.55, fontFace: FONT_BODY, italic: true, fontSize: 13.5, color: GRAY, isTextBox: true, margin: 0 }
  );

  pageNum(slide, 10);

  slide.addNotes(
    "This table is the whole project in one place. Two paradigms clearly helped — plain supervised " +
    "learning was strong throughout, and transfer learning gave a real, meaningful boost when Amherst " +
    "data was scarce. Two paradigms did NOT help, and I want to be upfront about that rather than bury " +
    "it — dimension reduction hurt performance at every size I tried, and semi-supervised learning's gain " +
    "was essentially zero. I think that mix is actually the most honest and most useful outcome a project " +
    "like this can produce — it tells you which techniques are worth reaching for on this kind of problem, " +
    "and which aren't, based on real evidence instead of assuming every advanced technique automatically " +
    "helps. Let's dig into the single most interesting result — one that almost didn't turn out this way."
  );
}

// ===========================================================================
// SLIDE 11 — MOST INTERESTING FINDING
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Most Interesting Finding", "One Learning Rate, Two Opposite Outcomes");

  slide.addText(
    "The exact same transfer setup — same cities, same 10 Amherst samples — produced either a severe failure or a clear win, depending only on the fine-tuning learning rate.",
    { x: MARGIN, y: 1.55, w: 12.2, h: 0.65, fontFace: FONT_BODY, fontSize: 15, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  slide.addChart(pres.ChartType.bar, [
    {
      name: "RMSE at k=10 (kW)",
      labels: ["Too-small LR\n(1e-4)", "Matched LR\n(1e-3)", "Few-shot\nbaseline"],
      values: [140.9, 27.2, 38.2],
    },
  ], {
    x: MARGIN, y: 2.4, w: 6.0, h: 3.9,
    chartColors: [TERRACOTTA, TEAL, GRAY],
    showTitle: true, title: "Fine-tuning RMSE, k=10, seed=42 (diagnostic)", titleFontSize: 13, titleColor: NAVY, titleFontFace: FONT_BODY,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: CHARCOAL, dataLabelFontSize: 12,
    catAxisLabelColor: CHARCOAL, catAxisLabelFontSize: 11.5, catAxisLabelFontFace: FONT_BODY,
    valAxisLabelColor: GRAY, valAxisLabelFontSize: 10,
    valGridLine: { color: LINE, size: 1 }, catGridLine: { style: "none" },
    showLegend: false, barGapWidthPct: 40,
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 6.75, y: 2.4, w: 6.03, h: 3.9, rectRadius: 0.08,
    fill: { color: NAVY }, line: { type: "none" },
  });
  bulletList(slide, 7.0, 2.65, 5.55, 3.5, [
    "Too small a fine-tuning learning rate (following the assignment's own suggested strategy) couldn't shift the model's output from Davis's ~164 kW scale to Amherst's ~64 kW scale in time",
    "Matching the pretraining learning rate instead let it adapt — RMSE dropped from 140.9 to 27.2 kW",
    "This is why the final, reported result (30.03 kW, mean of 3 seeds) clearly beats few-shot, not loses to it",
  ], { fontSize: 13.5, color: "E7ECF2", spaceAfter: 14 });

  pageNum(slide, 11);

  slide.addNotes(
    "Here's the single most interesting thing I found in this project. My very first version of transfer " +
    "learning used a smaller learning rate for fine-tuning than for pretraining — actually what the " +
    "assignment itself suggested as a reasonable strategy. With that setup, transfer learning was WORSE " +
    "than training from scratch on the same ten samples — 140.9 kilowatts of RMSE versus 38.2. That's " +
    "negative transfer. Instead of accepting that number, I checked what the model was actually " +
    "predicting, and found its average output was stuck near Davis's scale — about 164 kilowatts — because " +
    "the learning rate was too small to shift it down to Amherst's roughly 64 kilowatt scale in time. " +
    "Matching the fine-tuning learning rate to the pretraining rate instead dropped RMSE to 27.2, and " +
    "transfer clearly won. That one hyperparameter was the entire difference between failure and success. " +
    "Now, a real failure I want to walk through the same way."
  );
}

// ===========================================================================
// SLIDE 12 — FAILURE ANALYSIS
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Being Honest About What Didn't Work", "Failure Analysis");

  slide.addShape(pres.ShapeType.roundRect, {
    x: MARGIN, y: 1.6, w: 12.23, h: 1.15, rectRadius: 0.08,
    fill: { color: "FBEEED" }, line: { color: TERRACOTTA, width: 1 },
  });
  slide.addText("Main failure: Dimension reduction hurt every task, at every dimension tested", {
    x: MARGIN + 0.25, y: 1.78, w: 11.7, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 15, color: TERRACOTTA, isTextBox: true, margin: 0,
  });
  slide.addText(
    "Even at d=10 (using most of the original information, 94.6% explained variance for PCA), compressed features still lost real predictive power on both classification and regression.",
    { x: MARGIN + 0.25, y: 2.2, w: 11.7, h: 0.5, fontFace: FONT_BODY, fontSize: 13, color: CHARCOAL, isTextBox: true, margin: 0 }
  );

  slide.addText("Why?", { x: MARGIN, y: 3.0, w: 3, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 15, color: NAVY, isTextBox: true, margin: 0 });
  bulletList(slide, MARGIN, 3.45, 5.85, 2.7, [
    "Cloud Type is categorical and highly predictive, but doesn't compress well into a few continuous dimensions",
    "Verified directly: removing Cloud Type raised explained variance but lowered both reconstruction quality and downstream accuracy",
  ], { fontSize: 14, spaceAfter: 14 });

  slide.addText("What did I learn from it?", { x: 6.75, y: 3.0, w: 5.9, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 15, color: NAVY, isTextBox: true, margin: 0 });
  bulletList(slide, 6.75, 3.45, 5.85, 2.7, [
    "\u201cUnsupervised compression should be harmless\u201d is an assumption, not a guarantee — it has to be tested for each dataset",
    "A secondary failure (Problem 4): SSL's pseudo-labels were 100% accurate but redundant — a different, subtler kind of failure than being simply wrong",
  ], { fontSize: 14, spaceAfter: 14 });

  pageNum(slide, 12);

  slide.addNotes(
    "I want to spend a slide specifically on what didn't work, because that's just as important as the " +
    "successes. The clearest failure is dimension reduction — compressing the feature space hurt " +
    "performance on both classification and regression, at every size I tried, even when keeping ninety-" +
    "five percent of the variance. I traced it to Cloud Type specifically: categorical, and one of the " +
    "most useful individual features I have, but it doesn't survive being squeezed into a handful of " +
    "continuous numbers. I proved this by removing it from the compression step and watching both " +
    "reconstruction quality and downstream accuracy get worse, even though explained variance technically " +
    "went UP. The bigger lesson: \"compression is basically free\" is an assumption you have to test, not " +
    "assume. Problem four's SSL result is a different flavor of failure — redundant, not wrong — worth " +
    "distinguishing. Let's talk about the project's limitations more broadly."
  );
}

// ===========================================================================
// SLIDE 13 — LIMITATIONS + FUTURE WORK
// ===========================================================================
{
  const slide = newSlide();
  contentTitle(slide, "Scope", "Limitations & Future Work");

  slide.addShape(pres.ShapeType.roundRect, { x: MARGIN, y: 1.6, w: 5.95, h: 5.0, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 } });
  slide.addText("Limitations", { x: MARGIN + 0.25, y: 1.85, w: 5.5, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 16, color: TERRACOTTA, isTextBox: true, margin: 0 });
  bulletList(slide, MARGIN + 0.25, 2.35, 5.45, 4.1, [
    "5 cities, mostly 6 years — enough to demonstrate methods, not to generalize broadly",
    "Different PV plant scales across cities are a recurring complication, not a solved problem",
    "Deliberately small hyperparameter searches (3-5 options) throughout",
    "No GPU in development — all models trained on CPU (will use CUDA automatically)",
    "No real-time weather forecast input, no live deployment testing",
  ], { fontSize: 13.5, spaceAfter: 13 });

  slide.addShape(pres.ShapeType.roundRect, { x: 6.83, y: 1.6, w: 5.95, h: 5.0, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 } });
  slide.addText("Future Work", { x: 7.08, y: 1.85, w: 5.5, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 16, color: TEAL, isTextBox: true, margin: 0 });
  bulletList(slide, 7.08, 2.35, 5.45, 4.1, [
    "A larger hyperparameter search, now that a working baseline exists for each paradigm",
    "Investigate the Wind Speed domain-shift anomaly flagged in Problem 5",
    "Extend transfer learning to Huron, Santa Barbara, and La Jolla (currently zero-shot only)",
    "Longer forecasting horizons and real weather-forecast inputs, not just historical readings",
    "Uncertainty estimation alongside point predictions",
  ], { fontSize: 13.5, spaceAfter: 13 });

  pageNum(slide, 13);

  slide.addNotes(
    "Quickly, the honest limitations: five cities and mostly six years of data is enough to demonstrate " +
    "these methods, but not enough to claim they generalize broadly. The different plant scales across " +
    "cities kept showing up as a complication I had to work around, not something I fully solved. I kept " +
    "hyperparameter searches deliberately small throughout, and there was no GPU available during " +
    "development, though the code will use one automatically if run on GPU hardware. For future work, the " +
    "most natural next steps are a larger hyperparameter search now that I have working baselines, digging " +
    "into that Wind Speed anomaly I flagged rather than just noting it, and extending the transfer-learning " +
    "approach to the other data-scarce cities that were only tested zero-shot in problem two. Let's wrap up " +
    "with the big-picture takeaways."
  );
}

// ===========================================================================
// SLIDE 14 — CONCLUSION
// ===========================================================================
{
  const slide = newSlide(NAVY);
  slide.addText("CONCLUSION", { x: MARGIN, y: 0.55, w: 10, h: 0.35, fontFace: FONT_BODY, bold: true, fontSize: 13, color: GOLD, charSpacing: 1.5, isTextBox: true, margin: 0 });
  slide.addText("Five Takeaways", { x: MARGIN, y: 0.9, w: 11, h: 0.7, fontFace: FONT_HEAD, bold: true, fontSize: 30, color: "FFFFFF", isTextBox: true, margin: 0 });

  const takeaways = [
    ["1", "PV output can be predicted very effectively from weather and irradiance alone", "R² = 0.953 for same-city Davis regression"],
    ["2", "Different ML paradigms provide genuinely different, non-interchangeable benefits", "Supervised learning, transfer learning, dimension reduction, and SSL all behaved differently"],
    ["3", "Cross-city transfer is possible, but output-scale mismatch has to be handled explicitly", "Raw zero-shot transfer failed everywhere it was tried"],
    ["4", "Transfer learning helped substantially; semi-supervised learning did not, here", "+21.4% RMSE at k=10 vs. an SSL gain of essentially zero"],
    ["5", "Understanding WHY a technique fails is as valuable as the number itself", "Every negative result in this project was traced to a specific, explainable cause"],
  ];
  let ty = 1.85;
  takeaways.forEach((t) => {
    slide.addShape(pres.ShapeType.ellipse, { x: MARGIN, y: ty, w: 0.5, h: 0.5, fill: { color: GOLD }, line: { type: "none" } });
    slide.addText(t[0], { x: MARGIN, y: ty, w: 0.5, h: 0.5, fontFace: FONT_HEAD, bold: true, fontSize: 16, color: NAVY, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    slide.addText(t[1], { x: MARGIN + 0.7, y: ty - 0.03, w: 11.3, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 15, color: "FFFFFF", isTextBox: true, margin: 0 });
    slide.addText(t[2], { x: MARGIN + 0.7, y: ty + 0.37, w: 11.3, h: 0.35, fontFace: FONT_BODY, fontSize: 12.5, italic: true, color: "AEC0D4", isTextBox: true, margin: 0 });
    ty += 1.0;
  });

  pageNum(slide, 14, true);

  slide.addNotes(
    "So, five takeaways. First, PV output can be predicted very effectively from weather and irradiance " +
    "alone — an R-squared of point nine five three is strong by any standard. Second, the five " +
    "ML paradigms genuinely behaved differently — there's no single technique that's " +
    "universally best. Third, cross-city transfer is possible, but only once you handle the output-" +
    "scale mismatch explicitly — raw zero-shot transfer failed everywhere I tried it. Fourth, transfer " +
    "learning helped substantially here, while semi-supervised learning essentially didn't — two techniques " +
    "that sound similar in spirit but produced opposite results. And fifth, maybe the most " +
    "important lesson: understanding WHY something failed — a learning rate, a categorical " +
    "feature, a redundant pseudo-label — was at least as valuable as the headline numbers " +
    "themselves. That's the project. Happy to take questions."
  );
}

// ===========================================================================
// SLIDE 15 — QUESTIONS
// ===========================================================================
{
  const slide = newSlide(NAVY);
  slide.addShape(pres.ShapeType.ellipse, { x: 10.3, y: -1.3, w: 4.6, h: 4.6, fill: { color: NAVY_LIGHT }, line: { type: "none" } });
  slide.addShape(pres.ShapeType.ellipse, { x: 10.9, y: -0.7, w: 3.4, h: 3.4, fill: { color: "3A5A82" }, line: { type: "none" } });
  slide.addShape(pres.ShapeType.ellipse, { x: 11.5, y: -0.1, w: 2.2, h: 2.2, fill: { color: GOLD }, line: { type: "none" } });

  slide.addText("Questions?", { x: MARGIN, y: 3.1, w: 9, h: 1.1, fontFace: FONT_HEAD, bold: true, fontSize: 48, color: "FFFFFF", isTextBox: true, margin: 0 });
  slide.addText("Photovoltaic Power Prediction on Large-Scale Spatiotemporal Data", {
    x: MARGIN, y: 4.15, w: 9, h: 0.5, fontFace: FONT_BODY, italic: true, fontSize: 15, color: "CADCFC", isTextBox: true, margin: 0,
  });

  slide.addNotes(
    "Thanks — I'm glad to answer questions about any of the five problems, the dataset, the leakage-" +
    "prevention setup, or the reproducibility checks I ran."
  );
}

pres.writeFile({ fileName: "FINAL_PRESENTATION.pptx" }).then(() => console.log("Wrote FINAL_PRESENTATION.pptx"));
