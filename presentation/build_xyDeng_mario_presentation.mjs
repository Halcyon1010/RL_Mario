import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const repoRoot = path.resolve(process.argv[2] || process.cwd());
const outPath = path.resolve(
  process.argv[3] ||
    path.join(repoRoot, "presentation", "xyDeng_Mario_RL_ICM_Disagreement_1_1.pptx"),
);
const previewDir = path.resolve(
  process.argv[4] || path.join(repoRoot, "presentation", "preview_xyDeng"),
);

const C = {
  bg: "#F4F7FA",
  ink: "#202938",
  muted: "#607084",
  line: "#D9E1EA",
  panel: "#FFFFFF",
  blue: "#005CA8",
  purple: "#6243B6",
  green: "#059669",
};

const font = "Aptos";
const mono = "Arial";

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function imageBytes(relPath) {
  return fs.readFile(path.join(repoRoot, relPath));
}

function addBox(slide, position, fill = C.panel, line = C.line, radius = 28) {
  return slide.shapes.add({
    geometry: "roundRect",
    position,
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
    shadow: "shadow-sm",
  });
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: style.typeface || font,
    fontSize: style.fontSize ?? 18,
    color: style.color || C.ink,
    bold: style.bold ?? false,
    alignment: style.alignment,
  };
  return shape;
}

function header(slide, tag, title, subtitle, color) {
  slide.background.fill = C.bg;
  const pill = slide.shapes.add({
    geometry: "roundRect",
    position: { left: 40, top: 27, width: 112, height: 31 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
    borderRadius: 5,
    shadow: "shadow-sm",
  });
  pill.text = tag;
  pill.text.style = { typeface: font, fontSize: 14, color: "#FFFFFF", bold: true, alignment: "center" };

  addText(slide, title, { left: 170, top: 20, width: 780, height: 38 }, {
    fontSize: 31,
    bold: true,
    color: C.ink,
  });
  addText(slide, subtitle, { left: 170, top: 64, width: 760, height: 22 }, {
    fontSize: 14,
    color: C.muted,
  });

  const note = addBox(slide, { left: 1012, top: 22, width: 224, height: 53 }, "#EFF5FB", "#D7E2EF", 10);
  note.text = "Mario RL / PPO intrinsic rewards\n1M steps | seed 1 | World 1-1";
  note.text.style = { typeface: font, fontSize: 10, color: C.muted, alignment: "center" };

  slide.shapes.add({
    geometry: "rect",
    position: { left: 40, top: 92, width: 1198, height: 1.2 },
    fill: "#DDE6EF",
    line: { style: "solid", fill: "#DDE6EF", width: 0 },
  });
}

function label(slide, text, x, y, color, width = 150) {
  const shape = slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width, height: 30 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
    borderRadius: 5,
    shadow: "shadow-sm",
  });
  shape.text = text;
  shape.text.style = { typeface: font, fontSize: 14, bold: true, color: "#FFFFFF", alignment: "center" };
  return shape;
}

function footer(slide, text) {
  addText(slide, text, { left: 230, top: 690, width: 860, height: 16 }, {
    fontSize: 9,
    color: C.muted,
    alignment: "center",
  });
}

function table(slide, values, x, y, w, h) {
  const t = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: x,
    top: y,
    width: w,
    height: h,
    values,
  });
  t.styleOptions = { headerRow: true, bandedRows: false };
  t.borders.assign({ style: "solid", fill: "#111827", width: 0.8 });
  for (let r = 0; r < values.length; r += 1) {
    for (let c = 0; c < values[0].length; c += 1) {
      const cell = t.getCell(r, c);
      cell.text.style = {
        typeface: r === 0 || c === 0 ? font : mono,
        fontSize: r === 0 ? 10 : 9,
        bold: r === 0,
        color: C.ink,
        alignment: "center",
      };
      if (r === 0) cell.fill = "#E8EEF5";
    }
  }
  return t;
}

function metric(slide, labelText, value, sub, x, y, w, color) {
  addBox(slide, { left: x, top: y, width: w, height: 66 }, "#FFFFFF", C.line, 8);
  addText(slide, labelText, { left: x + 12, top: y + 8, width: w - 20, height: 14 }, {
    fontSize: 10,
    color: C.muted,
  });
  addText(slide, value, { left: x + 12, top: y + 27, width: w - 20, height: 26 }, {
    fontSize: w < 100 ? 18 : 21,
    bold: true,
    color,
    typeface: mono,
  });
  if (sub) {
    addText(slide, sub, { left: x + 12, top: y + 52, width: w - 20, height: 12 }, {
      fontSize: 8,
      color: C.muted,
    });
  }
}

function progress(slide, x, y, w, color) {
  addBox(slide, { left: x, top: y, width: w, height: 67 }, "#F8FBFE", C.line, 8);
  addText(slide, "Clear / progress", { left: x + 16, top: y + 9, width: 120, height: 17 }, {
    fontSize: 11,
    bold: true,
    color: C.muted,
  });
  addText(slide, "Reached x=3161", { left: x + 124, top: y + 8, width: w - 140, height: 22 }, {
    fontSize: 16,
    bold: true,
    color,
    alignment: "center",
  });
  addText(slide, "max_x = 3161 / 3161", { left: x + 120, top: y + 32, width: w - 160, height: 12 }, {
    fontSize: 8,
    color: C.muted,
    alignment: "center",
  });
  slide.shapes.add({
    geometry: "roundRect",
    position: { left: x + 14, top: y + 45, width: w - 28, height: 9 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
    borderRadius: 4,
  });
}

function flowNode(slide, text, x, y, w, h, fill = "#E7F1FB") {
  const n = addBox(slide, { left: x, top: y, width: w, height: h }, fill, "#CAD8E8", 8);
  n.text = text;
  n.text.style = { typeface: font, fontSize: 13, bold: true, color: C.ink, alignment: "center" };
  return n;
}

function line(slide, x1, y1, x2, y2) {
  slide.shapes.add({
    geometry: "line",
    position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 },
    fill: "none",
    line: { style: "solid", fill: "#7D8FA5", width: 1.4 },
  });
}

function methodSlide(presentation, method) {
  const slide = presentation.slides.add();
  header(slide, method.tag, method.title, "Method, setup, and intrinsic reward interface", method.color);

  addBox(slide, { left: 50, top: 113, width: 389, height: 211 });
  label(slide, "1. Role", 69, 129, method.color, 120);
  addText(slide, method.role, { left: 82, top: 174, width: 320, height: 88 }, {
    fontSize: 13,
    alignment: "center",
  });
  addText(slide, "Shared backbone: PPO + CNN Actor-Critic. Only the intrinsic reward module changes.", {
    left: 78,
    top: 282,
    width: 325,
    height: 28,
  }, { fontSize: 11, color: C.muted, alignment: "center" });

  addBox(slide, { left: 463, top: 113, width: 320, height: 211 });
  label(slide, "2. Key settings", 482, 129, method.color, 130);
  table(slide, method.params, 482, 166, 281, 136);

  addBox(slide, { left: 809, top: 113, width: 424, height: 211 });
  label(slide, "3. Protocol", 828, 129, method.color, 120);
  table(slide, [
    ["Item", "Value"],
    ["Train env", "SuperMarioBros-1-1-v0"],
    ["Input", "4-frame stack, 84x84"],
    ["Actions", "SIMPLE_MOVEMENT, 7 actions"],
    ["Budget", "1,000,000 timesteps"],
    ["Eval", "sample / greedy; best by sample validation"],
    ["Seed", "1"],
  ], 828, 166, 382, 136);

  addBox(slide, { left: 50, top: 344, width: 523, height: 255 });
  addText(slide, "Training flow", { left: 70, top: 360, width: 160, height: 22 }, {
    fontSize: 14,
    bold: true,
    color: method.color,
  });
  flowNode(slide, "Observation\n4x84x84", 69, 400, 130, 53);
  flowNode(slide, "CNN\nActor-Critic", 228, 400, 137, 53);
  flowNode(slide, "Policy / Value\nPPO update", 396, 400, 145, 53);
  flowNode(slide, `${method.short}\nintrinsic module`, 228, 493, 137, 53, "#EFFCF7");
  flowNode(slide, "r_total = r_ext\n+ beta*r_int", 396, 493, 145, 53, "#FFF7EB");
  line(slide, 199, 426, 228, 426);
  line(slide, 365, 426, 396, 426);
  line(slide, 466, 453, 466, 493);
  line(slide, 199, 474, 228, 517);
  line(slide, 365, 519, 396, 519);

  addBox(slide, { left: 597, top: 344, width: 636, height: 255 });
  label(slide, "4. Signals", 616, 360, method.color, 120);
  return slide;
}

async function methodWithPlots(presentation, method) {
  const slide = methodSlide(presentation, method);
  slide.images.add({
    blob: await imageBytes("xyDeng/plots/1m_comparison/val_max_x_pos.png"),
    contentType: "image/png",
    alt: `${method.short} validation max_x plot`,
    fit: "contain",
    position: { left: 624, top: 407, width: 350, height: 118 },
  });
  slide.images.add({
    blob: await imageBytes("xyDeng/plots/1m_comparison/train_int_reward.png"),
    contentType: "image/png",
    alt: `${method.short} intrinsic reward plot`,
    fit: "contain",
    position: { left: 993, top: 408, width: 210, height: 118 },
  });
  metric(slide, "validation", method.valMean, `best step ${method.bestStep}`, 616, 545, 156, method.color);
  metric(slide, "test", method.testMean, "sample mean", 782, 545, 156, method.color);
  metric(slide, "train terminal", "x=3161", method.completionShort, 948, 545, 166, method.color);
  metric(slide, "device", method.deviceLabel, method.speed, 1124, 545, 90, method.color);
  footer(slide, "Evidence: selected 1M World 1-1 runs, best_summary.json, one_million_run_metrics.csv, and training_completion_audit.md.");
}

async function outcomeSlide(presentation, method) {
  const slide = presentation.slides.add();
  header(slide, method.tag, `${method.title}: World 1-1 outcomes`, "Progress, reward, and video evidence", method.color);

  addBox(slide, { left: 50, top: 108, width: 1181, height: 57 });
  addText(slide, "Scope", { left: 76, top: 126, width: 72, height: 18 }, {
    fontSize: 13,
    bold: true,
    color: method.color,
  });
  addText(slide, "Train, validation, and test are all on SuperMarioBros-1-1-v0. Best checkpoints use sample-mode validation.", {
    left: 152,
    top: 124,
    width: 890,
    height: 22,
  }, { fontSize: 12, color: C.ink });

  addBox(slide, { left: 50, top: 184, width: 376, height: 214 });
  label(slide, "1-1 training evidence", 69, 198, method.color, 182);
  addText(slide, method.evidence, { left: 79, top: 250, width: 310, height: 58 }, {
    fontSize: 16,
    bold: true,
    color: method.color,
    alignment: "center",
  });
  addText(slide, method.evidenceNote, { left: 79, top: 325, width: 315, height: 42 }, {
    fontSize: 12,
    color: C.muted,
    alignment: "center",
  });

  metric(slide, "episodes", method.episodes, "training", 445, 184, 84, method.color);
  metric(slide, "reward", method.bestReward, "best train", 536, 184, 90, method.color);
  metric(slide, "max_x", "3161", "train max", 633, 184, 90, method.color);
  metric(slide, "reward/prog.", "Diff.", "video < train", 730, 184, 114, method.color);
  progress(slide, 445, 254, 399, method.color);
  table(slide, method.resultTable, 444, 331, 399, 58);

  addBox(slide, { left: 50, top: 430, width: 376, height: 214 });
  label(slide, "best saved video", 69, 443, method.color, 168);
  addText(slide, method.videoMetric, { left: 80, top: 491, width: 310, height: 42 }, {
    fontSize: 27,
    bold: true,
    color: method.color,
    alignment: "center",
    typeface: mono,
  });
  addText(slide, method.videoFile, { left: 75, top: 542, width: 318, height: 44 }, {
    fontSize: 10,
    color: C.ink,
    alignment: "center",
    typeface: mono,
  });
  addText(slide, "Saved videos show rollouts, not confirmed clears. Training logs provide terminal x evidence.", {
    left: 79,
    top: 596,
    width: 310,
    height: 28,
  }, { fontSize: 11, color: C.muted, alignment: "center" });

  addBox(slide, { left: 865, top: 184, width: 365, height: 460 });
  label(slide, "Reward-progress consistency", 884, 198, method.color, 250);
  slide.images.add({
    blob: await imageBytes("xyDeng/plots/1m_comparison/train_max_x_pos.png"),
    contentType: "image/png",
    alt: `${method.short} train max_x plot`,
    fit: "contain",
    position: { left: 893, top: 244, width: 303, height: 170 },
  });
  slide.images.add({
    blob: await imageBytes("xyDeng/plots/1m_comparison/train_total_reward.png"),
    contentType: "image/png",
    alt: `${method.short} train total reward plot`,
    fit: "contain",
    position: { left: 893, top: 449, width: 303, height: 170 },
  });

  addBox(slide, { left: 49, top: 654, width: 1182, height: 28 }, "#EAF2FB", "#D7E2EF", 6);
  addText(slide, method.bottomLine, { left: 120, top: 660, width: 1030, height: 16 }, {
    fontSize: 11,
    color: C.ink,
    alignment: "center",
  });
  footer(slide, "Evidence: final_selected_runs_1_1 and xyDeng/videos. Completion claims are conservative.");
}

async function compareSlide(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Compare", "ICM vs Disagreement: World 1-1", "Same PPO backbone, same 1M-step budget, same environment", C.green);

  addBox(slide, { left: 50, top: 113, width: 384, height: 520 });
  label(slide, "Key metrics", 69, 129, C.green, 110);
  table(slide, [
    ["Metric", "PPO+ICM", "PPO+Dis."],
    ["Device", "CUDA", "CPU"],
    ["Episodes", "7480", "7994"],
    ["Best step", "900k", "950k"],
    ["Val mean max_x", "1237.2", "1140.8"],
    ["Test mean max_x", "785.0", "784.6"],
    ["Train max_x", "3161", "3161"],
    ["Best video max_x", "1435", "2009"],
  ], 72, 174, 338, 286);
  addText(slide, "ICM has slightly higher mean validation progress. Disagreement has the farther saved rollout. Both reach terminal x=3161 during training.", {
    left: 76,
    top: 488,
    width: 316,
    height: 88,
  }, { fontSize: 15, color: C.ink, alignment: "center" });

  addBox(slide, { left: 463, top: 113, width: 770, height: 250 });
  label(slide, "Curves", 482, 129, C.green, 100);
  slide.images.add({
    blob: await imageBytes("xyDeng/plots/1m_comparison/val_max_x_pos.png"),
    contentType: "image/png",
    alt: "Validation max_x comparison",
    fit: "contain",
    position: { left: 486, top: 169, width: 350, height: 166 },
  });
  slide.images.add({
    blob: await imageBytes("xyDeng/plots/1m_comparison/test_max_x_pos.png"),
    contentType: "image/png",
    alt: "Test max_x comparison",
    fit: "contain",
    position: { left: 854, top: 169, width: 350, height: 166 },
  });

  addBox(slide, { left: 463, top: 383, width: 370, height: 250 });
  label(slide, "Speed", 482, 399, C.green, 100);
  table(slide, [
    ["Method", "CPU sec/1k", "CUDA sec/1k", "Used"],
    ["ICM", "11.88", "8.54", "CUDA"],
    ["Disagreement", "12.01", "13.43", "CPU"],
  ], 486, 443, 322, 92);
  addText(slide, "Device choice follows the 10k-step benchmark: CUDA for ICM, CPU for Disagreement.", {
    left: 493,
    top: 555,
    width: 300,
    height: 40,
  }, { fontSize: 12, color: C.muted, alignment: "center" });

  addBox(slide, { left: 862, top: 383, width: 371, height: 250 });
  label(slide, "Fair read", 884, 399, C.green, 112);
  addText(slide, "ICM", { left: 900, top: 448, width: 120, height: 22 }, {
    fontSize: 20,
    bold: true,
    color: C.blue,
  });
  addText(slide, "Forward-model error gives dense curiosity. This run shows stronger mean validation progress.", {
    left: 900,
    top: 476,
    width: 290,
    height: 48,
  }, { fontSize: 12, color: C.ink });
  addText(slide, "Disagreement", { left: 900, top: 536, width: 170, height: 22 }, {
    fontSize: 20,
    bold: true,
    color: C.purple,
  });
  addText(slide, "Ensemble uncertainty gives the best saved rollout, reaching max_x=2009.", {
    left: 900,
    top: 564,
    width: 290,
    height: 48,
  }, { fontSize: 12, color: C.ink });
  footer(slide, "Evidence: run_summary.md, device_benchmark_10k.csv, one_million_run_metrics.csv, and xyDeng/videos.");
}

function conclusionSlide(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Summary", "What we can claim", "Claim boundary, evidence, and next steps", C.green);

  addBox(slide, { left: 50, top: 113, width: 380, height: 250 });
  label(slide, "1. Main claim", 69, 129, C.green, 120);
  addText(slide, "Both methods learn useful exploration on World 1-1 and reach terminal x_pos=3161 during training.", {
    left: 84,
    top: 180,
    width: 302,
    height: 76,
  }, { fontSize: 18, bold: true, color: C.ink, alignment: "center" });
  addText(slide, "Do not present sampled videos as full-clear proof. They are policy visualizations.", {
    left: 83,
    top: 280,
    width: 303,
    height: 38,
  }, { fontSize: 12, color: C.muted, alignment: "center" });

  addBox(slide, { left: 463, top: 113, width: 370, height: 250 });
  label(slide, "2. Best method", 482, 129, C.green, 120);
  addText(slide, "Use ICM for mean validation progress. Use Disagreement for the farthest saved rollout.", {
    left: 494,
    top: 185,
    width: 300,
    height: 70,
  }, { fontSize: 18, bold: true, color: C.ink, alignment: "center" });
  addText(slide, "Fair wording: the methods are close. One seed is not enough for a strong ranking.", {
    left: 493,
    top: 283,
    width: 302,
    height: 34,
  }, { fontSize: 12, color: C.muted, alignment: "center" });

  addBox(slide, { left: 863, top: 113, width: 370, height: 250 });
  label(slide, "3. Evidence", 884, 129, C.green, 125);
  addText(slide, "Use World 1-1 only:\n- ICM best-sample video\n- Disagreement last-sample20 video\n- 1m_comparison plots", {
    left: 893,
    top: 174,
    width: 304,
    height: 118,
  }, { fontSize: 13, color: C.ink, alignment: "center", typeface: mono });

  addBox(slide, { left: 50, top: 384, width: 1183, height: 250 });
  label(slide, "4. To record a confirmed clear", 69, 400, C.green, 260);
  const items = [
    ["Checkpoint search", "Run more sample episodes on 400k, 800k, best, and last checkpoints. Save the farthest flag_get=1 video."],
    ["Flag logging", "Log episode_flag_get in training and evaluation. Do not rely only on x_pos."],
    ["More seeds", "Use at least 3 seeds to report mean and variance, not one-run luck."],
    ["Longer run", "Continue from a strong 1-1 checkpoint, then search with greedy and sample policies."],
  ];
  let x = 80;
  for (const [title, body] of items) {
    addBox(slide, { left: x, top: 456, width: 260, height: 125 }, "#F8FBFE", C.line, 10);
    addText(slide, title, { left: x + 18, top: 477, width: 224, height: 22 }, {
      fontSize: 17,
      bold: true,
      color: C.green,
      alignment: "center",
    });
    addText(slide, body, { left: x + 20, top: 514, width: 220, height: 52 }, {
      fontSize: 12,
      color: C.ink,
      alignment: "center",
    });
    x += 284;
  }
  footer(slide, "Claim boundary: current saved videos are rollouts, not strict full-clear evidence.");
}

async function main() {
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  const icm = {
    tag: "Method A",
    short: "ICM",
    title: "PPO + ICM",
    color: C.blue,
    deviceLabel: "GPU",
    speed: "CUDA | 8.54 sec/1k",
    bestStep: "900k",
    valMean: "x=1237.2",
    testMean: "x=785.0",
    completionShort: "4 inferred terminal episodes",
    role:
      "Learns inverse and forward dynamics in latent space.\nIntrinsic reward is forward prediction error.\nPPO uses r_total = r_ext + beta*r_ICM.",
    params: [
      ["Item", "Value"],
      ["Intrinsic type", "ICM"],
      ["beta", "0.001"],
      ["latent_dim", "128"],
      ["module lr", "0.0001"],
      ["batch/update", "32 / 1.0"],
    ],
    episodes: "7480",
    bestReward: "3033",
    evidence: "4 episodes reached\nfinal_x=3161",
    evidenceNote: "No flag_get column in the selected run. Completion is inferred from terminal x_pos and high reward.",
    resultTable: [
      ["1M steps", "mean", "best", "max"],
      ["val max_x", "1237.2", "2008", "3161"],
      ["test max_x", "785.0", "1442", "3161"],
    ],
    videoMetric: "best video x=1435",
    videoFile: "xyDeng_ppo_icm_1m_cuda_seed1_1780669071_ppo_step_800000_sample20.mp4",
    bottomLine: "ICM gives stronger mean validation progress, but its saved rollout does not show a confirmed clear.",
  };

  const disagreement = {
    tag: "Method B",
    short: "Disagreement",
    title: "PPO + Disagreement",
    color: C.purple,
    deviceLabel: "CPU",
    speed: "12.01 sec/1k",
    bestStep: "950k",
    valMean: "x=1140.8",
    testMean: "x=784.6",
    completionShort: "1 inferred terminal episode",
    role:
      "Trains an ensemble of forward models.\nIntrinsic reward is prediction disagreement.\nPPO explores states with high model uncertainty.",
    params: [
      ["Item", "Value"],
      ["Intrinsic type", "Disagreement"],
      ["beta", "0.001"],
      ["ensemble_size", "4"],
      ["latent_dim", "128"],
      ["batch/update", "32 / 1.0"],
    ],
    episodes: "7994",
    bestReward: "3020",
    evidence: "1 episode reached\nmax_x=3161",
    evidenceNote: "The selected run gives inferred terminal-progress evidence. The deck uses only World 1-1 results.",
    resultTable: [
      ["1M steps", "mean", "best", "max"],
      ["val max_x", "1140.8", "2021", "3161"],
      ["test max_x", "784.6", "1634", "3161"],
    ],
    videoMetric: "best video x=2009",
    videoFile: "xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739_last_sample20.mp4",
    bottomLine: "Disagreement gives the farthest saved rollout, reaching x=2009 in the recorded videos.",
  };

  await methodWithPlots(presentation, icm);
  await outcomeSlide(presentation, icm);
  await methodWithPlots(presentation, disagreement);
  await outcomeSlide(presentation, disagreement);
  await compareSlide(presentation);
  conclusionSlide(presentation);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(previewDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(previewDir, `${stem}.layout.json`), await layout.text());
  }
  await writeBlob(path.join(previewDir, "montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(outPath);
  console.log(outPath);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
