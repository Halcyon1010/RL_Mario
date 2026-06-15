import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, Presentation, PresentationFile } from "@oai/artifact-tool";

const repoRoot = path.resolve(process.argv[2] || process.cwd());
const outPath = path.resolve(
  process.argv[3] ||
    path.join(repoRoot, "presentation", "xyDeng_Mario_RL_ICM_Disagreement_1_1.pptx"),
);
const previewDir = path.resolve(
  process.argv[4] || path.join(repoRoot, "presentation", "preview_xyDeng"),
);

const W = 1280;
const H = 720;
const C = {
  bg: "#F4F7FA",
  ink: "#202938",
  muted: "#607084",
  line: "#D9E1EA",
  panel: "#FFFFFF",
  soft: "#EAF2FB",
  blue: "#005CA8",
  purple: "#6243B6",
  green: "#059669",
  red: "#D9483B",
  amber: "#E0A11A",
};

const font = "Microsoft YaHei";
const mono = "Arial";

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function imageBytes(relPath) {
  const filePath = path.join(repoRoot, relPath);
  return fs.readFile(filePath);
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
  const s = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = text;
  s.text.style = {
    typeface: style.typeface || font,
    fontSize: style.fontSize ?? 18,
    color: style.color || C.ink,
    bold: style.bold ?? false,
    alignment: style.alignment,
  };
  return s;
}

function addHeader(slide, tag, title, subtitle, color) {
  slide.background.fill = C.bg;
  const pill = slide.shapes.add({
    geometry: "roundRect",
    position: { left: 40, top: 27, width: 102, height: 31 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
    borderRadius: 5,
    shadow: "shadow-sm",
  });
  pill.text = tag;
  pill.text.style = { typeface: font, fontSize: 15, color: "#FFFFFF", bold: true, alignment: "center" };

  addText(slide, title, { left: 160, top: 20, width: 760, height: 38 }, {
    fontSize: 31,
    bold: true,
    color: C.ink,
  });
  addText(slide, subtitle, { left: 160, top: 64, width: 740, height: 22 }, {
    fontSize: 14,
    color: C.muted,
  });

  const note = addBox(slide, { left: 1012, top: 22, width: 224, height: 53 }, "#EFF5FB", "#D7E2EF", 10);
  note.shadow = "shadow-sm";
  note.text = "Mario RL / PPO intrinsic rewards\n1M steps · seed 1 · 1-1 only";
  note.text.style = { typeface: font, fontSize: 10, color: C.muted, alignment: "center" };

  slide.shapes.add({
    geometry: "rect",
    position: { left: 40, top: 92, width: 1198, height: 1.2 },
    fill: "#DDE6EF",
    line: { style: "solid", fill: "#DDE6EF", width: 0 },
  });
}

function sectionLabel(slide, text, x, y, color, width = 150) {
  const p = slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width, height: 30 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
    borderRadius: 5,
    shadow: "shadow-sm",
  });
  p.text = text;
  p.text.style = { typeface: font, fontSize: 14, bold: true, color: "#FFFFFF", alignment: "center" };
  return p;
}

function footer(slide, text) {
  addText(slide, text, { left: 252, top: 690, width: 820, height: 16 }, {
    fontSize: 9,
    color: C.muted,
    alignment: "center",
  });
}

function table(slide, values, x, y, w, h, color) {
  const t = slide.tables.add({ rows: values.length, columns: values[0].length, left: x, top: y, width: w, height: h, values });
  t.styleOptions = { headerRow: true, bandedRows: false };
  t.borders.assign({ style: "solid", fill: "#111827", width: 0.8 });
  for (let r = 0; r < values.length; r++) {
    for (let c = 0; c < values[0].length; c++) {
      const cell = t.getCell(r, c);
      cell.text.style = {
        typeface: r === 0 ? font : mono,
        fontSize: r === 0 ? 10 : 9,
        bold: r === 0,
        color: C.ink,
        alignment: "center",
      };
      if (r === 0) cell.fill = "#E8EEF5";
      if (c === 0 && r > 0) cell.text.style = { typeface: font, fontSize: 9, color: C.ink };
    }
  }
  return t;
}

function metric(slide, label, value, sub, x, y, w, color) {
  addBox(slide, { left: x, top: y, width: w, height: 66 }, "#FFFFFF", C.line, 8);
  addText(slide, label, { left: x + 12, top: y + 8, width: w - 20, height: 14 }, { fontSize: 10, color: C.muted });
  addText(slide, value, { left: x + 12, top: y + 27, width: w - 20, height: 26 }, {
    fontSize: w < 85 ? 16 : w < 100 ? 18 : 21,
    bold: true,
    color,
    typeface: mono,
  });
  if (sub) addText(slide, sub, { left: x + 12, top: y + 52, width: w - 20, height: 12 }, { fontSize: 8, color: C.muted });
}

function progress(slide, x, y, w, ratio, color, label) {
  addBox(slide, { left: x, top: y, width: w, height: 67 }, "#F8FBFE", C.line, 8);
  addText(slide, "通关 / 进度", { left: x + 16, top: y + 9, width: 120, height: 17 }, { fontSize: 11, bold: true, color: C.muted });
  addText(slide, label, { left: x + 124, top: y + 8, width: w - 140, height: 22 }, {
    fontSize: 16,
    bold: true,
    color,
    alignment: "center",
  });
  addText(slide, `max_x = ${Math.round(ratio * 3161)} / 3161`, { left: x + 120, top: y + 32, width: w - 160, height: 12 }, {
    fontSize: 8,
    color: C.muted,
    alignment: "center",
  });
  slide.shapes.add({
    geometry: "roundRect",
    position: { left: x + 14, top: y + 45, width: w - 28, height: 9 },
    fill: "#DDE6EF",
    line: { style: "solid", fill: "#DDE6EF", width: 0 },
    borderRadius: 4,
  });
  slide.shapes.add({
    geometry: "roundRect",
    position: { left: x + 14, top: y + 45, width: Math.max(8, (w - 28) * ratio), height: 9 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
    borderRadius: 4,
  });
}

function flowNode(slide, text, x, y, w, h, fill = "#E7F1FB") {
  const n = addBox(slide, { left: x, top: y, width: w, height: h }, fill, "#CAD8E8", 8);
  n.shadow = "shadow-sm";
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
  addHeader(slide, method.tag, method.title, "方法机制、训练设置与内在奖励接入方式", method.color);

  addBox(slide, { left: 50, top: 113, width: 389, height: 211 });
  sectionLabel(slide, "1. 方法定位", 69, 129, method.color, 120);
  addText(slide, method.description, { left: 82, top: 173, width: 320, height: 88 }, {
    fontSize: 13,
    alignment: "center",
  });
  addText(slide, "共同 backbone: PPO + CNN Actor-Critic；方法差异集中在 intrinsic reward module.", {
    left: 78, top: 282, width: 325, height: 28,
  }, { fontSize: 11, color: C.muted, alignment: "center" });

  addBox(slide, { left: 463, top: 113, width: 320, height: 211 });
  sectionLabel(slide, "2. 关键超参数", 482, 129, method.color, 120);
  table(slide, method.params, 482, 166, 281, 136, method.color);

  addBox(slide, { left: 809, top: 113, width: 424, height: 211 });
  sectionLabel(slide, "3. 实验协议", 828, 129, method.color, 120);
  table(slide, [
    ["项", "值"],
    ["训练环境", "SuperMarioBros-1-1-v0"],
    ["输入", "4-frame stack, 84x84"],
    ["动作", "SIMPLE_MOVEMENT, 7 actions"],
    ["训练步数", "1,000,000 timesteps"],
    ["评估", "sample / greedy; best by sample validation"],
    ["seed", "1"],
  ], 828, 166, 382, 136, method.color);

  addBox(slide, { left: 50, top: 344, width: 523, height: 255 });
  addText(slide, "训练数据流", { left: 70, top: 360, width: 160, height: 22 }, { fontSize: 14, bold: true, color: method.color });
  flowNode(slide, "Observation\n4×84×84", 69, 400, 130, 53);
  flowNode(slide, "CNN\nActor-Critic", 228, 400, 137, 53);
  flowNode(slide, "Policy / Value\nPPO update", 396, 400, 145, 53);
  flowNode(slide, `${method.short}\nintrinsic module`, 228, 493, 137, 53, "#EFFCF7");
  flowNode(slide, "r_total = r_ext\n+ β·r_int", 396, 493, 145, 53, "#FFF7EB");
  line(slide, 199, 426, 228, 426);
  line(slide, 365, 426, 396, 426);
  line(slide, 466, 453, 466, 493);
  line(slide, 199, 474, 228, 517);
  line(slide, 365, 519, 396, 519);

  addBox(slide, { left: 597, top: 344, width: 636, height: 255 });
  sectionLabel(slide, "4. 训练观测", 616, 360, method.color, 120);
  return slide;
}

async function methodWithPlots(presentation, method, plotA, plotB) {
  const slide = methodSlide(presentation, method);
  slide.images.add({
    blob: await imageBytes(plotA),
    contentType: "image/png",
    alt: `${method.short} validation max_x plot`,
    fit: "contain",
    position: { left: 624, top: 407, width: 350, height: 118 },
  });
  slide.images.add({
    blob: await imageBytes(plotB),
    contentType: "image/png",
    alt: `${method.short} intrinsic reward plot`,
    fit: "contain",
    position: { left: 993, top: 408, width: 210, height: 118 },
  });
  metric(slide, "validation sample", method.valMetric, `best step ${method.bestStep}`, 616, 545, 156, method.color);
  metric(slide, "test sample", method.testMetric, "5-episode sample mean", 782, 545, 156, method.color);
  metric(slide, "train terminal", "x=3161", method.completionShort, 948, 545, 166, method.color);
  metric(slide, "device", method.device, method.speed, 1124, 545, 90, method.color);
  footer(slide, "Evidence: xyDeng 1M selected 1-1 runs; metrics from best_summary.json, one_million_run_metrics.csv, and training_completion_audit.md.");
}

async function outcomeSlide(presentation, method, plot, secondaryPlot) {
  const slide = presentation.slides.add();
  addHeader(slide, method.tag, `${method.title}: 1-1 sample outcomes`, "通关/进度、total_reward 与 max_x_pos 观察", method.color);

  addBox(slide, { left: 50, top: 108, width: 1181, height: 57 });
  addText(slide, "评估口径", { left: 76, top: 126, width: 72, height: 18 }, { fontSize: 13, bold: true, color: method.color });
  addText(slide, "训练与验证均限定 SuperMarioBros-1-1-v0；sample 模式用于 best checkpoint；视频为 sample20 / best-sample 搜索结果。", {
    left: 152, top: 124, width: 890, height: 22,
  }, { fontSize: 12, color: C.ink });

  addBox(slide, { left: 50, top: 184, width: 376, height: 214 });
  sectionLabel(slide, "1-1 training evidence", 69, 198, method.color, 182);
  addText(slide, method.evidenceText, { left: 79, top: 244, width: 310, height: 70 }, {
    fontSize: 16,
    bold: true,
    color: method.color,
    alignment: "center",
  });
  addText(slide, method.evidenceSub, { left: 79, top: 325, width: 315, height: 42 }, {
    fontSize: 12,
    color: C.muted,
    alignment: "center",
  });

  metric(slide, "ep", method.episodes, "training episodes", 445, 184, 84, method.color);
  metric(slide, "R", method.bestReward, "best training reward", 536, 184, 90, method.color);
  metric(slide, "x", "3161", "max x_pos", 633, 184, 90, method.color);
  metric(slide, "reward vs progress", method.sync, method.syncSub, 730, 184, 114, method.color);
  progress(slide, 445, 254, 399, 1, method.color, "达到 x=3161");

  table(slide, method.resultTable, 444, 331, 399, 58, method.color);

  addBox(slide, { left: 50, top: 430, width: 376, height: 214 });
  sectionLabel(slide, "best recorded video", 69, 443, method.color, 168);
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
  addText(slide, "录制视频未观察到完整通关；训练日志提供终点 x 证据。", {
    left: 79, top: 596, width: 310, height: 28,
  }, { fontSize: 11, color: C.muted, alignment: "center" });

  addBox(slide, { left: 865, top: 184, width: 365, height: 460 });
  sectionLabel(slide, "Reward–progress consistency", 884, 198, method.color, 250);
  slide.images.add({
    blob: await imageBytes(plot),
    contentType: "image/png",
    alt: `${method.short} train max_x plot`,
    fit: "contain",
    position: { left: 893, top: 244, width: 303, height: 170 },
  });
  slide.images.add({
    blob: await imageBytes(secondaryPlot),
    contentType: "image/png",
    alt: `${method.short} total reward plot`,
    fit: "contain",
    position: { left: 893, top: 449, width: 303, height: 170 },
  });

  addBox(slide, { left: 49, top: 654, width: 1182, height: 28 }, "#EAF2FB", "#D7E2EF", 6);
  addText(slide, method.bottomLine, { left: 120, top: 660, width: 1030, height: 16 }, {
    fontSize: 11,
    color: C.ink,
    alignment: "center",
  });
  footer(slide, "Evidence: final_selected_runs_1_1 plus xyDeng/videos; training completion is reported conservatively from x_pos unless flag_get is logged.");
}

async function compareSlide(presentation) {
  const slide = presentation.slides.add();
  addHeader(slide, "对比", "ICM vs Disagreement: 1-1 results", "同一 PPO backbone、同一 1M steps、同一 SuperMarioBros-1-1-v0", C.green);

  addBox(slide, { left: 50, top: 113, width: 384, height: 520 });
  sectionLabel(slide, "核心指标", 69, 129, C.green, 100);
  table(slide, [
    ["Metric", "PPO+ICM", "PPO+Disagreement"],
    ["Device", "CUDA", "CPU"],
    ["Episodes", "7480", "7994"],
    ["Best step", "900k", "950k"],
    ["Val mean max_x", "1237.2", "1140.8"],
    ["Test mean max_x", "785.0", "784.6"],
    ["Train max_x", "3161", "3161"],
    ["Best video max_x", "1435", "2009"],
  ], 72, 174, 338, 286, C.green);
  addText(slide, "结论：ICM 的 validation mean max_x 略高；Disagreement 的 best recorded rollout 更远。两者训练中均到达 terminal x=3161。", {
    left: 76, top: 488, width: 316, height: 88,
  }, { fontSize: 15, color: C.ink, alignment: "center" });

  addBox(slide, { left: 463, top: 113, width: 770, height: 250 });
  sectionLabel(slide, "曲线证据", 482, 129, C.green, 100);
  slide.images.add({
    blob: await imageBytes("xyDeng/plots/1m_comparison/val_max_x_pos.png"),
    contentType: "image/png",
    alt: "Validation max x comparison",
    fit: "contain",
    position: { left: 486, top: 169, width: 350, height: 166 },
  });
  slide.images.add({
    blob: await imageBytes("xyDeng/plots/1m_comparison/test_max_x_pos.png"),
    contentType: "image/png",
    alt: "Test max x comparison",
    fit: "contain",
    position: { left: 854, top: 169, width: 350, height: 166 },
  });

  addBox(slide, { left: 463, top: 383, width: 370, height: 250 });
  sectionLabel(slide, "训练速度", 482, 399, C.green, 100);
  table(slide, [
    ["Method", "CPU sec/1k", "CUDA sec/1k", "Used"],
    ["ICM", "11.88", "8.54", "CUDA"],
    ["Disagreement", "12.01", "13.43", "CPU"],
  ], 486, 443, 322, 92, C.green);
  addText(slide, "设备选择来自 10k-step benchmark：ICM 在 CUDA 更快，Disagreement 在 CPU 更快。", {
    left: 493, top: 555, width: 300, height: 40,
  }, { fontSize: 12, color: C.muted, alignment: "center" });

  addBox(slide, { left: 862, top: 383, width: 371, height: 250 });
  sectionLabel(slide, "公正比较", 884, 399, C.green, 112);
  addText(slide, "ICM", { left: 900, top: 448, width: 120, height: 22 }, { fontSize: 20, bold: true, color: C.blue });
  addText(slide, "Forward model prediction error gives dense curiosity; in this run it produced slightly stronger validation progress.", {
    left: 900, top: 476, width: 290, height: 48,
  }, { fontSize: 12, color: C.ink });
  addText(slide, "Disagreement", { left: 900, top: 536, width: 170, height: 22 }, { fontSize: 20, bold: true, color: C.purple });
  addText(slide, "Ensemble uncertainty encouraged longer sampled rollouts; best recorded video reached max_x=2009.", {
    left: 900, top: 564, width: 290, height: 48,
  }, { fontSize: 12, color: C.ink });

  footer(slide, "Evidence: xyDeng/notes/run_summary.md, device_benchmark_10k.csv, one_million_run_metrics.csv, and videos under xyDeng/videos.");
}

async function conclusionSlide(presentation) {
  const slide = presentation.slides.add();
  addHeader(slide, "结论", "What we can claim for the report", "结果边界、展示素材与后续训练方向", C.green);

  addBox(slide, { left: 50, top: 113, width: 380, height: 250 });
  sectionLabel(slide, "1. 可确认结论", 69, 129, C.green, 120);
  addText(slide, "Both methods learned useful exploration behavior on World 1-1 and reached terminal x_pos=3161 during training.", {
    left: 84, top: 177, width: 302, height: 74,
  }, { fontSize: 18, bold: true, color: C.ink, alignment: "center" });
  addText(slide, "PPT 中不要把 sampled video 说成通关视频；它们是展示策略行为的可视化证据。", {
    left: 83, top: 275, width: 303, height: 38,
  }, { fontSize: 12, color: C.muted, alignment: "center" });

  addBox(slide, { left: 463, top: 113, width: 370, height: 250 });
  sectionLabel(slide, "2. 哪个更好", 482, 129, C.green, 110);
  addText(slide, "If the report emphasizes mean validation progress, choose ICM. If it emphasizes best demonstrable rollout video, choose Disagreement.", {
    left: 494, top: 177, width: 300, height: 86,
  }, { fontSize: 17, bold: true, color: C.ink, alignment: "center" });
  addText(slide, "公正表述：两者接近，优势维度不同；单 seed 不足以做强排名。", {
    left: 493, top: 283, width: 302, height: 34,
  }, { fontSize: 12, color: C.muted, alignment: "center" });

  addBox(slide, { left: 863, top: 113, width: 370, height: 250 });
  sectionLabel(slide, "3. 展示素材", 884, 129, C.green, 125);
  addText(slide, "Use 1-1 only:\n• xyDeng_ppo_icm_1m_cuda_best_sample.mp4\n• xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739_last_sample20.mp4\n• 1m_comparison plots", {
    left: 893, top: 174, width: 304, height: 118,
  }, { fontSize: 13, color: C.ink, alignment: "center", typeface: mono });

  addBox(slide, { left: 50, top: 384, width: 1183, height: 250 });
  sectionLabel(slide, "4. 后续若要录到明确通关", 69, 400, C.green, 260);
  const items = [
    ["Checkpoint search", "对 400k/800k/best/last 增加 sample episodes，保存最远且 flag_get=1 的视频。"],
    ["Flag logging", "训练与评估都记录 episode_flag_get，避免只靠 x_pos 推断。"],
    ["More seeds", "至少 3 seeds 比较均值和方差，避免单次运行偶然性。"],
    ["Longer / staged run", "在 1-1 上继续训练或从较好 checkpoint resume，再用 greedy+sample 搜索通关视频。"],
  ];
  let x = 80;
  for (const [title, body] of items) {
    addBox(slide, { left: x, top: 456, width: 260, height: 125 }, "#F8FBFE", C.line, 10);
    addText(slide, title, { left: x + 18, top: 477, width: 224, height: 22 }, { fontSize: 17, bold: true, color: C.green, alignment: "center" });
    addText(slide, body, { left: x + 20, top: 514, width: 220, height: 45 }, { fontSize: 12, color: C.ink, alignment: "center" });
    x += 284;
  }
  footer(slide, "Claim boundary: current selected 1-1 videos are visual rollouts, not strict full-clear evidence; training logs provide terminal-progress evidence.");
}

async function main() {
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  const icm = {
    tag: "方法 A",
    short: "ICM",
    title: "PPO + ICM",
    color: C.blue,
    device: "GPU",
    speed: "CUDA · 8.54 sec / 1k",
    bestStep: "900k",
    valMetric: "x=1237.2",
    testMetric: "x=785.0",
    completionShort: "4 inferred terminal episodes",
    description:
      "Learns inverse and forward dynamics in latent space.\nCuriosity reward is the forward prediction error.\nPPO receives r_total = r_ext + β·r_ICM.",
    params: [
      ["项", "值"],
      ["Intrinsic type", "ICM"],
      ["beta", "0.001"],
      ["latent_dim", "128"],
      ["module lr", "0.0001"],
      ["batch/update", "32 / 1.0"],
    ],
    episodes: "7480",
    bestReward: "3033",
    sync: "不同步",
    syncSub: "video best < train terminal",
    evidenceText: "4 episodes reached\nfinal_x=3161",
    evidenceSub: "Original selected run has no flag_get column; completion is inferred from terminal x_pos and high reward.",
    resultTable: [
      ["1000000 steps", "mean", "best", "max"],
      ["val max_x", "1237.2", "2008", "3161"],
      ["test max_x", "785.0", "1442", "3161"],
    ],
    videoMetric: "best video x=1435",
    videoFile: "xyDeng_ppo_icm_1m_cuda_seed1_1780669071_ppo_step_800000_sample20.mp4",
    bottomLine: "ICM has stronger mean validation progress in the selected run, but the best sampled video did not visibly complete 1-1.",
  };

  const dis = {
    tag: "方法 B",
    short: "Disagreement",
    title: "PPO + Disagreement",
    color: C.purple,
    device: "CPU",
    speed: "12.01 sec / 1k",
    bestStep: "950k",
    valMetric: "x=1140.8",
    testMetric: "x=784.6",
    completionShort: "1 inferred terminal episode",
    description:
      "Trains an ensemble of forward models.\nIntrinsic reward is prediction variance / disagreement.\nPPO is pushed toward states with high model uncertainty.",
    params: [
      ["项", "值"],
      ["Intrinsic type", "Disagreement"],
      ["beta", "0.001"],
      ["ensemble_size", "4"],
      ["latent_dim", "128"],
      ["batch/update", "32 / 1.0"],
    ],
    episodes: "7994",
    bestReward: "3020",
    sync: "不同步",
    syncSub: "best video from last.pt",
    evidenceText: "1 episode reached\nmax_x=3161",
    evidenceSub: "Original selected 1-1 run has inferred completion evidence; broader notes include later flag logging but this PPT uses selected 1-1 evidence.",
    resultTable: [
      ["1000000 steps", "mean", "best", "max"],
      ["val max_x", "1140.8", "2021", "3161"],
      ["test max_x", "784.6", "1634", "3161"],
    ],
    videoMetric: "best video x=2009",
    videoFile: "xyDeng_ppo_disagreement_1m_cpu_seed1_1780679739_last_sample20.mp4",
    bottomLine: "Disagreement produced the strongest recorded sample rollout, reaching x=2009 in the saved videos.",
  };

  await methodWithPlots(presentation, icm, "xyDeng/plots/1m_comparison/val_max_x_pos.png", "xyDeng/plots/1m_comparison/train_int_reward.png");
  await outcomeSlide(presentation, icm, "xyDeng/plots/1m_comparison/train_max_x_pos.png", "xyDeng/plots/1m_comparison/train_total_reward.png");
  await methodWithPlots(presentation, dis, "xyDeng/plots/1m_comparison/val_max_x_pos.png", "xyDeng/plots/1m_comparison/train_int_reward.png");
  await outcomeSlide(presentation, dis, "xyDeng/plots/1m_comparison/train_max_x_pos.png", "xyDeng/plots/1m_comparison/train_total_reward.png");
  await compareSlide(presentation);
  await conclusionSlide(presentation);

  for (const [idx, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(idx + 1).padStart(2, "0")}`;
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
