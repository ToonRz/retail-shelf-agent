#!/usr/bin/env node
// Assemble one release folder: the exact set of files a GitHub Release carries.
//
//   node scripts/package-release.mjs --version v0.1.0 [--model-dir models] [--dist dist]
//        [--ort node_modules/onnxruntime-web/dist] [--out release]
//
// models/ must hold what retail-shelf-detection's src/export_agent_assets.py wrote:
// model.onnx, labels.json, model.json. The output folder is flat (GitHub release assets
// have no directories) and ends with manifest.json and SHA256SUMS (sha256sum -c format).
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

export const CONTRACT = "shelf-agent-js/1";
export const ORT_FILE = /^ort-wasm-simd-threaded\.jsep\.(wasm|mjs)$/;
export const DEFAULTS = { conf: 0.25, iou: 0.7, max_det: 300 }; // = src/infer.py / Ultralytics predict

const sha = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");

function readModelMeta(dir) {
  for (const f of ["model.onnx", "labels.json", "model.json"]) {
    const p = join(dir, f);
    if (!existsSync(p)) throw new Error(`missing ${p}`);
  }
  const info = JSON.parse(readFileSync(join(dir, "model.json"), "utf8"));
  const labels = JSON.parse(readFileSync(join(dir, "labels.json"), "utf8"));
  if (!Array.isArray(labels) || labels.length !== info.classes) {
    throw new Error(`labels.json has ${labels.length} names, model.json says ${info.classes} classes`);
  }
  return info;
}

export function packageRelease({ version, modelDir = "models", shelfModelDir, dist = "dist", ort = "node_modules/onnxruntime-web/dist", out = "release" }) {
  if (!/^v\d+\.\d+\.\d+$/.test(version ?? "")) throw new Error(`--version must be vX.Y.Z, got ${version}`);
  if (!existsSync(join(dist, "shelf-agent.js"))) throw new Error(`missing ${join(dist, "shelf-agent.js")}`);
  const info = readModelMeta(modelDir);
  const shelfInfo = shelfModelDir ? readModelMeta(shelfModelDir) : null;

  // Only the JSEP build: one .wasm that runs both the WebGPU and the WASM execution providers
  // (what the default `onnxruntime-web` import loads). The other 3 variants would add ~60 MB.
  const ortFiles = existsSync(ort) ? readdirSync(ort).filter((f) => ORT_FILE.test(f)) : [];
  if (!ortFiles.some((f) => f.endsWith(".wasm"))) throw new Error(`no ort-wasm-simd-threaded.jsep.wasm in ${ort}`);

  const dir = join(out, version);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  copyFileSync(join(dist, "shelf-agent.js"), join(dir, "shelf-agent.js"));
  copyFileSync(join(modelDir, "model.onnx"), join(dir, "model.onnx"));
  copyFileSync(join(modelDir, "labels.json"), join(dir, "labels.json"));
  if (shelfModelDir) {
    copyFileSync(join(shelfModelDir, "model.onnx"), join(dir, "shelf-model.onnx"));
    copyFileSync(join(shelfModelDir, "labels.json"), join(dir, "shelf-labels.json"));
  }
  for (const f of ortFiles) copyFileSync(join(ort, f), join(dir, f));

  const files = {};
  for (const f of readdirSync(dir).sort()) files[f] = sha(join(dir, f));
  const manifest = {
    version,
    contract: CONTRACT,
    entry: "shelf-agent.js",
    model: {
      file: "model.onnx",
      labels: "labels.json",
      name: info.name,
      dataset: shelfInfo ? `${info.dataset} + ${shelfInfo.dataset}` : info.dataset,
      classes: info.classes,
      imgsz: info.imgsz,
      ...DEFAULTS,
      sha256: files["model.onnx"],
    },
    ...(shelfInfo
      ? {
          shelfModel: {
            file: "shelf-model.onnx",
            labels: "shelf-labels.json",
            name: shelfInfo.name,
            dataset: shelfInfo.dataset,
            classes: shelfInfo.classes,
            imgsz: shelfInfo.imgsz,
            ...DEFAULTS,
            sha256: files["shelf-model.onnx"],
          },
        }
      : {}),
    files,
  };
  writeFileSync(join(dir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

  const sums = readdirSync(dir).filter((f) => f !== "SHA256SUMS").sort()
    .map((f) => `${sha(join(dir, f))}  ${f}`).join("\n");
  writeFileSync(join(dir, "SHA256SUMS"), `${sums}\n`);
  return { dir, manifest };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { values } = parseArgs({
    options: {
      version: { type: "string" },
      "model-dir": { type: "string" },
      "shelf-model-dir": { type: "string" },
      dist: { type: "string" },
      ort: { type: "string" },
      out: { type: "string" },
    },
  });
  const shelfModelDir = values["shelf-model-dir"]
    ?? (!values["model-dir"] && existsSync(join("models/shelf", "model.onnx")) ? "models/shelf" : undefined);
  try {
    const { dir, manifest } = packageRelease({
      version: values.version,
      modelDir: values["model-dir"],
      shelfModelDir,
      dist: values.dist,
      ort: values.ort,
      out: values.out,
    });
    console.log(`${dir}: ${Object.keys(manifest.files).length} files + manifest.json + SHA256SUMS`);
  } catch (e) {
    console.error(`package-release: ${e.message}`);
    process.exit(1);
  }
}
