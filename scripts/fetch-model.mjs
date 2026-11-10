#!/usr/bin/env node
// Download the model named in model.lock.json into models/ and verify every sha256.
//   node scripts/fetch-model.mjs
// The lock points at a GitHub release that holds model.onnx, labels.json, model.json
// (written by retail-shelf-detection's src/export_agent_assets.py). Switching to a new
// model = upload it there, then update model.lock.json in a PR.
//   node scripts/fetch-model.mjs --lock test/fixtures/coco-dev/model.lock.json --out models/coco-dev
// fetches another lock (the COCO dev model the coco-dev parity fixtures were made with).
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({ options: { lock: { type: "string", default: "model.lock.json" }, out: { type: "string", default: "models" } } });
const lock = JSON.parse(readFileSync(args.lock, "utf8"));
mkdirSync(args.out, { recursive: true });
const sha = (buf) => createHash("sha256").update(buf).digest("hex");

for (const [name, want] of Object.entries(lock.files)) {
  const dst = `${args.out}/${name}`;
  if (existsSync(dst) && sha(readFileSync(dst)) === want) {
    console.log(`ok (cached) ${dst}`);
    continue;
  }
  const res = await fetch(lock.source + name);
  if (!res.ok) {
    console.error(`fetch-model: ${lock.source + name}: HTTP ${res.status}`);
    process.exit(1);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  if (sha(buf) !== want) {
    console.error(`fetch-model: ${name} sha256 ${sha(buf)} != lock ${want}`);
    process.exit(1);
  }
  writeFileSync(dst, buf);
  console.log(`ok ${dst} (${buf.length} bytes)`);
}
