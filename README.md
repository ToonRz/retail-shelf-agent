# retail-shelf-agent

ตัวตรวจจับสินค้าบนชั้นวาง (YOLOv8n, 110 คลาส) ที่ **รันใน browser** ด้วย onnxruntime-web
ออกเป็น GitHub Release แล้ว [retail-shelf-web](https://github.com/ToonRz/retail-shelf-web) pin เวอร์ชันด้วย `agent.version` และ bump มาใช้
(แบบเดียวกับ zcrX ↔ zcrAgentX)

- Spec: [retail-shelf-detection/docs/agent-connection/SPEC.md](https://github.com/ToonRz/retail-shelf-detection/blob/main/docs/agent-connection/SPEC.md)
- งานที่เหลือและเจ้าของงาน: [TASKS.md](https://github.com/ToonRz/retail-shelf-detection/blob/main/docs/agent-connection/TASKS.md)
- Model มาจาก [retail-shelf-detection](https://github.com/ToonRz/retail-shelf-detection) (`src/export_agent_assets.py`)

## API (contract `shelf-agent-js/1`)

```js
import { load } from "/agent/v0.1.0/shelf-agent.js";
const agent = await load({ baseUrl: "/agent/v0.1.0/", expectedVersion: "v0.1.0", backend: "webgpu" });
const r = await agent.detect(imageBitmap);   // { boxes: [{ cls, classId, conf, xyxy }], latency_ms, ... }
```

Types อยู่ใน [`src/types.ts`](src/types.ts) · `load()` ปฏิเสธ (throw `AgentError`) ถ้าเวอร์ชันไม่ตรง pin, contract ไม่ตรง,
sha256 ของ model ไม่ตรง manifest หรือ labels ไม่ครบ

## สถานะ

| ส่วน | สถานะ |
|---|---|
| types, `parseManifest`, `load()` (manifest + sha256 + labels) | ✅ มี test |
| `scripts/package-release.mjs` (โฟลเดอร์ release + `manifest.json` + `SHA256SUMS`) | ✅ มี test |
| `scripts/fetch-model.mjs` + `model.lock.json` | ✅ ตอนนี้ชี้ model COCO สำหรับ dev |
| `createSession` / `detect` (`src/detect.ts`) | ⏳ งาน A |
| release workflow ตอน push tag | ⏳ งาน C1 |
| model จริง (sell-products v3, 110 คลาส) + fixture `sell-v3` | ⛔ งาน T2 รอ `best.pt` จาก Google Drive (T1) |
| release แรก `v0.1.0` | ⛔ งาน T4 รอ A, C1 และ T2 |

## Dev

```bash
npm ci
npm run fetch-model     # models/ ตาม model.lock.json (ตอนนี้ COCO yolov8n, dev only)
npm test                # vitest
npm run typecheck
npm run build           # dist/shelf-agent.js
npm run package -- --version v0.0.1   # release/v0.0.1/ (ต้อง build + fetch-model ก่อน)
```

`test/fixtures/coco-dev/` = รูป + `expected.json` ที่ Ultralytics ทายด้วย **model.onnx ตัวเดียวกับใน `model.lock.json`**
ใช้เป็นเกณฑ์ parity ของ `detect()` (งาน A4) จนกว่าจะมี fixture ของ model จริง

## License / ที่มา

Model ได้มาจาก Ultralytics YOLOv8 (AGPL-3.0) และ dataset Sell Products – O2O Minimart (CC BY 4.0)
รูป `bus.jpg` ใน fixture มาจาก Ultralytics assets
