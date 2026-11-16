/**
 * Pure-JS browser inference engine for poetry_gpt (RoPE + QK-Norm + SwiGLU + KV Cache).
 * Zero external dependencies — runs directly on GitHub Pages static sites.
 */

export class PoetryTokenizerJS {
  constructor(meta) {
    this.specialTokens = meta.special_tokens || ["<PAD>", "<BOS>", "<EOS>"];
    this.itos = [...this.specialTokens, ...meta.chars];
    this.stoi = new Map(this.itos.map((tok, idx) => [tok, idx]));
    this.padId = this.stoi.get("<PAD>");
    this.bosId = this.stoi.get("<BOS>");
    this.eosId = this.stoi.get("<EOS>");
    this.userId = this.stoi.get("<USER>");
    this.botId = this.stoi.get("<BOT>");
    this.vocabSize = this.itos.length;
  }

  encodeChatPrompt(text) {
    const ids = [];
    if (this.userId !== undefined) ids.push(this.userId);
    for (const ch of text) {
      if (this.stoi.has(ch)) ids.push(this.stoi.get(ch));
    }
    if (this.botId !== undefined) ids.push(this.botId);
    return ids;
  }

  decodeToken(id) {
    if (id < 0 || id >= this.vocabSize) return "";
    const tok = this.itos[id];
    if (this.specialTokens.includes(tok)) return "";
    return tok;
  }
}

function dequantizeTensor(buffer, entry) {
  if (entry.dtype === "float32") {
    return new Float32Array(buffer, entry.offset, entry.len / 4);
  }
  const [rows, cols] = entry.shape;
  const qView = new Int8Array(buffer, entry.offset, entry.q_len);
  const scales = new Float32Array(buffer, entry.s_offset, entry.s_len / 4);
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const s = scales[r];
    const base = r * cols;
    for (let c = 0; c < cols; c++) {
      out[base + c] = qView[base + c] * s;
    }
  }
  return out;
}

function rmsNorm(out, x, weight, offset, dim, eps = 1e-5) {
  let sumSq = 0;
  for (let i = 0; i < dim; i++) {
    const v = x[offset + i];
    sumSq += v * v;
  }
  const invRms = 1.0 / Math.sqrt(sumSq / dim + eps);
  for (let i = 0; i < dim; i++) {
    out[i] = x[offset + i] * invRms * weight[i];
  }
}

function matvec(out, x, weight, outDim, inDim) {
  for (let r = 0; r < outDim; r++) {
    let acc = 0;
    const base = r * inDim;
    for (let c = 0; c < inDim; c++) {
      acc += weight[base + c] * x[c];
    }
    out[r] = acc;
  }
}

function silu(x) {
  return x / (1.0 + Math.exp(-x));
}

function gelu(x) {
  return 0.5 * x * (1.0 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

export class PoetryGPTJS {
  constructor(meta, weightsBuffer) {
    this.cfg = meta.config;
    this.tokenizer = new PoetryTokenizerJS(meta.tokenizer);
    this.weights = new Map();
    for (const entry of meta.tensors) {
      this.weights.set(entry.name, dequantizeTensor(weightsBuffer, entry));
    }

    // 预计算 RoPE cos / sin 缓存
    const { block_size: maxSeq, head_dim: hd, rope_theta: theta = 10000.0 } = this.cfg;
    const half = hd >> 1;
    const invFreq = new Float32Array(half);
    for (let i = 0; i < half; i++) {
      invFreq[i] = 1.0 / Math.pow(theta, (2 * i) / hd);
    }
    this.ropeCos = new Float32Array(maxSeq * hd);
    this.ropeSin = new Float32Array(maxSeq * hd);
    for (let p = 0; p < maxSeq; p++) {
      const base = p * hd;
      for (let i = 0; i < half; i++) {
        const ang = p * invFreq[i];
        const c = Math.cos(ang);
        const s = Math.sin(ang);
        this.ropeCos[base + i] = c;
        this.ropeCos[base + half + i] = c;
        this.ropeSin[base + i] = s;
        this.ropeSin[base + half + i] = s;
      }
    }
  }

  static async load(metaUrl, weightsUrl) {
    const [metaRes, binRes] = await Promise.all([fetch(metaUrl), fetch(weightsUrl)]);
    if (!metaRes.ok || !binRes.ok) {
      throw new Error("模型文件加载失败");
    }
    const [meta, buf] = await Promise.all([metaRes.json(), binRes.arrayBuffer()]);
    return new PoetryGPTJS(meta, buf);
  }

  createKVCache() {
    const { n_layer: nLayer, block_size: maxSeq, n_embd: dim } = this.cfg;
    return Array.from({ length: nLayer }, () => ({
      k: new Float32Array(maxSeq * dim),
      v: new Float32Array(maxSeq * dim),
    }));
  }

  forwardStep(tokenId, pos, kvCache) {
    const {
      n_layer: nLayer,
      n_head: nHead,
      head_dim: hd,
      n_embd: dim,
      vocab_size: vocabSize,
      pe_type: peType,
      mlp_type: mlpType,
      qk_norm: useQkNorm,
    } = this.cfg;

    const wte = this.weights.get("wte.weight");
    const x = new Float32Array(dim);
    const wteBase = tokenId * dim;
    for (let i = 0; i < dim; i++) x[i] = wte[wteBase + i];

    if (peType === "learned" && this.weights.has("pos_emb.wpe.weight")) {
      const wpe = this.weights.get("pos_emb.wpe.weight");
      const wpeBase = pos * dim;
      for (let i = 0; i < dim; i++) x[i] += wpe[wpeBase + i];
    }

    const xNorm = new Float32Array(dim);
    const qkv = new Float32Array(3 * dim);
    const attnConcat = new Float32Array(dim);
    const attnOut = new Float32Array(dim);
    const attnScores = new Float32Array(pos + 1);
    const headQ = new Float32Array(hd);
    const headK = new Float32Array(hd);
    const half = hd >> 1;
    const scale = 1.0 / Math.sqrt(hd);

    for (let li = 0; li < nLayer; li++) {
      const p = `blocks.${li}.`;
      rmsNorm(xNorm, x, this.weights.get(p + "norm1.weight"), 0, dim);
      matvec(qkv, xNorm, this.weights.get(p + "attn.qkv_proj.weight"), 3 * dim, dim);

      const qNormW = useQkNorm ? this.weights.get(p + "attn.q_norm.weight") : null;
      const kNormW = useQkNorm ? this.weights.get(p + "attn.k_norm.weight") : null;
      const layerKV = kvCache[li];
      const cacheBase = pos * dim;

      // 逐头执行 QK-Norm -> RoPE -> 存入 KV Cache -> Causal Attention
      for (let h = 0; h < nHead; h++) {
        const hs = h * hd;
        if (useQkNorm) {
          rmsNorm(headQ, qkv, qNormW, hs, hd);
          rmsNorm(headK, qkv, kNormW, dim + hs, hd);
        } else {
          for (let j = 0; j < hd; j++) {
            headQ[j] = qkv[hs + j];
            headK[j] = qkv[dim + hs + j];
          }
        }

        if (peType === "rope") {
          const ropeBase = pos * hd;
          for (let j = 0; j < half; j++) {
            const q0 = headQ[j];
            const q1 = headQ[j + half];
            const c0 = this.ropeCos[ropeBase + j];
            const s0 = this.ropeSin[ropeBase + j];
            headQ[j] = q0 * c0 - q1 * s0;
            headQ[j + half] = q1 * c0 + q0 * s0;

            const k0 = headK[j];
            const k1 = headK[j + half];
            headK[j] = k0 * c0 - k1 * s0;
            headK[j + half] = k1 * c0 + k0 * s0;
          }
        }

        // 写入当前位置的 K / V 到 Cache
        for (let j = 0; j < hd; j++) {
          layerKV.k[cacheBase + hs + j] = headK[j];
          layerKV.v[cacheBase + hs + j] = qkv[2 * dim + hs + j];
        }

        // 计算对 0..pos 所有历史 Key 的注意力分数
        let maxScore = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let dot = 0;
          const kBase = t * dim + hs;
          for (let j = 0; j < hd; j++) {
            dot += headQ[j] * layerKV.k[kBase + j];
          }
          const sc = dot * scale;
          attnScores[t] = sc;
          if (sc > maxScore) maxScore = sc;
        }

        let sumExp = 0;
        for (let t = 0; t <= pos; t++) {
          const e = Math.exp(attnScores[t] - maxScore);
          attnScores[t] = e;
          sumExp += e;
        }
        const invSum = 1.0 / sumExp;
        for (let j = 0; j < hd; j++) {
          let val = 0;
          for (let t = 0; t <= pos; t++) {
            val += attnScores[t] * invSum * layerKV.v[t * dim + hs + j];
          }
          attnConcat[hs + j] = val;
        }
      }

      matvec(attnOut, attnConcat, this.weights.get(p + "attn.out_proj.weight"), dim, dim);
      for (let i = 0; i < dim; i++) x[i] += attnOut[i];

      // MLP 块
      rmsNorm(xNorm, x, this.weights.get(p + "norm2.weight"), 0, dim);
      const mlpOut = attnOut;
      if (mlpType === "swiglu") {
        const wGate = this.weights.get(p + "mlp.w_gate.weight");
        const wUp = this.weights.get(p + "mlp.w_up.weight");
        const wDown = this.weights.get(p + "mlp.w_down.weight");
        const hiddenDim = wGate.length / dim;
        const gateBuf = new Float32Array(hiddenDim);
        const upBuf = new Float32Array(hiddenDim);
        matvec(gateBuf, xNorm, wGate, hiddenDim, dim);
        matvec(upBuf, xNorm, wUp, hiddenDim, dim);
        for (let i = 0; i < hiddenDim; i++) {
          gateBuf[i] = silu(gateBuf[i]) * upBuf[i];
        }
        matvec(mlpOut, gateBuf, wDown, dim, hiddenDim);
      } else {
        const w1 = this.weights.get(p + "mlp.fc1.weight");
        const w2 = this.weights.get(p + "mlp.fc2.weight");
        const hiddenDim = w1.length / dim;
        const hBuf = new Float32Array(hiddenDim);
        matvec(hBuf, xNorm, w1, hiddenDim, dim);
        for (let i = 0; i < hiddenDim; i++) {
          hBuf[i] = mlpType === "relu" ? Math.max(0, hBuf[i]) : gelu(hBuf[i]);
        }
        matvec(mlpOut, hBuf, w2, dim, hiddenDim);
      }
      for (let i = 0; i < dim; i++) x[i] += mlpOut[i];
    }

    rmsNorm(xNorm, x, this.weights.get("norm_f.weight"), 0, dim);
    const logits = new Float32Array(vocabSize);
    matvec(logits, xNorm, wte, vocabSize, dim);
    return logits;
  }

  sampleTopK(logits, temperature = 0.75, topK = 20) {
    const invT = 1.0 / Math.max(1e-4, temperature);
    const n = logits.length;
    const items = [];
    for (let i = 0; i < n; i++) {
      items.push({ id: i, val: logits[i] * invT });
    }
    items.sort((a, b) => b.val - a.val);
    const k = Math.min(topK > 0 ? topK : n, n);
    const maxV = items[0].val;
    let sumExp = 0;
    const probs = new Float32Array(k);
    for (let i = 0; i < k; i++) {
      const e = Math.exp(items[i].val - maxV);
      probs[i] = e;
      sumExp += e;
    }
    let r = Math.random() * sumExp;
    for (let i = 0; i < k; i++) {
      r -= probs[i];
      if (r <= 0) return items[i].id;
    }
    return items[k - 1].id;
  }

  async generateStream(promptText, options = {}, onToken = null) {
    const {
      maxNewTokens = 96,
      temperature = 0.75,
      topK = 20,
      yieldPerToken = true,
    } = options;

    const promptIds = this.tokenizer.encodeChatPrompt(promptText);
    const kvCache = this.createKVCache();
    let logits = null;

    for (let pos = 0; pos < promptIds.length; pos++) {
      logits = this.forwardStep(promptIds[pos], pos, kvCache);
    }

    const outChars = [];
    const maxSteps = Math.min(maxNewTokens, this.cfg.block_size - promptIds.length);
    let curPos = promptIds.length;

    for (let step = 0; step < maxSteps; step++) {
      const nextId = this.sampleTopK(logits, temperature, topK);
      if (nextId === this.tokenizer.eosId) break;
      const ch = this.tokenizer.decodeToken(nextId);
      if (ch) {
        outChars.push(ch);
        if (onToken) onToken(ch, outChars.join(""));
      }
      if (curPos + 1 >= this.cfg.block_size) break;
      logits = this.forwardStep(nextId, curPos, kvCache);
      curPos++;
      if (yieldPerToken) {
        await new Promise((resolve) => setTimeout(resolve, 18));
      }
    }
    return outChars.join("");
  }
}
