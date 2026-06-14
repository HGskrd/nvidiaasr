import { modelAssetUrls } from "./model-config";

const SPIECE_UNDERLINE = "\u2581";

type TokenizerProgress = (progress: { stage: string; detail?: string; level?: "info" | "warn" | "error"; data?: unknown }) => void;

export class NemotronTokenizer {
  private readonly vocab: string[];
  private encodeIndex?: { lookup: Map<string, number>; maxLength: number };

  private constructor(vocab: string[]) {
    this.vocab = vocab;
  }

  private buildEncodeIndex(): { lookup: Map<string, number>; maxLength: number } {
    if (this.encodeIndex) return this.encodeIndex;
    const lookup = new Map<string, number>();
    let maxLength = 1;
    for (let id = 0; id < this.vocab.length; id++) {
      const token = this.vocab[id];
      // Skip special markers like <blank>/<unk>/<bg-BG>; keep real subwords only.
      if (!token || (token.startsWith("<") && token.endsWith(">"))) continue;
      if (!lookup.has(token)) lookup.set(token, id);
      if (token.length > maxLength) maxLength = token.length;
    }
    this.encodeIndex = { lookup, maxLength };
    return this.encodeIndex;
  }

  /**
   * Approximate SentencePiece encoding via greedy longest-match over the vocab.
   * The model ships only vocab.txt (no unigram scores), so this can differ from
   * the canonical segmentation for rare words; it is good enough to seed context
   * biasing, where any plausible subword path the model emits will match.
   */
  encode(text: string): number[] {
    const normalized = SPIECE_UNDERLINE + text.trim().replace(/\s+/g, SPIECE_UNDERLINE);
    if (normalized === SPIECE_UNDERLINE) return [];
    const { lookup, maxLength } = this.buildEncodeIndex();

    const ids: number[] = [];
    let pos = 0;
    while (pos < normalized.length) {
      let matched = -1;
      let matchLength = 0;
      const limit = Math.min(maxLength, normalized.length - pos);
      for (let len = limit; len >= 1; len--) {
        const candidate = lookup.get(normalized.slice(pos, pos + len));
        if (candidate !== undefined) {
          matched = candidate;
          matchLength = len;
          break;
        }
      }
      if (matched === -1) {
        // Unknown character: skip it so a single bad glyph can't abort the word.
        pos += 1;
        continue;
      }
      ids.push(matched);
      pos += matchLength;
    }
    return ids;
  }

  static async fromHuggingFace(onProgress?: TokenizerProgress): Promise<NemotronTokenizer> {
    const started = performance.now();
    onProgress?.({ stage: "tokenizer", detail: "Fetching vocab.txt" });
    let response: Response | undefined;
    let lastError: unknown;
    for (const url of modelAssetUrls("vocab.txt")) {
      try {
        response = await fetch(url);
      } catch (error) {
        lastError = error;
        onProgress?.({ stage: "tokenizer", level: "warn", detail: "vocab.txt fetch failed", data: error });
        continue;
      }

      const contentType = response.headers.get("content-type") ?? "";
      if (response.ok && !contentType.includes("text/html")) break;

      lastError = new Error(`Failed to fetch vocab.txt: ${response.status} ${response.statusText}`);
      response = undefined;
    }
    if (!response) {
      throw lastError instanceof Error ? lastError : new Error("Failed to fetch vocab.txt");
    }
    onProgress?.({
      stage: "tokenizer",
      detail: `vocab.txt response ${response.status} ${response.statusText}`,
      data: {
        elapsedMs: Math.round(performance.now() - started),
        responseType: response.type,
        contentLength: response.headers.get("content-length"),
        contentType: response.headers.get("content-type")
      }
    });
    if (!response.ok) {
      throw new Error(`Failed to fetch vocab.txt: ${response.status} ${response.statusText}`);
    }
    const text = await response.text();
    onProgress?.({
      stage: "tokenizer",
      detail: `Loaded vocab.txt with ${text.length.toLocaleString()} characters`,
      data: { elapsedMs: Math.round(performance.now() - started) }
    });
    return new NemotronTokenizer(text.split(/\r?\n/).filter(Boolean));
  }

  decodeToken(id: number): string {
    const token = this.vocab[id];
    if (!token || token === "<blank>" || token === "<unk>") return "";
    if (token.startsWith("<") && token.endsWith(">")) return `${token} `;
    return token.replaceAll(SPIECE_UNDERLINE, " ");
  }

  decode(ids: number[]): string {
    return ids.map((id) => this.decodeToken(id)).join("").replace(/\s+/g, " ").trim();
  }
}
